"""Tests for shadow-mode capture (src/firefly/shadow.py).

Covers each of the six MVP layers:
1. Custom op pass-through semantics
2. Tapper records when active, no-ops when inactive
3. Tap-name regex filter
4. FullTensorPolicy decisions
5. Buffer + drain thread + LocalLogSink end-to-end
6. aggregate() distribution rollup
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from firefly import shadow

# ---------------------------------------------------------------------------
# Custom op
# ---------------------------------------------------------------------------


def test_capture_is_pass_through() -> None:
    x = torch.randn(2, 4)
    y = torch.ops.firefly.capture(x, "any.name")
    assert torch.equal(x, y)


def test_capture_with_no_active_tapper_is_noop() -> None:
    """No buffer is touched if no Tapper is active in the current thread."""
    x = torch.randn(2, 4)
    # No Tapper context here — call should still succeed.
    y = torch.ops.firefly.capture(x, "any.name")
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Tapper
# ---------------------------------------------------------------------------


def test_tapper_records_into_buffer() -> None:
    buf = shadow._Buffer(max_size=100)
    with shadow.Tapper(buf):
        torch.ops.firefly.capture(torch.ones(2, 3), "layer.0.mlp")
    event = buf.pop(timeout=0.01)
    assert event is not None
    assert event.tap_name == "layer.0.mlp"
    assert event.stats["shape"] == [2, 3]
    assert event.stats["abs_mean"] == pytest.approx(1.0)


def test_tapper_filters_by_regex() -> None:
    buf = shadow._Buffer(max_size=100)
    with shadow.Tapper(buf, taps=r"layer\.7\.self_attn"):
        torch.ops.firefly.capture(torch.ones(1), "layer.0.mlp")
        torch.ops.firefly.capture(torch.ones(1), "layer.7.self_attn")
        torch.ops.firefly.capture(torch.ones(1), "final_norm")

    # Drain everything from the buffer.
    events: list = []
    while True:
        e = buf.pop(timeout=0.01)
        if e is None:
            break
        events.append(e)
    assert len(events) == 1
    assert events[0].tap_name == "layer.7.self_attn"


def test_tapper_is_not_reentrant() -> None:
    buf = shadow._Buffer(max_size=100)
    with (
        shadow.Tapper(buf),
        pytest.raises(RuntimeError, match="not re-entrant"),
        shadow.Tapper(buf),  # this entry should raise
    ):
        pass


def test_tapper_clears_thread_local_on_exit() -> None:
    buf = shadow._Buffer(max_size=100)
    with shadow.Tapper(buf):
        assert shadow._active_tapper() is not None
    assert shadow._active_tapper() is None


# ---------------------------------------------------------------------------
# FullTensorPolicy
# ---------------------------------------------------------------------------


def test_full_tensor_policy_first_n_steps() -> None:
    p = shadow.FullTensorPolicy(first_n_steps=2)
    assert p.should_record_full(0) is True
    assert p.should_record_full(1) is True
    assert p.should_record_full(2) is False
    assert p.should_record_full(100) is False


def test_full_tensor_policy_every_n_steps() -> None:
    p = shadow.FullTensorPolicy(every_n_steps=100)
    assert p.should_record_full(0) is True  # 0 % 100 == 0
    assert p.should_record_full(99) is False
    assert p.should_record_full(100) is True
    assert p.should_record_full(200) is True


def test_full_tensor_policy_compose() -> None:
    """first_n_steps OR every_n_steps both fire."""
    p = shadow.FullTensorPolicy(first_n_steps=2, every_n_steps=100)
    assert p.should_record_full(0) is True   # both fire
    assert p.should_record_full(1) is True   # first_n_steps fires
    assert p.should_record_full(2) is False
    assert p.should_record_full(100) is True  # every_n_steps fires


def test_full_tensor_policy_default_never_records_full() -> None:
    p = shadow.FullTensorPolicy()
    for step in (0, 1, 100, 10_000):
        assert p.should_record_full(step) is False


def test_tapper_records_full_tensor_per_policy() -> None:
    buf = shadow._Buffer(max_size=100)
    policy = shadow.FullTensorPolicy(first_n_steps=2)
    with shadow.Tapper(buf, full_tensor=policy):
        for _ in range(4):
            torch.ops.firefly.capture(torch.ones(2), "layer.0")
    events = _drain(buf)
    assert len(events) == 4
    assert events[0].tensor is not None
    assert events[1].tensor is not None
    assert events[2].tensor is None
    assert events[3].tensor is None


# ---------------------------------------------------------------------------
# Tap decorator
# ---------------------------------------------------------------------------


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 8)

    @shadow.tap("decorated_layer")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_decorator_inserts_capture_at_module_boundary() -> None:
    buf = shadow._Buffer(max_size=100)
    layer = _TinyLayer()
    x = torch.randn(1, 4)
    with shadow.Tapper(buf):
        y = layer(x)
    assert y.shape == (1, 8)
    events = _drain(buf)
    assert len(events) == 1
    assert events[0].tap_name == "decorated_layer"


def test_decorator_handles_tuple_returns() -> None:
    """A forward returning (tensor, extras) gets its first elem captured."""

    class TupleReturning(nn.Module):
        @shadow.tap("tuple_layer")
        def forward(self, x):
            return (x * 2, "extra")

    buf = shadow._Buffer(max_size=100)
    m = TupleReturning()
    with shadow.Tapper(buf):
        y, extra = m(torch.ones(3))
    assert y.tolist() == [2.0, 2.0, 2.0]
    assert extra == "extra"
    events = _drain(buf)
    assert len(events) == 1
    assert events[0].tap_name == "tuple_layer"


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


def test_buffer_drops_on_overflow() -> None:
    buf = shadow._Buffer(max_size=3)
    with shadow.Tapper(buf):
        for _ in range(10):
            torch.ops.firefly.capture(torch.ones(1), "tap")
    drained = _drain(buf)
    assert len(drained) == 3
    assert buf.dropped == 7


# ---------------------------------------------------------------------------
# LocalLogSink + DrainThread
# ---------------------------------------------------------------------------


def test_sink_writes_stats_jsonl(tmp_path: Path) -> None:
    handle = shadow.start_sink(tmp_path / "logs")
    with shadow.Tapper(handle.buffer):
        torch.ops.firefly.capture(torch.ones(2), "tap_a")
        torch.ops.firefly.capture(torch.zeros(4), "tap_b")
    # Give the drain thread a moment to flush.
    _wait_for_lines(tmp_path / "logs" / "stats.jsonl", expected=2)
    shadow.stop_sink(handle)

    lines = (tmp_path / "logs" / "stats.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    tap_names = sorted(r["tap_name"] for r in records)
    assert tap_names == ["tap_a", "tap_b"]


def test_sink_writes_blob_when_policy_fires(tmp_path: Path) -> None:
    handle = shadow.start_sink(tmp_path / "logs")
    policy = shadow.FullTensorPolicy(first_n_steps=1)
    with shadow.Tapper(handle.buffer, full_tensor=policy):
        torch.ops.firefly.capture(torch.tensor([1.0, 2.0, 3.0]), "tap")
        torch.ops.firefly.capture(torch.tensor([4.0, 5.0, 6.0]), "tap")  # step 1, no blob
    _wait_for_lines(tmp_path / "logs" / "stats.jsonl", expected=2)
    shadow.stop_sink(handle)

    records = [json.loads(line) for line in (tmp_path / "logs" / "stats.jsonl").read_text().splitlines()]
    # First event recorded a blob; second did not.
    with_blob = [r for r in records if "blob_path" in r]
    assert len(with_blob) == 1
    blob_full_path = tmp_path / "logs" / with_blob[0]["blob_path"]
    assert blob_full_path.exists()
    loaded = torch.load(blob_full_path, weights_only=True)
    assert loaded.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# aggregate()
# ---------------------------------------------------------------------------


def test_aggregate_groups_by_tap_and_computes_percentiles(tmp_path: Path) -> None:
    handle = shadow.start_sink(tmp_path / "logs")
    with shadow.Tapper(handle.buffer):
        for i in range(20):
            # Increasing magnitude per call: 1, 2, 3, ..., 20
            torch.ops.firefly.capture(torch.ones(1) * (i + 1), "layer.0")
        for _ in range(10):
            torch.ops.firefly.capture(torch.ones(1) * 5.0, "layer.1")
    _wait_for_lines(tmp_path / "logs" / "stats.jsonl", expected=30)
    shadow.stop_sink(handle)

    agg = shadow.aggregate(tmp_path / "logs")
    assert set(agg.keys()) == {"layer.0", "layer.1"}

    layer0 = agg["layer.0"]
    assert layer0.n_events == 20
    # abs_means are sorted [1, 2, ..., 20]. p50 = index 10 = 11; max = 20.
    assert layer0.abs_mean_p50 == pytest.approx(11.0)
    assert layer0.abs_mean_max == pytest.approx(20.0)

    layer1 = agg["layer.1"]
    assert layer1.n_events == 10
    assert layer1.abs_mean_p50 == pytest.approx(5.0)
    assert layer1.abs_mean_max == pytest.approx(5.0)


def test_aggregate_records_full_tensor_blobs(tmp_path: Path) -> None:
    handle = shadow.start_sink(tmp_path / "logs")
    policy = shadow.FullTensorPolicy(first_n_steps=2)
    with shadow.Tapper(handle.buffer, full_tensor=policy):
        for _ in range(5):
            torch.ops.firefly.capture(torch.ones(1), "layer.0")
    _wait_for_lines(tmp_path / "logs" / "stats.jsonl", expected=5)
    shadow.stop_sink(handle)

    agg = shadow.aggregate(tmp_path / "logs")
    assert len(agg["layer.0"].full_tensor_blobs) == 2


# ---------------------------------------------------------------------------
# torch.compile integration check (the load-bearing question from the spike)
# ---------------------------------------------------------------------------


def test_op_survives_torch_compile_without_graph_break() -> None:
    """The spike's load-bearing claim, codified as a test.

    Uses backend='aot_eager' to skip Inductor codegen (which has a macOS
    libc++.1.dylib dlopen issue on uv-installed Python). The frontend
    question — does Dynamo trace the op as opaque — is what we care
    about for shadow-mode, and that's what aot_eager exercises.
    """

    class TwoLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)

        def forward(self, x):
            h = self.fc1(x)
            h = torch.ops.firefly.capture(h, "layer.0.mlp")
            return self.fc2(h)

    model = TwoLayer().eval()
    x = torch.randn(1, 8)
    with torch.inference_mode():
        explanation = torch._dynamo.explain(model)(x)

    assert getattr(explanation, "graph_break_count", -1) == 0, (
        f"Custom op forced a graph break. break_reasons="
        f"{getattr(explanation, 'break_reasons', None)}"
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _drain(buf: shadow._Buffer) -> list:
    events: list = []
    while True:
        e = buf.pop(timeout=0.01)
        if e is None:
            break
        events.append(e)
    return events


def _wait_for_lines(path: Path, expected: int, timeout: float = 2.0) -> None:
    """Poll for the drain thread to flush at least ``expected`` records."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and len(path.read_text().splitlines()) >= expected:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"Expected {expected} lines in {path} within {timeout}s; "
        f"got {len(path.read_text().splitlines()) if path.exists() else 'no file'}"
    )


# ---------------------------------------------------------------------------
# CUDA-graph-mode capture (API surface tests — run on CPU)
#
# The actual Triton kernel launch + CUDA-graph capture-and-replay test
# lives in scripts/spike_cuda_graph.py and runs on Modal GPU. These tests
# cover the surface that's exercisable without CUDA: op registration,
# StaticTapper bookkeeping, tap_static decorator dispatch, drain flow on
# fake counter values.
# ---------------------------------------------------------------------------


def test_capture_static_op_is_registered() -> None:
    """The op exists in the dispatcher. Actual kernel execution requires CUDA;
    that's covered by scripts/spike_cuda_graph.py running on Modal."""
    assert hasattr(torch.ops.firefly, "capture_static")
    # The op's schema can be queried — verifies registration was clean.
    schema = torch.ops.firefly.capture_static.default._schema
    assert "Tensor x" in str(schema)
    assert "Tensor(a1!) stats_buf" in str(schema)
    assert "Tensor(a2!) counter" in str(schema)


def test_static_tapper_writes_tap_index_json(tmp_path: Path) -> None:
    names = {0: "layer.0.mlp", 7: "layer.7.self_attn"}
    # Force CPU device so the Tapper doesn't try to touch CUDA.
    tapper = shadow.StaticTapper(tmp_path / "logs", names, device="cpu")
    tap_index = json.loads((tmp_path / "logs" / "tap_index.json").read_text())
    # JSON keys are strings; compare as such.
    assert tap_index == {"0": "layer.0.mlp", "7": "layer.7.self_attn"}
    assert tapper.stats_buf.shape == (10_000, 5)
    assert tapper.counter.shape == (1,)


def test_static_tapper_is_not_reentrant(tmp_path: Path) -> None:
    t = shadow.StaticTapper(tmp_path / "logs", {0: "tap_a"}, device="cpu")
    with (
        t,
        pytest.raises(RuntimeError, match="not re-entrant"),
        shadow.StaticTapper(tmp_path / "logs2", {0: "tap_b"}, device="cpu"),
    ):
        pass


def test_static_tapper_drains_synthetic_rows_to_sink(tmp_path: Path) -> None:
    """End-to-end on CPU: fake the GPU kernel by directly writing buffer
    rows + bumping the counter, then verify the drain produces correct
    JSONL records with names re-attached via the index_to_name map."""
    names = {0: "layer.0.mlp", 7: "layer.7.self_attn"}
    with shadow.StaticTapper(
        tmp_path / "logs", names, buffer_size=100, drain_interval_s=0.01, device="cpu"
    ) as t:
        # Fake 3 captures: tap 0, tap 7, tap 0. Buffer layout per row:
        # [mean, abs_mean, abs_max, std, tap_idx]
        t.stats_buf[0] = torch.tensor([0.1, 0.2, 0.5, 0.05, 0.0])
        t.stats_buf[1] = torch.tensor([1.0, 1.1, 2.0, 0.30, 7.0])
        t.stats_buf[2] = torch.tensor([0.0, 0.1, 0.4, 0.02, 0.0])
        t.counter[0] = 3
        # Wait for drain to flush.
        _wait_for_lines(tmp_path / "logs" / "stats.jsonl", expected=3)

    # Outside the with-block: tapper has cleaned up and flushed.
    lines = (tmp_path / "logs" / "stats.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 3
    names_seen = [r["tap_name"] for r in records]
    assert names_seen == ["layer.0.mlp", "layer.7.self_attn", "layer.0.mlp"]
    # Verify the stats round-tripped correctly.
    assert records[0]["stats"]["abs_max"] == pytest.approx(0.5)
    assert records[1]["stats"]["mean"] == pytest.approx(1.0)


def test_static_tapper_handles_buffer_wraparound(tmp_path: Path) -> None:
    """If the counter advances by more than buffer_size between drains,
    we lose data — but the drain shouldn't crash, and the next drain
    should pick up cleanly."""
    names = {0: "tap"}
    t = shadow.StaticTapper(
        tmp_path / "logs", names, buffer_size=10, device="cpu"
    )
    # Manually drive the drain without entering the context manager so
    # we can step through wraparound scenarios deterministically.
    # Pre-write 10 rows (full buffer) and bump counter to 25 (15 lost).
    for i in range(10):
        t.stats_buf[i] = torch.tensor([float(i), 0.0, 0.0, 0.0, 0.0])
    t.counter[0] = 25
    n_drained = t._drain_once()
    # Drain returns the buffer-size cap, not 25.
    assert n_drained == 10
    t.sink.close()


def test_tap_static_decorator_is_noop_without_active_tapper() -> None:
    """The decorator must not crash if the model is called outside a
    StaticTapper context (so the same instrumented model works both ways)."""

    class M(nn.Module):
        @shadow.tap_static(idx=0, name="tap")
        def forward(self, x):
            return x * 2

    m = M()
    # No StaticTapper active. The decorator should pass through to fn().
    y = m(torch.ones(3))
    assert y.tolist() == [2.0, 2.0, 2.0]


def test_static_full_tensor_policy_default_no_blob_buffers(tmp_path: Path) -> None:
    """No policy → 1x1 placeholder blob buffers (negligible memory)."""
    t = shadow.StaticTapper(tmp_path / "logs", {0: "t"}, device="cpu")
    assert t.blob_buf.shape == (1, 1)
    assert t.blob_meta.shape == (1, 2)


def test_static_full_tensor_policy_first_n_allocates_blob_buffers(tmp_path: Path) -> None:
    policy = shadow.StaticFullTensorPolicy(first_n_steps=5, max_blob_numel=128)
    t = shadow.StaticTapper(
        tmp_path / "logs", {0: "t"}, full_tensor_policy=policy, device="cpu"
    )
    assert t.blob_buf.shape == (5, 128)
    assert t.blob_meta.shape == (5, 2)


def test_static_full_tensor_policy_every_n_auto_sizes_slots(tmp_path: Path) -> None:
    """every_n_steps with no n_blob_slots → auto-size to 16 (steady-state floor)."""
    policy = shadow.StaticFullTensorPolicy(every_n_steps=100, max_blob_numel=128)
    t = shadow.StaticTapper(
        tmp_path / "logs", {0: "t"}, full_tensor_policy=policy, device="cpu"
    )
    assert t.blob_buf.shape == (16, 128)


def test_static_full_tensor_policy_explicit_n_blob_slots(tmp_path: Path) -> None:
    """n_blob_slots overrides the auto-size heuristic."""
    policy = shadow.StaticFullTensorPolicy(
        every_n_steps=100, n_blob_slots=32, max_blob_numel=128
    )
    t = shadow.StaticTapper(
        tmp_path / "logs", {0: "t"}, full_tensor_policy=policy, device="cpu"
    )
    assert t.blob_buf.shape == (32, 128)


def test_static_drain_attaches_blob_tensors_when_first_n_policy_fires(tmp_path: Path) -> None:
    """End-to-end blob drain on CPU: pre-write stats + blob rows + meta,
    drain, verify each blob is correctly attached via global_idx matching."""
    policy = shadow.StaticFullTensorPolicy(first_n_steps=3, max_blob_numel=10)
    t = shadow.StaticTapper(
        tmp_path / "logs",
        {0: "layer.0", 7: "layer.7"},
        buffer_size=100,
        full_tensor_policy=policy,
        device="cpu",
    )
    # 4 stats rows (3 with blob, 1 without).
    t.stats_buf[0] = torch.tensor([0.0, 0.0, 1.0, 0.5, 0.0])
    t.stats_buf[1] = torch.tensor([1.0, 1.0, 2.0, 0.5, 7.0])
    t.stats_buf[2] = torch.tensor([0.5, 0.5, 0.6, 0.1, 0.0])
    t.stats_buf[3] = torch.tensor([2.0, 2.0, 3.0, 0.5, 0.0])  # past first_n_steps
    # Blob layout (ring buffer indexed by blob_count % n_slots):
    # blob_meta[i] = (n_valid, global_idx); for first_n_steps mode the
    # blob_count and global_idx coincide on the first 3 captures.
    t.blob_buf[0, :4] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    t.blob_meta[0] = torch.tensor([4, 0], dtype=torch.int32)
    t.blob_buf[1, :3] = torch.tensor([7.0, 7.1, 7.2])
    t.blob_meta[1] = torch.tensor([3, 1], dtype=torch.int32)
    t.blob_buf[2, :2] = torch.tensor([0.7, 0.8])
    t.blob_meta[2] = torch.tensor([2, 2], dtype=torch.int32)
    t.counter[0] = 4
    t.blob_counter[0] = 3

    n_drained = t._drain_once()
    assert n_drained == 4
    t.sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "stats.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    blobs = [r.get("blob_path") for r in records]
    assert all(b is not None for b in blobs[:3])
    assert blobs[3] is None
    blob0 = torch.load(tmp_path / "logs" / blobs[0], weights_only=True)
    assert blob0.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    blob1 = torch.load(tmp_path / "logs" / blobs[1], weights_only=True)
    assert blob1.tolist() == pytest.approx([7.0, 7.1, 7.2])


def test_static_drain_attaches_sparse_blobs_via_global_idx(tmp_path: Path) -> None:
    """every_n_steps produces blobs sparse in the stats stream — drain
    must match by global_idx, not by stats row position."""
    policy = shadow.StaticFullTensorPolicy(
        every_n_steps=2, n_blob_slots=4, max_blob_numel=5
    )
    t = shadow.StaticTapper(
        tmp_path / "logs",
        {0: "tap"},
        buffer_size=100,
        full_tensor_policy=policy,
        device="cpu",
    )
    # 5 stats rows; only global_idx in {0, 2, 4} would have blobs under
    # every_n_steps=2 (idx % 2 == 0).
    for i in range(5):
        t.stats_buf[i] = torch.tensor([float(i), float(i), float(i), 0.0, 0.0])
    t.counter[0] = 5
    # Three blobs, ring-buffer slots 0, 1, 2 (blob_count 0..2). meta = (numel, global_idx).
    t.blob_buf[0, :2] = torch.tensor([100.0, 100.1])
    t.blob_meta[0] = torch.tensor([2, 0], dtype=torch.int32)
    t.blob_buf[1, :2] = torch.tensor([102.0, 102.1])
    t.blob_meta[1] = torch.tensor([2, 2], dtype=torch.int32)
    t.blob_buf[2, :2] = torch.tensor([104.0, 104.1])
    t.blob_meta[2] = torch.tensor([2, 4], dtype=torch.int32)
    t.blob_counter[0] = 3

    t._drain_once()
    t.sink.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "stats.jsonl").read_text().splitlines()
    ]
    assert len(records) == 5
    blobs = [r.get("blob_path") for r in records]
    # Records 0, 2, 4 have blobs; 1 and 3 don't.
    assert blobs[0] is not None
    assert blobs[1] is None
    assert blobs[2] is not None
    assert blobs[3] is None
    assert blobs[4] is not None
    # Verify the blob content matches the recorded global_idx.
    blob0 = torch.load(tmp_path / "logs" / blobs[0], weights_only=True)
    blob4 = torch.load(tmp_path / "logs" / blobs[4], weights_only=True)
    assert blob0.tolist() == pytest.approx([100.0, 100.1])
    assert blob4.tolist() == pytest.approx([104.0, 104.1])


def test_tap_static_decorator_registers_name_lazily(tmp_path: Path, monkeypatch) -> None:
    """If the user passes ``name`` and that idx isn't in the Tapper's
    map yet, the decorator should register it lazily on first call.

    The op execution itself requires CUDA, so we patch the op call to
    a no-op pass-through. The name-registration behavior is pure Python
    state and testable on CPU.
    """

    class M(nn.Module):
        @shadow.tap_static(idx=42, name="layer.42.mlp")
        def forward(self, x):
            return x * 2

    tapper = shadow.StaticTapper(
        tmp_path / "logs", {}, buffer_size=10, device="cpu"
    )

    def _noop_capture_static(x, *_args):
        return x

    monkeypatch.setattr(torch.ops.firefly, "capture_static", _noop_capture_static)
    with tapper:
        m = M()
        m(torch.ones(3))
    assert tapper.index_to_name[42] == "layer.42.mlp"


# Silence the re-import-of-shadow-via-firefly bookkeeping check: torch's
# custom_op registry is global, so re-importing the module would attempt
# to re-register the op and raise. Importing once at module scope (above)
# is sufficient.
_ = re  # silence "imported but unused" if a future refactor drops the regex test
