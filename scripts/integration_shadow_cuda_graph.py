"""End-to-end CUDA-graph integration test for shadow-mode.

The Spike (scripts/spike_cuda_graph.py) verifies the Triton kernel + CUDA
graph capture/replay layer in isolation. This script wires the whole
firefly.shadow stack together — StaticTapper + @tap_static decorator + drain
thread + LocalLogSink — and verifies the output JSONL contains the
expected records after replays.

What this catches that the spike doesn't:
  - @tap_static decorator dispatches to the op correctly inside a graph
  - StaticTapper's drain thread keeps up with replay throughput
  - LocalLogSink writes valid JSONL with correct tap names + blob paths
  - tap_index → name resolution works at drain time
  - Blob tensors round-trip through the filesystem correctly
  - StaticTapper.__exit__ flushes remaining captures before the sink closes

Run:  uv run modal run scripts/integration_shadow_cuda_graph.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-integration-shadow-cuda-graph")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)


@app.function(image=image, gpu="A10G", timeout=600)
def integration() -> dict:
    """Runs the full stack on the GPU container; returns a verdict dict."""
    import json
    import time
    from pathlib import Path

    import torch
    import torch.nn as nn

    from firefly import shadow

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    # ------------------------------------------------------------------
    # 1. A 3-layer MLP with shadow taps at two intermediate sites
    # ------------------------------------------------------------------
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc1(x)
            h = self._tap_fc1(h)
            h = self.fc2(h)
            h = self._tap_fc2(h)
            return self.fc3(h)

        @shadow.tap_static(idx=0, name="layer.0.fc1")
        def _tap_fc1(self, x: torch.Tensor) -> torch.Tensor:
            return x

        @shadow.tap_static(idx=1, name="layer.1.fc2")
        def _tap_fc2(self, x: torch.Tensor) -> torch.Tensor:
            return x

    device = "cuda"
    model = Tiny().eval().to(device)
    x = torch.randn(1, 16, device=device)

    log_dir = Path("/tmp/firefly-shadow-integration")
    if log_dir.exists():
        # Clean stale logs from prior runs in the same container.
        import shutil
        shutil.rmtree(log_dir)

    # ------------------------------------------------------------------
    # 2. StaticTapper with both first_n and every_n blob modes
    # ------------------------------------------------------------------
    N_REPLAYS = 100
    # Each model() forward triggers 2 capture_static calls → 200 stats
    # events from 100 replays. With first_n=2 + every_n=10, blobs land
    # at global_idx in {0, 1, 10, 20, 30, ..., 190} = 1+1+19 = 21 blobs.
    policy = shadow.StaticFullTensorPolicy(
        first_n_steps=2,
        every_n_steps=10,
        n_blob_slots=32,        # >21 expected blobs; no ring-buffer wrap
        max_blob_numel=64,      # 32 (fc1 out) and 64 (fc2 out) both fit
    )

    with (
        shadow.StaticTapper(
            log_dir,
            index_to_name={0: "layer.0.fc1", 1: "layer.1.fc2"},
            buffer_size=10_000,
            drain_interval_s=0.05,
            full_tensor_policy=policy,
            device=device,
        ) as tapper,
        torch.inference_mode(),
    ):
        # Warmup so Triton JIT compilation happens before graph capture
        # (compilation is CPU-side and would otherwise leak into capture).
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()

        # CUDA graph requires capture-time warmup on a side stream.
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(x)
        torch.cuda.current_stream().wait_stream(s)

        # Reset GPU state so the post-replay verification sees only the
        # replays' contribution. Captured buffer pointers are unaffected
        # by the value reset.
        tapper.counter.zero_()
        tapper.blob_counter.zero_()
        tapper.stats_buf.zero_()
        tapper.blob_buf.zero_()
        tapper.blob_meta.zero_()

        with torch.cuda.graph(g):
            _ = model(x)

        for _ in range(N_REPLAYS):
            g.replay()
        torch.cuda.synchronize()

        # Give the drain thread a moment to catch up to the GPU buffer
        # (it polls at drain_interval_s=0.05).
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # 3. After the StaticTapper context exits: drain has finalized,
    #    sink has closed. Verify the output JSONL + blobs/.
    # ------------------------------------------------------------------
    stats_path = log_dir / "stats.jsonl"
    tap_index_path = log_dir / "tap_index.json"

    if not stats_path.exists():
        return {"error": "stats.jsonl not produced", "log_dir_contents": list_dir(log_dir)}

    lines = stats_path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    print(f"records: {len(records)}")

    tap_counts: dict[str, int] = {}
    for r in records:
        tap_counts[r["tap_name"]] = tap_counts.get(r["tap_name"], 0) + 1
    print(f"tap counts: {tap_counts}")

    records_with_blobs = [r for r in records if r.get("blob_path")]
    print(f"records with blobs: {len(records_with_blobs)}")
    blob_global_idxs = sorted(r["step"] for r in records_with_blobs)
    print(f"blob global_idxs: {blob_global_idxs}")

    # Verify a few blob round-trips
    blob_round_trips: list[bool] = []
    for r in records_with_blobs[:3]:
        blob_path = log_dir / r["blob_path"]
        try:
            t = torch.load(blob_path, weights_only=True)
            # The tensor was the output of fc1 (32 elements) or fc2 (64).
            expected_numel = 32 if r["tap_name"] == "layer.0.fc1" else 64
            blob_round_trips.append(t.numel() == expected_numel)
        except Exception as e:
            print(f"blob load failed for {blob_path}: {e}")
            blob_round_trips.append(False)

    # tap_index.json sidecar
    tap_index = json.loads(tap_index_path.read_text())
    print(f"tap_index.json: {tap_index}")

    # ------------------------------------------------------------------
    # 4. Pass conditions
    # ------------------------------------------------------------------
    expected_records = N_REPLAYS * 2  # 2 taps per forward
    # Blob pattern: global_idx in {0, 1, 10, 20, ..., 190}
    expected_blob_idxs = sorted({0, 1, *list(range(10, expected_records, 10))})

    pass_n_records = len(records) == expected_records
    pass_tap_balance = (
        tap_counts.get("layer.0.fc1", 0) == N_REPLAYS
        and tap_counts.get("layer.1.fc2", 0) == N_REPLAYS
    )
    pass_n_blobs = len(records_with_blobs) == len(expected_blob_idxs)
    pass_blob_idxs = blob_global_idxs == expected_blob_idxs
    pass_blob_round_trip = all(blob_round_trips)
    pass_tap_index = tap_index == {"0": "layer.0.fc1", "1": "layer.1.fc2"}

    verdict = {
        "n_records": len(records),
        "expected_records": expected_records,
        "tap_counts": tap_counts,
        "n_records_with_blobs": len(records_with_blobs),
        "expected_n_blobs": len(expected_blob_idxs),
        "blob_global_idxs": blob_global_idxs,
        "expected_blob_idxs": expected_blob_idxs,
        "blob_round_trips": blob_round_trips,
        "tap_index_json": tap_index,
        "pass_n_records": pass_n_records,
        "pass_tap_balance": pass_tap_balance,
        "pass_n_blobs": pass_n_blobs,
        "pass_blob_idxs": pass_blob_idxs,
        "pass_blob_round_trip": pass_blob_round_trip,
        "pass_tap_index": pass_tap_index,
        "overall_pass": (
            pass_n_records and pass_tap_balance and pass_n_blobs
            and pass_blob_idxs and pass_blob_round_trip and pass_tap_index
        ),
        "first_record": records[0] if records else None,
        "last_record": records[-1] if records else None,
    }

    print()
    print("=" * 60)
    print(f"OVERALL: {'PASS' if verdict['overall_pass'] else 'FAIL'}")
    print("=" * 60)
    return verdict


def list_dir(path) -> list[str]:
    """Helper for debugging when expected files don't exist."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    return sorted(str(child.relative_to(p)) for child in p.rglob("*"))


@app.local_entrypoint()
def main() -> None:
    import json
    verdict = integration.remote()
    print()
    print("verdict:")
    print(json.dumps(verdict, indent=2, default=str))
