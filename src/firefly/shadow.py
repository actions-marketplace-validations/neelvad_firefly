"""Shadow-mode capture: passive activation observation for production inference.

Shadow-mode is the production-side complement to Firefly's CI-time
``firefly capture`` / ``firefly check`` flow. The CI flow uses forward
hooks (``register_forward_hook``) which are incompatible with
``torch.compile`` (Dynamo can't trace through Python-level hooks) and
CUDA graphs (no callback hook point after graph capture). Shadow-mode
sidesteps both by routing observation through a registered custom op
that Dynamo treats as opaque.

Architecture (see ``project_firefly_shadow_mode_design`` memory for the
full sketch and the six decisions baked in):

::

    user model.forward(...)
        torch.ops.firefly.capture(x, "layer.7.mlp")  ← pass-through op
            │
            ▼
        Tapper (thread-local) — filters by tap-name regex, decides
        whether to record stats / full tensor for this call
            │
            ▼
        _Buffer — thread-safe queue, drops on overflow (best-effort)
            │
            ▼ (separate drain thread)
        Sink — currently LocalLogSink, writes stats.jsonl + blobs/

Decisions baked in (from the design memory):

* Pass-through op semantics: ``y = capture(x, name)`` returns ``x``.
* Summary stats always, full tensors gated by a policy (every N steps,
  first N steps, on-alert TBD).
* User-controlled tap-name regex filter, not per-request sampling.
* Mutex-protected ring buffer first (correctness over performance).
* Local log sink for prototyping; S3/GCS/Azure sinks reuse v2 storage
  backends later.
* Manual + ``@tap`` decorator instrumentation; FX-rewrite deferred.

MVP scope: targets eager + torch.compile. CUDA-graph compat deferred.
"""

from __future__ import annotations

import functools
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# 1. The custom op (pass-through, Dynamo-opaque)
# ---------------------------------------------------------------------------


@torch.library.custom_op("firefly::capture", mutates_args=())
def capture(x: torch.Tensor, name: str) -> torch.Tensor:
    """Observe ``x`` under name ``name``; return ``x`` unchanged.

    The actual recording (summary stats, optional full tensor) happens
    in the active :class:`Tapper`. If no Tapper is active, this is
    cheap: a regex check and a return.
    """
    tapper = _active_tapper()
    if tapper is not None:
        tapper._record(x, name)
    return x.clone()


@capture.register_fake
def _capture_fake(x: torch.Tensor, name: str) -> torch.Tensor:
    """Abstract impl for Dynamo / FX tracing. Never executes."""
    return torch.empty_like(x)


def _capture_backward(ctx, grad: torch.Tensor) -> tuple:
    """Pass-through gradient. Lets the op compose with autograd graph capture."""
    return grad, None


capture.register_autograd(_capture_backward)


# ---------------------------------------------------------------------------
# 2. Tapper: thread-local context for an inference call
# ---------------------------------------------------------------------------


_TLS = threading.local()


def _active_tapper() -> Tapper | None:
    return getattr(_TLS, "tapper", None)


@dataclass
class FullTensorPolicy:
    """When (if ever) to record a full tensor in addition to summary stats.

    Modes are composable: setting both ``every_n_steps=100`` and
    ``first_n_steps=2`` records full tensors at steps 0, 1, 100, 200, ...
    """

    first_n_steps: int = 0
    """Record full tensors during the first N steps (warmup)."""
    every_n_steps: int = 0
    """Record full tensors every Nth step. 0 = disabled."""
    on_alert: bool = False
    """Reserved for alert-driven full-tensor capture (not yet implemented)."""

    def should_record_full(self, step: int) -> bool:
        if step < self.first_n_steps:
            return True
        return self.every_n_steps > 0 and step % self.every_n_steps == 0


@dataclass
class _TapState:
    """Per-tap-name bookkeeping inside a Tapper."""

    step: int = 0


class Tapper:
    """Thread-local context that enables shadow capture for one inference call.

    Usage::

        with Tapper(buffer, taps=r"layer\\.(7|15)\\.self_attn", full_tensor=FullTensorPolicy(first_n_steps=2)):
            output = model(input_ids)

    Inside the ``with`` block, every :func:`capture` call inside the
    model's forward checks this Tapper. Names matching ``taps`` get
    summary stats recorded; the full-tensor policy decides whether to
    also persist the raw tensor.
    """

    def __init__(
        self,
        buffer: _Buffer,
        taps: str = ".*",
        full_tensor: FullTensorPolicy | None = None,
        request_id: str | None = None,
    ) -> None:
        self._buffer = buffer
        self._taps_re = re.compile(taps)
        self._full_tensor = full_tensor or FullTensorPolicy()
        self._request_id = request_id
        self._step_by_tap: dict[str, _TapState] = {}

    def __enter__(self) -> Tapper:
        if _active_tapper() is not None:
            raise RuntimeError(
                "Tapper is not re-entrant within a single thread. "
                "Nested Tappers would interleave captures unpredictably."
            )
        _TLS.tapper = self
        return self

    def __exit__(self, *_exc) -> None:
        _TLS.tapper = None

    def _record(self, x: torch.Tensor, name: str) -> None:
        if not self._taps_re.search(name):
            return

        state = self._step_by_tap.setdefault(name, _TapState())
        record_full = self._full_tensor.should_record_full(state.step)
        stats = _summary_stats(x)
        full_tensor = x.detach().cpu().contiguous() if record_full else None
        self._buffer.push(
            _Event(
                request_id=self._request_id,
                tap_name=name,
                step=state.step,
                stats=stats,
                tensor=full_tensor,
                timestamp=time.time(),
            )
        )
        state.step += 1


# ---------------------------------------------------------------------------
# 3. Decorator: @tap("name") on nn.Module.forward
# ---------------------------------------------------------------------------


def tap(name: str):
    """Wrap a ``forward`` method to call :func:`capture` on its return value.

    Usage::

        class MyLayer(nn.Module):
            @firefly.shadow.tap("layer.7.mlp")
            def forward(self, x):
                return self.mlp(x)

    The decorator returns the captured tensor (pass-through), so wrapping
    a forward is non-invasive to the data flow.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            out = fn(self, *args, **kwargs)
            if isinstance(out, torch.Tensor):
                return torch.ops.firefly.capture(out, name)
            # Forward returned a non-tensor (e.g., tuple of (output, hidden));
            # capture the first tensor element and pass the rest through.
            if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                return (torch.ops.firefly.capture(out[0], name), *out[1:])
            return out

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# 4. Summary statistics — what gets logged on every capture call
# ---------------------------------------------------------------------------


def _summary_stats(x: torch.Tensor) -> dict[str, Any]:
    """Compact per-call summary. ~100 bytes serialized.

    Computed on the device the tensor lives on, then read back as Python
    floats to avoid keeping GPU tensors in the buffer.
    """
    fx = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "mean": fx.mean().item(),
        "abs_mean": fx.abs().mean().item(),
        "abs_max": fx.abs().max().item(),
        "std": fx.std().item() if fx.numel() > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# 5. Ring buffer + drain thread + sink
# ---------------------------------------------------------------------------


@dataclass
class _Event:
    """One captured observation queued for the drain thread."""

    request_id: str | None
    tap_name: str
    step: int
    stats: dict[str, Any]
    tensor: torch.Tensor | None
    timestamp: float


class _Buffer:
    """Thread-safe bounded queue. Drops events when full (best-effort capture)."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._q: queue.Queue[_Event] = queue.Queue(maxsize=max_size)
        self._dropped = 0
        self._dropped_lock = threading.Lock()

    def push(self, event: _Event) -> None:
        try:
            self._q.put_nowait(event)
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1

    def pop(self, timeout: float = 0.1) -> _Event | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def dropped(self) -> int:
        with self._dropped_lock:
            return self._dropped


class LocalLogSink:
    """Persist events to a local directory.

    Layout::

        <dir>/
            stats.jsonl       one line per captured event
            blobs/
                <event_id>.pt   serialized full-tensor when policy fires

    The JSONL design is intentional: ``firefly aggregate`` reads it
    line-by-line, no need for atomic-ness or full deserialization. Each
    blob is referenced from the JSONL line via ``blob_path``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.blobs_dir = self.path / "blobs"
        self.blobs_dir.mkdir(exist_ok=True)
        self.stats_path = self.path / "stats.jsonl"
        self._fh = self.stats_path.open("a", buffering=1)  # line-buffered
        self._blob_counter = 0
        self._counter_lock = threading.Lock()

    def write(self, event: _Event) -> None:
        record: dict[str, Any] = {
            "request_id": event.request_id,
            "tap_name": event.tap_name,
            "step": event.step,
            "stats": event.stats,
            "timestamp": event.timestamp,
        }
        if event.tensor is not None:
            with self._counter_lock:
                idx = self._blob_counter
                self._blob_counter += 1
            blob_path = self.blobs_dir / f"{idx:08d}.pt"
            torch.save(event.tensor, blob_path)
            record["blob_path"] = str(blob_path.relative_to(self.path))
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()


class DrainThread(threading.Thread):
    """Background thread that drains the buffer and forwards to the sink."""

    def __init__(self, buffer: _Buffer, sink: LocalLogSink) -> None:
        super().__init__(daemon=True, name="firefly-shadow-drain")
        self._buffer = buffer
        self._sink = sink
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            event = self._buffer.pop(timeout=0.1)
            if event is not None:
                self._sink.write(event)
        # Drain any remaining events on shutdown.
        while True:
            event = self._buffer.pop(timeout=0.0)
            if event is None:
                break
            self._sink.write(event)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# 6. Aggregation — collapse shadow logs into per-tap distributions
# ---------------------------------------------------------------------------


@dataclass
class TapAggregate:
    """Aggregated statistics for one tap name across all captures."""

    tap_name: str
    n_events: int
    abs_mean_p50: float
    abs_mean_p95: float
    abs_mean_max: float
    abs_max_p50: float
    abs_max_p95: float
    abs_max_max: float
    full_tensor_blobs: list[str] = field(default_factory=list)


def aggregate(shadow_log_dir: str | Path) -> dict[str, TapAggregate]:
    """Read a shadow log directory, group by tap, compute per-tap distributions.

    For each tap name observed, returns a :class:`TapAggregate` with
    p50 / p95 / max of the abs-mean and abs-max series, plus the list
    of full-tensor blobs available for that tap. Downstream code can
    compare these distributions against a reference's deterministic
    per-tap values, with tolerance derived from the spread.

    This is the thin-end of the shadow-vs-reference comparison flow.
    A future ``firefly check-shadow`` would consume this output, load
    the reference's per-tap means, and report taps whose live p95 has
    drifted past tolerance.
    """
    path = Path(shadow_log_dir)
    by_tap: dict[str, dict[str, list]] = {}
    with (path / "stats.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            t = rec["tap_name"]
            slot = by_tap.setdefault(
                t, {"abs_mean": [], "abs_max": [], "blobs": []}
            )
            slot["abs_mean"].append(rec["stats"]["abs_mean"])
            slot["abs_max"].append(rec["stats"]["abs_max"])
            if "blob_path" in rec:
                slot["blobs"].append(rec["blob_path"])

    out: dict[str, TapAggregate] = {}
    for tap_name, slot in by_tap.items():
        abs_means = sorted(slot["abs_mean"])
        abs_maxes = sorted(slot["abs_max"])
        n = len(abs_means)
        if n == 0:
            continue
        out[tap_name] = TapAggregate(
            tap_name=tap_name,
            n_events=n,
            abs_mean_p50=abs_means[n // 2],
            abs_mean_p95=abs_means[min(n - 1, int(n * 0.95))],
            abs_mean_max=abs_means[-1],
            abs_max_p50=abs_maxes[n // 2],
            abs_max_p95=abs_maxes[min(n - 1, int(n * 0.95))],
            abs_max_max=abs_maxes[-1],
            full_tensor_blobs=sorted(slot["blobs"]),
        )
    return out


# ---------------------------------------------------------------------------
# Convenience: start_sink / stop_sink for callers that don't want to manage
# the buffer + drain thread directly.
# ---------------------------------------------------------------------------


@dataclass
class SinkHandle:
    """Returned by :func:`start_sink`; pass to :func:`stop_sink` to clean up."""

    buffer: _Buffer
    sink: LocalLogSink
    drain: DrainThread


def start_sink(log_dir: str | Path, buffer_size: int = 10_000) -> SinkHandle:
    """Create a buffer + sink + drain thread; start the drain thread."""
    buffer = _Buffer(max_size=buffer_size)
    sink = LocalLogSink(log_dir)
    drain = DrainThread(buffer, sink)
    drain.start()
    return SinkHandle(buffer=buffer, sink=sink, drain=drain)


def stop_sink(handle: SinkHandle) -> None:
    """Stop the drain thread and flush remaining events to the sink."""
    handle.drain.stop()
    handle.drain.join(timeout=5.0)
    handle.sink.close()
