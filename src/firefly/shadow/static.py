"""CUDA-graph-mode capture: StaticTapper + @tap_static."""

from __future__ import annotations

import functools
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from firefly.shadow.buffer import _Event
from firefly.shadow.eager import _TLS, _TapperContextBase
from firefly.shadow.sinks import make_sink


@dataclass
class StaticFullTensorPolicy:
    """CUDA-graph-mode version of :class:`FullTensorPolicy`.

    Two composable modes:

    * ``first_n_steps`` — record full tensors during the first N captures
      (warmup). After idx >= first_n_steps, only stats accumulate.
    * ``every_n_steps`` — record full tensors when idx % every_n_steps == 0
      (periodic sampling). Combines with first_n_steps via OR — both
      conditions firing still produces one record.

    The blob ring buffer holds the ``n_blob_slots`` most recently recorded
    full tensors. Older blobs get overwritten as the recording continues;
    the drain copies new rows out periodically. If ``n_blob_slots == 0``
    (the default), it's auto-sized to ``max(first_n_steps, every_n_blobs_floor)``
    where ``every_n_blobs_floor=16`` covers typical drain-interval headroom
    for steady-state every_n_steps recording.

    ``max_blob_numel`` is the per-slot capacity in floats. For an LLM with
    hidden_dim=4096 and a 10-token prompt, a per-layer activation is 40k
    floats = 160 KB. Default of 200_000 covers most production LLM taps
    with headroom.

    ``on_alert`` is reserved (not yet implemented in either mode).
    """

    first_n_steps: int = 0
    every_n_steps: int = 0
    on_alert: bool = False
    """If True, the kernel honors :attr:`StaticTapper.alert_flag`. Caller
    toggles via :meth:`StaticTapper.set_alert`."""
    n_blob_slots: int = 0
    max_blob_numel: int = 200_000

    def is_recording(self) -> bool:
        return self.first_n_steps > 0 or self.every_n_steps > 0 or self.on_alert

    def effective_n_blob_slots(self) -> int:
        """Resolve auto-sized slot counts."""
        if self.n_blob_slots > 0:
            return self.n_blob_slots
        # on_alert needs a larger ring; alerts can fire many times consecutively.
        return max(
            self.first_n_steps,
            16 if (self.every_n_steps > 0 or self.on_alert) else 0,
        )

class StaticTapper(_TapperContextBase):
    """CUDA-graph-mode counterpart to :class:`Tapper`.

    Pre-allocates GPU buffers (stats + counter, optionally blob + blob_meta)
    whose pointers get captured into a CUDA graph. The Triton kernel inside
    :func:`capture_static` writes to these buffers on every replay. A
    CPU-side drain thread periodically reads the GPU stats and writes
    to the local log sink, re-attaching tap names via the index → name
    map this Tapper holds.

    Full-tensor capture is controlled by ``full_tensor_policy``. With
    ``StaticFullTensorPolicy(first_n_steps=N)``, the first N captures
    across the whole tapper's lifetime get their full tensors persisted
    to ``blobs/``; later captures are stats-only.

    Usage::

        names = {0: "layer.0.mlp", 1: "layer.7.self_attn"}
        policy = StaticFullTensorPolicy(first_n_steps=5)
        with StaticTapper(log_dir, names, full_tensor_policy=policy) as tapper:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = model(static_input)
            for _ in range(N):
                g.replay()
    """

    def __init__(
        self,
        log_dir: str | Path,
        index_to_name: dict[int, str],
        buffer_size: int = 10_000,
        drain_interval_s: float = 0.1,
        full_tensor_policy: StaticFullTensorPolicy | None = None,
        device: str = "cuda",
    ) -> None:
        # Sink construction handles its own directory/object setup so this
        # works for both local paths and cloud URIs.
        self.log_dir = log_dir  # kept for diagnostics; not Path-coerced
        self.sink = make_sink(log_dir)

        # Persist the index → name map so the drain can re-attach names.
        # For local sinks this writes ``tap_index.json`` to disk; for cloud
        # sinks it uploads the same object alongside the stats shards.
        self.index_to_name = dict(index_to_name)
        self.sink.write_sidecar(
            "tap_index.json",
            json.dumps(self.index_to_name, indent=2, sort_keys=True),
        )

        self.buffer_size = buffer_size
        self.drain_interval_s = drain_interval_s
        self.full_tensor_policy = full_tensor_policy or StaticFullTensorPolicy()
        self.stats_buf = torch.zeros((buffer_size, 5), device=device, dtype=torch.float32)
        self.counter = torch.zeros((1,), device=device, dtype=torch.int32)

        # Blob buffers: real if policy.is_recording(), else 1x1 placeholders.
        # Placeholders keep the op signature constant; the kernel masks off
        # all blob writes when both step counts are 0.
        if self.full_tensor_policy.is_recording():
            n_slots = self.full_tensor_policy.effective_n_blob_slots()
            self.blob_buf = torch.zeros(
                (n_slots, self.full_tensor_policy.max_blob_numel),
                device=device, dtype=torch.float32,
            )
            self.blob_meta = torch.zeros((n_slots, 2), device=device, dtype=torch.int32)
            self.n_blob_slots = n_slots
        else:
            self.blob_buf = torch.zeros((1, 1), device=device, dtype=torch.float32)
            self.blob_meta = torch.zeros((1, 2), device=device, dtype=torch.int32)
            self.n_blob_slots = 1
        self.blob_counter = torch.zeros((1,), device=device, dtype=torch.int32)
        # Alert flag: CPU toggles via :meth:`set_alert`, kernel reads each
        # call. Lives on GPU so it's accessible to the captured kernel.
        self.alert_flag = torch.zeros((1,), device=device, dtype=torch.int32)

        # (sink already constructed above; only the blob/stats GPU buffers
        # had to wait for the policy decision.)
        self._drained_count = 0
        self._drained_blob_count = 0
        self._drain_thread: _StaticDrainThread | None = None

    def set_alert(self, alerting: bool) -> None:
        """Toggle the GPU-resident alert flag.

        When the policy has ``on_alert=True``, the kernel reads this flag
        every call and records a full tensor while it's set. Caller is
        responsible for clearing the flag (e.g. after the alert source
        clears, or after recording the desired number of post-alert frames).

        Note on CUDA-graph synchronization: ``alert_flag.fill_(...)`` is an
        async memcpy. Between setting the flag from CPU and the next
        ``g.replay()`` there is normally enough latency for the write to
        land; in tight loops, call ``torch.cuda.synchronize()`` between
        ``set_alert`` and ``g.replay()`` to guarantee visibility.
        """
        self.alert_flag.fill_(1 if alerting else 0)

    _tls_attr = "static_tapper"
    _kind = "StaticTapper"

    def _on_enter(self) -> None:
        self._drain_thread = _StaticDrainThread(self)
        self._drain_thread.start()

    def _on_exit(self) -> None:
        if self._drain_thread is not None:
            self._drain_thread.stop()
            self._drain_thread.join(timeout=5.0)
        # Final drain pass to flush remaining captures.
        self._drain_once()
        self.sink.close()

    def _drain_once(self) -> int:
        """Copy any new stats rows from GPU to CPU, write to sink.

        Also drains any new blob rows (full-tensor captures) up to
        ``first_n_steps`` total across the tapper's lifetime; blob writes
        are attached to the corresponding stats row's drain event via the
        sink's ``write`` method.

        Returns the number of stats rows drained.
        """
        current = int(self.counter.item())
        if current <= self._drained_count:
            return 0
        n_to_drain = current - self._drained_count
        # If we've lost data (n_to_drain > buffer_size), advance the
        # drained-count to recover whatever's still in the buffer.
        if n_to_drain > self.buffer_size:
            self._drained_count = current - self.buffer_size
            n_to_drain = self.buffer_size
        # Read n_to_drain stats rows starting at start_slot, with wraparound.
        start_slot = self._drained_count % self.buffer_size
        if start_slot + n_to_drain <= self.buffer_size:
            slots = self.stats_buf[start_slot : start_slot + n_to_drain].cpu().tolist()
        else:
            first_chunk = self.stats_buf[start_slot:].cpu().tolist()
            n_remaining = n_to_drain - len(first_chunk)
            slots = first_chunk + self.stats_buf[:n_remaining].cpu().tolist()

        # Blob drain. With the ring buffer, blobs are written sparsely
        # across the stats stream (only when first_n or every_n policy
        # fires). The drain reads the new blob_counter range, copies the
        # corresponding ring-buffer slots to CPU, and builds a dict
        # {global_idx → tensor} keyed by the global_idx field of blob_meta.
        # Stats events whose global_idx hits the dict get a tensor attached.
        blob_by_global_idx: dict[int, torch.Tensor] = {}
        if self.full_tensor_policy.is_recording():
            blob_current = int(self.blob_counter.item())
            if blob_current > self._drained_blob_count:
                n_new = blob_current - self._drained_blob_count
                # Lost data — recover what's still in the ring buffer.
                if n_new > self.n_blob_slots:
                    self._drained_blob_count = blob_current - self.n_blob_slots
                    n_new = self.n_blob_slots
                # Read blob_meta + blob_buf for the new range.
                # Each blob occupies one ring-buffer slot computed from
                # (drained_blob_count + i) % n_blob_slots.
                blob_meta_cpu = self.blob_meta.cpu()
                blob_buf_cpu = self.blob_buf.cpu()
                for i in range(n_new):
                    slot = (self._drained_blob_count + i) % self.n_blob_slots
                    n_valid = int(blob_meta_cpu[slot, 0].item())
                    global_idx = int(blob_meta_cpu[slot, 1].item())
                    if n_valid > 0:
                        blob_by_global_idx[global_idx] = (
                            blob_buf_cpu[slot, :n_valid].clone()
                        )
                self._drained_blob_count = blob_current

        for offset, row in enumerate(slots):
            tap_idx = int(row[4])
            global_idx = self._drained_count + offset
            tensor = blob_by_global_idx.get(global_idx)
            self.sink.write(
                _Event(
                    request_id=None,
                    tap_name=self.index_to_name.get(tap_idx, f"tap_{tap_idx}"),
                    step=global_idx,
                    stats={
                        "shape": [],          # not captured in CUDA-graph mode
                        "dtype": "float32",
                        "mean": row[0],
                        "abs_mean": row[1],
                        "abs_max": row[2],
                        "std": row[3],
                    },
                    tensor=tensor,
                    timestamp=time.time(),
                )
            )
        self._drained_count = current
        return len(slots)

def _active_static_tapper() -> StaticTapper | None:
    return getattr(_TLS, "static_tapper", None)

class _StaticDrainThread(threading.Thread):
    """Polls the GPU counter, copies new stats rows to CPU, writes to sink."""

    def __init__(self, tapper: StaticTapper) -> None:
        super().__init__(daemon=True, name="firefly-static-drain")
        self._tapper = tapper
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._tapper._drain_once()
            self._stop_event.wait(self._tapper.drain_interval_s)

    def stop(self) -> None:
        self._stop_event.set()

def tap_static(idx: int, name: str | None = None):
    """Decorator for CUDA-graph mode capture.

    Wraps an ``nn.Module.forward`` to call :func:`capture_static` with the
    active :class:`StaticTapper`'s buffers. If no StaticTapper is active,
    the decorator is a no-op pass-through (so the instrumented model can
    also be called outside of CUDA-graph mode).

    Usage::

        class MyLayer(nn.Module):
            @firefly.shadow.tap_static(idx=7, name="layer.7.mlp")
            def forward(self, x):
                return self.mlp(x)

    The ``name`` argument is optional and is used to populate the active
    Tapper's ``index_to_name`` map at decoration time (so the user
    doesn't have to maintain it separately).
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            out = fn(self, *args, **kwargs)
            t = _active_static_tapper()
            if t is None:
                return out
            # Register name lazily on first call if not yet known.
            if name is not None and idx not in t.index_to_name:
                t.index_to_name[idx] = name
            first_n = t.full_tensor_policy.first_n_steps
            every_n = t.full_tensor_policy.every_n_steps
            on_alert_enabled = 1 if t.full_tensor_policy.on_alert else 0
            if isinstance(out, torch.Tensor):
                return torch.ops.firefly.capture_static(
                    out, t.stats_buf, t.counter, idx,
                    t.blob_buf, t.blob_meta, t.blob_counter, t.alert_flag,
                    first_n, every_n, on_alert_enabled,
                )
            if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                head = torch.ops.firefly.capture_static(
                    out[0], t.stats_buf, t.counter, idx,
                    t.blob_buf, t.blob_meta, t.blob_counter, t.alert_flag,
                    first_n, every_n, on_alert_enabled,
                )
                return (head, *out[1:])
            return out

        return wrapper

    return decorator
