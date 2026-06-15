"""Shadow-mode capture: passive activation observation for production inference.

Shadow-mode is the production-side complement to Firefly's CI-time
``firefly capture`` / ``firefly check`` flow. The CI flow uses forward
hooks (``register_forward_hook``) which are incompatible with
``torch.compile`` (Dynamo can't trace through Python-level hooks) and
CUDA graphs (no callback hook point after graph capture). Shadow-mode
sidesteps both by routing observation through a registered custom op
that Dynamo treats as opaque.

Architecture sketch (the design decisions baked in are listed below):

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

Decisions baked in:

* Pass-through op semantics: ``y = capture(x, name)`` returns ``x``.
* Summary stats always, full tensors gated by a policy (first N steps,
  every N steps, on-alert — all implemented).
* User-controlled tap-name regex filter, not per-request sampling.
* Mutex-protected ring buffer first (correctness over performance).
* Local log sink plus streaming S3/GCS/Azure sinks (sharded JSONL).
* Instrumentation: manual ``@tap`` decorators, or ``instrument(model,
  pattern, ...)`` which auto-wires capture ops via ``torch.fx`` with a
  ``named_modules`` fallback.

Scope: eager + torch.compile via the ``capture`` op, and CUDA-graph
replay via the Triton-backed ``capture_static`` op (see below). What is
*not* built is the production loop that drives this against a live
serving stack — the mechanism is here; the ``firefly check-shadow``
product flow is future work.
"""

from __future__ import annotations

import functools
import io
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


class _TapperContextBase:
    """Re-entrance check + thread-local registration; subclass hooks for setup.

    Both :class:`Tapper` and :class:`StaticTapper` use the same context-
    manager pattern: enforce one-active-per-thread, register on a TLS
    attribute, then do whatever subclass-specific setup is needed (drain
    thread for StaticTapper). Override ``_tls_attr`` and ``_kind`` in
    each subclass, plus optional ``_on_enter`` / ``_on_exit`` hooks.
    """

    _tls_attr: str = ""    # override: TLS attribute name
    _kind: str = "Tapper"  # override: label for the re-entrance error

    def __enter__(self):
        current = getattr(_TLS, self._tls_attr, None)
        if current is not None:
            raise RuntimeError(
                f"{self._kind} is not re-entrant within a single thread. "
                f"Nested instances would interleave captures unpredictably."
            )
        setattr(_TLS, self._tls_attr, self)
        self._on_enter()
        return self

    def __exit__(self, *_exc) -> None:
        self._on_exit()
        setattr(_TLS, self._tls_attr, None)

    def _on_enter(self) -> None:
        """Subclass hook for additional setup (e.g. start drain thread)."""

    def _on_exit(self) -> None:
        """Subclass hook for additional teardown (e.g. stop drain, flush)."""


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
    """If True, record full tensors whenever the active Tapper's
    ``alerting`` flag is set. Caller toggles via :meth:`Tapper.set_alert`."""

    def should_record_full(self, step: int, alerting: bool = False) -> bool:
        if step < self.first_n_steps:
            return True
        if self.every_n_steps > 0 and step % self.every_n_steps == 0:
            return True
        return self.on_alert and alerting


@dataclass
class _TapState:
    """Per-tap-name bookkeeping inside a Tapper."""

    step: int = 0


class Tapper(_TapperContextBase):
    """Thread-local context that enables shadow capture for one inference call.

    Usage::

        with Tapper(buffer, taps=r"layer\\.(7|15)\\.self_attn", full_tensor=FullTensorPolicy(first_n_steps=2)):
            output = model(input_ids)

    Inside the ``with`` block, every :func:`capture` call inside the
    model's forward checks this Tapper. Names matching ``taps`` get
    summary stats recorded; the full-tensor policy decides whether to
    also persist the raw tensor.
    """

    _tls_attr = "tapper"
    _kind = "Tapper"

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
        self._alerting = False

    def set_alert(self, alerting: bool) -> None:
        """Toggle the alert flag.

        When the policy has ``on_alert=True``, every subsequent capture
        records a full tensor as long as this flag is set. Caller is
        responsible for clearing the flag (e.g. after N seconds of
        recording, or after the underlying alert condition clears).
        """
        self._alerting = alerting

    def _record(self, x: torch.Tensor, name: str) -> None:
        if not self._taps_re.search(name):
            return

        state = self._step_by_tap.setdefault(name, _TapState())
        record_full = self._full_tensor.should_record_full(state.step, self._alerting)
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

    def write_sidecar(self, name: str, content: str) -> None:
        """Write a small companion file alongside stats.jsonl.

        Used by :class:`StaticTapper` to persist the tap_index → name map.
        """
        (self.path / name).write_text(content)


# ---------------------------------------------------------------------------
# Cloud streaming sinks: S3, GCS, Azure
#
# Shape differences from LocalLogSink:
#   - Stats output is *sharded* (stats-00000.jsonl, stats-00001.jsonl, ...)
#     instead of a single growing file. Each shard is flushed periodically
#     as a complete object. Shard rotation criteria: every N events OR every
#     K seconds, whichever comes first.
#   - Blobs are uploaded synchronously (one PUT per blob) since they're
#     small and individually named.
#   - Errors during upload don't crash inference — we log and drop.
#
# The aggregate() function reads either layout (single file or sharded)
# transparently via :func:`_iter_jsonl_records`.
# ---------------------------------------------------------------------------


_SHARD_FLUSH_EVERY_N = 500    # flush a shard once it accumulates N records
_SHARD_FLUSH_EVERY_S = 5.0    # ...or when this many seconds have elapsed


class _CloudSinkBase:
    """Shared scaffolding for cloud sinks: sharded stats, background flusher.

    Subclasses implement ``_put_object(key, body)`` and override
    ``_blob_uri(idx)`` if needed.
    """

    def __init__(self) -> None:
        self._stats_buffer: list[str] = []
        self._blob_seq = 0
        self._shard_seq = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flusher = threading.Thread(
            target=self._flush_loop, daemon=True, name="firefly-cloud-flusher"
        )
        self._flusher.start()

    def write(self, event: _Event) -> None:
        record: dict[str, Any] = {
            "request_id": event.request_id,
            "tap_name": event.tap_name,
            "step": event.step,
            "stats": event.stats,
            "timestamp": event.timestamp,
        }
        if event.tensor is not None:
            with self._lock:
                idx = self._blob_seq
                self._blob_seq += 1
            blob_key = self._blob_key(idx)
            try:
                buf = io.BytesIO()
                torch.save(event.tensor, buf)
                self._put_object(blob_key, buf.getvalue())
                record["blob_path"] = self._blob_relpath(idx)
            except Exception as e:  # noqa: BLE001 — cloud SDKs raise many types
                import sys
                print(f"[firefly] cloud sink blob upload failed: {e}", file=sys.stderr)
        line = json.dumps(record) + "\n"
        with self._lock:
            self._stats_buffer.append(line)
            if len(self._stats_buffer) >= _SHARD_FLUSH_EVERY_N:
                self._flush_locked()

    def close(self) -> None:
        self._stop_event.set()
        self._flusher.join(timeout=5.0)
        with self._lock:
            self._flush_locked()

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(_SHARD_FLUSH_EVERY_S)
            with self._lock:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._stats_buffer:
            return
        body = "".join(self._stats_buffer).encode("utf-8")
        shard_key = self._shard_key(self._shard_seq)
        self._shard_seq += 1
        try:
            self._put_object(shard_key, body)
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[firefly] cloud sink shard upload failed: {e}", file=sys.stderr)
        self._stats_buffer.clear()

    def write_sidecar(self, name: str, content: str) -> None:
        """Upload a small companion object alongside the sharded stats."""
        try:
            self._put_object(self._sidecar_key(name), content.encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[firefly] cloud sink sidecar upload failed: {e}", file=sys.stderr)

    # --- subclass hooks ----------------------------------------------------

    def _put_object(self, key: str, body: bytes) -> None:
        raise NotImplementedError

    def _shard_key(self, seq: int) -> str:
        raise NotImplementedError

    def _blob_key(self, seq: int) -> str:
        raise NotImplementedError

    def _sidecar_key(self, name: str) -> str:
        raise NotImplementedError

    def _blob_relpath(self, seq: int) -> str:
        """Returned in JSONL as ``blob_path``. Relative to the sink prefix."""
        return f"blobs/{seq:08d}.pt"


class S3Sink(_CloudSinkBase):
    """Streaming S3 sink for shadow capture.

    Layout under ``s3://<bucket>/<prefix>/``:

    * ``stats-NNNNN.jsonl`` — sharded stats; one shard PUT per flush
    * ``blobs/<seq>.pt`` — individual full-tensor PUTs

    Uses boto3's default credential chain. Errors during upload don't
    crash the inference process — they're logged to stderr and dropped.
    """

    def __init__(self, uri: str) -> None:
        super().__init__()
        from firefly.storage import _S3_REGEX

        m = _S3_REGEX.match(uri)
        if not m:
            raise ValueError(
                f"Invalid S3 URI {uri!r}. Expected format: s3://<bucket>/<prefix>"
            )
        self._bucket = m.group("bucket")
        raw_prefix = (m.group("prefix") or "").strip("/")
        self._prefix = f"{raw_prefix}/" if raw_prefix else ""

        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "boto3 is required for s3:// shadow sinks. Install with "
                "pip install 'firefly[s3]'."
            ) from e
        self._client = boto3.client("s3")

    def _put_object(self, key: str, body: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=body)

    def _shard_key(self, seq: int) -> str:
        return f"{self._prefix}stats-{seq:05d}.jsonl"

    def _blob_key(self, seq: int) -> str:
        return f"{self._prefix}blobs/{seq:08d}.pt"

    def _sidecar_key(self, name: str) -> str:
        return f"{self._prefix}{name}"


class GCSSink(_CloudSinkBase):
    """Streaming GCS sink. Same shape as :class:`S3Sink`."""

    def __init__(self, uri: str) -> None:
        super().__init__()
        from firefly.storage import _GCS_REGEX

        m = _GCS_REGEX.match(uri)
        if not m:
            raise ValueError(
                f"Invalid GCS URI {uri!r}. Expected format: gs://<bucket>/<prefix>"
            )
        bucket_name = m.group("bucket")
        raw_prefix = (m.group("prefix") or "").strip("/")
        self._prefix = f"{raw_prefix}/" if raw_prefix else ""

        try:
            from google.cloud import storage as gcs_storage
        except ImportError as e:
            raise ImportError(
                "google-cloud-storage is required for gs:// shadow sinks. "
                "Install with pip install 'firefly[gcs]'."
            ) from e
        self._client = gcs_storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def _put_object(self, key: str, body: bytes) -> None:
        self._bucket.blob(key).upload_from_string(body)

    def _shard_key(self, seq: int) -> str:
        return f"{self._prefix}stats-{seq:05d}.jsonl"

    def _blob_key(self, seq: int) -> str:
        return f"{self._prefix}blobs/{seq:08d}.pt"

    def _sidecar_key(self, name: str) -> str:
        return f"{self._prefix}{name}"


class AzureSink(_CloudSinkBase):
    """Streaming Azure Blob sink. Same shape as :class:`S3Sink`."""

    def __init__(self, uri: str) -> None:
        super().__init__()
        from firefly.storage import _AZURE_REGEX, _azure_client

        m = _AZURE_REGEX.match(uri)
        if not m:
            raise ValueError(
                f"Invalid Azure URI {uri!r}. Expected format: "
                f"az://<account>/<container>/<prefix>"
            )
        account = m.group("account")
        container_name = m.group("container")
        raw_prefix = (m.group("prefix") or "").strip("/")
        self._prefix = f"{raw_prefix}/" if raw_prefix else ""

        try:
            client = _azure_client(account)
        except ImportError as e:
            raise ImportError(
                "azure-storage-blob is required for az:// shadow sinks. "
                "Install with pip install 'firefly[azure]'."
            ) from e
        self._container_client = client.get_container_client(container_name)

    def _put_object(self, key: str, body: bytes) -> None:
        self._container_client.upload_blob(name=key, data=body, overwrite=True)

    def _shard_key(self, seq: int) -> str:
        return f"{self._prefix}stats-{seq:05d}.jsonl"

    def _blob_key(self, seq: int) -> str:
        return f"{self._prefix}blobs/{seq:08d}.pt"

    def _sidecar_key(self, name: str) -> str:
        return f"{self._prefix}{name}"


def make_sink(uri: str | Path):
    """Pick the right sink implementation based on the URI scheme.

    * local paths → :class:`LocalLogSink`
    * ``s3://`` → :class:`S3Sink`
    * ``gs://`` / ``gcs://`` → :class:`GCSSink`
    * ``az://`` / ``azure://`` → :class:`AzureSink`
    """
    from firefly.storage import _extract_scheme

    raw = str(uri)
    scheme = _extract_scheme(raw)
    if scheme is None:
        return LocalLogSink(raw)
    if scheme == "s3":
        return S3Sink(raw)
    if scheme in ("gs", "gcs"):
        return GCSSink(raw)
    if scheme in ("az", "azure"):
        return AzureSink(raw)
    raise ValueError(f"Unknown sink scheme {scheme!r}")


class DrainThread(threading.Thread):
    """Background thread that drains the buffer and forwards to the sink."""

    def __init__(self, buffer: _Buffer, sink) -> None:  # noqa: ANN001 — duck-typed Sink
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

    Handles two on-disk layouts:

    * Single-file (eager-mode / local sink default): ``stats.jsonl``
    * Sharded (cloud-streaming sink default): ``stats-00000.jsonl``,
      ``stats-00001.jsonl``, ...

    Shards are read in lexical order so events stay in capture order
    within each shard; cross-shard ordering preserves the wall-clock
    order the drain wrote them in.

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
    for rec in _iter_jsonl_records(path):
        t = rec["tap_name"]
        slot = by_tap.setdefault(t, {"abs_mean": [], "abs_max": [], "blobs": []})
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


def _iter_jsonl_records(path: Path):
    """Yield JSONL records from a shadow-log directory.

    Layout dispatch:
    * ``stats.jsonl`` present → read it as a single stream
    * ``stats-NNNNN.jsonl`` shards → read them in lexical order

    If both are present (e.g. legacy + new-style cohabiting), the
    sharded stream is preferred and the singleton is logged as
    skipped.
    """
    shards = sorted(path.glob("stats-*.jsonl"))
    if shards:
        if (path / "stats.jsonl").exists():
            import sys
            print(
                f"[firefly] aggregate: both stats.jsonl and shards present in "
                f"{path}; reading shards, ignoring singleton.",
                file=sys.stderr,
            )
        for shard in shards:
            with shard.open() as f:
                for line in f:
                    yield json.loads(line)
        return
    # No shards. Fall back to single file.
    stats_path = path / "stats.jsonl"
    if stats_path.exists():
        with stats_path.open() as f:
            for line in f:
                yield json.loads(line)


def load_tap_index(shadow_log_dir: str | Path) -> dict[int, str]:
    """Read the ``tap_index.json`` sidecar a :class:`StaticTapper` writes.

    Returns an empty dict if no sidecar is present (eager-mode logs
    don't write one — the tap names live directly in each event).
    Useful for downstream tools that want to enumerate all known taps
    without scanning the whole stats stream.
    """
    path = Path(shadow_log_dir) / "tap_index.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    # JSON keys are strings; restore to int.
    return {int(k): v for k, v in raw.items()}


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
    """Create a buffer + sink + drain thread; start the drain thread.

    ``log_dir`` can be a local path or one of the supported cloud URIs
    (``s3://``, ``gs://``, ``az://``); :func:`make_sink` picks the right
    sink implementation by scheme.
    """
    buffer = _Buffer(max_size=buffer_size)
    sink = make_sink(log_dir)
    drain = DrainThread(buffer, sink)
    drain.start()
    return SinkHandle(buffer=buffer, sink=sink, drain=drain)


def stop_sink(handle: SinkHandle) -> None:
    """Stop the drain thread and flush remaining events to the sink."""
    handle.drain.stop()
    handle.drain.join(timeout=5.0)
    handle.sink.close()


# ---------------------------------------------------------------------------
# CUDA-graph-mode capture (validated by scripts/spike_cuda_graph.py)
#
# The eager / torch.compile path above does CPU-side Python in its op body
# (regex match, queue push). CUDA graphs capture a fixed sequence of GPU
# kernel launches and replay them with zero CPU work between launches —
# so the CPU-side body never runs during replay, and shadow data goes
# silent the moment the model enters CUDA-graph replay mode.
#
# The CUDA-graph-mode capture op routes the work through a Triton kernel
# that runs entirely on GPU: compute stats, atomically increment a counter,
# write to buffer[idx]. The kernel launch gets captured into the graph and
# re-executes on every replay. A separate CPU-side drain thread polls the
# GPU buffer between replays and writes to the sink.
#
# Two key design differences from the eager path:
#  1. Buffers are pre-allocated GPU tensors with fixed lifetime, not a
#     Python queue. Their pointers are captured into the graph.
#  2. Tap-name filter logic moves from "regex match inside the op" to
#     "compile-time decision about which sites get instrumented." The
#     instrumented sites pass an integer `tap_idx`; the sidecar JSON maps
#     idx → name at drain time.
# ---------------------------------------------------------------------------


# Lazy Triton import — keeps the module importable on CPU-only machines.
# The CUDA-graph path requires Triton at runtime but not at import time.
def _import_triton():
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        raise ImportError(
            "triton is required for CUDA-graph-mode shadow capture but isn't "
            "installed. On Linux/CUDA, triton is bundled with torch — if it's "
            "missing, your torch install may be CPU-only."
        ) from e
    return triton, tl


# Triton kernel + Python wrapper, both lazily constructed at first use so
# the import-time cost is zero on CPU-only machines.
_STATS_KERNEL = None


def _get_stats_kernel():
    """Lazily JIT-compile and cache the Triton stats kernel.

    Always writes summary stats. Conditionally writes the full tensor to
    ``blob_buf[idx]`` when ``idx < first_n_steps`` AND ``idx < max_blob_slots``
    AND ``blob_buf`` is non-placeholder. This implements the "first N steps"
    mode of :class:`StaticFullTensorPolicy`. Other modes (every_n_steps,
    on_alert) are not supported in CUDA-graph mode in this MVP.

    When the caller doesn't want blob recording, they pass placeholder
    blob_buf / blob_meta buffers and first_n_steps=0; the kernel runs but
    the blob writes are all masked off.
    """
    global _STATS_KERNEL
    if _STATS_KERNEL is not None:
        return _STATS_KERNEL

    triton, tl = _import_triton()

    @triton.jit
    def _shadow_stats_kernel(
        x_ptr,                            # input tensor
        stats_buf_ptr,                    # [N, 5] stats buffer
        counter_ptr,                      # [1] int32 stats atomic counter
        tap_idx,                          # int — which tap
        blob_buf_ptr,                     # [n_blob_slots, max_blob_numel]
        blob_meta_ptr,                    # [n_blob_slots, 2] int32 (numel, global_idx)
        blob_counter_ptr,                 # [1] int32 blob-ring-buffer atomic counter
        alert_flag_ptr,                   # [1] int32 — CPU-toggled alert flag
        first_n_steps,                    # int — record blob if idx < this
        every_n_steps,                    # int — record blob if idx % this == 0; 0 disables
        on_alert_enabled,                 # int — 1: consult alert_flag; 0: ignore
        n_blob_slots,                     # int — ring buffer capacity
        max_blob_numel,                   # int — per-slot blob capacity
        n_elements,                       # numel(x)
        BLOCK_SIZE: tl.constexpr,         # power-of-2 >= n_elements
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        x_f = x.to(tl.float32)
        s_mean = tl.sum(x_f, axis=0) / n_elements
        abs_x = tl.abs(x_f)
        s_abs_mean = tl.sum(abs_x, axis=0) / n_elements
        s_abs_max = tl.max(abs_x, axis=0)
        diff = x_f - s_mean
        s_std = tl.sqrt(tl.sum(diff * diff, axis=0) / n_elements)

        idx = tl.atomic_add(counter_ptr, 1)

        # Stats write (always).
        tl.store(stats_buf_ptr + idx * 5 + 0, s_mean)
        tl.store(stats_buf_ptr + idx * 5 + 1, s_abs_mean)
        tl.store(stats_buf_ptr + idx * 5 + 2, s_abs_max)
        tl.store(stats_buf_ptr + idx * 5 + 3, s_std)
        # tap_idx arrives as a Python int (not a Triton scalar) so we
        # coerce to float by multiplying by 1.0 — implicit type
        # promotion is the most version-portable way across Triton
        # releases.
        tl.store(stats_buf_ptr + idx * 5 + 4, tap_idx * 1.0)

        # Blob recording decision (composable first_n + every_n + on_alert).
        # The alert flag is GPU-resident and CPU-toggled. Each kernel call
        # reads the current value; the captured graph references the
        # tensor by pointer, so subsequent CPU writes between replays are
        # visible (modulo memcpy ordering — see StaticTapper.set_alert).
        record_first_n = (first_n_steps > 0) & (idx < first_n_steps)
        record_every_n = (every_n_steps > 0) & ((idx % every_n_steps) == 0)
        alert_now = tl.load(alert_flag_ptr) != 0
        record_alert = (on_alert_enabled != 0) & alert_now
        should_record_blob = record_first_n | record_every_n | record_alert

        # Atomic blob-slot allocation, gated by should_record. When the
        # mask is off, the atomic_add is skipped (its return value is
        # undefined but we don't read it).
        blob_count = tl.atomic_add(blob_counter_ptr, 1, mask=should_record_blob)
        blob_slot = blob_count % n_blob_slots
        safe_blob_slot = tl.minimum(blob_slot, n_blob_slots - 1)

        # Blob tensor data write — mask gates each element's store.
        blob_offset_in_range = offsets < tl.minimum(n_elements, max_blob_numel)
        blob_write_mask = should_record_blob & blob_offset_in_range
        tl.store(
            blob_buf_ptr + safe_blob_slot * max_blob_numel + offsets,
            x_f,
            mask=blob_write_mask,
        )

        # Blob metadata: (numel, global_idx) lets the drain attach the
        # blob to the corresponding stats row by matching global_idx.
        tl.store(blob_meta_ptr + safe_blob_slot * 2 + 0, n_elements, mask=should_record_blob)
        tl.store(blob_meta_ptr + safe_blob_slot * 2 + 1, idx, mask=should_record_blob)

    _STATS_KERNEL = _shadow_stats_kernel
    return _shadow_stats_kernel


# Custom op for CUDA-graph mode. ``mutates_args`` tells Dynamo /
# torch.compile that all four mutated buffers are written, so the op
# survives both ``torch.compile`` AND ``torch.cuda.CUDAGraph`` capture.
@torch.library.custom_op(
    "firefly::capture_static",
    mutates_args=("stats_buf", "counter", "blob_buf", "blob_meta", "blob_counter"),
)
def capture_static(
    x: torch.Tensor,
    stats_buf: torch.Tensor,
    counter: torch.Tensor,
    tap_idx: int,
    blob_buf: torch.Tensor,
    blob_meta: torch.Tensor,
    blob_counter: torch.Tensor,
    alert_flag: torch.Tensor,
    first_n_steps: int,
    every_n_steps: int,
    on_alert_enabled: int,
) -> torch.Tensor:
    """CUDA-graph-safe pass-through capture.

    Always writes summary stats to ``stats_buf[counter]`` and atomically
    increments ``counter``. Conditionally writes the full tensor to a
    ring-buffer slot ``blob_buf[blob_counter % n_blob_slots]`` when any
    of three policies fires: ``first_n_steps``, ``every_n_steps``, or
    ``on_alert_enabled & alert_flag[0]``. Returns ``x`` unchanged.

    To opt out of full-tensor recording, pass 1x1 placeholder
    ``blob_buf`` / ``blob_meta`` buffers and all three policy ints at 0;
    the kernel runs but all blob writes are masked off. The
    :class:`StaticTapper` handles this setup automatically.
    """
    triton, _ = _import_triton()
    kernel = _get_stats_kernel()
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    n_blob_slots = blob_buf.shape[0]
    max_blob_numel = blob_buf.shape[1]
    kernel[(1,)](
        x, stats_buf, counter, tap_idx,
        blob_buf, blob_meta, blob_counter, alert_flag,
        first_n_steps, every_n_steps, on_alert_enabled,
        n_blob_slots, max_blob_numel,
        n, BLOCK_SIZE=BLOCK,
    )
    return x.clone()


@capture_static.register_fake
def _capture_static_fake(
    x: torch.Tensor,
    stats_buf: torch.Tensor,
    counter: torch.Tensor,
    tap_idx: int,
    blob_buf: torch.Tensor,
    blob_meta: torch.Tensor,
    blob_counter: torch.Tensor,
    alert_flag: torch.Tensor,
    first_n_steps: int,
    every_n_steps: int,
    on_alert_enabled: int,
) -> torch.Tensor:
    return torch.empty_like(x)


# Note: no autograd formula. ``mutates_args`` and ``register_autograd`` are
# mutually exclusive in torch.library — mutating ops can't have backward
# formulas because backward through a mutation needs the pre-mutation
# state. CUDA-graph-mode shadow capture is inference-only by design, so
# this is fine; the eager ``capture`` op above keeps its pass-through
# gradient for the (rarer) training-time debugging case.


# ---------------------------------------------------------------------------
# StaticTapper: holds GPU buffers + drain thread for CUDA-graph mode
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Auto-instrumentation via torch.fx OR named_modules
#
# Manual `@tap("name")` / `@tap_static(idx, name)` is the most explicit way to
# wire shadow capture into a model — you mark exactly which forward methods
# get instrumented. But on a 32-layer Llama you'd be decorating 100+ sites by
# hand. `instrument(model, pattern, ...)` does the wiring by name-pattern.
#
# Two underlying methods are supported:
#
#   - "fx": torch.fx.symbolic_trace(model), walk the resulting Graph, insert
#     call_function nodes after every call_module whose target matches the
#     pattern. Most powerful; finds taps at sub-module boundaries even when
#     the forward method calls them mid-expression. Fails on dynamic control
#     flow (most production LLMs have `if attention_mask is not None`,
#     KV-cache None checks, etc.) — symbolic_trace will raise.
#
#   - "named_modules": walk model.named_modules(), monkey-patch matching
#     module forward methods to wrap their return value with the capture op.
#     Simpler; works on any model with nn.Module structure. Misses taps that
#     don't correspond to a named sub-module (rare in practice).
#
#   - "auto" (default): try fx first, fall back to named_modules if fx raises.
#     Gets the best available approach for each model without the user having
#     to know which to pick.
# ---------------------------------------------------------------------------


@torch.fx.wrap
def _firefly_eager_tap(x: torch.Tensor, name: str) -> torch.Tensor:
    """FX-opaque wrapper for eager-mode capture. Inserted at instrumented
    sites by :func:`_instrument_via_fx`."""
    return torch.ops.firefly.capture(x, name)


@torch.fx.wrap
def _firefly_static_tap(x: torch.Tensor, tap_idx: int) -> torch.Tensor:
    """FX-opaque wrapper for CUDA-graph-mode capture.

    Looks up the active StaticTapper at runtime; the lookup happens during
    graph capture and the resulting op call (with its concrete buffer
    pointers baked in) is what survives into the captured graph for replay.
    """
    t = _active_static_tapper()
    if t is None:
        return x
    return torch.ops.firefly.capture_static(
        x, t.stats_buf, t.counter, tap_idx,
        t.blob_buf, t.blob_meta, t.blob_counter, t.alert_flag,
        t.full_tensor_policy.first_n_steps,
        t.full_tensor_policy.every_n_steps,
        1 if t.full_tensor_policy.on_alert else 0,
    )


def instrument(
    model: torch.nn.Module,
    pattern: str,
    *,
    mode: str = "eager",
    method: str = "auto",
    tap_index_start: int = 0,
) -> tuple[torch.nn.Module, dict[int, str]]:
    """Insert shadow capture ops at module sites matching ``pattern``.

    Args:
        model: The model to instrument. Modified in place for the
            ``named_modules`` path; replaced with a GraphModule for the
            ``fx`` path. Either way, the returned model is the one to
            use for inference.
        pattern: Regex matched against module names (e.g.
            ``r"layer\\.(7|15)\\.self_attn$"``).
        mode: ``"eager"`` (calls :func:`capture` with the tap name) or
            ``"static"`` (calls :func:`capture_static` with a tap index).
        method: ``"fx"`` forces torch.fx; ``"named_modules"`` wraps
            forwards; ``"auto"`` (default) tries fx first then falls
            back to named_modules if symbolic tracing fails.
        tap_index_start: For ``mode="static"``: starting tap_idx (useful
            when instrumenting multiple sub-models with disjoint indices).

    Returns:
        ``(instrumented_model, index_to_name)`` — the second element is a
        ``{tap_idx: module_name}`` map populated for ``mode="static"``,
        empty for ``mode="eager"``. Pass it to
        :class:`StaticTapper(index_to_name=...)` so the drain can re-attach
        names.
    """
    if mode not in ("eager", "static"):
        raise ValueError(f"mode must be 'eager' or 'static', got {mode!r}")
    if method not in ("fx", "named_modules", "auto"):
        raise ValueError(
            f"method must be 'fx', 'named_modules', or 'auto', got {method!r}"
        )

    if method == "fx":
        return _instrument_via_fx(model, pattern, mode, tap_index_start)
    if method == "named_modules":
        return _instrument_via_named_modules(model, pattern, mode, tap_index_start)
    # auto: try fx, fall back to named_modules on tracing failure.
    try:
        return _instrument_via_fx(model, pattern, mode, tap_index_start)
    except Exception as e:  # noqa: BLE001 — FX raises many types
        import sys
        print(
            f"[firefly] instrument: torch.fx tracing failed ({type(e).__name__}: "
            f"{e}); falling back to named_modules method.",
            file=sys.stderr,
        )
        return _instrument_via_named_modules(model, pattern, mode, tap_index_start)


def _instrument_via_fx(
    model: torch.nn.Module,
    pattern: str,
    mode: str,
    tap_index_start: int,
) -> tuple[torch.nn.Module, dict[int, str]]:
    """torch.fx symbolic-trace then insert capture call_function nodes."""
    pat = re.compile(pattern)
    gm = torch.fx.symbolic_trace(model)
    index_to_name: dict[int, str] = {}
    next_idx = tap_index_start

    # Iterate over a list copy — we mutate the graph during iteration.
    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue
        target_str = str(node.target)
        if not pat.search(target_str):
            continue

        if mode == "eager":
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    _firefly_eager_tap, args=(node, target_str)
                )
        else:  # static
            tap_idx = next_idx
            index_to_name[tap_idx] = target_str
            next_idx += 1
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    _firefly_static_tap, args=(node, tap_idx)
                )
        # Redirect every downstream use of `node` to read from `new_node`
        # instead. The replacement runs over all current uses; we then
        # restore `new_node`'s reference to the original `node` since the
        # capture wrapper needs the captured tensor as input.
        node.replace_all_uses_with(new_node)
        new_node.args = (node, *new_node.args[1:])

    gm.recompile()
    return gm, index_to_name


def _instrument_via_named_modules(
    model: torch.nn.Module,
    pattern: str,
    mode: str,
    tap_index_start: int,
) -> tuple[torch.nn.Module, dict[int, str]]:
    """Monkey-patch forward() on every module whose name matches the pattern."""
    pat = re.compile(pattern)
    index_to_name: dict[int, str] = {}
    next_idx = tap_index_start

    for name, mod in model.named_modules():
        if not pat.search(name):
            continue
        if mode == "eager":
            _wrap_forward_eager(mod, name)
        else:
            index_to_name[next_idx] = name
            _wrap_forward_static(mod, next_idx)
            next_idx += 1
    return model, index_to_name


def _wrap_forward_eager(mod: torch.nn.Module, name: str) -> None:
    """Replace ``mod.forward`` so its return value is routed through the
    eager capture op. Tuple-returning forwards get their first tensor
    element captured."""
    original = mod.forward

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return torch.ops.firefly.capture(out, name)
        if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            return (torch.ops.firefly.capture(out[0], name), *out[1:])
        return out

    mod.forward = wrapped


def _wrap_forward_static(mod: torch.nn.Module, tap_idx: int) -> None:
    """Like :func:`_wrap_forward_eager` but for CUDA-graph mode."""
    original = mod.forward

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        t = _active_static_tapper()
        if t is None:
            return out
        first_n = t.full_tensor_policy.first_n_steps
        every_n = t.full_tensor_policy.every_n_steps
        on_alert_enabled = 1 if t.full_tensor_policy.on_alert else 0
        if isinstance(out, torch.Tensor):
            return torch.ops.firefly.capture_static(
                out, t.stats_buf, t.counter, tap_idx,
                t.blob_buf, t.blob_meta, t.blob_counter, t.alert_flag,
                first_n, every_n, on_alert_enabled,
            )
        if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            head = torch.ops.firefly.capture_static(
                out[0], t.stats_buf, t.counter, tap_idx,
                t.blob_buf, t.blob_meta, t.blob_counter, t.alert_flag,
                first_n, every_n, on_alert_enabled,
            )
            return (head, *out[1:])
        return out

    mod.forward = wrapped
