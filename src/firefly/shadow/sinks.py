"""Persistence sinks: local JSONL+blobs and streaming S3/GCS/Azure."""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from firefly.shadow.buffer import DrainThread, _Buffer, _Event


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
