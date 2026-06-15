"""Event records, the bounded ring buffer, and the drain thread."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

import torch


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
