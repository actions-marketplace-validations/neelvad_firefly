"""Eager / torch.compile capture: Tapper, @tap, FullTensorPolicy."""

from __future__ import annotations

import functools
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from firefly.shadow.buffer import _Buffer, _Event

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
