"""The pass-through custom ops (Dynamo/CUDA-graph opaque)."""

from __future__ import annotations

import torch

from firefly.shadow.eager import _active_tapper
from firefly.shadow.triton_stats import _get_stats_kernel, _import_triton


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
