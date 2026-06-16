"""The pass-through custom ops (Dynamo/CUDA-graph opaque)."""

from __future__ import annotations

import torch

from firefly.shadow.eager import _active_tapper
from firefly.shadow.triton_stats import (
    MAX_BLOCKS,
    REDUCE_BLOCK,
    _get_kernels,
    _import_triton,
    reduction_grid,
)


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
    mutates_args=(
        "stats_buf", "counter", "blob_buf", "blob_meta", "blob_counter",
        "partials", "decision",
    ),
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
    partials: torch.Tensor,
    decision: torch.Tensor,
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

    The reduction runs as a bounded, two-phase grid-stride kernel (see
    :mod:`firefly.shadow.triton_stats`): ``partials`` is the per-program
    scratch buffer and ``decision`` carries the blob slot from the finalize
    kernel to the copy kernel. Both are pre-allocated by :class:`StaticTapper`.

    To opt out of full-tensor recording, pass 1x1 placeholder
    ``blob_buf`` / ``blob_meta`` buffers and all three policy ints at 0; the
    blob-copy kernel is skipped entirely. :class:`StaticTapper` handles this.
    """
    _import_triton()
    partial_reduce, finalize, blob_copy = _get_kernels()
    n = x.numel()
    grid = reduction_grid(n)
    n_blob_slots = blob_buf.shape[0]
    max_blob_numel = blob_buf.shape[1]

    # Phase 1: bounded grid-stride partial reductions.
    partial_reduce[(grid,)](x, partials, n, BLOCK=REDUCE_BLOCK)
    # Phase 2: single-block combine + stats-row + blob decision.
    finalize[(1,)](
        partials, grid, n,
        stats_buf, counter, tap_idx,
        blob_meta, blob_counter, alert_flag,
        first_n_steps, every_n_steps, on_alert_enabled, n_blob_slots,
        decision, MAXB=MAX_BLOCKS,
    )
    # Phase 3: copy the full tensor into the chosen slot, only when this
    # tapper records full tensors at all (real, non-placeholder blob buffer).
    if max_blob_numel > 1:
        blob_copy[(grid,)](x, blob_buf, decision, n, max_blob_numel, BLOCK=REDUCE_BLOCK)
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
    partials: torch.Tensor,
    decision: torch.Tensor,
    first_n_steps: int,
    every_n_steps: int,
    on_alert_enabled: int,
) -> torch.Tensor:
    return torch.empty_like(x)
