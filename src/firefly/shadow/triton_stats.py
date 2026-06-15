"""Lazily-built Triton kernel for CUDA-graph-mode summary stats."""

from __future__ import annotations


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
