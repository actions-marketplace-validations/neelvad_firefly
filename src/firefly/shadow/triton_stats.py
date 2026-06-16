"""Lazily-built Triton kernels for CUDA-graph-mode summary stats.

The reduction is a standard two-phase, grid-stride design so it scales to
arbitrarily large taps. An earlier version launched a single block sized to
``next_power_of_2(numel)`` — fine for a per-token hidden vector, but it fails
to compile / runs out of resources once a tap is more than a block's worth of
elements (e.g. a full-prefill ``[1, seq, hidden]`` activation). The three
kernels here split the work:

1. ``_partial_reduce`` — a bounded grid of ``<= MAX_BLOCKS`` programs, each
   grid-striding over its share of the tensor and writing one partial
   ``(sum, abs_sum, sum_sq, abs_max)`` row.
2. ``_finalize`` — a single program reduces the (at most ``MAX_BLOCKS``)
   partials into the final stats, allocates the stats-row index, and makes the
   blob-recording decision, publishing the chosen slot for kernel 3.
3. ``_blob_copy`` — the same bounded grid-stride pattern, copying the full
   tensor into the chosen ring-buffer slot when the policy fired.

The grid is *capped* at ``MAX_BLOCKS`` and shapes are static under CUDA graphs,
so every buffer (partials, decision) has a fixed shape that can be captured.
"""

from __future__ import annotations

#: Phase-1 block width (elements per program per grid-stride step).
REDUCE_BLOCK = 2048
#: Upper bound on phase-1 programs == rows in the partials buffer. The finalize
#: kernel reduces this many partials in a single block, so it is also the
#: finalize block width (a power of two). 1024 partials covers ~2M elements in
#: one stride step and any larger tap via grid-stride looping.
MAX_BLOCKS = 1024


def reduction_grid(n_elements: int) -> int:
    """Number of phase-1 programs: enough to cover ``n_elements`` in blocks of
    ``REDUCE_BLOCK``, but never more than ``MAX_BLOCKS`` (excess elements are
    handled by each program's grid-stride loop). At least 1.

    Pure and CPU-testable — this is the bound that prevents the single-block
    blow-up, independent of any GPU.
    """
    blocks = (n_elements + REDUCE_BLOCK - 1) // REDUCE_BLOCK
    return max(1, min(blocks, MAX_BLOCKS))


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


_KERNELS = None


def _get_kernels():
    """Lazily JIT-compile and cache the three reduction kernels.

    Returns ``(partial_reduce, finalize, blob_copy)``. Cached so the JIT cost
    is paid once per process.
    """
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS

    triton, tl = _import_triton()

    @triton.jit
    def _partial_reduce(
        x_ptr,                       # input tensor
        partials_ptr,                # [MAX_BLOCKS, 4] -> (sum, abs_sum, sum_sq, abs_max)
        n_elements,                  # numel(x)
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        acc_sum = 0.0
        acc_abs = 0.0
        acc_sq = 0.0
        acc_max = 0.0
        step = nprog * BLOCK
        # Grid-stride loop: each program covers indices pid*BLOCK, then strides
        # by (nprog*BLOCK) until the tensor is exhausted.
        for off in range(pid * BLOCK, n_elements, step):
            offs = off + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            ax = tl.abs(x)
            acc_sum += tl.sum(x, axis=0)
            acc_abs += tl.sum(ax, axis=0)
            acc_sq += tl.sum(x * x, axis=0)
            acc_max = tl.maximum(acc_max, tl.max(ax, axis=0))
        tl.store(partials_ptr + pid * 4 + 0, acc_sum)
        tl.store(partials_ptr + pid * 4 + 1, acc_abs)
        tl.store(partials_ptr + pid * 4 + 2, acc_sq)
        tl.store(partials_ptr + pid * 4 + 3, acc_max)

    @triton.jit
    def _finalize(
        partials_ptr,                # [MAX_BLOCKS, 4]
        n_partials,                  # how many partial rows are live this call
        n_elements,                  # numel(x) — the population size for means
        stats_buf_ptr,               # [N, 5] stats ring buffer
        counter_ptr,                 # [1] int32 stats atomic counter
        tap_idx,                     # int — which tap
        blob_meta_ptr,               # [n_slots, 2] int32 (numel, global_idx)
        blob_counter_ptr,            # [1] int32 blob-ring atomic counter
        alert_flag_ptr,              # [1] int32 — CPU-toggled alert flag
        first_n_steps,               # int — record blob if idx < this
        every_n_steps,               # int — record blob if idx % this == 0; 0 disables
        on_alert_enabled,            # int — 1: consult alert_flag; 0: ignore
        n_blob_slots,                # int — ring capacity
        decision_ptr,                # [2] int32 -> (slot, should_record) for kernel 3
        MAXB: tl.constexpr,          # power-of-2 >= MAX_BLOCKS
    ):
        offs = tl.arange(0, MAXB)
        m = offs < n_partials
        psum = tl.sum(tl.load(partials_ptr + offs * 4 + 0, mask=m, other=0.0), axis=0)
        pabs = tl.sum(tl.load(partials_ptr + offs * 4 + 1, mask=m, other=0.0), axis=0)
        psq = tl.sum(tl.load(partials_ptr + offs * 4 + 2, mask=m, other=0.0), axis=0)
        pmax = tl.max(tl.load(partials_ptr + offs * 4 + 3, mask=m, other=0.0), axis=0)

        mean = psum / n_elements
        abs_mean = pabs / n_elements
        # Population variance via E[x^2] - E[x]^2 (one-pass; matches the
        # previous kernel's divide-by-n semantics). Clamp tiny negatives from
        # floating-point cancellation before the sqrt.
        var = psq / n_elements - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)

        idx = tl.atomic_add(counter_ptr, 1)
        tl.store(stats_buf_ptr + idx * 5 + 0, mean)
        tl.store(stats_buf_ptr + idx * 5 + 1, abs_mean)
        tl.store(stats_buf_ptr + idx * 5 + 2, pmax)
        tl.store(stats_buf_ptr + idx * 5 + 3, std)
        # tap_idx arrives as a Python int; coerce to float via *1.0 — the most
        # version-portable promotion across Triton releases.
        tl.store(stats_buf_ptr + idx * 5 + 4, tap_idx * 1.0)

        # Blob-recording decision (composable first_n + every_n + on_alert).
        record_first = (first_n_steps > 0) & (idx < first_n_steps)
        record_every = (every_n_steps > 0) & ((idx % every_n_steps) == 0)
        alert_now = tl.load(alert_flag_ptr) != 0
        record_alert = (on_alert_enabled != 0) & alert_now
        should = record_first | record_every | record_alert

        # Allocate a ring slot only when recording. When the mask is off the
        # atomic is skipped (its return is undefined, so we clamp before use
        # and gate every dependent store on `should`).
        blob_count = tl.atomic_add(blob_counter_ptr, 1, mask=should)
        blob_slot = blob_count % n_blob_slots
        safe = tl.maximum(tl.minimum(blob_slot, n_blob_slots - 1), 0)
        tl.store(blob_meta_ptr + safe * 2 + 0, n_elements, mask=should)
        tl.store(blob_meta_ptr + safe * 2 + 1, idx, mask=should)

        # Publish the decision for the (separately launched) blob-copy kernel.
        tl.store(decision_ptr + 0, safe)
        tl.store(decision_ptr + 1, tl.where(should, 1, 0))

    @triton.jit
    def _blob_copy(
        x_ptr,                       # input tensor
        blob_buf_ptr,                # [n_slots, max_blob_numel]
        decision_ptr,                # [2] int32 -> (slot, should_record)
        n_elements,                  # numel(x)
        max_blob_numel,              # per-slot capacity
        BLOCK: tl.constexpr,
    ):
        should = tl.load(decision_ptr + 1)
        slot = tl.load(decision_ptr + 0)
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        cap = tl.minimum(n_elements, max_blob_numel)
        step = nprog * BLOCK
        for off in range(pid * BLOCK, cap, step):
            offs = off + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0).to(tl.float32)
            write_mask = (offs < cap) & (should != 0)
            tl.store(blob_buf_ptr + slot * max_blob_numel + offs, x, mask=write_mask)

    _KERNELS = (_partial_reduce, _finalize, _blob_copy)
    return _KERNELS
