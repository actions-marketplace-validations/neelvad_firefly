"""Spike: does a Triton-backed pass-through capture survive CUDA-graph capture + replay?

The shadow-mode CUDA-graph compat story (v1.1) hinges on this question.
The eager + torch.compile MVP works because the op body runs at Python
level — but CUDA graphs capture a fixed sequence of GPU kernel launches
and "replay" them with zero CPU work between launches. CPU-side Python
in the op body never runs during replay, so our existing Tapper-based
recording silently misses every captured forward.

The proposed CUDA-graph-safe design:
  1. Pre-allocated GPU buffers (stats + counter), lifetime = container
  2. A Triton kernel that computes stats + atomic-incs counter + writes
     to buffer[idx] — all on-device, captureable into a graph
  3. CPU-side drain reads the buffer between graph replays

Pass criterion:
  - We capture a small model (linear + capture + linear) into a CUDAGraph
  - We replay it N times
  - The counter increments to exactly N
  - The buffer contains N populated rows of stats

Run:  uv run modal run scripts/spike_cuda_graph.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-spike-cuda-graph")

# Lean image: just PyTorch + CUDA. Triton is a torch dep on Linux/CUDA
# so we get it for free without a separate install.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)


@app.function(image=image, gpu="A10G", timeout=600)
def spike() -> dict:
    """Runs entirely on the GPU container; returns a verdict dict.

    Tests two scenarios:
    1. Stats-only (first_n_steps=0, placeholder blob buffers): the kernel
       runs, stats accumulate, blob buffer stays untouched.
    2. Stats + blob (first_n_steps=5): the first 5 calls record both stats
       and the full input tensor; calls 6+ are stats-only.
    """
    import torch
    import torch.nn as nn
    import triton
    import triton.language as tl

    print(f"torch: {torch.__version__}")
    print(f"triton: {triton.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    device = "cuda"

    # ------------------------------------------------------------------
    # 1. Triton kernel: stats + conditional full-tensor blob write
    # ------------------------------------------------------------------
    @triton.jit
    def _shadow_kernel(
        x_ptr,
        stats_buf_ptr,              # [N, 4] stats
        counter_ptr,                # [1] int32 stats atomic
        blob_buf_ptr,               # [n_blob_slots, max_blob_numel]
        blob_meta_ptr,              # [n_blob_slots, 2] int32 (numel, global_idx)
        blob_counter_ptr,           # [1] int32 blob ring-buffer atomic
        first_n_steps,              # int — record blob if idx < this
        every_n_steps,              # int — record blob if idx % this == 0; 0 disables
        n_blob_slots,               # int — ring buffer capacity
        max_blob_numel,             # int — per-slot capacity
        n_elements,
        BLOCK_SIZE: tl.constexpr,
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
        tl.store(stats_buf_ptr + idx * 4 + 0, s_mean)
        tl.store(stats_buf_ptr + idx * 4 + 1, s_abs_mean)
        tl.store(stats_buf_ptr + idx * 4 + 2, s_abs_max)
        tl.store(stats_buf_ptr + idx * 4 + 3, s_std)

        # Composable blob recording: first_n OR every_n.
        record_first_n = (first_n_steps > 0) & (idx < first_n_steps)
        record_every_n = (every_n_steps > 0) & ((idx % every_n_steps) == 0)
        should_record_blob = record_first_n | record_every_n

        # Atomic blob slot (gated by mask).
        blob_count = tl.atomic_add(blob_counter_ptr, 1, mask=should_record_blob)
        blob_slot = blob_count % n_blob_slots
        safe_blob_slot = tl.minimum(blob_slot, n_blob_slots - 1)

        # Blob data + metadata writes.
        blob_offset_in_range = offsets < tl.minimum(n_elements, max_blob_numel)
        blob_write_mask = should_record_blob & blob_offset_in_range
        tl.store(
            blob_buf_ptr + safe_blob_slot * max_blob_numel + offsets,
            x_f,
            mask=blob_write_mask,
        )
        tl.store(blob_meta_ptr + safe_blob_slot * 2 + 0, n_elements, mask=should_record_blob)
        tl.store(blob_meta_ptr + safe_blob_slot * 2 + 1, idx, mask=should_record_blob)

    # ------------------------------------------------------------------
    # 2. Pre-allocated buffers (lifetime = container)
    # ------------------------------------------------------------------
    BUF_SIZE = 10_000
    FIRST_N_STEPS = 2      # record full tensors for the first 2 captures
    EVERY_N_STEPS = 10     # AND every 10th capture
    N_BLOB_SLOTS = 16      # ring buffer for blobs
    MAX_BLOB_NUMEL = 64    # 2-layer MLP intermediate is 32 elements; 64 has headroom
    stats_buffer = torch.zeros((BUF_SIZE, 4), device=device, dtype=torch.float32)
    counter = torch.zeros((1,), device=device, dtype=torch.int32)
    blob_buffer = torch.zeros((N_BLOB_SLOTS, MAX_BLOB_NUMEL), device=device, dtype=torch.float32)
    blob_meta = torch.zeros((N_BLOB_SLOTS, 2), device=device, dtype=torch.int32)
    blob_counter = torch.zeros((1,), device=device, dtype=torch.int32)

    def shadow_capture(x: torch.Tensor) -> torch.Tensor:
        """Pass-through capture: launches the Triton kernel and returns x."""
        n = x.numel()
        BLOCK = triton.next_power_of_2(n)
        _shadow_kernel[(1,)](
            x, stats_buffer, counter,
            blob_buffer, blob_meta, blob_counter,
            FIRST_N_STEPS, EVERY_N_STEPS, N_BLOB_SLOTS, MAX_BLOB_NUMEL,
            n, BLOCK_SIZE=BLOCK,
        )
        return x

    # ------------------------------------------------------------------
    # 3. Tiny model with one capture site
    # ------------------------------------------------------------------
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc1(x)
            h = shadow_capture(h)        # ← the captureable kernel
            return self.fc2(h)

    model = Tiny().eval().to(device)
    x = torch.randn(1, 16, device=device)

    # ------------------------------------------------------------------
    # 4. Warmup pass — gets Triton to JIT-compile its kernel before
    #    CUDA-graph capture (compilation is CPU-side and would otherwise
    #    happen inside capture, breaking the graph).
    # ------------------------------------------------------------------
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        print(f"after eager warmup: counter={counter.item()}")
        # Reset for a clean graph-capture test.
        counter.zero_()

        # ----------------------------------------------------------------
        # 5. Capture into CUDA graph (PyTorch's required side-stream dance)
        # ----------------------------------------------------------------
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(x)
        torch.cuda.current_stream().wait_stream(s)
        # Reset again — warmup pass on side stream populated the counter.
        counter.zero_()

        with torch.cuda.graph(g):
            _ = model(x)
        print(f"after graph capture: counter={counter.item()}  (record-only)")
        # Reset all GPU state so the replay-phase verification sees only
        # the replays' contribution. The captured kernels reference these
        # buffers by pointer, so the resets don't break the graph.
        counter.zero_()
        blob_counter.zero_()
        blob_buffer.zero_()
        blob_meta.zero_()

        # ----------------------------------------------------------------
        # 6. Replay N times
        # ----------------------------------------------------------------
        N_REPLAYS = 100
        for _ in range(N_REPLAYS):
            g.replay()
        torch.cuda.synchronize()

        final_counter = counter.item()
        print(f"after {N_REPLAYS} replays: counter={final_counter}")
        print()

    # ------------------------------------------------------------------
    # 7. Verify stats
    # ------------------------------------------------------------------
    populated = (stats_buffer[:N_REPLAYS].abs().sum(dim=1) > 0).sum().item()
    sentinel_row_is_zero = stats_buffer[N_REPLAYS].abs().sum().item() == 0.0

    print(f"populated stats rows in [0, {N_REPLAYS}): {populated}")
    print(f"stats row {N_REPLAYS} is zero (sentinel): {sentinel_row_is_zero}")
    print()

    # ------------------------------------------------------------------
    # 8. Verify blob recording
    # ------------------------------------------------------------------
    # Expected blob captures across 100 replays with FIRST_N_STEPS=2,
    # EVERY_N_STEPS=10:
    #   first_n fires at idx 0, 1                  → 2 captures
    #   every_n fires at idx 0, 10, 20, ..., 90    → 10 captures
    # OR-combined: {0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90} → 11 distinct.
    # Each fires the atomic counter once; total blob_count = 11.
    expected_blob_count = 11
    final_blob_count = blob_counter.item()

    blob_meta_cpu = blob_meta.cpu().tolist()
    blob_populated_meta = sum(1 for row in blob_meta_cpu if row[0] > 0)
    expected_n_elements = 32  # fc1 output is (1, 32) → 32 elements
    blob_meta_correct = all(row[0] == expected_n_elements for row in blob_meta_cpu if row[0] > 0)

    print(f"blob_counter (expected {expected_blob_count}): {final_blob_count}")
    print(f"populated blob_meta entries: {blob_populated_meta}")
    print(f"blob_meta values: {blob_meta_cpu}")
    print(f"blob_meta numel all equal {expected_n_elements}: {blob_meta_correct}")

    # The 11 blob captures wrote to ring-buffer slots [0..10] (since
    # 11 < N_BLOB_SLOTS=16, no wraparound). Verify the global_idx field
    # of blob_meta matches the expected pattern.
    global_idxs = sorted(row[1] for row in blob_meta_cpu if row[0] > 0)
    expected_global_idxs = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    blob_global_idxs_correct = global_idxs == expected_global_idxs
    print(f"blob global_idx values (sorted): {global_idxs}")
    print(f"expected global_idx values:      {expected_global_idxs}")
    print(f"global_idxs match: {blob_global_idxs_correct}")
    print()

    pass_count = final_counter == N_REPLAYS
    pass_populated = populated == N_REPLAYS
    pass_sentinel = sentinel_row_is_zero
    pass_blob_count = final_blob_count == expected_blob_count
    pass_blob_global_idxs = blob_global_idxs_correct
    pass_blob_meta_numel = blob_meta_correct

    verdict = {
        "final_counter": final_counter,
        "expected_counter": N_REPLAYS,
        "populated_stats_rows": populated,
        "sentinel_zero": sentinel_row_is_zero,
        "final_blob_count": final_blob_count,
        "expected_blob_count": expected_blob_count,
        "blob_global_idxs": global_idxs,
        "expected_global_idxs": expected_global_idxs,
        "pass_count": pass_count,
        "pass_populated": pass_populated,
        "pass_sentinel": pass_sentinel,
        "pass_blob_count": pass_blob_count,
        "pass_blob_global_idxs": pass_blob_global_idxs,
        "pass_blob_meta_numel": pass_blob_meta_numel,
        "overall_pass": (
            pass_count and pass_populated and pass_sentinel
            and pass_blob_count and pass_blob_global_idxs and pass_blob_meta_numel
        ),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
    }

    print("=" * 60)
    print(f"OVERALL: {'PASS' if verdict['overall_pass'] else 'FAIL'}")
    print("=" * 60)
    return verdict


@app.local_entrypoint()
def main() -> None:
    verdict = spike.remote()
    print()
    print("verdict:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
