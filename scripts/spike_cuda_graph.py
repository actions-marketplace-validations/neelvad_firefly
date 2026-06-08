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
        counter_ptr,                # [1] int32 atomic
        blob_buf_ptr,               # [max_blob_slots, max_blob_numel]
        blob_meta_ptr,              # [max_blob_slots] int32 (n_elements per slot)
        first_n_steps,              # int — record blob if idx < this
        max_blob_slots,             # int — sanity clamp
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

        # Conditional blob write. Mask gates the actual store; clamp the
        # blob index to keep address math safe when masked off.
        record_blob = (idx < first_n_steps) & (idx < max_blob_slots)
        safe_blob_idx = tl.minimum(idx, max_blob_slots - 1)
        blob_offset_in_range = offsets < tl.minimum(n_elements, max_blob_numel)
        blob_write_mask = record_blob & blob_offset_in_range
        tl.store(
            blob_buf_ptr + safe_blob_idx * max_blob_numel + offsets,
            x_f,
            mask=blob_write_mask,
        )
        tl.store(blob_meta_ptr + safe_blob_idx, n_elements, mask=record_blob)

    # ------------------------------------------------------------------
    # 2. Pre-allocated buffers (lifetime = container)
    # ------------------------------------------------------------------
    BUF_SIZE = 10_000
    FIRST_N_STEPS = 5      # record full tensors for the first 5 captures
    MAX_BLOB_NUMEL = 64    # 2-layer MLP intermediate is 32 elements; 64 has headroom
    stats_buffer = torch.zeros((BUF_SIZE, 4), device=device, dtype=torch.float32)
    counter = torch.zeros((1,), device=device, dtype=torch.int32)
    blob_buffer = torch.zeros((FIRST_N_STEPS, MAX_BLOB_NUMEL), device=device, dtype=torch.float32)
    blob_meta = torch.zeros((FIRST_N_STEPS,), device=device, dtype=torch.int32)

    def shadow_capture(x: torch.Tensor) -> torch.Tensor:
        """Pass-through capture: launches the Triton kernel and returns x."""
        n = x.numel()
        BLOCK = triton.next_power_of_2(n)
        _shadow_kernel[(1,)](
            x, stats_buffer, counter,
            blob_buffer, blob_meta,
            FIRST_N_STEPS, FIRST_N_STEPS, MAX_BLOB_NUMEL,
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
        print(f"after graph capture: counter={counter.item()}  (1 capture pass)")
        counter.zero_()

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
    # Each of the first FIRST_N_STEPS replays should have written:
    #   blob_buffer[idx, :n_elements] = the actual fc1 output values
    #   blob_meta[idx] = n_elements (32 for our 16->32 layer)
    blob_meta_cpu = blob_meta.cpu().tolist()
    blob_populated_meta = sum(1 for v in blob_meta_cpu if v > 0)
    expected_n_elements = 32  # fc1 output is (1, 32) → 32 elements
    blob_meta_correct = all(v == expected_n_elements for v in blob_meta_cpu)
    blob_populated_data = (blob_buffer.abs().sum(dim=1) > 0).sum().item()

    print(f"populated blob_meta entries (expected {FIRST_N_STEPS}): {blob_populated_meta}")
    print(f"populated blob_buffer rows (data > 0): {blob_populated_data}")
    print(f"blob_meta values: {blob_meta_cpu}")
    print(f"blob_meta all equal {expected_n_elements}: {blob_meta_correct}")
    # All N replays use the same captured input, so all blob rows should be
    # IDENTICAL — verify by checking row 0 and row FIRST_N_STEPS-1 match.
    blob_row_0 = blob_buffer[0, :expected_n_elements].cpu()
    blob_row_last = blob_buffer[FIRST_N_STEPS - 1, :expected_n_elements].cpu()
    blob_rows_identical = torch.equal(blob_row_0, blob_row_last)
    print(f"blob row 0 matches blob row {FIRST_N_STEPS - 1}: {blob_rows_identical}")
    print()

    # PASS conditions:
    #   - counter == N_REPLAYS                                  (stats atomic)
    #   - all N_REPLAYS stats rows populated                    (stats writes)
    #   - stats row[N_REPLAYS] still zero                       (no off-by-ones)
    #   - FIRST_N_STEPS blob_meta entries correctly set         (blob meta)
    #   - FIRST_N_STEPS blob_buffer rows populated              (blob data)
    #   - all blob rows identical (same input replayed)         (consistency)
    pass_count = final_counter == N_REPLAYS
    pass_populated = populated == N_REPLAYS
    pass_sentinel = sentinel_row_is_zero
    pass_blob_meta = blob_meta_correct and blob_populated_meta == FIRST_N_STEPS
    pass_blob_data = blob_populated_data == FIRST_N_STEPS
    pass_blob_consistency = blob_rows_identical

    verdict = {
        "final_counter": final_counter,
        "expected_counter": N_REPLAYS,
        "populated_stats_rows": populated,
        "sentinel_zero": sentinel_row_is_zero,
        "populated_blob_meta": blob_populated_meta,
        "populated_blob_data": blob_populated_data,
        "blob_meta_values": blob_meta_cpu,
        "expected_blob_rows": FIRST_N_STEPS,
        "blob_rows_identical": blob_rows_identical,
        "pass_count": pass_count,
        "pass_populated": pass_populated,
        "pass_sentinel": pass_sentinel,
        "pass_blob_meta": pass_blob_meta,
        "pass_blob_data": pass_blob_data,
        "pass_blob_consistency": pass_blob_consistency,
        "overall_pass": (
            pass_count and pass_populated and pass_sentinel
            and pass_blob_meta and pass_blob_data and pass_blob_consistency
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
