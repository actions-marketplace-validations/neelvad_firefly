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
    """Runs entirely on the GPU container; returns a verdict dict."""
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
    # 1. Triton kernel: compute stats + atomic-write to buffer
    # ------------------------------------------------------------------
    @triton.jit
    def _shadow_kernel(
        x_ptr,                       # input tensor
        stats_buf_ptr,               # [N, 4] stats buffer
        counter_ptr,                 # [1] int32 atomic counter
        n_elements,                  # number of input elements
        BLOCK_SIZE: tl.constexpr,    # power-of-2 >= n_elements
    ):
        # Single-program reduction over the whole input tensor.
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # 4 stats. Cast to float32 to keep math in fp32 regardless of input dtype.
        x_f = x.to(tl.float32)
        s_mean = tl.sum(x_f, axis=0) / n_elements
        abs_x = tl.abs(x_f)
        s_abs_mean = tl.sum(abs_x, axis=0) / n_elements
        s_abs_max = tl.max(abs_x, axis=0)
        diff = x_f - s_mean
        s_std = tl.sqrt(tl.sum(diff * diff, axis=0) / n_elements)

        # Atomic slot acquisition.
        idx = tl.atomic_add(counter_ptr, 1)

        # Write the 4 stats to slot idx.
        tl.store(stats_buf_ptr + idx * 4 + 0, s_mean)
        tl.store(stats_buf_ptr + idx * 4 + 1, s_abs_mean)
        tl.store(stats_buf_ptr + idx * 4 + 2, s_abs_max)
        tl.store(stats_buf_ptr + idx * 4 + 3, s_std)

    # ------------------------------------------------------------------
    # 2. Pre-allocated buffers (lifetime = container)
    # ------------------------------------------------------------------
    BUF_SIZE = 10_000
    stats_buffer = torch.zeros((BUF_SIZE, 4), device=device, dtype=torch.float32)
    counter = torch.zeros((1,), device=device, dtype=torch.int32)

    def shadow_capture(x: torch.Tensor) -> torch.Tensor:
        """Pass-through capture: launches the Triton kernel and returns x."""
        n = x.numel()
        BLOCK = triton.next_power_of_2(n)
        _shadow_kernel[(1,)](x, stats_buffer, counter, n, BLOCK_SIZE=BLOCK)
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
    # 7. Verify
    # ------------------------------------------------------------------
    populated = (stats_buffer[:N_REPLAYS].abs().sum(dim=1) > 0).sum().item()
    sentinel_row_is_zero = stats_buffer[N_REPLAYS].abs().sum().item() == 0.0

    first_row = stats_buffer[0].tolist()
    last_populated_row = stats_buffer[N_REPLAYS - 1].tolist()

    print(f"populated rows in [0, {N_REPLAYS}): {populated}")
    print(f"row {N_REPLAYS} is zero (sentinel): {sentinel_row_is_zero}")
    print(f"first row stats: {first_row}")
    print(f"row {N_REPLAYS - 1} stats: {last_populated_row}")
    print()

    # PASS conditions:
    #   - counter == N_REPLAYS  (atomic increments captured)
    #   - all N_REPLAYS rows populated  (writes captured)
    #   - row[N_REPLAYS] still zero  (no off-by-ones)
    pass_count = final_counter == N_REPLAYS
    pass_populated = populated == N_REPLAYS
    pass_sentinel = sentinel_row_is_zero

    verdict = {
        "final_counter": final_counter,
        "expected_counter": N_REPLAYS,
        "populated_rows": populated,
        "sentinel_zero": sentinel_row_is_zero,
        "pass_count": pass_count,
        "pass_populated": pass_populated,
        "pass_sentinel": pass_sentinel,
        "overall_pass": pass_count and pass_populated and pass_sentinel,
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
