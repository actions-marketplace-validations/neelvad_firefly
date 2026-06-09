"""Minimal standalone repro: FlashInfer returns all-zero outputs for two
attention heads of Qwen2.5-7B layer 27 (BF16, prefill).

This is the self-contained script for the upstream issue — no Firefly
dependency. The core logic (between the markers) is what goes in the issue;
the Modal scaffolding just provides the GPU + pinned environment.

Run:
    uv run modal run scripts/repro_flashinfer_zero_heads.py --backend FLASHINFER
    uv run modal run scripts/repro_flashinfer_zero_heads.py --backend FLASH_ATTN
"""

from __future__ import annotations

import modal

app = modal.App("firefly-repro-flashinfer-zero-heads")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("vllm==0.22.1", "transformers>=4.55", "huggingface_hub>=0.24")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def repro(backend: str = "FLASHINFER") -> None:
    # --- begin standalone repro (paste into upstream issue) -----------------
    import os

    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # collective_rpc callables

    from vllm import LLM, SamplingParams

    N_HEADS, HEAD_DIM, LAYER = 28, 128, 27  # Qwen2.5-7B: 28 q-heads, 4 kv-heads

    def install_hook(worker):
        model = worker.model_runner.model
        store = {}

        def hook(_mod, inputs, _out):
            # o_proj input = concatenated per-head attention outputs,
            # shape (num_tokens, n_heads * head_dim) — the only place
            # per-head outputs are observable before they're mixed.
            store["x"] = inputs[0].detach()

        model.model.layers[LAYER].self_attn.o_proj.register_forward_hook(hook)
        worker._repro_store = store
        return True

    def read_per_head_max(worker):
        x = worker._repro_store["x"].float()
        heads = x.reshape(x.shape[0], N_HEADS, HEAD_DIM)
        return [float(heads[:, h, :].abs().max()) for h in range(N_HEADS)]

    llm = LLM(
        model="Qwen/Qwen2.5-7B",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.7,
        attention_backend=backend,  # "FLASHINFER" or "FLASH_ATTN"
    )
    llm.collective_rpc(install_hook)
    llm.generate(
        ["the quick brown fox jumps over the lazy dog"],
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    per_head_max = llm.collective_rpc(read_per_head_max)[0]

    print(f"\nbackend={backend}  layer={LAYER}  per-head max|attention output|:")
    zeroed = []
    for h, v in enumerate(per_head_max):
        marker = "   <-- ALL ZERO" if v == 0.0 else ""
        print(f"  head {h:2d}: {v:10.4f}{marker}")
        if v == 0.0:
            zeroed.append(h)
    print(f"\nzeroed heads: {zeroed or 'none'}")
    # --- end standalone repro ------------------------------------------------


@app.local_entrypoint()
def main(backend: str = "FLASHINFER") -> None:
    repro.remote(backend=backend)
