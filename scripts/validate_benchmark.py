"""GPU validation for the vLLM benchmarker — measured QPS + memory.

Proves the measured-cost primitive end to end on a real GPU: stand a model up in
vLLM (CUDA graphs on), time prefill + steady-state decode, read peak/weight
memory from the worker, for an unquantized baseline AND a vLLM-native quant
(fp8). The point is the *measurement* the analytic cost model can't make —
whether the quant actually serves faster, and by how much, at this batch/seqlen.

Reading: weight memory should drop under fp8; decode throughput is the QPS the
user feels and is regime-dependent (so it must be measured, not assumed).

Run:  uv run modal run scripts/validate_benchmark.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-benchmark")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "transformers>=4.44")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# (label, quantization-or-None) — one per container (vLLM holds ~90% of the GPU
# and caches engine state per process, so configs must not share a process).
CONFIGS = [("bf16 baseline", None), ("fp8", "fp8")]


@app.function(
    image=image, gpu=GPU, timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def bench_one(label: str, quant: str | None) -> dict:
    import torch

    from firefly.bench import BenchmarkConfig, get_benchmarker

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  [{label}]")
    cfg = BenchmarkConfig(batch_size=16, input_len=512, output_len=128, n_iters=3, warmup_iters=1)
    try:
        r = get_benchmarker("vllm").benchmark(MODEL, cfg, dtype="bfloat16", quantization=quant)
        entry = {
            "label": label,
            "decode_tok_s": round(r.decode_throughput_tok_s, 1),
            "prefill_tok_s": round(r.prefill_throughput_tok_s, 1),
            "ttft_ms": round(r.ttft_ms, 1),
            "weight_mb": round(r.weight_memory_bytes / 1e6, 1) if r.weight_memory_bytes else None,
            "peak_mb": round(r.peak_memory_bytes / 1e6, 1) if r.peak_memory_bytes else None,
        }
        print(f"  decode {entry['decode_tok_s']} tok/s | prefill {entry['prefill_tok_s']} tok/s | "
              f"TTFT {entry['ttft_ms']} ms | weights {entry['weight_mb']} MB | peak {entry['peak_mb']} MB")
    except Exception as e:  # noqa: BLE001 — resilient: a config that won't load is skipped
        entry = {"label": label, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"  SKIPPED: {entry['error']}")
    return entry


@app.local_entrypoint()
def main() -> None:
    import json

    # One container per config (.starmap) → no shared vLLM process.
    out = list(bench_one.starmap(CONFIGS))
    print(f"\n{'=' * 70}\nMEASURED-COST SUMMARY ({MODEL})\n{'=' * 70}")
    for e in out:
        if "error" in e:
            print(f"  {e['label']:16s}  SKIPPED ({e['error'][:50]})")
        else:
            print(f"  {e['label']:16s}  {e['decode_tok_s']:8.1f} dec tok/s  "
                  f"{e['weight_mb']} MB weights  {e['peak_mb']} MB peak")
    print(json.dumps(out, indent=2, default=str))
