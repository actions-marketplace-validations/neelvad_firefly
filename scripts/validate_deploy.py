"""GPU validation for the deploy loop — apply → save → vLLM-load → benchmark.

Proves Gap 2 end to end: a uniform w8a8 recipe is exported to a real torchao
checkpoint (firefly.quant.deploy.export_deployable), then that *saved directory*
is loaded by vLLM (quantization=torchao) and benchmarked — confirming the
artifact actually serves, and measuring the speedup/footprint vs the bf16
baseline. This is the difference between handing someone a recipe and handing
them a faster model.

Each config runs in its own container (vLLM holds ~90% of the GPU per process).

Run:  uv run modal run scripts/validate_deploy.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-deploy")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "torchao>=0.7", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# (label, export-as-w8a8?)
CONFIGS = [("bf16 baseline", False), ("w8a8 exported", True)]


@app.function(
    image=image, gpu=GPU, timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def bench_one(label: str, do_export: bool) -> dict:
    import gc

    import torch

    from firefly.bench import BenchmarkConfig, get_benchmarker
    from firefly.quant.deploy import classify_recipe, export_deployable
    from firefly.quant.intervention import RTNQuantizer
    from firefly.quant.recipe_io import Recipe, serialize_intervention

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  [{label}]")
    cfg = BenchmarkConfig(batch_size=16, input_len=512, output_len=128, n_iters=3, warmup_iters=1)
    bench = get_benchmarker("vllm")

    try:
        if do_export:
            recipe = Recipe(
                model_id=MODEL, scheme="w8a8", group_size=32, granularity="layer",
                quantize_fqns=[], kept_fp_fqns=[], pre_transforms=[],
                quantizer=serialize_intervention(RTNQuantizer()),
            )
            status, reason = classify_recipe(recipe)
            print(f"  deployability: {status} — {reason}")
            art = export_deployable(recipe, "/tmp/q_w8a8", dtype="bfloat16", device="cuda")
            print(f"  exported → {art.path}\n  serve: {art.serve_command}")
            # Free the transformers export model before vLLM grabs the GPU.
            gc.collect()
            torch.cuda.empty_cache()
            target, quant = str(art.path), "torchao"
        else:
            target, quant = MODEL, None

        r = bench.benchmark(target, cfg, dtype="bfloat16", quantization=quant)
        entry = {
            "label": label,
            "decode_tok_s": round(r.decode_throughput_tok_s, 1),
            "prefill_tok_s": round(r.prefill_throughput_tok_s, 1),
            "ttft_ms": round(r.ttft_ms, 1),
            "weight_mb": round(r.weight_memory_bytes / 1e6, 1) if r.weight_memory_bytes else None,
        }
        print(f"  decode {entry['decode_tok_s']} tok/s | prefill {entry['prefill_tok_s']} tok/s | "
              f"weights {entry['weight_mb']} MB")
    except Exception as e:  # noqa: BLE001 — resilient
        entry = {"label": label, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"  FAILED: {entry['error']}")
    return entry


@app.local_entrypoint()
def main() -> None:
    import json

    out = list(bench_one.starmap(CONFIGS))
    print(f"\n{'=' * 70}\nDEPLOY LOOP — apply→save→vLLM-load→benchmark ({MODEL})\n{'=' * 70}")
    for e in out:
        if "error" in e:
            print(f"  {e['label']:16s}  FAILED ({e['error'][:60]})")
        else:
            print(f"  {e['label']:16s}  {e['decode_tok_s']:8.1f} dec tok/s  {e['weight_mb']} MB weights")
    print(json.dumps(out, indent=2, default=str))
