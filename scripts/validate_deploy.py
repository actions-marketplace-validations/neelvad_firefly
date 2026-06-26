"""GPU validation for the deploy loop — apply → save → vLLM-load → benchmark.

Proves Gap 2 end to end: a uniform recipe is exported to a portable
compressed-tensors checkpoint (firefly.quant.deploy.export_deployable, via
llm-compressor), then that *saved directory* is loaded by vLLM and benchmarked —
confirming the artifact actually serves, and measuring the speedup/footprint vs
the bf16 baseline. This is the difference between handing someone a recipe and
handing them a faster model.

Deployment goes through compressed-tensors (vLLM-native, portable), NOT torchao:
a GPU run confirmed torchao's quantized subclasses don't serialize through
save_pretrained in the current stack. compressed-tensors handles w8a8 / int8 /
int4 cleanly. (torchao stays the measurement backend.)

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
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# (label, scheme-or-None) — None = bf16 baseline; a scheme exports + serves it.
CONFIGS = [("bf16 baseline", None), ("int8wo (W8A16)", "int8wo"), ("w8a8 (W8A8)", "w8a8")]


@app.function(
    image=image, gpu=GPU, timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def bench_one(label: str, scheme: str | None) -> dict:
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
        if scheme is not None:
            recipe = Recipe(
                model_id=MODEL, scheme=scheme, group_size=128, granularity="layer",
                quantize_fqns=[], kept_fp_fqns=[], pre_transforms=[],
                quantizer=serialize_intervention(RTNQuantizer()),
            )
            status, reason = classify_recipe(recipe)
            print(f"  deployability: {status} — {reason}")
            art = export_deployable(recipe, f"/tmp/q_{scheme}")
            print(f"  exported → {art.path}  ({art.compressed_tensors_scheme})\n  serve: {art.serve_command}")
            # Free the export model before vLLM grabs the GPU.
            gc.collect()
            torch.cuda.empty_cache()
            target = str(art.path)
        else:
            target = MODEL

        # vLLM auto-detects compressed-tensors from the checkpoint config.
        r = bench.benchmark(target, cfg, dtype="bfloat16", quantization=None)
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
            print(f"  {e['label']:18s}  FAILED ({e['error'][:60]})")
        else:
            print(f"  {e['label']:18s}  {e['decode_tok_s']:8.1f} dec tok/s  {e['weight_mb']} MB weights")
    print(json.dumps(out, indent=2, default=str))
