"""Dense-family auto-quant sweep — evangelism material.

Runs the deterministic auto-quant loop (diagnose → route → verify) on int4 across
several dense ≤9B families, to show the mechanism generalizes: the sensors
localize the outlier structure per model, the router sends salient-weight cases
to AWQ, and the verify step reports the measured recovery (and rejects what
doesn't help). Resilient — a model that won't load/quantize is skipped, not fatal.

The frontier 2026 opens (GLM-5.2 = 744B MoE, Gemma-4 = MoE/multimodal) are out of
this stack's reach (dense, single-GPU, weight-only); this is the reachable lane.

Run:  uv run modal run scripts/demo_dense_sweep.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-demo-dense-sweep")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "torchao>=0.7", "transformers>=4.44", "accelerate", "sentencepiece",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
SCHEME = "int4wo"
GROUP_SIZE = 128
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]

_CALIB = [
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
    "In 1969, Apollo 11 landed the first humans on the Moon during the Space Race.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Supply and demand determine prices in a competitive market economy.",
]
_EVAL = [
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Gradient descent iteratively updates parameters to minimize a loss function.",
    "Mount Everest, on the border of Nepal and Tibet, is Earth's highest peak.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "A hash table offers average constant-time lookups using a hash function.",
    "DNA encodes genetic instructions using four nucleotide bases.",
    "The transformer architecture relies on self-attention to model sequences.",
    "Jupiter is the largest planet in the solar system, a gas giant with many moons.",
    "Recursion solves a problem by reducing it to smaller instances of itself.",
    "Electrons occupy discrete energy levels around an atomic nucleus.",
    "A neural network learns features hierarchically across its layers.",
    "Compound interest grows savings exponentially over long horizons.",
    "Entropy measures the disorder of a thermodynamic system.",
    "Batch normalization stabilizes training by rescaling layer activations.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Antibiotics kill bacteria but have no effect on viral infections.",
    "Newton's three laws describe the motion of objects under forces.",
    "The Industrial Revolution began in Britain in the late 18th century.",
    "Vaccines train the immune system to recognize specific pathogens.",
]


@app.function(
    image=image, gpu=GPU, timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def run() -> list[dict]:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.quant.auto import auto_quant

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  scheme={SCHEME}")
    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    out: list[dict] = []
    for model_id in MODELS:
        print(f"\n{'=' * 70}\n{model_id}\n{'=' * 70}")
        try:
            r = auto_quant(
                model_id, inputs, _EVAL, scheme=SCHEME, group_size=GROUP_SIZE,
                device="cuda", dtype="bfloat16", max_length=64,
            )
            p = r["perplexity"]
            entry = {
                "model": model_id,
                "diagnosis": r["diagnosis_summary"],
                "routed_quantizer": r["recipe"]["quantizer"],
                "pre_transforms": r["recipe"]["pre_transforms"],
                "perplexity_fp": round(p["fp"], 3),
                "perplexity_plain_int4": round(p["plain"], 3),
                "perplexity_routed": round(p["routed"], 3),
                "recovery": round(r["recovery"], 4),
                "accepted": r["accepted"],
                "top_salience": r["attribution_worst_taps"][:3],
            }
            print(f"  {r['recipe']['quantizer']}: ppl {p['fp']:.2f} fp / {p['plain']:.2f} plain / "
                  f"{p['routed']:.2f} routed → {r['recovery']:.0%} recovered, accepted={r['accepted']}")
        except Exception as e:  # noqa: BLE001 — resilient: skip a model that won't load/quantize
            entry = {"model": model_id, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  SKIPPED: {entry['error']}")
        out.append(entry)
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\nSWEEP SUMMARY (scheme={SCHEME})\n{'=' * 70}")
    for e in out:
        if "error" in e:
            print(f"  {e['model']:42s}  SKIPPED ({e['error'][:40]})")
        else:
            print(f"  {e['model']:42s}  {e['routed_quantizer']:4s}  "
                  f"ppl {e['perplexity_plain_int4']:.1f}→{e['perplexity_routed']:.1f}  "
                  f"{e['recovery']:.0%} rec  {'ACCEPT' if e['accepted'] else 'reject'}")
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
