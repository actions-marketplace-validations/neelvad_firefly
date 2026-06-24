"""Sanity check: why is plain int4wo on Mistral-7B perplexity ~185k (broken)?

Isolates the cause by measuring perplexity across the precision ladder on the
SAME loaded model, plus a Qwen control (known-good) in the same run:
  - fp           : is the baseline sane? (rules out eval/load issues)
  - w8a8         : is quant generally broken, or only int4? (rules out the harness)
  - int4wo gs128 : reproduce the sweep's broken number
  - int4wo gs64/256 : is it group-size sensitive (a config issue)?

Reading: fp sane + w8a8 sane + int4wo broken → int4wo specifically breaks Mistral
(real fragility or an int4-path arch issue). fp/w8a8 also broken → eval/harness.
Qwen int4wo gs128 sane (matches the sweep's 15.3) → the harness is fine.

Run:  uv run modal run scripts/sanity_mistral_int4.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-sanity-mistral-int4")

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
_EVAL = [
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Gradient descent iteratively updates parameters to minimize a loss function.",
    "Mount Everest, on the border of Nepal and Tibet, is Earth's highest peak.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "DNA encodes genetic instructions using four nucleotide bases.",
    "Jupiter is the largest planet in the solar system, a gas giant with many moons.",
    "Electrons occupy discrete energy levels around an atomic nucleus.",
    "Compound interest grows savings exponentially over long horizons.",
    "Newton's three laws describe the motion of objects under forces.",
]

# (model, [(scheme, group_size), ...]); group_size ignored for w8a8/fp
PLAN = [
    ("mistralai/Mistral-7B-Instruct-v0.3", [("w8a8", 32), ("int4wo", 128), ("int4wo", 64), ("int4wo", 256)]),
    ("Qwen/Qwen2.5-7B-Instruct", [("int4wo", 128)]),  # control: should reproduce the sweep's ~15.3
]


@app.function(
    image=image, gpu=GPU, timeout=5400,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def run() -> dict:
    import copy

    import torch

    from firefly.capture import load_model_and_tokenizer
    from firefly.quant.evaluate import perplexity_evaluator
    from firefly.quant.intervention import Pipeline, PrecisionPolicy, RTNQuantizer
    from firefly.quant.sensitivity import discover_units

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    ev = perplexity_evaluator(_EVAL, max_length=64)
    results: dict[str, dict[str, float]] = {}

    for model_id, configs in PLAN:
        print(f"\n=== {model_id} ===")
        model, tok = load_model_and_tokenizer(model_id, device="cuda", dtype=torch.bfloat16)
        all_fqns = {f for fqns in discover_units(model, "layer").values() for f in fqns}
        n_quantizable = len(all_fqns)
        res = {"fp": round(ev(model, tok), 3), "_n_linears": n_quantizable}
        print(f"  fp perplexity: {res['fp']}   ({n_quantizable} quantizable Linears)")
        for scheme, gs in configs:
            policy = PrecisionPolicy(scheme=scheme, group_size=gs, quantize=set(all_fqns))
            q = Pipeline(quantizer=RTNQuantizer()).run(copy.deepcopy(model), policy)
            ppl = round(ev(q, tok), 3)
            res[f"{scheme}_gs{gs}"] = ppl
            print(f"  {scheme} gs{gs}: perplexity {ppl}")
            del q
            torch.cuda.empty_cache()
        results[model_id] = res
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    for m, res in results.items():
        print(f"{m}: {res}")
    return results


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
