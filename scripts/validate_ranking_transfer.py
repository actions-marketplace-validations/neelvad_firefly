"""Does the cheap (torchao) per-layer fragility RANKING transfer to served quality?

The load-bearing assumption for attribution-guided per-layer mixed precision: the
*ranking* of which layers are fragile is model-intrinsic, so a cheap torchao
measurement predicts which layers — kept higher precision — recover the *served*
(compressed-tensors) model. Absolute quality doesn't transfer across backends
(int4 diverged ~29%); the question is whether the ORDER does.

Decisive cheap test (3 exports, not N ablations). Rank layers by torchao int4
isolated sensitivity, then build three int4-GPTQ checkpoints that each keep K=4
layers at fp16, differing only in WHICH four:
  - top-4    : the torchao-flagged most-sensitive layers
  - random-4 : a random four (the control)
  - bottom-4 : the torchao-flagged least-sensitive layers
Serve + re-eval each.

Verdict: top-4 served-ppl << random-4 ≈ bottom-4  → the ranking transfers (keeping
the flagged layers genuinely recovers the served model; the per-layer signal is
real and backend-stable). top-4 ≈ random-4 → the ranking is worthless when served
(any mixed precision helps equally) → fall back to method-coupled sensitivity.
The random-4 control is the whole point.

Run:  uv run modal run scripts/validate_ranking_transfer.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-ranking-transfer")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "torchao>=0.7", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
K = 4  # layers kept fp16 in each mixed checkpoint

_CALIB = [
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
    "In 1969, Apollo 11 landed the first humans on the Moon during the Space Race.",
    "Supply and demand determine prices in a competitive market economy.",
    "A binary search halves the search space at every comparison step.",
    "The water cycle moves moisture through evaporation, condensation, and precipitation.",
    "Shakespeare wrote tragedies, comedies, and histories in early modern English.",
    "Plate tectonics explains the slow drift of continents over geological time.",
    "A compiler translates source code into machine instructions before execution.",
    "The immune system distinguishes self from foreign antigens to fight infection.",
    "Inflation erodes purchasing power when the money supply grows faster than output.",
    "Genes encode proteins through transcription into RNA and translation by ribosomes.",
    "The speed of light in a vacuum is a fundamental constant of the universe.",
    "Object-oriented programming organizes code around encapsulated data and methods.",
    "Tectonic stress released along faults produces earthquakes and seismic waves.",
    "Markets allocate scarce resources through the price signals of buyers and sellers.",
    "Neurons communicate via electrochemical signals across synaptic junctions.",
    "The Renaissance revived classical art and learning across fifteenth-century Europe.",
    "A database index trades storage for faster lookups on a queried column.",
    "Thermodynamics governs how energy is conserved and dispersed in physical systems.",
    "Natural selection favors heritable traits that improve reproductive success.",
    "Encryption protects data by transforming it into a form only a key can reverse.",
    "Fiscal policy uses government spending and taxation to steer the economy.",
]
_EVAL = [
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Gradient descent iteratively updates parameters to minimize a loss function.",
    "Mount Everest, on the border of Nepal and Tibet, is Earth's highest peak.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "Jupiter is the largest planet in the solar system, a gas giant with many moons.",
    "Compound interest grows savings exponentially over long horizons.",
    "Newton's three laws describe the motion of objects under forces.",
    "A neural network learns features hierarchically across its layers.",
    "Entropy measures the disorder of a thermodynamic system.",
]


@app.function(
    image=image, gpu=GPU, timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def run() -> dict:
    import copy
    import gc
    import json
    import random
    import tempfile
    from pathlib import Path

    import torch

    from firefly.capture import (
        load_golden_inputs,
        load_model_and_tokenizer,
        parse_dtype,
        run_capture,
    )
    from firefly.determinism import set_deterministic
    from firefly.quant.deploy import evaluate_deployed, export_deployable
    from firefly.quant.intervention import RTNQuantizer
    from firefly.quant.recipe_io import Recipe, serialize_intervention
    from firefly.quant.sensitivity import discover_units
    from firefly.quant.torchao import quantize_model, rel_l1

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    # --- cheap signal: torchao int4 isolated per-layer sensitivity ---
    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(MODEL, device="cuda", dtype=parse_dtype("bfloat16"))
    batch = load_golden_inputs(inputs, tok, "cuda")
    ref_out = run_capture(fp_model, batch)["final_norm"]
    units = discover_units(fp_model, "layer")  # {layer.i: [linear fqns]}

    sens: dict[str, float] = {}
    for unit, fqns in units.items():
        fqn_set = set(fqns)
        q = copy.deepcopy(fp_model)
        quantize_model(q, "int4wo", 128, module_filter=lambda _m, n, s=fqn_set: n in s)
        sens[unit] = rel_l1(ref_out, run_capture(q, batch)["final_norm"])
        del q
        gc.collect()
        torch.cuda.empty_cache()
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    ranked = sorted(sens, key=sens.get, reverse=True)  # most-sensitive first
    top = ranked[:K]
    bottom = ranked[-K:]
    rng = random.Random(0)
    random_k = rng.sample(ranked, K)
    print("\ntorchao int4 per-layer sensitivity (top 6):")
    for u in ranked[:6]:
        print(f"  {u:12s}  {sens[u]:.4f}")
    print(f"  ... bottom: {[(u, round(sens[u], 4)) for u in bottom]}")

    all_fqns = {f for fqns in units.values() for f in fqns}

    def recipe_for(kept_units: list[str]) -> Recipe:
        kept = sorted({f for u in kept_units for f in units[u]})
        return Recipe(
            model_id=MODEL, scheme="int4wo", group_size=128, granularity="layer",
            quantize_fqns=sorted(all_fqns - set(kept)), kept_fp_fqns=kept,
            pre_transforms=[], quantizer=serialize_intervention(RTNQuantizer()),
        )

    configs = [("all-int4 (K=0)", []), ("top-4 kept", top), ("random-4 kept", random_k), ("bottom-4 kept", bottom)]
    results: dict[str, float] = {}
    for label, kept_units in configs:
        out_dir = f"/tmp/rt_{label.split()[0].replace('-', '')}"
        try:
            export_deployable(recipe_for(kept_units), out_dir, calib_texts=_CALIB, calib_max_length=64)
            served = round(evaluate_deployed(out_dir, _EVAL, max_length=64, device="cuda", dtype="bfloat16"), 3)
            results[label] = served
            print(f"  {label:16s}  served ppl {served}   (kept {[u for u in kept_units]})")
        except Exception as e:  # noqa: BLE001
            results[label] = None
            print(f"  {label:16s}  FAILED: {type(e).__name__}: {str(e)[:160]}")
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 64}\nDOES THE torchao RANKING TRANSFER TO SERVED int4? ({MODEL})\n{'=' * 64}")
    for label, served in results.items():
        print(f"  {label:16s}  served ppl {served}")
    return {
        "sensitivity": {u: round(sens[u], 4) for u in ranked},
        "top": top, "bottom": bottom, "random": random_k,
        "served_ppl": results,
    }


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
