"""Probe: does the per-layer ranking's margin over random grow with scale + tight budget?

The 1.5B ranking-transfer test (validate_ranking_transfer.py) showed the torchao
ranking transfers but the margin over random was moderate (top-4 recovered ~1.5×
random-4). Hypothesis: that margin GROWS with scale (outliers sharpen — channel
concentration 633× @1.5B → 1247× @7B) and with a TIGHTER budget (small K forces
picking the right few). This probes both on Qwen2.5-7B:

  all-int4 (K=0), then for K in {2, 4, 8}: top-K vs random-K kept fp16 (+ bottom-4).

Read: if top-K beats random-K by MORE at 7B than at 1.5B — and by more at small K —
the ranking's value scales toward the real targets (fp4 / heterogeneous recsys),
justifying the per-layer feature. If the gap stays ~1.5× regardless, it's modest
everywhere on homogeneous LLMs.

Run:  uv run modal run experiments/probe_ranking_scale.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-probe-ranking-scale")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "torchao>=0.7", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
K_VALUES = [2, 4]  # the decisive pair: tight-budget vs the 1.5B K=4 comparison

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
    image=image, gpu=GPU, timeout=5400,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def rank() -> dict:
    """Compute the torchao int4 per-layer sensitivity ranking (one container)."""
    import copy
    import gc
    import json
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
    from firefly.quant.sensitivity import discover_units
    from firefly.quant.torchao import quantize_model, rel_l1

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  {MODEL}")
    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(MODEL, device="cuda", dtype=parse_dtype("bfloat16"))
    batch = load_golden_inputs(inputs, tok, "cuda")
    ref_out = run_capture(fp_model, batch)["final_norm"]
    units = discover_units(fp_model, "layer")

    sens: dict[str, float] = {}
    for unit, fqns in units.items():
        fqn_set = set(fqns)
        q = copy.deepcopy(fp_model)
        quantize_model(q, "int4wo", 128, module_filter=lambda _m, n, s=fqn_set: n in s)
        sens[unit] = rel_l1(ref_out, run_capture(q, batch)["final_norm"])
        del q
        gc.collect()
        torch.cuda.empty_cache()

    ranked = sorted(sens, key=sens.get, reverse=True)
    print("sensitivity (top 6):", [(u, round(sens[u], 4)) for u in ranked[:6]])
    return {"ranked": ranked, "units": {u: units[u] for u in units}, "sens": {u: round(sens[u], 4) for u in ranked}}


@app.function(
    image=image, gpu=GPU, timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def export_eval(kept_units: list[str], units: dict, tag: str) -> dict:
    """Export one int4 checkpoint (K kept-units fp16) and re-eval — isolated container."""
    import gc

    import torch

    from firefly.quant.deploy import evaluate_deployed, export_deployable
    from firefly.quant.intervention import RTNQuantizer
    from firefly.quant.recipe_io import Recipe, serialize_intervention

    all_fqns = {f for fqns in units.values() for f in fqns}
    kept = sorted({f for u in kept_units for f in units[u]})
    recipe = Recipe(
        model_id=MODEL, scheme="int4wo", group_size=128, granularity="layer",
        quantize_fqns=sorted(all_fqns - set(kept)), kept_fp_fqns=kept,
        pre_transforms=[], quantizer=serialize_intervention(RTNQuantizer()),
    )
    try:
        export_deployable(recipe, f"/tmp/{tag}", calib_texts=_CALIB, calib_max_length=64)
        v = round(evaluate_deployed(f"/tmp/{tag}", _EVAL, max_length=64, device="cuda", dtype="bfloat16"), 3)
    except Exception as e:  # noqa: BLE001
        v = None
        print(f"{tag} FAILED: {type(e).__name__}: {str(e)[:160]}")
    gc.collect()
    torch.cuda.empty_cache()
    return {"tag": tag, "served": v}


@app.local_entrypoint()
def main() -> None:
    import json
    import random

    r = rank.remote()
    ranked, units = r["ranked"], r["units"]
    rng = random.Random(0)
    configs = [([], "all_int4")]
    for k in K_VALUES:
        configs.append((ranked[:k], f"top{k}"))
        configs.append((rng.sample(ranked, k), f"rand{k}"))
        if k == 4:
            configs.append((ranked[-k:], "bot4"))

    # Each export in its own container (no memory accumulation), all in parallel.
    served = {d["tag"]: d["served"] for d in export_eval.starmap([(kept, units, tag) for kept, tag in configs])}

    base = served.get("all_int4")
    print(f"\n{'=' * 66}\nRANKING MARGIN vs SCALE + BUDGET ({MODEL})\n{'=' * 66}")
    print(f"  sensitivity top-6: {[(u, r['sens'][u]) for u in ranked[:6]]}")
    print(f"  all-int4 (K=0): {base}")
    for k in K_VALUES:
        t, rr = served.get(f"top{k}"), served.get(f"rand{k}")
        rec_t = round(base - t, 3) if (base and t) else None
        rec_r = round(base - rr, 3) if (base and rr) else None
        ratio = round(rec_t / rec_r, 2) if (rec_t and rec_r and rec_r > 0) else None
        extra = f"  bottom {served.get('bot4')}" if k == 4 else ""
        print(f"  K={k}:  top {t} (rec {rec_t})  random {rr} (rec {rec_r})  top/random={ratio}×{extra}")
    print(json.dumps({"ranked": ranked, "served": served}, indent=2, default=str))
