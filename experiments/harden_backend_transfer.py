"""Cross-family hardening for the two sharp write-up findings.

Finding 1 (backend gap): the same int4-RTN scheme measured in torchao vs served
in compressed-tensors diverges (Qwen: 21.45 vs 18.73). Does it hold cross-family?

Finding 2 (SmoothQuant no-op): w8a8 SmoothQuant recovers in torchao but is a
bit-identical no-op in compressed-tensors serving (a property of CT's W8A8
activation-quant granularity, so it SHOULD be universal). Does it replicate?

Three architecturally-distinct small families where w8a8 actually degrades (so
SmoothQuant has something to recover in torchao): Qwen (Qwen2.5-1.5B), Llama-arch
(SmolLM2-1.7B), Gemma (gemma-2-2b). Each model in its own container.

Per model, all measured as served-equivalent perplexity on the same eval:
  fp | torchao{int4, w8a8, w8a8+SQ} | compressed-tensors{int4-RTN, w8a8, w8a8+SQ}

Run:  uv run modal run experiments/harden_backend_transfer.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-harden-backend-transfer")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "torchao>=0.7", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "Llama-arch"),
    ("google/gemma-2-2b-it", "Gemma"),
]

_CALIB = [
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
    "In 1969, Apollo 11 landed the first humans on the Moon during the Space Race.",
    "Supply and demand determine prices in a competitive market economy.",
    "A binary search halves the search space at every comparison step.",
    "Photosynthesis converts sunlight into chemical energy in green plants.",
    "The water cycle moves moisture through evaporation, condensation, and precipitation.",
]
_EVAL = [
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Gradient descent iteratively updates parameters to minimize a loss function.",
    "Mount Everest, on the border of Nepal and Tibet, is Earth's highest peak.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "DNA encodes genetic instructions using four nucleotide bases.",
    "Jupiter is the largest planet in the solar system, a gas giant with many moons.",
    "Newton's three laws describe the motion of objects under forces.",
    "A neural network learns features hierarchically across its layers.",
    "Entropy measures the disorder of a thermodynamic system.",
]


@app.function(
    image=image, gpu=GPU, timeout=5400,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def sweep_model(model_id: str, family: str) -> dict:
    import copy
    import gc
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.capture import load_golden_inputs, load_model_and_tokenizer, parse_dtype
    from firefly.determinism import set_deterministic
    from firefly.quant.deploy import _build_calib_dataset, evaluate_deployed
    from firefly.quant.evaluate import perplexity_evaluator
    from firefly.quant.intervention import RTNQuantizer
    from firefly.quant.recipe_io import Recipe, apply_recipe, serialize_intervention
    from firefly.quant.sensitivity import discover_units
    from firefly.quant.smoothquant import SmoothQuant
    from firefly.quant.torchao import quantize_model

    print(f"\n=== {family}: {model_id} ===")
    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(model_id, device="cuda", dtype=parse_dtype("bfloat16"))
    batch = load_golden_inputs(inputs, tok, "cuda")
    ev = perplexity_evaluator(_EVAL, max_length=64)
    all_fqns = {f for fqns in discover_units(fp_model, "layer").values() for f in fqns}
    r: dict[str, float] = {"fp": round(ev(fp_model, tok), 3)}

    def ta(base, scheme, sq=False):  # torchao perplexity on a fresh copy of `base`
        q = copy.deepcopy(base)
        if sq:
            rec = Recipe(model_id, scheme, 128, "layer", quantize_fqns=sorted(all_fqns),
                         kept_fp_fqns=[], pre_transforms=[serialize_intervention(SmoothQuant(alpha=0.5))],
                         quantizer=serialize_intervention(RTNQuantizer()))
            q = apply_recipe(rec, q, batch)
        else:
            quantize_model(q, scheme, 128)
        v = round(ev(q, tok), 3)
        del q
        gc.collect()
        torch.cuda.empty_cache()
        return v

    r["ta_int4"] = ta(fp_model, "int4wo")
    r["ta_w8a8"] = ta(fp_model, "w8a8")
    r["ta_w8a8_sq"] = ta(fp_model, "w8a8", sq=True)
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    def ct(tag, modifiers, calib=False):  # compressed-tensors served perplexity
        out = f"/tmp/{tag}"
        try:
            if calib:
                ds = _build_calib_dataset(model_id, _CALIB, 64)
                oneshot(model=model_id, recipe=modifiers, dataset=ds,
                        num_calibration_samples=len(_CALIB), max_seq_length=64, output_dir=out)
            else:
                oneshot(model=model_id, recipe=modifiers, output_dir=out)
            v = round(evaluate_deployed(out, _EVAL, max_length=64, device="cuda", dtype="bfloat16"), 3)
        except Exception as e:  # noqa: BLE001
            print(f"  ct {tag} FAILED: {type(e).__name__}: {str(e)[:120]}")
            v = None
        gc.collect()
        torch.cuda.empty_cache()
        return v

    ig = ["lm_head"]
    r["ct_int4"] = ct("ct_int4", [QuantizationModifier(targets="Linear", scheme="W4A16", ignore=ig)])
    r["ct_w8a8"] = ct("ct_w8a8", [QuantizationModifier(targets="Linear", scheme="W8A8", ignore=ig)])
    r["ct_w8a8_sq"] = ct("ct_w8a8_sq", [SmoothQuantModifier(smoothing_strength=0.5),
                                        QuantizationModifier(targets="Linear", scheme="W8A8", ignore=ig)], calib=True)

    print(f"  {family}: {r}")
    return {"family": family, "model": model_id, **r}


@app.local_entrypoint()
def main() -> None:
    import json

    out = list(sweep_model.starmap(MODELS))

    def pct(a, b):
        return f"{(b - a) / a * 100:+.0f}%" if (a and b) else "n/a"

    print(f"\n{'=' * 90}\nFINDING 1 — int4 backend gap (torchao vs compressed-tensors, same RTN scheme)\n{'=' * 90}")
    for e in out:
        print(f"  {e['family']:12s}  fp {e['fp']:.2f}  torchao-int4 {e['ta_int4']}  ct-int4 {e['ct_int4']}  "
              f"gap {pct(e['ta_int4'], e['ct_int4'])} of torchao")

    print(f"\n{'=' * 90}\nFINDING 2 — SmoothQuant w8a8: recovers in torchao, no-op served (compressed-tensors)\n{'=' * 90}")
    for e in out:
        print(f"  {e['family']:12s}  torchao w8a8 {e['ta_w8a8']} → +SQ {e['ta_w8a8_sq']} ({pct(e['ta_w8a8'], e['ta_w8a8_sq'])})   "
              f"||  served w8a8 {e['ct_w8a8']} → +SQ {e['ct_w8a8_sq']} ({pct(e['ct_w8a8'], e['ct_w8a8_sq'])})")
    print(json.dumps(out, indent=2, default=str))
