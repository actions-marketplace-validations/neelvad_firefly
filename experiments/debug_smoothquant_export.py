"""Diagnostic: what actually recovers compressed-tensors W8A8 on Qwen2.5-1.5B?

The recovery run showed SmoothQuant calibrates + applies (per the logs) yet the
served perplexity is identical to plain W8A8 (18.37). So the question isn't
"did SmoothQuant run" but "does SmoothQuant move compressed-tensors W8A8 at all,
or do we need GPTQ?" This maps the space in ONE run, each config exported +
re-evaluated via transformers (apples-to-apples, the same path optimize uses):

  1. plain W8A8 (QuantizationModifier RTN)        — the baseline (~18.4 expected)
  2. SmoothQuant(0.8) + W8A8 RTN                   — does SmoothQuant help RTN?
  3. GPTQ W8A8 (GPTQModifier, calibration weight)  — does GPTQ alone recover?
  4. SmoothQuant(0.8) + GPTQ W8A8                  — the canonical llm-compressor combo

A real calibration set (24 samples) so SmoothQuant/GPTQ have enough to estimate.

Run:  uv run modal run experiments/debug_smoothquant_export.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-debug-smoothquant")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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
    "The Doppler effect shifts the observed frequency of a moving wave source.",
    "Fiscal policy uses government spending and taxation to steer the economy.",
    "Photovoltaic cells convert sunlight directly into electrical current.",
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
def run() -> list[dict]:
    import gc

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from firefly.quant.evaluate import perplexity_evaluator

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    ev = perplexity_evaluator(_EVAL, max_length=64)

    def ppl_of(path_or_id: str) -> float:
        m = AutoModelForCausalLM.from_pretrained(path_or_id, torch_dtype=torch.bfloat16, device_map="cuda")
        tok = AutoTokenizer.from_pretrained(path_or_id)
        out = round(ev(m, tok), 3)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return out

    fp = ppl_of(MODEL)
    print(f"fp perplexity: {fp}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [tok(t, truncation=True, max_length=128) for t in _CALIB]
    ds = Dataset.from_list([{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows])

    def export(label: str, modifiers: list, out: str) -> dict:
        from llmcompressor import oneshot
        try:
            oneshot(model=MODEL, recipe=modifiers, dataset=ds,
                    num_calibration_samples=len(_CALIB), max_seq_length=128, output_dir=out)
            served = ppl_of(out)
            print(f"  {label:24s} served perplexity {served}")
            return {"config": label, "served": served}
        except Exception as e:  # noqa: BLE001
            print(f"  {label:24s} FAILED: {type(e).__name__}: {str(e)[:150]}")
            return {"config": label, "error": f"{type(e).__name__}: {str(e)[:150]}"}

    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    ig = ["lm_head"]
    out: list[dict] = [{"config": "fp", "served": fp}]
    out.append(export("plain W8A8 (RTN)",
                      [QuantizationModifier(targets="Linear", scheme="W8A8", ignore=ig)], "/tmp/d1"))
    out.append(export("SmoothQuant0.8 + W8A8 RTN",
                      [SmoothQuantModifier(smoothing_strength=0.8),
                       QuantizationModifier(targets="Linear", scheme="W8A8", ignore=ig)], "/tmp/d2"))
    out.append(export("GPTQ W8A8",
                      [GPTQModifier(targets="Linear", scheme="W8A8", ignore=ig)], "/tmp/d3"))
    out.append(export("SmoothQuant0.8 + GPTQ W8A8",
                      [SmoothQuantModifier(smoothing_strength=0.8),
                       GPTQModifier(targets="Linear", scheme="W8A8", ignore=ig)], "/tmp/d4"))

    print(f"\n{'=' * 60}\nWHAT RECOVERS compressed-tensors W8A8 ({MODEL})\n{'=' * 60}")
    for e in out:
        v = e.get("served", e.get("error"))
        print(f"  {e['config']:28s}  {v}")
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
