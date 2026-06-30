"""GPU validation: the quality bar drives the multi-scheme choice both ways.

optimize_over_schemes({int4wo, int8wo}) ships the MOST-compressed scheme that
meets the bar. On Qwen2.5-1.5B (int4 serves ~+25%, int8wo ~+3%):
  - a TIGHT bar (0.10) → int4 misses, ships int8wo (2×).
  - a LOOSE bar (0.30) → int4 clears it, ships int4wo (4×).
Same model, same code — only the bar changes the answer. Selection uses
export + served re-eval (no vLLM), so both runs fit one container.

Run:  uv run modal run scripts/validate_multi_scheme.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-multi-scheme")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "torchao>=0.7", "transformers>=4.44", "accelerate")
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
def run() -> list[dict]:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.quant.optimize import optimize_over_schemes
    from firefly.report import render_optimize_schemes

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    out: list[dict] = []
    for bar in (0.10, 0.30):
        print(f"\n{'=' * 70}\nbar = {bar:.0%}\n{'=' * 70}")
        r = optimize_over_schemes(
            MODEL, inputs, _EVAL, schemes=("int4wo", "int8wo"), quality_bar=bar,
            group_size=128, device="cuda", dtype="bfloat16", max_length=64,
            out_dir=f"/tmp/ms_{int(bar * 100)}", benchmark=False,
        )
        print(render_optimize_schemes(r))
        out.append({
            "bar": bar, "chosen": r["chosen_scheme"], "met_bar": r["met_bar"],
            "per_scheme": [(s["scheme"], round(s["served_rel_to_fp"], 3), s["meets_bar"])
                           for s in r["per_scheme"]],
        })

    print(f"\n{'=' * 70}\nBAR DRIVES THE SCHEME CHOICE\n{'=' * 70}")
    for e in out:
        print(f"  bar {e['bar']:.0%} → chosen {e['chosen']}  (met_bar={e['met_bar']})  {e['per_scheme']}")
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
