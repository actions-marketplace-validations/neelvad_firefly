"""Cross-backend quality re-eval experiment — does what we MEASURE match what we SHIP?

Firefly *selects* a recipe by measuring perplexity on a **torchao**-quantized
model, but *deploys* a **compressed-tensors** one (same scheme, different
implementation). This runs optimize(reeval_quality=True) across schemes and puts
the two perplexities side by side, so we learn whether the backend handoff is
faithful — and where it isn't.

Hypothesis: int8wo (per-channel int8 weight RTN) should agree closely; w8a8
(int8 dynamic activations) and int4wo (group packing) are the likely places the
two backends diverge — which would mean the bar must be checked on the *served*
model, not the torchao proxy.

Run:  uv run modal run scripts/validate_reeval.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-reeval")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "torchao>=0.7", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SCHEMES = ["int8wo", "w8a8", "int4wo"]

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
    "DNA encodes genetic instructions using four nucleotide bases.",
    "Jupiter is the largest planet in the solar system, a gas giant with many moons.",
    "Electrons occupy discrete energy levels around an atomic nucleus.",
    "Compound interest grows savings exponentially over long horizons.",
    "Newton's three laws describe the motion of objects under forces.",
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

    from firefly.quant.optimize import optimize

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    out: list[dict] = []
    for scheme in SCHEMES:
        print(f"\n{'=' * 70}\n{scheme}\n{'=' * 70}")
        try:
            r = optimize(
                MODEL, inputs, _EVAL, scheme=scheme, group_size=128,
                device="cuda", dtype="bfloat16", max_length=64,
                quality_bar=0.10, out_dir=f"/tmp/opt_{scheme}", reeval_quality=True, benchmark=False,
            )
            q = r["quality"]
            entry = {
                "scheme": scheme,
                "fp": round(q["fp"], 3),
                "torchao": round(q["shipped"], 3),
                "served_ct": round(q["served"], 3),
                "backend_delta": round(q["backend_delta"], 3),
                "served_rel_pct": round(q["served_rel_to_fp"] * 100, 1),
                "meets_bar": r["meets_bar"],
            }
            print(f"  fp {entry['fp']} | torchao {entry['torchao']} | compressed-tensors {entry['served_ct']} "
                  f"| Δ {entry['backend_delta']:+.2f} | served {entry['served_rel_pct']:+.1f}% vs fp")
        except Exception as e:  # noqa: BLE001 — resilient per-scheme
            entry = {"scheme": scheme, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  SKIPPED: {entry['error']}")
        out.append(entry)
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\nCROSS-BACKEND RE-EVAL ({MODEL})\n{'=' * 70}")
    print(f"  {'scheme':8s}  {'fp':>7s}  {'torchao':>8s}  {'comp-ten':>8s}  {'Δ':>7s}  {'served%':>8s}")
    for e in out:
        if "error" in e:
            print(f"  {e['scheme']:8s}  SKIPPED")
        else:
            print(f"  {e['scheme']:8s}  {e['fp']:7.2f}  {e['torchao']:8.2f}  {e['served_ct']:8.2f}  "
                  f"{e['backend_delta']:+7.2f}  {e['served_rel_pct']:+7.1f}%")
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
