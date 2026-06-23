"""The deterministic auto-quant harness on a real model — the "win" demo.

Given only Firefly's measurements (no human picking the technique), the agent:
diagnoses the model's quant failure mode → routes the diagnosis to a recipe →
applies + VERIFIES it against the plain-quant baseline → ships it only if the
measurement says it helped, with the explanation.

The honest win: on Qwen2.5-7B int4, plain int4 is the *distributed-damage* regime
the mixed-precision recipe could only recover ~9% of; the router diagnoses
salient weight channels, routes to AWQ, and verification accepts it (~91%
recovery). On a tiny model where the routed intervention doesn't generalize, the
gate REJECTS it and ships plain — the measurement is ground truth, not the rule.

Run:  uv run modal run scripts/demo_auto_quant.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-demo-auto-quant")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "torchao>=0.7", "transformers>=4.44", "accelerate",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPU = "A100-80GB"
SCHEME = "int4wo"
GROUP_SIZE = 128

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
]


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/root/.cache/huggingface": hf_cache})
def run() -> dict:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.quant.auto import auto_quant
    from firefly.report import render_auto

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  model={MODEL} scheme={SCHEME}")

    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    result = auto_quant(
        MODEL, inputs, _EVAL, scheme=SCHEME, group_size=GROUP_SIZE,
        device="cuda", dtype="bfloat16", max_length=64,
    )
    print("\n" + render_auto(result))
    result.pop("recipe_obj", None)  # not JSON-serializable
    return result


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
