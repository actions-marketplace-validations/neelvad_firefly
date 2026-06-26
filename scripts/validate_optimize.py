"""GPU validation for `firefly optimize` — the whole adoption loop in one call.

Runs firefly.quant.optimize.optimize() end to end on a real GPU: select (diagnose
→ route → measurement-gate) → export the deployable compressed-tensors checkpoint
→ benchmark the *served artifact*'s real QPS/memory → report quality vs the bar +
the serve command. This is the "stranger runs one command, gets a faster servable
model + the evidence" path.

Run:  uv run modal run scripts/validate_optimize.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-optimize")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "torchao>=0.7", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SCHEME = "int8wo"  # gentle weight-only; deployable + should meet a 10% bar

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
    image=image, gpu=GPU, timeout=5400,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def run() -> dict:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.bench import BenchmarkConfig
    from firefly.quant.optimize import optimize
    from firefly.report import render_optimize

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  scheme={SCHEME}")
    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    result = optimize(
        MODEL, inputs, _EVAL, scheme=SCHEME, group_size=128,
        device="cuda", dtype="bfloat16", max_length=64,
        quality_bar=0.10, out_dir="/tmp/optimized", benchmark=True,
        bench_config=BenchmarkConfig(batch_size=16, input_len=512, output_len=128),
    )
    print("\n" + render_optimize(result))

    # Confirm the artifact + manifest are real and carry the measurement.
    manifest = json.loads((Path("/tmp/optimized") / "firefly_serving.json").read_text())
    print(f"\nmanifest serve_command: {manifest['serve_command']}")
    print(f"manifest measured: {manifest.get('measured')}")
    result.pop("auto", None)  # drop the heavy nested dict from the returned summary
    return result


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
