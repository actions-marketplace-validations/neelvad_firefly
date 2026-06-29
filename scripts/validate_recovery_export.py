"""GPU validation: a SmoothQuant RECOVERY recipe now ships, not just headroom.

The unlock. On a model where plain w8a8 degrades, `optimize` should: diagnose
activation outliers → route SmoothQuant → gate-accept it → **ship the SmoothQuant
recipe** (exported via llm-compressor's SmoothQuantModifier into compressed-tensors,
not falling back to uniform quant) → re-eval the served model to confirm the
recovery actually transferred.

What we're checking: ship == 'routed' with SmoothQuant in the treatments, and the
*served* (compressed-tensors) perplexity is the recovered one (near fp), not the
wrecked plain-w8a8 number. That's the difference between recommending a recovery
and serving it.

Run:  uv run modal run scripts/validate_recovery_export.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-recovery-export")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "llmcompressor>=0.3", "torchao>=0.7", "transformers>=4.44", "accelerate")
    .add_local_python_source("firefly")
)

GPU = "A100-80GB"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SCHEME = "w8a8"  # degrades on a 1.5B model → SmoothQuant has real damage to recover

_CALIB = [
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
    "In 1969, Apollo 11 landed the first humans on the Moon during the Space Race.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Supply and demand determine prices in a competitive market economy.",
    "Photosynthesis converts sunlight into chemical energy in green plants.",
    "A binary search halves the search space at every comparison step.",
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
        quality_bar=0.50, out_dir="/tmp/recovery", reeval_quality=True, benchmark=True,
        bench_config=BenchmarkConfig(batch_size=16, input_len=512, output_len=128),
    )
    print("\n" + render_optimize(result))

    q = result["quality"]
    art = result["artifact"]
    print(f"\nship={result['ship']}  treatments={art.get('treatments')}")
    print(f"fp={q['fp']:.2f}  plain-w8a8(torchao,selection)={result['auto']['perplexity']['plain']:.2f}  "
          f"routed-SQ(torchao)={q['shipped']:.2f}  served-SQ(comp-tensors)={q['served']:.2f}")
    manifest = json.loads((Path("/tmp/recovery") / "firefly_serving.json").read_text())
    print(f"manifest treatments: {manifest.get('treatments')}")

    result.pop("auto", None)
    return result


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
