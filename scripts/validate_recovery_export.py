"""GPU validation: an int4 RECOVERY recipe ships and serves recovered.

The unlock (now pointed at the recovery that actually transfers). Diagnostics
showed plain W4A16 serves wrecked (+113%) but GPTQ recovers it ~96% / AWQ ~94%,
while w8a8 SmoothQuant is a no-op for serving. So `optimize --scheme int4wo`
should: select an int4 recipe → export it via GPTQ (or AWQ if routed) into
compressed-tensors → re-eval the served model and confirm it recovered.

What we're checking: the manifest method is gptq/awq, and the *served*
(compressed-tensors) perplexity is the recovered one (~11-12, near fp ~10.7),
NOT the wrecked plain-int4 number (~21). That's the difference between
recommending a recovery and serving it.

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
SCHEME = "int4wo"  # plain int4 serves wrecked (+113%); GPTQ/AWQ recover it ~96%

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
        quality_bar=0.20, out_dir="/tmp/recovery", reeval_quality=True, benchmark=True,
        bench_config=BenchmarkConfig(batch_size=16, input_len=512, output_len=128),
    )
    print("\n" + render_optimize(result))

    q = result["quality"]
    manifest = json.loads((Path("/tmp/recovery") / "firefly_serving.json").read_text())
    print(f"\nship={result['ship']}  method={manifest.get('method')}  "
          f"dropped_pre_transforms={manifest.get('dropped_pre_transforms')}")
    print(f"fp={q['fp']:.2f}  plain-int4(torchao,selection)={result['auto']['perplexity']['plain']:.2f}  "
          f"shipped(torchao)={q['shipped']:.2f}  served(comp-tensors {manifest.get('method')})={q['served']:.2f}")

    result.pop("auto", None)
    return result


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
