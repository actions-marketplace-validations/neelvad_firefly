"""w8a8 cross-family recovery sweep — the clean evangelism demo.

w8a8 is robust across architectures (unlike the Qwen-overfit int4/AWQ path), and
small models degrade enough under w8a8 that SmoothQuant has real damage to
recover. So this sweep gives the clean cross-family story the int4 sweep
couldn't: per model, diagnose → route SmoothQuant → verify the recovery (and the
gate accepts/rejects honestly). Small models (≤2-3B) on a cheap A10G.

Run:  uv run modal run scripts/demo_w8a8_recovery_sweep.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-demo-w8a8-recovery-sweep")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "torchao>=0.7", "transformers>=4.44", "accelerate", "sentencepiece",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

GPU = "A10G"
SCHEME = "w8a8"
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # open, Llama-arch, different lab
    "google/gemma-2-2b-it",
    "tiiuae/Falcon3-1B-Instruct",  # open, Falcon family
]

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
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Antibiotics kill bacteria but have no effect on viral infections.",
    "Newton's three laws describe the motion of objects under forces.",
    "The Industrial Revolution began in Britain in the late 18th century.",
    "Vaccines train the immune system to recognize specific pathogens.",
]


@app.function(
    image=image, gpu=GPU, timeout=5400,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def run() -> list[dict]:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.quant.auto import auto_quant

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  scheme={SCHEME}")
    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    out: list[dict] = []
    for model_id in MODELS:
        print(f"\n{'=' * 70}\n{model_id}\n{'=' * 70}")
        try:
            r = auto_quant(
                model_id, inputs, _EVAL, scheme=SCHEME, group_size=32,
                device="cuda", dtype="bfloat16", max_length=64,
            )
            p = r["perplexity"]
            entry = {
                "model": model_id,
                "routed": "+".join([*r["recipe"]["pre_transforms"], r["recipe"]["quantizer"]]),
                "perplexity_fp": round(p["fp"], 3),
                "perplexity_plain_w8a8": round(p["plain"], 3),
                "perplexity_routed": round(p["routed"], 3),
                "recovery": round(r["recovery"], 4),
                "accepted": r["accepted"],
            }
            print(f"  ppl {p['fp']:.2f} fp / {p['plain']:.2f} plain w8a8 / {p['routed']:.2f} routed "
                  f"→ {r['recovery']:.0%} recovered, accepted={r['accepted']}")
        except Exception as e:  # noqa: BLE001 — resilient
            entry = {"model": model_id, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  SKIPPED: {entry['error']}")
        out.append(entry)
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\nw8a8 SmoothQuant RECOVERY SWEEP\n{'=' * 70}")
    for e in out:
        if "error" in e:
            print(f"  {e['model']:38s}  SKIPPED")
        else:
            print(f"  {e['model']:38s}  ppl {e['perplexity_fp']:.1f} fp → "
                  f"{e['perplexity_plain_w8a8']:.1f} w8a8 → {e['perplexity_routed']:.1f} +SQ  "
                  f"{e['recovery']:.0%} rec  {'ACCEPT' if e['accepted'] else 'reject'}")
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
