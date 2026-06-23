"""LLM agent demo: min memory at a perplexity bar (the reasoning-beats-router case).

An external LLM (Anthropic tool-use) is the proposer in Firefly's harness slot;
Firefly is the oracle. Goal: clear a perplexity bar at MINIMUM weight memory — a
composition/tradeoff the deterministic router can't do. The LLM reads the
measurements (diagnosis, salience, per-step residual attribution), emits a
compact recipe (sandboxed: only validated interventions), Firefly verifies it,
and the LLM refines toward the cheapest in-bar recipe.

Validated (Qwen2.5-1.5B int4, ~50 eval texts, bar within 10% of fp): AWQ-all
alone FAILS the bar (14.20 vs ≤12.62); the LLM composes AWQ + an
attribution-guided keep-fp set (layers 1,2,15-20,27) and navigates the memory
frontier to a verified 12.56 at 1301 MB = 2.0x smaller than fp — a composition
the deterministic router's one-shot signature→treatment can't produce.

Scope, honestly: the win needs a meaningful int4 gap + a de-noised eval. On the
7B (AWQ-all only +4.4%) the bar sat below the achievable/noise floor and the
agent couldn't clear it — the value is regime-dependent. Bump MODEL/GPU to the
7B to see that honest counter-case.

Run:  uv run modal run scripts/demo_llm_agent.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-demo-llm-agent")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
anthropic_secret = modal.Secret.from_dict({"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "torchao>=0.7", "transformers>=4.44", "accelerate", "anthropic>=0.40",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GPU = "A10G"
SCHEME = "int4wo"
BAR_REL = 0.10
BUDGET = 6

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

_EVAL += [
    "The speed of light in a vacuum is about 300,000 kilometers per second.",
    "Shakespeare wrote both tragedies and comedies in Elizabethan England.",
    "A stack is a last-in, first-out data structure used in many algorithms.",
    "The human heart pumps blood through arteries, veins, and capillaries.",
    "Continental drift is driven by convection currents in the mantle.",
    "Bayes' theorem relates conditional probabilities of two events.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Antibiotics kill bacteria but have no effect on viral infections.",
    "A compiler translates source code into machine instructions.",
    "The Renaissance revived classical art and learning across Europe.",
    "Newton's three laws describe the motion of objects under forces.",
    "Glucose is broken down in glycolysis to release usable energy.",
    "TCP guarantees ordered, reliable delivery of network packets.",
    "Volcanoes form where magma reaches the surface through the crust.",
    "Supervised learning trains models on labeled input-output pairs.",
    "The moon's gravity is the primary cause of ocean tides on Earth.",
    "A prime number has exactly two distinct positive divisors.",
    "Photoreceptors in the retina convert light into neural signals.",
    "The Industrial Revolution began in Britain in the late 18th century.",
    "Encryption transforms readable data into ciphertext using a key.",
    "Plate boundaries are where most earthquakes and volcanoes occur.",
    "An enzyme lowers the activation energy of a biochemical reaction.",
    "Dynamic programming solves problems by reusing subproblem solutions.",
    "Saturn is famous for its prominent system of icy rings.",
    "The Magna Carta limited the power of the English monarchy in 1215.",
    "Diffusion moves particles from high to low concentration regions.",
    "A graph consists of vertices connected by edges.",
    "Vaccines train the immune system to recognize specific pathogens.",
    "The Krebs cycle produces energy carriers in cellular respiration.",
    "Latency is the delay before a transfer of data begins.",
    "Erosion gradually wears down rock and soil over time.",
    "Reinforcement learning optimizes actions through trial and reward.",
    "The Sahara is the largest hot desert on the planet.",
    "Osmosis is the movement of water across a semipermeable membrane.",
    "A binary tree node has at most two children.",
]


@app.function(
    image=image, gpu=GPU, timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[anthropic_secret],
)
def run() -> dict:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.quant.evaluate import AccuracyBar
    from firefly.quant.llm import propose_with_llm
    from firefly.quant.search import min_memory_search
    from firefly.report import render_search

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  model={MODEL} scheme={SCHEME}")

    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    result = min_memory_search(
        MODEL, inputs, _EVAL, propose=propose_with_llm, scheme=SCHEME,
        bar=AccuracyBar("rel", BAR_REL), group_size=128, device="cuda", dtype="bfloat16",
        max_length=64, budget=BUDGET,
    )
    print("\n" + render_search(result))
    if result["best"]:
        result["best"].pop("recipe", None)  # Recipe obj not JSON-serializable
    return result


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
