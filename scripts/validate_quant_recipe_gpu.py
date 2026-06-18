"""GPU + int4 validation of the attribution-guided mixed-precision loop.

The CPU demos used W8A8 on SmolLM, where layer.28 dominated so the cheap
filter (isolated) was already near-optimal and greedy barely helped. This runs
the same machinery on a bigger GQA model (Qwen2.5-0.5B) with **int4 weight-only**
— more aggressive quant, sensitivity likely more distributed — to answer:

  1. Does the whole loop (sensitivity → recipe → verify) run on real int4
     kernels on GPU? (the kernel path is tile_packed_to_4d; see
     scripts/probe_int4_torchao.py)
  2. Does greedy (wrapper) pull ahead of isolated (filter) here, as the
     feature-selection analogy predicts when interactions are stronger?
  3. Does per-Linear granularity work on int4, and what does it surface?

Run:  uv run modal run scripts/validate_quant_recipe_gpu.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-quant-recipe-gpu")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "torchao>=0.7", "transformers>=4.44", "accelerate",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

MODEL = "Qwen/Qwen2.5-0.5B"
SCHEME = "int4wo"
K_VALUES = [1, 2, 4, 8]


@app.function(
    image=image, gpu="A10G", timeout=2400,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run() -> dict:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.quant.sensitivity import compute_recipe, compute_sensitivity

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  model={MODEL} scheme={SCHEME}")

    inputs = Path(tempfile.mkdtemp()) / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox jumps over the lazy dog"], "max_length": 16}))

    common = dict(device="cuda", dtype="bfloat16", scheme=SCHEME)

    # --- 1+2: layer-granularity strategy comparison on int4 -----------------
    curves: dict[str, dict[int, float]] = {}
    full_div = None
    for strat in ("isolated", "marginal", "greedy"):
        print(f"running strategy={strat} ...")
        r = compute_recipe(MODEL, inputs, strategy=strat, k_values=K_VALUES, **common)
        curves[strat] = {p.k: round(p.recovery, 4) for p in r.curve}
        full_div = r.sensitivity.full_quant_divergence

    print(f"\nint4 all-quantized output divergence: {full_div:.2%}")
    print(f"{'k':>3}  {'isolated':>10}  {'marginal':>10}  {'greedy':>10}")
    for k in K_VALUES:
        print(f"{k:>3}  {curves['isolated'][k]:>10.1%}  "
              f"{curves['marginal'][k]:>10.1%}  {curves['greedy'][k]:>10.1%}")

    greedy_vs_isolated = {
        k: round(curves["greedy"][k] - curves["isolated"][k], 4) for k in K_VALUES
    }
    greedy_pulls_ahead = any(v > 0.01 for v in greedy_vs_isolated.values())

    # --- 3: per-Linear granularity on int4 (isolated, top units) ------------
    print("\nrunning per-Linear (isolated) ...")
    lin = compute_sensitivity(MODEL, inputs, granularity="linear", strategy="isolated", **common)
    top_linear = [(u.unit, round(u.sensitivity, 4)) for u in lin.ranked[:10]]

    verdict = {
        "model": MODEL,
        "scheme": SCHEME,
        "full_quant_divergence": round(full_div, 4),
        "recovery_curves": curves,
        "greedy_minus_isolated": greedy_vs_isolated,
        "greedy_pulls_ahead_of_isolated": greedy_pulls_ahead,
        "n_linear_units": len(lin.units),
        "per_linear_top10": top_linear,
    }
    print("\n" + "=" * 60)
    print(f"greedy pulls ahead of isolated (int4): {greedy_pulls_ahead}")
    print(f"per-Linear top units: {[u for u, _ in top_linear[:5]]}")
    print("=" * 60)
    return verdict


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
