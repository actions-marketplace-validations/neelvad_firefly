"""End-to-end demo + validation of the quant mechanism on a REAL model (GPU).

Everything else is validated on SmolLM-135M (tiny, CPU) — which proves
correctness but not that the mechanism *matters* at scale. This runs the whole
product loop on a real model through the actual library code and reports the
numbers that decide whether the mechanism is real:

  1. Does w8a8 actually hurt? (baseline degradation — divergence + perplexity)
  2. Does SmoothQuant recover it at scale? (the headline intervention)
  3. Do the SENSORS flag the right layers? (channel_concentration diagnosis +
     weight-salience / AWQ signal)
  4. Does the int4 mixed-precision recipe recover most of the loss by keeping a
     few layers fp? (the attribution-guided recipe)

Start cheap (1.5B on A10G) to de-risk the GPU path, then bump MODEL/GPU to a 7B
on A100 for the headline demo.

Run:  uv run modal run scripts/demo_quant_real_model.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-demo-quant-real-model")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "torchao>=0.7", "transformers>=4.44", "accelerate",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # bump to Qwen/Qwen2.5-7B-Instruct + A100-40GB for the headline
GPU = "A10G"
K_VALUES = [1, 2, 4]

_CALIB = [
    "The mitochondria is the powerhouse of the cell.",
    "In 1969, Apollo 11 landed the first humans on the Moon.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris, a city on the river Seine.",
]
_EVAL = [
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Quantization reduces model size by storing weights in lower precision.",
]


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/root/.cache/huggingface": hf_cache})
def run() -> dict:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.capture import (
        capture_reference,
        load_golden_inputs,
        load_model_and_tokenizer,
        run_capture,
    )
    from firefly.quant.diagnose import diagnose
    from firefly.quant.evaluate import perplexity_evaluator
    from firefly.quant.recipe import compute_recipe
    from firefly.quant.risk import analyze_quant_risk
    from firefly.quant.salience import weight_salience
    from firefly.quant.sensitivity import _apply_policy, discover_units
    from firefly.quant.smoothquant import SmoothQuant
    from firefly.quant.torchao import rel_l1
    from firefly.reference import read_reference

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  model={MODEL}")

    work = Path(tempfile.mkdtemp())
    calib = work / "calib.json"
    calib.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    model, tok = load_model_and_tokenizer(MODEL, device="cuda", dtype=torch.bfloat16)
    batch = load_golden_inputs(calib, tok, "cuda")
    ref_out = run_capture(model, batch)["final_norm"]
    units = discover_units(model, "layer")
    all_fqns = {f for fqns in units.values() for f in fqns}

    ev = perplexity_evaluator(_EVAL, max_length=64)
    ppl_fp = ev(model, tok)

    # --- weight-salience (AWQ signal), while the fp model is loaded ---------
    lin_fqns = {f for fqns in discover_units(model, "linear").values() for f in fqns}
    salience = weight_salience(model, lin_fqns, batch)[:6]

    # --- 1+2: w8a8 baseline degradation vs SmoothQuant recovery -------------
    def measure(m) -> tuple[float, float]:
        return rel_l1(ref_out, run_capture(m, batch)["final_norm"]), ev(m, tok)

    m_plain = _apply_policy(model, "w8a8", 32, all_fqns)
    d_plain, ppl_plain = measure(m_plain)
    del m_plain
    torch.cuda.empty_cache()

    m_sq = _apply_policy(model, "w8a8", 32, all_fqns, [SmoothQuant()], batch)
    d_sq, ppl_sq = measure(m_sq)
    del m_sq, model
    torch.cuda.empty_cache()

    # --- 3: diagnosis from a captured reference (channel_concentration) -----
    ref_dir = work / "ref"
    capture_reference(MODEL, calib, ref_dir, device="cuda", dtype=torch.bfloat16)
    manifest, tensors = read_reference(ref_dir)
    diag = diagnose(tensors, manifest.tap_points, bits=8)
    risks = sorted(
        analyze_quant_risk(tensors, manifest.tap_points, bits=8),
        key=lambda r: r.channel_concentration, reverse=True,
    )[:6]

    # --- 4: int4 mixed-precision recipe -------------------------------------
    rec = compute_recipe(MODEL, calib, scheme="int4wo", k_values=K_VALUES, device="cuda", dtype="bfloat16")

    recovery = (d_plain - d_sq) / d_plain if d_plain else 0.0
    verdict = {
        "model": MODEL,
        "perplexity": {"fp": round(ppl_fp, 3), "w8a8": round(ppl_plain, 3), "w8a8_smoothquant": round(ppl_sq, 3)},
        "w8a8_divergence": {"plain": round(d_plain, 4), "smoothquant": round(d_sq, 4),
                            "smoothquant_recovery": round(recovery, 4)},
        "salience_top": [(s.fqn, round(s.salience_concentration, 1)) for s in salience],
        "diagnosis_findings": [(f.signature, f.location) for f in diag.findings[:8]],
        "channel_concentration_top": [(r.tap_name, round(r.channel_concentration, 1)) for r in risks],
        "int4_recipe": {
            "full_quant_divergence": round(rec.sensitivity.full_quant_divergence, 4),
            "recovery_curve": {p.k: round(p.recovery, 4) for p in rec.curve},
            "recommended_k": rec.recommended_k,
        },
    }
    print("\n" + "=" * 64)
    print(f"w8a8 divergence {d_plain:.1%} → SmoothQuant {d_sq:.1%}  ({recovery:.0%} recovered)")
    print(f"perplexity  fp {ppl_fp:.2f} → w8a8 {ppl_plain:.2f} → +SmoothQuant {ppl_sq:.2f}")
    print(f"int4 all-quant divergence {rec.sensitivity.full_quant_divergence:.1%}; "
          f"recovery curve {verdict['int4_recipe']['recovery_curve']}")
    print(f"top salience: {[s for s, _ in verdict['salience_top'][:3]]}")
    print("=" * 64)
    return verdict


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
