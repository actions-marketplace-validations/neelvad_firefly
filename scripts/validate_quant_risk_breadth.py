"""GPU breadth sweep for the quant-risk -> real-torchao claim.

The local CPU validation established +0.71 (channel_concentration vs local
W8A8 error) on ONE tiny model (SmolLM-135M). This sweep answers two questions
on real GPU hardware:

1. **Does it generalize?** Run W8A8 `validate_against_torchao` across scale
   (135M -> 7B) and two families. The claim is only sellable if the
   correlation survives scale — n=1 on a 135M model isn't.

2. **Where's the boundary?** quant-risk analyzes *activations*; W8A8 quantizes
   activations (per-token), which is why concentration predicts its error. The
   dominant inference deployment is **int4 weight-only** (W4A16) — activations
   stay in fp, only weights are quantized. quant-risk should NOT predict that.
   Running int4wo confirms the tool's scope empirically: we expect
   w8a8_rho >> int4wo_rho.

It also watches the known imperfection: SwiGLU `down_proj` over-weighting (high
channel_concentration but modest real error) dragged SmolLM's rho to 0.71 not
~0.95. We report rho with and without down_proj to see if it worsens at scale.

Models are ungated (no HF token needed). Qwen2.5 0.5/1.5/7B gives a clean
within-family scale axis; SmolLM-135M anchors a second family at the small end.

Run:  uv run modal run scripts/validate_quant_risk_breadth.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-quant-risk-breadth")

# Persist the HF cache across runs so re-runs don't re-download 7B weights.
hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        "torchao>=0.7",
        "transformers>=4.44",
        "accelerate",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

MODELS = [
    "HuggingFaceTB/SmolLM-135M",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B",
]
SCHEMES = ["w8a8", "int4wo"]


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def sweep() -> dict:
    import torch

    from firefly.quant.torchao import spearman, validate_against_torchao

    # Descriptive threshold for "a strong rank correlation" in this research
    # sweep — NOT a product gate (the predictor was falsified; see the reframe).
    STRONG_RHO = 0.5

    print(f"torch: {torch.__version__}  device: {torch.cuda.get_device_name(0)}")
    print(f"STRONG_RHO={STRONG_RHO}")
    print()

    def analyze(result) -> dict:
        recs = result.records
        no_dp = [r for r in recs if "down_proj" not in r.name]
        rho_no_dp = spearman(
            [r.channel_concentration for r in no_dp],
            [r.actual_local_err for r in no_dp],
        )
        worst = max(recs, key=lambda r: r.actual_local_err) if recs else None
        return {
            "n_linears": len(recs),
            "rho_concentration": round(result.spearman_concentration, 3),
            "rho_per_tensor": round(result.spearman_per_tensor, 3),
            "rho_concentration_no_downproj": round(rho_no_dp, 3),
            "strong": result.spearman_concentration > STRONG_RHO,
            "worst_layer": worst.name if worst else None,
            "worst_layer_err": round(worst.actual_local_err, 4) if worst else None,
            "worst_layer_conc": round(worst.channel_concentration, 1) if worst else None,
        }

    results: dict = {}
    for model_id in MODELS:
        results[model_id] = {}
        for scheme in SCHEMES:
            tag = f"{model_id}  [{scheme}]"
            try:
                res = validate_against_torchao(
                    model_id, device="cuda", scheme=scheme, dtype=torch.bfloat16
                )
                a = analyze(res)
                results[model_id][scheme] = a
                print(f"{tag:48s} rho_conc={a['rho_concentration']:+.3f} "
                      f"(no_dp={a['rho_concentration_no_downproj']:+.3f}) "
                      f"n={a['n_linears']} pass={a['passed']}")
            except Exception as e:  # report-and-continue: a failure is a data point
                results[model_id][scheme] = {"error": f"{type(e).__name__}: {e}"}
                print(f"{tag:48s} ERROR {type(e).__name__}: {e}")
            finally:
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary: does W8A8 generalize, and is the int4wo boundary confirmed?
    # ------------------------------------------------------------------
    def rho(m, s):
        return results[m][s].get("rho_concentration") if "error" not in results[m][s] else None

    w8a8_rhos = {m: rho(m, "w8a8") for m in MODELS}
    int4wo_rhos = {m: rho(m, "int4wo") for m in MODELS}

    w8a8_vals = [v for v in w8a8_rhos.values() if v is not None]
    w8a8_generalizes = bool(w8a8_vals) and all(v > STRONG_RHO for v in w8a8_vals)
    # Boundary: where both ran, W8A8 should clearly out-predict int4 weight-only.
    gaps = [
        w8a8_rhos[m] - int4wo_rhos[m]
        for m in MODELS
        if w8a8_rhos[m] is not None and int4wo_rhos[m] is not None
    ]
    boundary_confirmed = bool(gaps) and all(g > 0.25 for g in gaps)

    summary = {
        "w8a8_rho_by_model": w8a8_rhos,
        "int4wo_rho_by_model": int4wo_rhos,
        "w8a8_generalizes_across_scale": w8a8_generalizes,
        "int4wo_boundary_confirmed": boundary_confirmed,
        "w8a8_rho_gaps_over_int4wo": [round(g, 3) for g in gaps],
    }

    print()
    print("=" * 64)
    print(f"W8A8 generalizes (all rho > {STRONG_RHO}): {w8a8_generalizes}")
    print(f"  by model: {w8a8_rhos}")
    print(f"int4wo boundary confirmed (W8A8 >> int4wo): {boundary_confirmed}")
    print(f"  int4wo by model: {int4wo_rhos}")
    print("=" * 64)

    return {"results": results, "summary": summary}


@app.local_entrypoint()
def main() -> None:
    import json

    verdict = sweep.remote()
    print()
    print(json.dumps(verdict, indent=2, default=str))
