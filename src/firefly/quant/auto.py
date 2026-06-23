"""The deterministic auto-quant harness: sensors → diagnose → route → verify.

Ties Firefly's measurement oracle to the deterministic router in one pass and
reports the win vs a plain-quant baseline. This is the agent-agnostic loop with
the *deterministic* proposer (the router); an LLM proposer plugs into the same
``diagnose -> recipe`` slot later. Thin on purpose — it just composes existing
library functions.
"""

from __future__ import annotations

import copy

from firefly.capture import (
    load_golden_inputs,
    load_model_and_tokenizer,
    parse_dtype,
    run_capture,
)
from firefly.compare import diff_captures
from firefly.determinism import set_deterministic
from firefly.quant.diagnose import diagnose
from firefly.quant.evaluate import perplexity_evaluator
from firefly.quant.recipe_io import apply_recipe
from firefly.quant.route import route_recipe
from firefly.quant.salience import weight_salience
from firefly.quant.sensitivity import _apply_policy, discover_units
from firefly.quant.torchao import rel_l1

_OUTPUT_TAP = "final_norm"


def auto_quant(
    model_id: str,
    inputs_path,
    eval_texts: list[str],
    *,
    scheme: str = "w8a8",
    group_size: int = 128,
    device: str = "cpu",
    dtype: str = "float32",
    max_length: int = 64,
    with_sensitivity: bool = False,
) -> dict:
    """Diagnose the model, route the diagnosis to a recipe, apply + verify it
    against a plain-quant baseline, and return the structured result."""
    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(model_id, device=device, dtype=parse_dtype(dtype))
    batch = load_golden_inputs(inputs_path, tok, device)

    ref_tensors = run_capture(fp_model, batch)
    ref_out = ref_tensors[_OUTPUT_TAP]
    tap_order = list(ref_tensors)
    units = discover_units(fp_model, "layer")
    all_fqns = {f for fqns in units.values() for f in fqns}
    lin_fqns = {f for fqns in discover_units(fp_model, "linear").values() for f in fqns}

    # --- sensors → diagnosis (cheap: channel_concentration + weight salience) ---
    salience = weight_salience(fp_model, lin_fqns, batch)
    bits = 4 if scheme == "int4wo" else 8
    sens = None
    if with_sensitivity:
        # Opt-in (an N-measurement sweep): emits SINGLE_UNIT_DOMINANCE → mixed
        # precision. Reuses the loaded model (no second load).
        from firefly.quant.sensitivity import _Ctx, _run_sensitivity

        ctx = _Ctx(model_id, fp_model, tok, batch, ref_out, units, all_fqns, scheme, group_size, "layer")
        sens = _run_sensitivity(ctx, "isolated")
    diagnosis = diagnose(ref_tensors, tap_order, sensitivity=sens, salience=salience, bits=bits)

    # --- route → recipe (the deterministic agent) ---
    recipe = route_recipe(
        diagnosis, model_id=model_id, scheme=scheme, group_size=group_size,
        all_fqns=all_fqns, unit_fqns=units, inputs_path=inputs_path, dtype=dtype, device=device,
    )

    ev = perplexity_evaluator(eval_texts, max_length=max_length)
    ppl_fp = ev(fp_model, tok)

    def measure(m) -> tuple[float, float]:
        return rel_l1(ref_out, run_capture(m, batch)[_OUTPUT_TAP]), ev(m, tok)

    # --- baseline: plain quant at the target scheme ---
    base_model = _apply_policy(fp_model, scheme, group_size, all_fqns)
    d_base, ppl_base = measure(base_model)
    del base_model

    # --- routed: the agent's recipe (apply + verify + attribute) ---
    routed_model = apply_recipe(recipe, copy.deepcopy(fp_model), batch)
    routed_caps = run_capture(routed_model, batch)
    d_routed = rel_l1(ref_out, routed_caps[_OUTPUT_TAP])
    ppl_routed = ev(routed_model, tok)
    worst = sorted(
        diff_captures(ref_tensors, routed_caps, tap_order), key=lambda x: x.rel_mean, reverse=True
    )[:5]

    from collections import Counter

    sig_counts = Counter(f.signature for f in diagnosis.findings)
    gap = ppl_base - ppl_fp
    recovery = (ppl_base - ppl_routed) / gap if gap > 0 else 0.0
    # Measurement-gated: ship the routed recipe only if it actually beat plain
    # quant. The verify step is the point — the router proposes by signature, the
    # measurement decides. (SmoothQuant, say, helps a 7B but hurts a tiny model.)
    accepted = ppl_routed < ppl_base
    return {
        "model": model_id,
        "scheme": scheme,
        "diagnosis_summary": dict(sig_counts),
        "diagnosis": [(f.signature, f.location) for f in diagnosis.findings[:8]],
        "routing": recipe.result["routing"],
        "recipe": {
            "quantizer": recipe.quantizer["name"],
            "pre_transforms": [p["name"] for p in recipe.pre_transforms],
            "quantize_fqns": len(recipe.quantize_fqns),
            "kept_fp_fqns": len(recipe.kept_fp_fqns),
        },
        "perplexity": {"fp": ppl_fp, "plain": ppl_base, "routed": ppl_routed},
        "divergence": {"plain": d_base, "routed": d_routed},
        "recovery": recovery,
        "accepted": accepted,
        "chosen": "routed" if accepted else "plain",
        "attribution_worst_taps": [(d.tap_name, d.rel_mean) for d in worst],
        "recipe_obj": recipe,
    }
