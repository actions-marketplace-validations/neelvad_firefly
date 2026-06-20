"""The agent's step primitive: apply a proposed policy, measure it, attribute it.

This is the one verb an external coding agent calls in a loop — Firefly is the
*oracle*, the agent is the searcher. The agent emits a structured ``Recipe``
(its only action space — it cannot run arbitrary code, only compose validated
interventions), and ``quant_step`` returns everything needed to choose the next
move as one machine-readable dict:

* **verify** — the real eval metric vs the fp baseline and whether it clears the bar,
* **attribution** — per-tap divergence of the quantized run vs the fp reference,
  so the agent learns *where* it still hurts (not just pass/fail), and
* **cost** — the weight footprint / compression.

Baseline is paid once, not per step: the per-tap fp activations come from a
captured ``reference/`` (free), and the fp eval metric can be passed back in via
``baseline_metric`` so the harness computes it a single time across the loop.
"""

from __future__ import annotations

import copy

from firefly.attribution import attribute_first_divergence
from firefly.capture import (
    load_golden_inputs,
    load_model_and_tokenizer,
    parse_dtype,
    run_capture,
)
from firefly.compare import diff_captures
from firefly.determinism import set_deterministic
from firefly.quant.cost import dtype_bits, linear_numels, memory_envelope, recipe_memory_bytes
from firefly.quant.evaluate import AccuracyBar, Evaluator
from firefly.quant.recipe_io import Recipe, apply_recipe
from firefly.quant.sensitivity import discover_units
from firefly.reference import read_reference


def quant_step(
    recipe: Recipe,
    reference_dir,
    inputs_path,
    *,
    evaluator: Evaluator,
    bar: AccuracyBar,
    device: str = "cpu",
    dtype: str = "float32",
    baseline_metric: float | None = None,
    top_n: int = 10,
) -> dict:
    """Apply ``recipe``, verify against the bar, attribute per-tap, return one dict.

    ``reference_dir`` supplies the fp activations the attribution diffs against
    (the baseline, captured once). Pass ``baseline_metric`` to skip recomputing
    the fp eval metric — the harness should compute it a single time and thread
    it through every step.
    """
    set_deterministic()
    manifest, ref_tensors = read_reference(reference_dir)

    fp_model, tok = load_model_and_tokenizer(
        recipe.model_id, device=device, dtype=parse_dtype(dtype)
    )
    batch = load_golden_inputs(inputs_path, tok, device)

    hib = evaluator.higher_is_better
    baseline = baseline_metric if baseline_metric is not None else evaluator(fp_model, tok)

    quantized = apply_recipe(recipe, copy.deepcopy(fp_model), batch)

    # Attribution: where did this policy drift from fp? (free baseline = the
    # stored reference activations on the same inputs).
    cand_tensors = run_capture(quantized, batch)
    divergences = diff_captures(ref_tensors, cand_tensors, manifest.tap_points)
    worst = sorted(divergences, key=lambda d: d.rel_mean, reverse=True)[:top_n]

    # Verify against the bar.
    cand_metric = evaluator(quantized, tok)
    passed = bar.passes(cand_metric, baseline, hib)
    threshold = bar.threshold(baseline, hib)

    # Cost of this policy.
    all_fqns = {fqn for fqns in discover_units(fp_model, recipe.granularity).values() for fqn in fqns}
    numels = linear_numels(fp_model, all_fqns)
    base_bits = dtype_bits(dtype)
    cost_bytes = recipe_memory_bytes(
        numels, set(recipe.quantize_fqns),
        base_bits=base_bits, scheme=recipe.scheme, group_size=recipe.group_size,
    )
    env = memory_envelope(numels, base_bits=base_bits, scheme=recipe.scheme, group_size=recipe.group_size)

    return {
        "passed": passed,
        "policy": {
            "scheme": recipe.scheme,
            "group_size": recipe.group_size,
            "granularity": recipe.granularity,
            "quantize_fqns": len(recipe.quantize_fqns),
            "kept_fp_fqns": len(recipe.kept_fp_fqns),
            "pre_transforms": [p["name"] for p in recipe.pre_transforms],
        },
        "metric": {
            "name": evaluator.name,
            "value": cand_metric,
            "baseline": baseline,
            "threshold": threshold,
            "higher_is_better": hib,
        },
        "cost": {
            "bytes": cost_bytes,
            "all_fp_bytes": env.all_fp_bytes,
            "compression": env.compression(cost_bytes),
        },
        "attribution": {
            "first_divergent_tap": attribute_first_divergence(divergences).first_divergent_tap,
            "worst_taps": [
                {"tap": d.tap_name, "rel_mean": d.rel_mean, "max_abs_diff": d.max_abs_diff}
                for d in worst
            ],
        },
    }
