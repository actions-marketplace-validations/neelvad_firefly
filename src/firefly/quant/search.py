"""Min-memory-at-bar search — the LLM harness loop (proposer-agnostic).

The scenario where an LLM beats the deterministic router: **clear a perplexity
bar at minimum weight memory**, which needs composition + tradeoff reasoning the
router's fixed signature→treatment rules can't do. The loop sets up the oracle
once (sensors + fp baseline), then repeatedly: propose a compact action → expand
to a Recipe → apply + verify (perplexity) + attribute (residual) + cost → feed
the result back. ``propose`` is any callable (the LLM, or a deterministic stub);
the loop tracks the lowest-memory recipe that clears the bar.
"""

from __future__ import annotations

import copy
from collections import Counter

from firefly.capture import (
    load_golden_inputs,
    load_model_and_tokenizer,
    parse_dtype,
    run_capture,
)
from firefly.compare import diff_captures
from firefly.determinism import set_deterministic
from firefly.quant.cost import dtype_bits, linear_numels, memory_envelope, recipe_memory_bytes
from firefly.quant.diagnose import diagnose
from firefly.quant.evaluate import AccuracyBar, perplexity_evaluator
from firefly.quant.llm import compact_to_recipe
from firefly.quant.recipe_io import apply_recipe
from firefly.quant.salience import weight_salience
from firefly.quant.sensitivity import _apply_policy, discover_units


def min_memory_search(
    model_id: str,
    inputs_path,
    eval_texts: list[str],
    *,
    propose,
    scheme: str = "int4wo",
    bar: AccuracyBar | None = None,
    group_size: int = 128,
    device: str = "cpu",
    dtype: str = "float32",
    max_length: int = 64,
    budget: int = 6,
) -> dict:
    """Search for the lowest-memory recipe that clears ``bar`` (perplexity, rel
    to fp). ``propose(bundle, history) -> action`` is the agent."""
    bar = bar or AccuracyBar("rel", 0.03)
    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(model_id, device=device, dtype=parse_dtype(dtype))
    batch = load_golden_inputs(inputs_path, tok, device)
    ref_tensors = run_capture(fp_model, batch)
    tap_order = list(ref_tensors)
    units = discover_units(fp_model, "layer")
    all_fqns = {f for fqns in units.values() for f in fqns}
    numels = linear_numels(fp_model, all_fqns)
    base_bits = dtype_bits(dtype)
    quant_bits = 4 if scheme == "int4wo" else 8

    lin_fqns = {f for fqns in discover_units(fp_model, "linear").values() for f in fqns}
    salience = weight_salience(fp_model, lin_fqns, batch)
    diagnosis = diagnose(ref_tensors, tap_order, salience=salience, bits=quant_bits)

    ev = perplexity_evaluator(eval_texts, max_length=max_length)
    ppl_fp = ev(fp_model, tok)
    threshold = bar.threshold(ppl_fp, higher_is_better=False)
    env = memory_envelope(numels, base_bits=base_bits, scheme=scheme, group_size=group_size)

    base_model = _apply_policy(fp_model, scheme, group_size, all_fqns)
    ppl_plain = ev(base_model, tok)
    del base_model

    # marginal memory of keeping a unit fp vs quantizing it (what each fp unit costs).
    unit_extra_mb = {
        u: round(sum(numels[f] for f in fqns) * (base_bits - quant_bits) / 8 / 1e6, 2)
        for u, fqns in units.items()
    }
    bundle = {
        "goal": "clear the perplexity bar at minimum weight memory",
        "scheme": scheme,
        "group_size": group_size,
        "bar": f"{bar.mode}:{bar.value}",
        "perplexity_fp": round(ppl_fp, 3),
        "perplexity_threshold": round(threshold, 3),
        "perplexity_plain_quant": round(ppl_plain, 3),
        "memory_all_fp_mb": round(env.all_fp_bytes / 1e6, 1),
        "memory_all_quant_mb": round(env.all_quant_bytes / 1e6, 1),
        "diagnosis": dict(Counter(f.signature for f in diagnosis.findings)),
        "salience_top_linears": [(s.fqn, round(s.salience_concentration, 1)) for s in salience[:8]],
        "extra_mb_per_kept_fp_unit": unit_extra_mb,
        "budget_steps": budget,
    }

    history: list[dict] = []
    best: dict | None = None
    for step in range(budget):
        action = propose(bundle, history)
        if action is None:
            break
        recipe = compact_to_recipe(
            action, model_id=model_id, scheme=scheme, all_fqns=all_fqns, unit_fqns=units,
            inputs_path=inputs_path, default_group_size=group_size, dtype=dtype, device=device,
        )
        model = apply_recipe(recipe, copy.deepcopy(fp_model), batch)
        caps = run_capture(model, batch)
        ppl = ev(model, tok)
        del model
        mem = recipe_memory_bytes(
            numels, set(recipe.quantize_fqns), base_bits=base_bits, scheme=scheme, group_size=group_size
        )
        worst = sorted(
            diff_captures(ref_tensors, caps, tap_order), key=lambda d: d.rel_mean, reverse=True
        )[:5]
        passed = ppl <= threshold
        history.append({
            "step": step,
            "action": action,
            "perplexity": round(ppl, 3),
            "memory_mb": round(mem / 1e6, 1),
            "passed_bar": passed,
            "residual_worst_taps": [(d.tap_name, round(d.rel_mean, 3)) for d in worst],
        })
        if passed and (best is None or mem < best["memory_bytes"]):
            best = {"step": step, "recipe": recipe, "perplexity": ppl, "memory_bytes": mem, "action": action}

    return {
        "model": model_id,
        "scheme": scheme,
        "bundle": bundle,
        "history": history,
        "best": best,
        "perplexity_fp": ppl_fp,
        "perplexity_plain": ppl_plain,
        "threshold": threshold,
        "all_fp_mb": env.all_fp_bytes / 1e6,
        "all_quant_mb": env.all_quant_bytes / 1e6,
    }
