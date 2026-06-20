"""The agent's step primitive — slow end-to-end on SmolLM (capture a reference,
build a policy, run quant_step, assert the oracle JSON shape + sane values).
The composed pieces (apply_recipe, diff_captures, evaluate, cost) are unit-tested
in their own modules; this asserts the step wires them into one result."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.mark.slow
def test_quant_step_smollm_oracle_output() -> None:
    pytest.importorskip("torchao", reason="needs the torchao extra")

    from firefly.capture import capture_reference, load_model_and_tokenizer
    from firefly.quant.evaluate import AccuracyBar, perplexity_evaluator
    from firefly.quant.recipe_io import Recipe
    from firefly.quant.sensitivity import discover_units
    from firefly.quant.step import quant_step

    work = Path(tempfile.mkdtemp())
    inputs = work / "calib.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox"], "max_length": 12}))
    ref_dir = work / "ref"
    capture_reference("HuggingFaceTB/SmolLM-135M", inputs, ref_dir, dtype=torch.float32)

    # A real mixed-precision policy: keep layer.0 fp, quantize the rest (w8a8).
    model, _ = load_model_and_tokenizer("HuggingFaceTB/SmolLM-135M", device="cpu")
    units = discover_units(model, "layer")
    kept = set(units["layer.0"])
    quantize = {fqn for fqns in units.values() for fqn in fqns} - kept
    recipe = Recipe(
        model_id="HuggingFaceTB/SmolLM-135M", scheme="w8a8", group_size=32,
        granularity="layer", quantize_fqns=sorted(quantize), kept_fp_fqns=sorted(kept),
    )

    evaluator = perplexity_evaluator(["a lazy dog sleeps"], max_length=12)
    res = quant_step(
        recipe, ref_dir, inputs, evaluator=evaluator, bar=AccuracyBar("rel", 0.5),
        device="cpu", dtype="float32", top_n=5,
    )

    # verify block
    assert isinstance(res["passed"], bool)
    assert res["metric"]["name"] == "perplexity"
    assert res["metric"]["value"] > 0 and res["metric"]["baseline"] > 0
    # cost block — quantizing 29/30 layers shrinks the footprint
    assert res["cost"]["compression"] > 1.0
    assert res["cost"]["bytes"] < res["cost"]["all_fp_bytes"]
    # attribution block — ranked residual divergence, where it still hurts
    assert res["attribution"]["first_divergent_tap"]
    worst = res["attribution"]["worst_taps"]
    assert 1 <= len(worst) <= 5
    assert all(worst[i]["rel_mean"] >= worst[i + 1]["rel_mean"] for i in range(len(worst) - 1))

    # baseline-once: passing the cached fp metric back skips recompute and matches.
    res2 = quant_step(
        recipe, ref_dir, inputs, evaluator=evaluator, bar=AccuracyBar("rel", 0.5),
        device="cpu", dtype="float32", baseline_metric=res["metric"]["baseline"],
    )
    assert res2["metric"]["baseline"] == res["metric"]["baseline"]
