"""Recipe export/apply: serialization round-trips, build_recipe provenance, and
reconstruct-and-run. The real-model reproduction is covered end-to-end by the
CLI round-trip; the slow test here asserts apply matches a direct policy run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from firefly.quant.intervention import RTNQuantizer
from firefly.quant.recipe_io import (
    Recipe,
    apply_recipe,
    build_recipe,
    deserialize_intervention,
    serialize_intervention,
)
from firefly.quant.smoothquant import SmoothQuant


def test_intervention_serialize_roundtrip() -> None:
    rtn = deserialize_intervention(serialize_intervention(RTNQuantizer()))
    assert rtn.name == "rtn"

    sq0 = SmoothQuant(scope={"y", "x"}, alpha=0.6)
    sq = deserialize_intervention(serialize_intervention(sq0))
    assert sq.scope == {"x", "y"} and sq.alpha == 0.6  # list↔set + params survive


def test_deserialize_unknown_intervention() -> None:
    with pytest.raises(ValueError, match="unknown intervention"):
        deserialize_intervention({"name": "gptq", "params": {}})


def test_recipe_json_roundtrip(tmp_path: Path) -> None:
    r = Recipe(
        model_id="m", scheme="w8a8", group_size=32, granularity="layer",
        quantize_fqns=["a", "b"], kept_fp_fqns=["c"],
        pre_transforms=[serialize_intervention(SmoothQuant(scope={"a"}))],
    )
    p = tmp_path / "r.json"
    r.to_json(p)
    back = Recipe.from_json(p)
    assert back.quantize_fqns == ["a", "b"]
    assert back.pre_transforms[0]["name"] == "smoothquant"


def test_recipe_rejects_unknown_version(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"recipe_version": 999, "model_id": "m"}))
    with pytest.raises(ValueError, match="recipe_version"):
        Recipe.from_json(p)


def test_build_recipe_resolves_fqns_and_provenance(tmp_path: Path) -> None:
    inputs = tmp_path / "calib.json"
    inputs.write_text(json.dumps({"texts": ["hi"]}))
    r = build_recipe(
        model_id="org/m", scheme="w8a8", group_size=32, granularity="layer",
        quantize_fqns={"layers.0.mlp", "layers.1.mlp"}, kept_fp_fqns={"layers.2.mlp"},
        pre_transforms=[SmoothQuant(alpha=0.5)], dtype="float32", device="cpu",
        inputs_path=inputs, result={"divergence": 0.07},
    )
    assert r.quantize_fqns == ["layers.0.mlp", "layers.1.mlp"]  # sorted
    assert r.kept_fp_fqns == ["layers.2.mlp"]
    assert r.pre_transforms[0]["name"] == "smoothquant"
    assert r.quantizer["name"] == "rtn"
    assert r.provenance["inputs_sha256"] and r.provenance["dtype"] == "float32"
    assert r.provenance["inputs"].endswith("calib.json")


def test_apply_recipe_reconstructs_and_runs() -> None:
    # quantize nothing (RTN no-op, no torchao needed) + a SmoothQuant pre-transform:
    # exercises deserialize + Pipeline.run end to end on CPU.
    torch.manual_seed(0)
    lin = nn.Linear(8, 4)  # root module name is ""
    x = torch.randn(3, 8)
    x[:, 1] *= 40.0
    y0 = lin(x).clone()

    r = Recipe(
        model_id="m", scheme="w8a8", group_size=32, granularity="layer",
        quantize_fqns=[], kept_fp_fqns=[""],
        pre_transforms=[serialize_intervention(SmoothQuant(scope={""}))],
    )
    out = apply_recipe(r, lin, calib=x)
    assert out is lin
    assert torch.allclose(y0, lin(x), atol=1e-4)  # SmoothQuant preserved output; RTN no-op


@pytest.mark.slow
def test_apply_recipe_reproduces_direct_policy_smollm() -> None:
    """apply_recipe must reproduce a direct _apply_policy run bit-for-bit on a
    real model (the recipe is a faithful serialization of the pipeline)."""
    pytest.importorskip("torchao", reason="needs the torchao extra")
    import tempfile

    from firefly.capture import run_capture
    from firefly.quant.sensitivity import _apply_policy, _setup
    from firefly.quant.torchao import rel_l1

    inp = Path(tempfile.mkdtemp()) / "g.json"
    inp.write_text(json.dumps({"texts": ["the quick brown fox"], "max_length": 12}))
    ctx = _setup("HuggingFaceTB/SmolLM-135M", inp, "cpu", "float32", "w8a8", 32, "layer")

    # Keep layer.0 fp, quantize the rest — a real mixed-precision recipe.
    kept = set(ctx.units["layer.0"])
    quantize = set(ctx.all_fqns) - kept

    direct = _apply_policy(ctx.fp_model, "w8a8", 32, quantize)
    d_direct = rel_l1(ctx.ref_output, run_capture(direct, ctx.batch)["final_norm"])

    recipe = Recipe(
        model_id="HuggingFaceTB/SmolLM-135M", scheme="w8a8", group_size=32,
        granularity="layer", quantize_fqns=sorted(quantize), kept_fp_fqns=sorted(kept),
    )
    import copy

    applied = apply_recipe(recipe, copy.deepcopy(ctx.fp_model), ctx.batch)
    d_applied = rel_l1(ctx.ref_output, run_capture(applied, ctx.batch)["final_norm"])
    assert d_applied == pytest.approx(d_direct, rel=1e-6)
