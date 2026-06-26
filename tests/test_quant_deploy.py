"""Pure-logic tests for the deploy seam — the deployability classification and
the serve command. The actual export (load + torchao quantize + save) needs the
torchao extra and a GPU, so it's validated on Modal (scripts/validate_deploy.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from firefly.quant.deploy import (
    DIRECTLY_DEPLOYABLE,
    NEEDS_FOLDING,
    NOT_YET,
    DeployabilityError,
    classify_recipe,
    export_deployable,
    serve_command,
)
from firefly.quant.intervention import RTNQuantizer
from firefly.quant.recipe_io import Recipe, serialize_intervention


def _recipe(*, scheme="int8wo", kept_fp=None, pre=None, quantizer=None) -> Recipe:
    return Recipe(
        model_id="some/model",
        scheme=scheme,
        group_size=32,
        granularity="layer",
        quantize_fqns=["model.layers.0.mlp.down_proj"],
        kept_fp_fqns=kept_fp or [],
        pre_transforms=pre or [],
        quantizer=quantizer or serialize_intervention(RTNQuantizer()),
    )


class TestClassify:
    def test_int8wo_uniform_rtn_is_directly_deployable(self):
        status, reason = classify_recipe(_recipe(scheme="int8wo"))
        assert status == DIRECTLY_DEPLOYABLE
        assert "torchao" in reason

    def test_int4wo_uniform_rtn_is_deployable(self):
        status, _ = classify_recipe(_recipe(scheme="int4wo"))
        assert status == DIRECTLY_DEPLOYABLE

    def test_w8a8_not_directly_deployable_serialization(self):
        # w8a8's dynamic-activation tensor subclass won't serialize → not servable
        # via this path (GPU-confirmed); reported, not faked.
        status, reason = classify_recipe(_recipe(scheme="w8a8"))
        assert status == NOT_YET
        assert "serialize" in reason and "weight-only" in reason

    def test_smoothquant_needs_folding(self):
        r = _recipe(pre=[{"name": "smoothquant", "params": {"alpha": 0.5}}])
        status, reason = classify_recipe(r)
        assert status == NEEDS_FOLDING
        assert "fold" in reason.lower()

    def test_mixed_precision_not_yet(self):
        status, reason = classify_recipe(_recipe(kept_fp=["model.layers.5.mlp.down_proj"]))
        assert status == NOT_YET
        assert "mixed precision" in reason

    def test_awq_not_yet(self):
        r = _recipe(quantizer={"name": "awq", "params": {"group_size": 128}})
        status, reason = classify_recipe(r)
        assert status == NOT_YET
        assert "AWQ" in reason

    def test_smoothquant_reported_before_mixed_precision(self):
        # A SmoothQuant + mixed-precision recipe surfaces the folding blocker first.
        r = _recipe(
            kept_fp=["model.layers.5.mlp.down_proj"],
            pre=[{"name": "smoothquant", "params": {"alpha": 0.5}}],
        )
        assert classify_recipe(r)[0] == NEEDS_FOLDING

    def test_unknown_scheme_not_yet(self):
        status, _ = classify_recipe(_recipe(scheme="fp6"))
        assert status == NOT_YET


class TestServeCommand:
    def test_includes_quantization_flag(self):
        cmd = serve_command(Path("/models/q"))
        assert "vllm serve /models/q" in cmd
        assert "--quantization torchao" in cmd

    def test_max_model_len_optional(self):
        assert "--max-model-len" not in serve_command(Path("/m"))
        assert "--max-model-len 4096" in serve_command(Path("/m"), max_model_len=4096)


class TestExportRefusesUndeployable:
    def test_export_raises_on_smoothquant(self, tmp_path):
        r = _recipe(pre=[{"name": "smoothquant", "params": {"alpha": 0.5}}])
        with pytest.raises(DeployabilityError, match="fold"):
            export_deployable(r, tmp_path)

    def test_export_raises_on_mixed_precision(self, tmp_path):
        with pytest.raises(DeployabilityError, match="mixed precision"):
            export_deployable(_recipe(kept_fp=["model.layers.5.mlp.down_proj"]), tmp_path)
