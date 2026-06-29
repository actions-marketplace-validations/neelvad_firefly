"""Pure-logic tests for the deploy seam — the deployability classification and
the serve command. The actual export (load + torchao quantize + save) needs the
torchao extra and a GPU, so it's validated on Modal (scripts/validate_deploy.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from firefly.quant.deploy import (
    DIRECTLY_DEPLOYABLE,
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
    @pytest.mark.parametrize(
        "scheme,ct", [("int8wo", "W8A16"), ("int4wo", "W4A16"), ("w8a8", "W8A8")]
    )
    def test_uniform_rtn_schemes_are_directly_deployable(self, scheme, ct):
        # All three uniform RTN schemes map to a compressed-tensors preset and
        # serve — including w8a8, which torchao's save_pretrained couldn't
        # serialize but compressed-tensors handles cleanly.
        status, reason = classify_recipe(_recipe(scheme=scheme))
        assert status == DIRECTLY_DEPLOYABLE
        assert ct in reason

    def test_smoothquant_is_deployable(self):
        # SmoothQuant now maps to llm-compressor's SmoothQuantModifier → servable.
        r = _recipe(pre=[{"name": "smoothquant", "params": {"alpha": 0.5}}])
        status, reason = classify_recipe(r)
        assert status == DIRECTLY_DEPLOYABLE
        assert "SmoothQuant" in reason

    def test_mixed_precision_is_deployable_via_ignore_list(self):
        status, reason = classify_recipe(_recipe(kept_fp=["model.layers.5.mlp.down_proj"]))
        assert status == DIRECTLY_DEPLOYABLE
        assert "fp-kept" in reason

    def test_smoothquant_plus_mixed_precision_is_deployable(self):
        r = _recipe(
            kept_fp=["model.layers.5.mlp.down_proj"],
            pre=[{"name": "smoothquant", "params": {"alpha": 0.5}}],
        )
        status, reason = classify_recipe(r)
        assert status == DIRECTLY_DEPLOYABLE
        assert "SmoothQuant" in reason and "fp-kept" in reason

    def test_awq_not_yet(self):
        r = _recipe(quantizer={"name": "awq", "params": {"group_size": 128}})
        status, reason = classify_recipe(r)
        assert status == NOT_YET
        assert "AWQ" in reason

    def test_unmapped_pre_transform_not_yet(self):
        status, reason = classify_recipe(_recipe(pre=[{"name": "magic-transform", "params": {}}]))
        assert status == NOT_YET
        assert "magic-transform" in reason

    def test_unknown_scheme_not_yet(self):
        status, _ = classify_recipe(_recipe(scheme="fp6"))
        assert status == NOT_YET


class TestServeCommand:
    def test_serves_the_dir_no_quant_flag(self):
        # vLLM auto-detects compressed-tensors from config.json — no flag needed.
        cmd = serve_command(Path("/models/q"))
        assert "vllm serve /models/q" in cmd
        assert "--quantization" not in cmd

    def test_max_model_len_optional(self):
        assert "--max-model-len" not in serve_command(Path("/m"))
        assert "--max-model-len 4096" in serve_command(Path("/m"), max_model_len=4096)


class TestExportRefuses:
    def test_export_raises_on_awq(self, tmp_path):
        # AWQ is still NOT_YET → refused before any model load.
        r = _recipe(quantizer={"name": "awq", "params": {"group_size": 128}})
        with pytest.raises(DeployabilityError, match="AWQ"):
            export_deployable(r, tmp_path)

    def test_smoothquant_export_without_calib_raises(self, tmp_path):
        # SmoothQuant is deployable, but its scales need calibration data — refuse
        # clearly rather than silently exporting a mis-calibrated checkpoint.
        r = _recipe(pre=[{"name": "smoothquant", "params": {"alpha": 0.5}}])
        with pytest.raises(DeployabilityError, match="calibration"):
            export_deployable(r, tmp_path)
