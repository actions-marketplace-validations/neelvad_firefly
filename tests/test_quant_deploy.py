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
    export_method,
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


class TestExportMethod:
    def test_int4_rtn_maps_to_gptq(self):
        # plain int4 RTN serves wrecked (+113%); GPTQ recovers ~96% → int4 uses GPTQ.
        assert export_method(_recipe(scheme="int4wo")) == "gptq"

    def test_int4_awq_maps_to_awq(self):
        r = _recipe(scheme="int4wo", quantizer={"name": "awq", "params": {"group_size": 128}})
        assert export_method(r) == "awq"

    @pytest.mark.parametrize("scheme", ["w8a8", "int8wo"])
    def test_int8_maps_to_rtn(self, scheme):
        assert export_method(_recipe(scheme=scheme)) == "rtn"


class TestClassify:
    @pytest.mark.parametrize(
        "scheme,ct,method", [("int8wo", "W8A16", "RTN"), ("w8a8", "W8A8", "RTN"), ("int4wo", "W4A16", "GPTQ")]
    )
    def test_schemes_are_directly_deployable_with_method(self, scheme, ct, method):
        status, reason = classify_recipe(_recipe(scheme=scheme))
        assert status == DIRECTLY_DEPLOYABLE
        assert ct in reason and method in reason

    def test_int4_awq_is_deployable(self):
        r = _recipe(scheme="int4wo", quantizer={"name": "awq", "params": {"group_size": 128}})
        status, reason = classify_recipe(r)
        assert status == DIRECTLY_DEPLOYABLE
        assert "AWQ" in reason

    def test_smoothquant_pre_transform_is_dropped_still_deployable(self):
        # SmoothQuant is a no-op for serving → dropped, recipe deploys by scheme.
        r = _recipe(scheme="w8a8", pre=[{"name": "smoothquant", "params": {"alpha": 0.5}}])
        status, reason = classify_recipe(r)
        assert status == DIRECTLY_DEPLOYABLE
        assert "RTN" in reason  # ships as plain w8a8; SmoothQuant not mentioned

    def test_mixed_precision_is_deployable_via_ignore_list(self):
        status, reason = classify_recipe(_recipe(kept_fp=["model.layers.5.mlp.down_proj"]))
        assert status == DIRECTLY_DEPLOYABLE
        assert "fp-kept" in reason

    def test_awq_on_non_int4_not_yet(self):
        # AWQ recovery is wired for int4 only.
        r = _recipe(scheme="int8wo", quantizer={"name": "awq", "params": {"group_size": 128}})
        status, reason = classify_recipe(r)
        assert status == NOT_YET
        assert "int4" in reason

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
    def test_export_raises_on_awq_non_int4(self, tmp_path):
        # AWQ on a non-int4 scheme is NOT_YET → refused before any model load.
        r = _recipe(scheme="int8wo", quantizer={"name": "awq", "params": {"group_size": 128}})
        with pytest.raises(DeployabilityError, match="int4"):
            export_deployable(r, tmp_path)

    def test_int4_export_without_calib_raises(self, tmp_path):
        # int4 GPTQ/AWQ derive corrections from calibration — refuse clearly rather
        # than silently exporting an uncalibrated (wrecked) int4 checkpoint.
        with pytest.raises(DeployabilityError, match="calibration"):
            export_deployable(_recipe(scheme="int4wo"), tmp_path)


class TestAWQMappingFallback:
    """AWQ's smooth-layer mappings resolve per-architecture inside llm-compressor;
    unknown architectures (e.g. day-one multimodal wrappers) fail with the
    'single smoothlayer' ValueError. The export must fall back to GPTQ (the
    mapping-free canonical int4 recovery) and record the downgrade."""

    def _install_fake_llmcompressor(self, monkeypatch, oneshot):
        import importlib.machinery
        import sys
        import types

        def _module(name: str) -> types.ModuleType:
            mod = types.ModuleType(name)
            # transformers' lazy-import machinery probes __spec__; a bare
            # ModuleType carries None, which importlib rejects.
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            return mod

        lc = _module("llmcompressor")
        lc.oneshot = oneshot
        quant_mod = _module("llmcompressor.modifiers.quantization")
        awq_mod = _module("llmcompressor.modifiers.awq")

        class _Modifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        quant_mod.GPTQModifier = type("GPTQModifier", (_Modifier,), {})
        quant_mod.QuantizationModifier = type("QuantizationModifier", (_Modifier,), {})
        awq_mod.AWQModifier = type("AWQModifier", (_Modifier,), {})
        modifiers = _module("llmcompressor.modifiers")
        for name, mod in {
            "llmcompressor": lc,
            "llmcompressor.modifiers": modifiers,
            "llmcompressor.modifiers.quantization": quant_mod,
            "llmcompressor.modifiers.awq": awq_mod,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)

        ds_mod = _module("datasets")

        class _Dataset:
            @staticmethod
            def from_list(rows):
                return rows

        ds_mod.Dataset = _Dataset
        monkeypatch.setitem(sys.modules, "datasets", ds_mod)

        import transformers

        class _Tok:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2], "attention_mask": [1, 1]}

            def save_pretrained(self, path):
                pass

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok()
        )

    def test_awq_mapping_failure_falls_back_to_gptq(self, tmp_path, monkeypatch):
        calls: list[type] = []

        def oneshot(*, recipe, **kwargs):
            calls.append(type(recipe[0]).__name__)
            if type(recipe[0]).__name__ == "AWQModifier":
                raise ValueError(
                    "AWQ needs to match a single smoothlayer for each mapping but got [...]"
                )

        self._install_fake_llmcompressor(monkeypatch, oneshot)
        r = _recipe(scheme="int4wo", quantizer={"name": "awq", "params": {"group_size": 128}})
        artifact = export_deployable(r, tmp_path, calib_texts=["a", "b"])

        assert calls == ["AWQModifier", "GPTQModifier"]
        assert artifact.manifest["method"] == "gptq"
        assert artifact.manifest["method_fallback"]["from"] == "awq"
        assert "smoothlayer" in artifact.manifest["method_fallback"]["reason"]
        assert "gptq" in artifact.manifest["treatments"]

    def test_non_mapping_awq_error_still_raises(self, tmp_path, monkeypatch):
        def oneshot(**kwargs):
            raise ValueError("out of memory")

        self._install_fake_llmcompressor(monkeypatch, oneshot)
        r = _recipe(scheme="int4wo", quantizer={"name": "awq", "params": {"group_size": 128}})
        with pytest.raises(ValueError, match="out of memory"):
            export_deployable(r, tmp_path, calib_texts=["a"])
