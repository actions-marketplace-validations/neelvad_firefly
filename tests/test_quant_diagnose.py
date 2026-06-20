"""Diagnosis: the sensor that emits detectable signatures from measurements.
Fast + deterministic (synthetic activations / a hand-built sensitivity result)."""

from __future__ import annotations

import torch

from firefly.quant.diagnose import (
    SIGNATURE_TREATMENTS,
    diagnose,
    diagnose_activation_outliers,
    diagnose_salient_weight_channels,
    diagnose_single_unit_dominance,
)
from firefly.quant.intervention import (
    ACTIVATION_OUTLIERS,
    SALIENT_WEIGHT_CHANNELS,
    SINGLE_UNIT_DOMINANCE,
)
from firefly.quant.salience import LinearSalience
from firefly.quant.sensitivity import SensitivityResult, UnitSensitivity


def test_only_detectable_signatures_exist() -> None:
    # de-theater: a signature is shipped only with BOTH a detector and a treatment.
    # SALIENT_WEIGHT_CHANNELS is now in (salience sensor + AWQ); GPTQ's stays out.
    import firefly.quant.intervention as iv

    assert not hasattr(iv, "DIFFUSE_WEIGHT_LOSS")
    assert hasattr(iv, "SALIENT_WEIGHT_CHANNELS")
    assert set(SIGNATURE_TREATMENTS) == {
        ACTIVATION_OUTLIERS, SINGLE_UNIT_DOMINANCE, SALIENT_WEIGHT_CHANNELS
    }
    assert SIGNATURE_TREATMENTS[SALIENT_WEIGHT_CHANNELS] == "awq"


def test_activation_outliers_detected_on_outlier_tap() -> None:
    torch.manual_seed(0)
    tensors = {"layer.5.mlp": torch.randn(4, 16), "layer.0.mlp": torch.randn(4, 16)}
    tensors["layer.5.mlp"][:, 3] *= 200.0  # a dominant outlier channel
    findings = diagnose_activation_outliers(tensors, ["layer.0.mlp", "layer.5.mlp"], bits=8)
    assert [f.location for f in findings] == ["layer.5.mlp"]
    f = findings[0]
    assert f.signature == ACTIVATION_OUTLIERS and f.recommend == "smoothquant"
    assert f.evidence["channel_concentration"] > 8.0
    assert "smoothquant" in f.explanation.lower()


def test_activation_outliers_skips_clean_taps() -> None:
    torch.manual_seed(1)
    tensors = {"layer.0.mlp": torch.randn(8, 32)}  # no outlier channel
    assert diagnose_activation_outliers(tensors, ["layer.0.mlp"], bits=8) == []


def _sens(units: list[tuple[str, float]]) -> SensitivityResult:
    return SensitivityResult(
        model_id="m", scheme="w8a8", strategy="isolated", granularity="layer",
        full_quant_divergence=0.6,
        units=[UnitSensitivity(u, s, raw_divergence=s, n_linears=7) for u, s in units],
    )


def test_single_unit_dominance_detected() -> None:
    sens = _sens([("layer.28", 0.40), ("layer.1", 0.02), ("layer.2", 0.03), ("layer.3", 0.02)])
    findings = diagnose_single_unit_dominance(sens, ratio_threshold=5.0)
    assert len(findings) == 1
    assert findings[0].signature == SINGLE_UNIT_DOMINANCE
    assert findings[0].location == "layer.28" and findings[0].recommend == "mixed-precision"


def test_single_unit_dominance_not_flagged_when_flat() -> None:
    sens = _sens([("a", 0.10), ("b", 0.09), ("c", 0.11), ("d", 0.10)])
    assert diagnose_single_unit_dominance(sens, ratio_threshold=5.0) == []


def test_salient_weight_channels_detected_and_routes_to_awq() -> None:
    sal = [
        LinearSalience("model.layers.0.mlp.down_proj", 800.0, 1536),
        LinearSalience("model.layers.5.self_attn.q_proj", 12.0, 896),  # below threshold
    ]
    findings = diagnose_salient_weight_channels(sal, concentration_threshold=50.0)
    assert [f.location for f in findings] == ["model.layers.0.mlp.down_proj"]
    f = findings[0]
    assert f.signature == SALIENT_WEIGHT_CHANNELS and f.recommend == "awq"
    assert "awq" in f.explanation.lower()


def test_salient_weight_channels_skips_uniform() -> None:
    sal = [LinearSalience("a", 2.0, 256), LinearSalience("b", 1.5, 256)]
    assert diagnose_salient_weight_channels(sal, concentration_threshold=50.0) == []


def test_diagnose_combines_all_three_sensors() -> None:
    torch.manual_seed(0)
    tensors = {"layer.5.mlp": torch.randn(4, 16)}
    tensors["layer.5.mlp"][:, 3] *= 200.0
    sens = _sens([("layer.28", 0.40), ("layer.1", 0.02), ("layer.2", 0.03)])
    sal = [LinearSalience("model.layers.0.mlp.down_proj", 800.0, 1536)]
    d = diagnose(tensors, ["layer.5.mlp"], sensitivity=sens, salience=sal, bits=8)
    sigs = {f.signature for f in d.findings}
    assert sigs == {ACTIVATION_OUTLIERS, SINGLE_UNIT_DOMINANCE, SALIENT_WEIGHT_CHANNELS}
    assert d.by_signature(SALIENT_WEIGHT_CHANNELS)[0].recommend == "awq"


def test_render_diagnosis() -> None:
    from firefly.report import render_diagnosis

    torch.manual_seed(0)
    tensors = {"layer.5.mlp": torch.randn(4, 16)}
    tensors["layer.5.mlp"][:, 3] *= 200.0
    d = diagnose(tensors, ["layer.5.mlp"], bits=8)
    out = render_diagnosis(d, model_id="org/m")
    assert "activation_outliers" in out and "layer.5.mlp" in out
    assert "verify:" in out and "--smoothquant" in out


def test_render_diagnosis_empty() -> None:
    from firefly.quant.diagnose import Diagnosis
    from firefly.report import render_diagnosis

    assert "No quant failure-mode" in render_diagnosis(Diagnosis())
