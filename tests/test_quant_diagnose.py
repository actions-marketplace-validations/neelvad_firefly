"""Diagnosis: the sensor that emits detectable signatures from measurements.
Fast + deterministic (synthetic activations / a hand-built sensitivity result)."""

from __future__ import annotations

import torch

from firefly.quant.diagnose import (
    SIGNATURE_TREATMENTS,
    diagnose,
    diagnose_activation_outliers,
    diagnose_single_unit_dominance,
)
from firefly.quant.intervention import ACTIVATION_OUTLIERS, SINGLE_UNIT_DOMINANCE
from firefly.quant.sensitivity import SensitivityResult, UnitSensitivity


def test_only_detectable_signatures_exist() -> None:
    # de-theater: the undetectable signatures were removed, not shipped as labels.
    import firefly.quant.intervention as iv

    assert not hasattr(iv, "DIFFUSE_WEIGHT_LOSS")
    assert not hasattr(iv, "SALIENT_WEIGHT_CHANNELS")
    assert set(SIGNATURE_TREATMENTS) == {ACTIVATION_OUTLIERS, SINGLE_UNIT_DOMINANCE}


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


def test_diagnose_combines_both_sensors() -> None:
    torch.manual_seed(0)
    tensors = {"layer.5.mlp": torch.randn(4, 16)}
    tensors["layer.5.mlp"][:, 3] *= 200.0
    sens = _sens([("layer.28", 0.40), ("layer.1", 0.02), ("layer.2", 0.03)])
    d = diagnose(tensors, ["layer.5.mlp"], sensitivity=sens, bits=8)
    sigs = {f.signature for f in d.findings}
    assert sigs == {ACTIVATION_OUTLIERS, SINGLE_UNIT_DOMINANCE}
    assert len(d.by_signature(ACTIVATION_OUTLIERS)) == 1


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
