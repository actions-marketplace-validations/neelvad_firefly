"""Tests for quantization-risk attribution.

The pure core simulates symmetric quantization of captured activations and
flags taps where per-tensor scaling breaks down (outlier-feature layers).
All offline on synthetic tensors; the CLI smoke test writes a tiny artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from firefly.quant_risk import (
    analyze_quant_risk,
    simulate_quant_error,
    tap_quant_risk,
)

# --- simulate_quant_error ---------------------------------------------------


def test_zero_tensor_quantizes_exactly() -> None:
    assert simulate_quant_error(torch.zeros(4, 8)) == 0.0
    assert simulate_quant_error(torch.zeros(4, 8), per_channel=True) == 0.0


def test_uniform_tensor_has_small_int8_error() -> None:
    torch.manual_seed(0)
    t = torch.randn(16, 32)
    err = simulate_quant_error(t, bits=8)
    # Gaussian data under int8: ~1% expected channel-mean relative error.
    assert 0 < err < 0.02


def test_lower_bits_means_higher_error() -> None:
    torch.manual_seed(0)
    t = torch.randn(16, 32)
    assert simulate_quant_error(t, bits=4) > simulate_quant_error(t, bits=8)


def test_rejects_invalid_bits() -> None:
    with pytest.raises(ValueError, match="bits must be >= 2"):
        simulate_quant_error(torch.randn(4, 4), bits=1)


def test_per_channel_rescues_outlier_channel() -> None:
    """One channel 1000× the others: per-tensor scale crushes the small
    channels; per-channel scaling quantizes each at its own resolution."""
    torch.manual_seed(0)
    t = torch.randn(64, 16)
    t[:, 3] *= 1000.0
    per_tensor = simulate_quant_error(t, bits=8, per_channel=False)
    per_channel = simulate_quant_error(t, bits=8, per_channel=True)
    assert per_tensor > 10 * per_channel


# --- tap_quant_risk ----------------------------------------------------------


def test_outlier_channel_flags_high_concentration_and_gain() -> None:
    torch.manual_seed(0)
    t = torch.randn(64, 16)
    t[:, 3] *= 1000.0

    risk = tap_quant_risk("layer.11.mlp", t)

    assert risk.channel_concentration > 100.0
    assert risk.mitigation_gain > 10.0
    assert risk.per_tensor_rel_err > risk.per_channel_rel_err


def test_uniform_tensor_is_low_risk() -> None:
    torch.manual_seed(0)
    t = torch.randn(64, 16)

    risk = tap_quant_risk("layer.0.mlp", t)

    # No outlier structure: channels within a few × of each other, and
    # per-channel scaling buys little.
    assert risk.channel_concentration < 3.0
    assert risk.mitigation_gain < 2.0
    assert risk.outlier_ratio < 10.0


def test_zero_tensor_edge_case() -> None:
    risk = tap_quant_risk("t", torch.zeros(4, 8))
    assert risk.per_tensor_rel_err == 0.0
    assert risk.per_channel_rel_err == 0.0
    assert risk.mitigation_gain == 1.0
    assert risk.channel_concentration == 1.0
    assert risk.outlier_ratio == 1.0


# --- analyze_quant_risk ------------------------------------------------------


def test_analyze_preserves_forward_order_and_skips_missing() -> None:
    torch.manual_seed(0)
    tensors = {
        "layer.0": torch.randn(4, 8),
        "layer.1": torch.randn(4, 8),
        "final_norm": torch.randn(4, 8),
    }
    tap_order = ["layer.0", "layer.1", "layer.2", "final_norm"]  # layer.2 missing

    risks = analyze_quant_risk(tensors, tap_order)

    assert [r.tap_name for r in risks] == ["layer.0", "layer.1", "final_norm"]


def test_analyze_passes_bits_through() -> None:
    torch.manual_seed(0)
    tensors = {"t": torch.randn(16, 32)}
    [r8] = analyze_quant_risk(tensors, ["t"], bits=8)
    [r4] = analyze_quant_risk(tensors, ["t"], bits=4)
    assert r4.per_tensor_rel_err > r8.per_tensor_rel_err


# --- report + CLI ------------------------------------------------------------


def test_render_quant_risk_flags_above_threshold() -> None:
    from firefly.report import render_quant_risk

    torch.manual_seed(0)
    outlier = torch.randn(64, 16)
    outlier[:, 3] *= 1000.0
    risks = analyze_quant_risk(
        {"layer.0": torch.randn(64, 16), "layer.11.mlp": outlier},
        ["layer.0", "layer.11.mlp"],
    )

    text = render_quant_risk(risks, bits=8, threshold=0.01)

    assert "layer.11.mlp" in text
    assert "Worst: layer.11.mlp" in text


def test_render_quant_risk_all_clean() -> None:
    from firefly.report import render_quant_risk

    torch.manual_seed(0)
    risks = analyze_quant_risk({"layer.0": torch.randn(64, 16)}, ["layer.0"])
    text = render_quant_risk(risks, bits=8, threshold=0.05)
    assert "within 5.0% simulated int8 error" in text


def test_quant_risk_cli(tmp_path: Path) -> None:
    """End-to-end through the typer CLI on a hand-written artifact."""
    from typer.testing import CliRunner

    from firefly.cli import app
    from firefly.reference import ReferenceManifest, write_reference

    torch.manual_seed(0)
    outlier = torch.randn(8, 16)
    outlier[:, 3] *= 1000.0
    tensors = {"layer.0": torch.randn(8, 16), "layer.11.mlp": outlier}
    manifest = ReferenceManifest(
        model_id="fake",
        model_fingerprint="abc",
        tap_points=["layer.0", "layer.11.mlp"],
        shapes={k: list(v.shape) for k, v in tensors.items()},
        dtypes={k: "float32" for k in tensors},
        captured_at="2026-06-09T00:00:00+00:00",
    )
    ref_dir = tmp_path / "reference"
    write_reference(ref_dir, manifest, tensors)

    runner = CliRunner()
    report_path = tmp_path / "quant_report.json"
    result = runner.invoke(
        app,
        [
            "quant-risk",
            "--reference", str(ref_dir),
            "--report-json", str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "quantization-risk report" in result.output

    payload = json.loads(report_path.read_text())
    assert payload["bits"] == 8
    by_name = {t["tap_name"]: t for t in payload["taps"]}
    assert by_name["layer.11.mlp"]["mitigation_gain"] > 10.0
    assert by_name["layer.0"]["per_tensor_rel_err"] < 0.02
