"""Tests for per-layer tolerance calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from firefly.calibrate import derive_tolerances
from firefly.compare import DEFAULT_TOLERANCE


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


# --- derive_tolerances unit tests --------------------------------------------


def test_zero_noise_floors_atol_at_default() -> None:
    """Identical captures (no noise) → noise_floor=0 → atol=DEFAULT_TOLERANCE."""
    ref = {"layer.0": _t(1.0, 2.0, 3.0)}
    captures = {"layer.0": [ref["layer.0"].clone(), ref["layer.0"].clone()]}

    tols = derive_tolerances(ref, captures, safety_factor=6.0, n_runs=2)

    assert tols["layer.0"].noise_floor == 0.0
    assert tols["layer.0"].atol == DEFAULT_TOLERANCE


def test_observed_deviation_drives_atol() -> None:
    """With known max deviation X, atol = safety_factor × X."""
    ref = {"layer.0": _t(0.0, 0.0, 0.0)}
    captures = {
        "layer.0": [
            _t(0.0, 0.0, 0.0),          # zero dev
            _t(0.0, 0.01, 0.0),         # 0.01 dev
            _t(0.0, 0.0, 0.05),         # 0.05 dev (worst)
        ]
    }

    tols = derive_tolerances(ref, captures, safety_factor=6.0, n_runs=3)

    assert tols["layer.0"].noise_floor == pytest.approx(0.05)
    assert tols["layer.0"].atol == pytest.approx(0.3)  # 6 × 0.05


def test_safety_factor_scales_atol() -> None:
    ref = {"layer.0": _t(0.0)}
    captures = {"layer.0": [_t(0.0), _t(0.1)]}

    tight = derive_tolerances(ref, captures, safety_factor=1.0, n_runs=2)
    loose = derive_tolerances(ref, captures, safety_factor=10.0, n_runs=2)

    assert tight["layer.0"].atol == pytest.approx(0.1)
    assert loose["layer.0"].atol == pytest.approx(1.0)


def test_marks_all_results_as_calibrated() -> None:
    ref = {"layer.0": _t(0.0), "layer.1": _t(0.0)}
    captures = {"layer.0": [_t(0.0)], "layer.1": [_t(0.0)]}

    tols = derive_tolerances(ref, captures, safety_factor=6.0, n_runs=1)

    for t in tols.values():
        assert t.source == "calibrated"


def test_records_n_runs_and_noise_floor() -> None:
    ref = {"layer.0": _t(0.0)}
    captures = {"layer.0": [_t(0.0), _t(0.02), _t(0.05)]}

    tols = derive_tolerances(ref, captures, safety_factor=6.0, n_runs=3)

    assert tols["layer.0"].n_calibration_runs == 3
    assert tols["layer.0"].noise_floor == pytest.approx(0.05)


def test_skips_taps_missing_from_reference() -> None:
    ref = {"layer.0": _t(0.0)}
    captures = {"layer.0": [_t(0.0)], "phantom": [_t(0.0)]}

    tols = derive_tolerances(ref, captures, safety_factor=6.0, n_runs=1)

    assert "layer.0" in tols
    assert "phantom" not in tols


def test_atol_floors_dominate_when_noise_is_below_default() -> None:
    """If safety_factor × noise_floor < DEFAULT_TOLERANCE, atol = DEFAULT_TOLERANCE."""
    ref = {"layer.0": _t(0.0)}
    captures = {"layer.0": [_t(0.0), _t(1e-9)]}

    tols = derive_tolerances(ref, captures, safety_factor=6.0, n_runs=2)

    assert tols["layer.0"].atol == DEFAULT_TOLERANCE
    assert tols["layer.0"].noise_floor == pytest.approx(1e-9)


# --- slow integration: calibrate against real SmolLM -------------------------


@pytest.mark.slow
def test_calibrate_no_noise_yields_default_tolerances(tmp_path: Path) -> None:
    """Pure determinism on CPU+fp32: every tap's noise_floor must be 0,
    and every atol falls to DEFAULT_TOLERANCE."""
    from firefly.calibrate import calibrate
    from firefly.capture import capture_reference

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    ref_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, ref_dir)
    tols = calibrate(ref_dir, inputs_path, runs=2)

    for t in tols.values():
        assert t.source == "calibrated"
        assert t.noise_floor == 0.0
        assert t.atol == DEFAULT_TOLERANCE
        assert t.n_calibration_runs == 2

    assert (ref_dir / "tolerances.json").exists()


@pytest.mark.slow
def test_calibrate_with_synthetic_noise_propagates_with_depth(tmp_path: Path) -> None:
    """Inject noise at layer.0; upstream taps should still have noise_floor=0,
    layer.0 + downstream should have noise_floor > 0 and growing roughly with depth.
    """
    from firefly.calibrate import calibrate
    from firefly.capture import capture_reference
    from firefly.noise import NoiseSpec

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    ref_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, ref_dir)

    noise = NoiseSpec(
        mode="synthetic",
        sigma=1e-3,
        inject_at="layer.0",
        base_seed=0,
    )
    tols = calibrate(ref_dir, inputs_path, runs=4, safety_factor=6.0, noise=noise)

    # Upstream of injection point: still zero noise.
    for upstream_tap in ("layer.0.self_attn", "layer.0.mlp"):
        assert tols[upstream_tap].noise_floor == 0.0, (
            f"{upstream_tap} should be unaffected by noise injection at layer.0"
        )

    # Injection point and downstream: nonzero noise that grows with depth.
    assert tols["layer.0"].noise_floor > 0
    assert tols["layer.5"].noise_floor > tols["layer.0"].noise_floor / 2, (
        "downstream noise should be in the same order of magnitude or larger"
    )

    # And the tolerances.json reads back correctly.
    from firefly.compare import read_tolerances

    on_disk = read_tolerances(ref_dir)
    assert on_disk is not None
    assert on_disk["layer.0"].source == "calibrated"
    assert on_disk["layer.0"].atol == tols["layer.0"].atol
