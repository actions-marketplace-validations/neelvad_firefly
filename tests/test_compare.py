"""Tests for the per-tap diff and the compare-to-reference orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from firefly.compare import DEFAULT_TOLERANCE, TapTolerance, diff_captures


def _t(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def test_diff_captures_zero_when_identical() -> None:
    a = {"x": _t([1.0, 2.0, 3.0]), "y": _t([4.0, 5.0])}
    b = {"x": _t([1.0, 2.0, 3.0]), "y": _t([4.0, 5.0])}

    divs = diff_captures(a, b, ["x", "y"])

    assert [d.max_abs_diff for d in divs] == [0.0, 0.0]
    assert all(not d.exceeds_tolerance for d in divs)


def test_diff_captures_flags_only_above_tolerance() -> None:
    a = {"x": _t([1.0]), "y": _t([1.0])}
    b = {"x": _t([1.0 + 1e-7]), "y": _t([1.0 + 1e-3])}

    divs = diff_captures(a, b, ["x", "y"])

    assert not divs[0].exceeds_tolerance  # 1e-7 < 1e-5 default
    assert divs[1].exceeds_tolerance      # 1e-3 > 1e-5 default


def test_diff_captures_respects_per_tap_tolerance() -> None:
    a = {"x": _t([1.0])}
    b = {"x": _t([1.0 + 1e-3])}

    relaxed = diff_captures(a, b, ["x"], tolerances={"x": TapTolerance(atol=1e-2)})
    strict = diff_captures(a, b, ["x"], tolerances={"x": TapTolerance(atol=1e-5)})

    assert not relaxed[0].exceeds_tolerance
    assert strict[0].exceeds_tolerance


def test_diff_captures_max_rel_error_loosens_tight_calibration() -> None:
    """A tap with a tight calibrated atol passes under a max_rel_error ceiling
    if its absolute diff is small relative to the reference magnitude."""
    a = {"x": _t([10.0, 10.0, 10.0])}
    b = {"x": _t([10.05, 10.05, 10.05])}  # diff = 0.05, ref_max = 10

    # Calibrated atol is much tighter than the diff
    tight = {"x": TapTolerance(atol=1e-5)}

    no_ceiling = diff_captures(a, b, ["x"], tolerances=tight)
    assert no_ceiling[0].exceeds_tolerance  # 0.05 > 1e-5
    assert no_ceiling[0].effective_atol == 1e-5

    # 1% of ref_max = 0.1, which is > 0.05 → tap should now pass
    with_ceiling = diff_captures(a, b, ["x"], tolerances=tight, max_rel_error=0.01)
    assert not with_ceiling[0].exceeds_tolerance
    assert with_ceiling[0].effective_atol == 0.1


def test_diff_captures_max_rel_error_doesnt_loosen_when_calibration_is_already_looser() -> None:
    """If calibrated atol is already greater than max_rel_error × max|ref|,
    the ceiling is a no-op (we always take the looser of the two)."""
    a = {"x": _t([1.0])}
    b = {"x": _t([1.5])}

    # Calibrated atol (10.0) is way looser than 0.01 × 1.0 = 0.01
    loose = {"x": TapTolerance(atol=10.0)}

    divs = diff_captures(a, b, ["x"], tolerances=loose, max_rel_error=0.01)
    assert not divs[0].exceeds_tolerance
    assert divs[0].effective_atol == 10.0  # calibrated wins


def test_diff_captures_records_tolerance_source() -> None:
    a = {"x": _t([1.0])}
    b = {"x": _t([1.0])}

    explicit = diff_captures(a, b, ["x"], tolerances={"x": TapTolerance(atol=1e-2, source="manual")})
    implicit = diff_captures(a, b, ["x"])

    assert explicit[0].tolerance.source == "manual"
    assert implicit[0].tolerance.source == "default"


def test_diff_captures_preserves_forward_order() -> None:
    a = {"layer.0": _t([0.0]), "layer.1": _t([0.0]), "layer.2": _t([0.0])}
    b = {"layer.0": _t([0.0]), "layer.1": _t([1.0]), "layer.2": _t([0.0])}

    divs = diff_captures(a, b, ["layer.0", "layer.1", "layer.2"])

    assert [d.tap_name for d in divs] == ["layer.0", "layer.1", "layer.2"]


def test_diff_captures_raises_on_shape_mismatch() -> None:
    a = {"x": torch.zeros(3)}
    b = {"x": torch.zeros(4)}

    with pytest.raises(ValueError, match="Shape mismatch"):
        diff_captures(a, b, ["x"])


def test_diff_captures_raises_on_missing_tap() -> None:
    a = {"x": _t([0.0])}
    b: dict[str, torch.Tensor] = {}

    with pytest.raises(ValueError, match="Candidate is missing tap"):
        diff_captures(a, b, ["x"])


def test_default_tolerance_is_documented_value() -> None:
    """Pin the default so a silent change to it triggers a test failure."""
    assert DEFAULT_TOLERANCE == 1e-5


@pytest.mark.slow
def test_compare_to_reference_auto_loads_tolerances(tmp_path: Path) -> None:
    """If tolerances.json exists in the reference dir and the caller passes no
    tolerances explicitly, compare_to_reference uses them instead of defaults."""
    from firefly.capture import capture_reference
    from firefly.compare import (
        TapTolerance,
        compare_to_reference,
        write_tolerances,
    )

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    ref_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, ref_dir)

    # Calibrated atol on one tap; everything else gets default.
    write_tolerances(
        ref_dir,
        {"layer.0": TapTolerance(atol=2.5e-5, source="calibrated", noise_floor=1.1e-7, n_calibration_runs=4)},
    )

    divs = compare_to_reference(ref_dir, "HuggingFaceTB/SmolLM-135M", inputs_path)
    by_tap = {d.tap_name: d for d in divs}

    assert by_tap["layer.0"].tolerance.atol == 2.5e-5
    assert by_tap["layer.0"].tolerance.source == "calibrated"
    # Untuned taps still get the default.
    assert by_tap["layer.1"].tolerance.source == "default"


@pytest.mark.slow
def test_compare_same_model_is_clean(tmp_path: Path) -> None:
    """Capturing then comparing the same model yields zero divergence."""
    from firefly.capture import capture_reference
    from firefly.compare import compare_to_reference

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    ref_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, ref_dir)
    divs = compare_to_reference(ref_dir, "HuggingFaceTB/SmolLM-135M", inputs_path)

    assert all(not d.exceeds_tolerance for d in divs)
    assert max(d.max_abs_diff for d in divs) == 0.0


@pytest.mark.slow
def test_compare_raises_on_fingerprint_mismatch(tmp_path: Path) -> None:
    """Hand-edit the manifest's fingerprint to simulate a model swap;
    compare_to_reference should refuse to proceed."""
    from firefly.capture import capture_reference
    from firefly.compare import FingerprintMismatchError, compare_to_reference

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    ref_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, ref_dir)

    manifest_path = ref_dir / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    manifest["model_fingerprint"] = "0000000000000000"
    with manifest_path.open("w") as f:
        json.dump(manifest, f)

    with pytest.raises(FingerprintMismatchError, match="Candidate fingerprint differs"):
        compare_to_reference(ref_dir, "HuggingFaceTB/SmolLM-135M", inputs_path)


@pytest.mark.slow
def test_compare_allow_fingerprint_mismatch_proceeds(tmp_path: Path) -> None:
    """With the escape hatch set, compare proceeds even when fingerprints differ."""
    from firefly.capture import capture_reference
    from firefly.compare import compare_to_reference

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    ref_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, ref_dir)

    manifest_path = ref_dir / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    manifest["model_fingerprint"] = "0000000000000000"
    with manifest_path.open("w") as f:
        json.dump(manifest, f)

    divs = compare_to_reference(
        ref_dir,
        "HuggingFaceTB/SmolLM-135M",
        inputs_path,
        allow_fingerprint_mismatch=True,
    )
    assert max(d.max_abs_diff for d in divs) == 0.0


def test_check_loads_candidate_at_reference_dtype() -> None:
    """compare_to_reference must load the candidate at the reference's recorded
    dtype by default — otherwise a bf16 reference vs an fp32-loaded candidate
    reports the dtype gap as divergence (the #2 correctness bug)."""
    from unittest.mock import MagicMock, patch

    from firefly.compare import compare_to_reference
    from firefly.reference import ReferenceManifest

    manifest = ReferenceManifest(
        model_id="fake",
        model_fingerprint="fp",
        tap_points=[],
        shapes={},
        dtypes={},
        captured_at="2026-06-15T00:00:00+00:00",
        dtype="bfloat16",
    )

    with (
        patch("firefly.compare.read_reference", return_value=(manifest, {})),
        patch("firefly.compare.read_tolerances", return_value=None),
        patch("firefly.compare.set_deterministic"),
        patch("firefly.compare.load_golden_inputs", return_value={}),
        patch("firefly.compare.run_capture", return_value={}),
        patch("firefly.compare.fingerprint_model", return_value="fp"),
        patch(
            "firefly.compare.load_model_and_tokenizer",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_load,
    ):
        compare_to_reference(Path("ref"), "cand", Path("inputs.json"))

    assert mock_load.call_args.kwargs["dtype"] is torch.bfloat16


def test_check_candidate_dtype_override() -> None:
    """--candidate-dtype overrides the reference dtype for cross-dtype runs."""
    from unittest.mock import MagicMock, patch

    from firefly.compare import compare_to_reference
    from firefly.reference import ReferenceManifest

    manifest = ReferenceManifest(
        model_id="fake",
        model_fingerprint="fp",
        tap_points=[],
        shapes={},
        dtypes={},
        captured_at="2026-06-15T00:00:00+00:00",
        dtype="bfloat16",
    )

    with (
        patch("firefly.compare.read_reference", return_value=(manifest, {})),
        patch("firefly.compare.read_tolerances", return_value=None),
        patch("firefly.compare.set_deterministic"),
        patch("firefly.compare.load_golden_inputs", return_value={}),
        patch("firefly.compare.run_capture", return_value={}),
        patch("firefly.compare.fingerprint_model", return_value="fp"),
        patch(
            "firefly.compare.load_model_and_tokenizer",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_load,
    ):
        compare_to_reference(
            Path("ref"), "cand", Path("inputs.json"), candidate_dtype="fp16"
        )

    assert mock_load.call_args.kwargs["dtype"] is torch.float16
