"""Tests for the per-tap diff and the compare-to-reference orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from firefly.compare import DEFAULT_TOLERANCE, diff_captures


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

    relaxed = diff_captures(a, b, ["x"], tolerances={"x": 1e-2})
    strict = diff_captures(a, b, ["x"], tolerances={"x": 1e-5})

    assert not relaxed[0].exceeds_tolerance
    assert strict[0].exceeds_tolerance


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
