"""Tests for the human + JSON report renderers."""

from __future__ import annotations

import json
from pathlib import Path

from firefly.attribution import AttributionResult
from firefly.compare import TapDivergence, TapTolerance
from firefly.report import render_human, write_json


def _div(name: str, max_d: float, exceeds: bool) -> TapDivergence:
    return TapDivergence(
        tap_name=name,
        max_abs_diff=max_d,
        mean_abs_diff=max_d / 10,
        tolerance=TapTolerance(atol=1e-5),
        exceeds_tolerance=exceeds,
    )


def _clean_result() -> AttributionResult:
    return AttributionResult(
        first_divergent_tap=None,
        any_exceeded=False,
        divergences=[_div("layer.0", 0.0, False), _div("layer.1", 0.0, False)],
    )


def _diverged_result() -> AttributionResult:
    return AttributionResult(
        first_divergent_tap="layer.1.mlp",
        any_exceeded=True,
        divergences=[
            _div("layer.0", 0.0, False),
            _div("layer.1.mlp", 1.2e-3, True),
            _div("layer.2", 5.0e-3, True),
        ],
    )


def test_render_human_clean_says_no_divergence() -> None:
    text = render_human(_clean_result())
    assert "No divergence detected" in text
    assert "layer.0" in text


def test_render_human_diverged_names_first_tap() -> None:
    text = render_human(_diverged_result())
    assert "First divergence" in text
    assert "layer.1.mlp" in text


def test_render_human_includes_all_taps() -> None:
    text = render_human(_diverged_result())
    for name in ("layer.0", "layer.1.mlp", "layer.2"):
        assert name in text


def test_write_json_payload_shape(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    write_json(_diverged_result(), out)

    with out.open() as f:
        payload = json.load(f)

    assert payload["first_divergent_tap"] == "layer.1.mlp"
    assert payload["any_exceeded"] is True
    assert len(payload["divergences"]) == 3
    assert payload["divergences"][1]["tap_name"] == "layer.1.mlp"
    assert payload["divergences"][1]["exceeds_tolerance"] is True
    # TapTolerance serializes as a nested object, not a bare float.
    assert payload["divergences"][1]["tolerance"]["atol"] == 1e-5
    assert payload["divergences"][1]["tolerance"]["source"] == "default"


def test_write_json_clean_run(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    write_json(_clean_result(), out)

    with out.open() as f:
        payload = json.load(f)

    assert payload["first_divergent_tap"] is None
    assert payload["any_exceeded"] is False
