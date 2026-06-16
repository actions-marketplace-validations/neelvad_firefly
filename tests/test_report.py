"""Tests for the human + JSON report renderers."""

from __future__ import annotations

import json
from pathlib import Path

from firefly.attribution import AttributionResult
from firefly.compare import TapDivergence, TapTolerance
from firefly.report import (
    render_human,
    render_markdown,
    render_quant_diff,
    write_json,
)


def _div(name: str, max_d: float, exceeds: bool) -> TapDivergence:
    return TapDivergence(
        tap_name=name,
        max_abs_diff=max_d,
        mean_abs_diff=max_d / 10,
        tolerance=TapTolerance(atol=1e-5),
        exceeds_tolerance=exceeds,
    )


def _qdiv(name: str, mean_d: float, ref_mean: float) -> TapDivergence:
    return TapDivergence(
        tap_name=name,
        max_abs_diff=mean_d * 2,
        mean_abs_diff=mean_d,
        tolerance=TapTolerance(atol=1e-5),
        exceeds_tolerance=True,
        ref_max_abs=ref_mean * 2,
        ref_mean_abs=ref_mean,
    )


def test_render_quant_diff_ranks_by_relative_not_absolute() -> None:
    # layer.28.mlp has the largest ABSOLUTE diff (20.0) but a tiny relative one
    # (0.2%, because its activations are huge); layer.5.mlp is the real worst at
    # 20% relative. The ranking must surface layer.5.mlp, not the abs outlier.
    result = AttributionResult(
        first_divergent_tap=None,
        any_exceeded=True,
        divergences=[
            _qdiv("layer.5.mlp", 2.0, 10.0),      # 20% rel
            _qdiv("layer.28.mlp", 20.0, 1.0e4),   # 0.2% rel, but biggest abs
            _qdiv("final_norm", 0.5, 50.0),       # 1% rel, forward-order last
        ],
    )
    out = render_quant_diff(result, scheme="int4wo", rel_threshold=0.05)

    assert "int4wo" in out
    assert "worst layer: layer.5.mlp" in out
    # Accumulated-at-output uses the forward-order-last tap, not the worst.
    assert "final_norm" in out
    # Exactly one tap (layer.5.mlp at 20%) exceeds the 5% threshold.
    assert "1 tap(s) exceed 5.0%" in out


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


def test_render_markdown_clean_has_pass_headline() -> None:
    text = render_markdown(_clean_result())
    assert "✅" in text
    assert "no divergence" in text.lower()


def test_render_markdown_diverged_names_first_tap_in_headline() -> None:
    text = render_markdown(_diverged_result())
    assert "❌" in text
    assert "`layer.1.mlp`" in text
    assert "2 of 3" in text  # 2 of 3 taps exceeded


def test_render_markdown_omits_passing_taps_from_table() -> None:
    text = render_markdown(_diverged_result())
    # only divergent taps go in the markdown table; layer.0 (clean) should not
    assert "`layer.1.mlp`" in text
    assert "`layer.2`" in text
    # The clean tap should not appear as a table row
    assert "| `layer.0` |" not in text


def test_render_markdown_truncates_long_diverged_lists() -> None:
    divs = [_div(f"layer.{i}", 1.0, True) for i in range(15)]
    result = AttributionResult(
        first_divergent_tap="layer.0",
        any_exceeded=True,
        divergences=divs,
    )
    text = render_markdown(result, max_rows=5)
    assert "and 10 more" in text


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
