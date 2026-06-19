"""Fast, model-free tests for the eval-loop pieces: accuracy-bar math, the eval
loader, evaluator resolution, the binary search, and the renderer. The
end-to-end optimize_to_bar on a real model is a slow test in
test_quant_sensitivity.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from firefly.quant.bar import BarPoint, BarRecipeResult, _bar_search
from firefly.quant.evaluate import (
    AccuracyBar,
    callable_evaluator,
    load_eval_texts,
    resolve_evaluator,
)

# --- AccuracyBar ---------------------------------------------------------------

def test_accuracy_bar_parse_valid() -> None:
    assert AccuracyBar.parse("rel:0.01") == AccuracyBar("rel", 0.01)
    assert AccuracyBar.parse("abs:0.5") == AccuracyBar("abs", 0.5)


@pytest.mark.parametrize("spec", ["0.01", "pct:0.01", "rel:", "rel:abc", "rel:-0.1"])
def test_accuracy_bar_parse_invalid(spec: str) -> None:
    with pytest.raises(ValueError):
        AccuracyBar.parse(spec)


def test_accuracy_bar_lower_is_better_is_a_ceiling() -> None:
    # perplexity: lower better → bar is a ceiling ABOVE the baseline.
    bar = AccuracyBar("rel", 0.01)
    assert bar.threshold(10.0, higher_is_better=False) == pytest.approx(10.1)
    assert bar.passes(10.05, 10.0, higher_is_better=False)
    assert not bar.passes(10.2, 10.0, higher_is_better=False)


def test_accuracy_bar_higher_is_better_is_a_floor() -> None:
    # accuracy: higher better → bar is a floor BELOW the baseline.
    bar = AccuracyBar("rel", 0.01)
    assert bar.threshold(0.9, higher_is_better=True) == pytest.approx(0.891)
    assert bar.passes(0.895, 0.9, higher_is_better=True)
    assert not bar.passes(0.80, 0.9, higher_is_better=True)


def test_accuracy_bar_abs_mode() -> None:
    bar = AccuracyBar("abs", 0.5)
    assert bar.threshold(10.0, higher_is_better=False) == pytest.approx(10.5)
    assert bar.threshold(0.9, higher_is_better=True) == pytest.approx(0.4)


# --- load_eval_texts -----------------------------------------------------------

def test_load_eval_texts_json_texts_key(tmp_path: Path) -> None:
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"texts": ["a", "b"]}))
    assert load_eval_texts(p) == ["a", "b"]


def test_load_eval_texts_json_bare_list(tmp_path: Path) -> None:
    p = tmp_path / "e.json"
    p.write_text(json.dumps(["a", "b"]))
    assert load_eval_texts(p) == ["a", "b"]


def test_load_eval_texts_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_text('"a"\n{"text": "b"}\n\n"c"\n')
    assert load_eval_texts(p) == ["a", "b", "c"]


def test_load_eval_texts_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"texts": []}))
    with pytest.raises(ValueError, match="non-empty list of strings"):
        load_eval_texts(p)


# --- resolve_evaluator ---------------------------------------------------------

def test_resolve_perplexity_requires_eval_set() -> None:
    with pytest.raises(ValueError, match="requires an --eval"):
        resolve_evaluator("perplexity", None)


def test_resolve_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        resolve_evaluator("bleu", None)


def _good_metric(model, tokenizer) -> float:  # noqa: ARG001 — signature contract
    return 0.42


_good_metric.higher_is_better = False  # type: ignore[attr-defined]
_good_metric.metric_name = "loss"  # type: ignore[attr-defined]


def test_callable_evaluator_resolves_and_reads_attrs() -> None:
    ev = callable_evaluator("tests.test_quant_evaluate:_good_metric")
    assert ev.name == "loss"
    assert ev.higher_is_better is False
    assert ev(None, None) == 0.42


def test_callable_evaluator_defaults_higher_is_better() -> None:
    # A plain function with no attrs is treated as accuracy-style (higher better).
    ev = callable_evaluator("math:isqrt")
    assert ev.higher_is_better is True


@pytest.mark.parametrize("spec", ["nomodule", "json:nonexistent_fn"])
def test_callable_evaluator_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        callable_evaluator(spec)


# --- _bar_search ---------------------------------------------------------------

def test_bar_search_all_pass_returns_zero() -> None:
    assert _bar_search(8, lambda k: True) == 0


def test_bar_search_only_full_precision_passes() -> None:
    assert _bar_search(8, lambda k: k >= 8) == 8


@pytest.mark.parametrize("threshold", [1, 3, 5, 7])
def test_bar_search_finds_monotonic_boundary(threshold: int) -> None:
    calls: list[int] = []

    def passes(k: int) -> bool:
        calls.append(k)
        return k >= threshold

    assert _bar_search(8, passes) == threshold
    assert len(calls) <= 8  # binary search, not a linear scan of 0..8


# --- render_bar_recipe ---------------------------------------------------------

def test_render_bar_recipe_smoke() -> None:
    from firefly.report import render_bar_recipe

    result = BarRecipeResult(
        model_id="m", scheme="w8a8", granularity="layer", strategy="isolated",
        metric_name="perplexity", higher_is_better=False, bar=AccuracyBar("rel", 0.01),
        baseline_metric=10.0, full_quant_metric=14.0, threshold=10.1, n_units=4,
        chosen_k=2, chosen_kept_units=["layer.11", "layer.0"], chosen_metric=10.05,
        evaluated=[
            BarPoint(0, [], 14.0, False),
            BarPoint(2, ["layer.11", "layer.0"], 10.05, True),
            BarPoint(4, ["layer.11", "layer.0", "layer.1", "layer.2"], 10.0, True),
        ],
        evals_used=4,
    )
    out = render_bar_recipe(result)
    assert "perplexity" in out
    assert "keep 2/4" in out
    assert "10.05" in out
