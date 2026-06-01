"""Unit tests for first-divergence attribution."""

from __future__ import annotations

from firefly.attribution import attribute_first_divergence
from firefly.compare import TapDivergence


def _div(name: str, exceeds: bool, max_d: float = 0.0) -> TapDivergence:
    return TapDivergence(
        tap_name=name,
        max_abs_diff=max_d,
        mean_abs_diff=max_d,
        tolerance=1e-5,
        exceeds_tolerance=exceeds,
    )


def test_returns_none_when_nothing_exceeds() -> None:
    divs = [_div("layer.0", False), _div("layer.1", False)]
    result = attribute_first_divergence(divs)

    assert result.first_divergent_tap is None
    assert result.any_exceeded is False


def test_picks_first_in_forward_order_not_largest() -> None:
    """Attribution must respect the forward-order semantics: the first
    breach is what matters, not the largest. A divergence at layer 7 that
    propagates to a larger one at layer 30 should still attribute to layer 7.
    """
    divs = [
        _div("layer.0", False, 0.0),
        _div("layer.7", True, 1e-3),     # first breach, smaller magnitude
        _div("layer.30", True, 1.0),     # downstream amplification, larger
    ]
    result = attribute_first_divergence(divs)

    assert result.first_divergent_tap == "layer.7"
    assert result.any_exceeded is True


def test_preserves_underlying_divergences() -> None:
    divs = [_div("a", False), _div("b", True)]
    result = attribute_first_divergence(divs)

    assert result.divergences == divs
