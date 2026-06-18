"""Per-unit quantization sensitivity (phase 2).

Fast tests cover the pure parts (unit discovery + granularity, the strategy
seam, greedy selection, the renderers). Slow tests run the real sweeps on
SmolLM-135M (need torchao); W8A8 runs on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch.nn as nn

from firefly.quant.sensitivity import (
    GRANULARITIES,
    ISOLATED,
    MARGINAL,
    STRATEGIES,
    RecipePoint,
    RecipeResult,
    SensitivityResult,
    UnitSensitivity,
    _recommend_k,
    _recovery,
    discover_units,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)


class _Tiny(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Block() for _ in range(n))
        self.lm_head = nn.Linear(8, 8)  # non-layer Linear, must be excluded


def test_discover_units_layer_granularity() -> None:
    units = discover_units(_Tiny(3), "layer")
    assert list(units) == ["layer.0", "layer.1", "layer.2"]
    assert units["layer.0"] == ["layers.0.attn", "layers.0.mlp"]
    assert all("lm_head" not in fqn for fqns in units.values() for fqn in fqns)


def test_discover_units_linear_granularity() -> None:
    units = discover_units(_Tiny(2), "linear")
    # Each Linear is its own unit, FQN as the name; ~2 layers * 2 Linears.
    assert list(units) == ["layers.0.attn", "layers.0.mlp", "layers.1.attn", "layers.1.mlp"]
    assert units["layers.0.attn"] == ["layers.0.attn"]
    # Same Linear set as layer granularity → same all-quantized baseline.
    layer_fqns = {f for fqns in discover_units(_Tiny(2), "layer").values() for f in fqns}
    assert set(units) == layer_fqns


def test_discover_units_rejects_unknown_granularity() -> None:
    assert set(GRANULARITIES) == {"layer", "linear"}
    with pytest.raises(ValueError, match="unknown granularity"):
        discover_units(_Tiny(1), "bogus")


def test_isolated_strategy_targets_and_score() -> None:
    unit = {"layers.1.attn", "layers.1.mlp"}
    all_fqns = unit | {"layers.0.attn", "layers.2.mlp"}
    assert ISOLATED.targets(unit, all_fqns) == unit
    assert ISOLATED.score(0.42, 0.99) == 0.42
    assert "isolated" in STRATEGIES


def test_marginal_strategy_targets_and_score() -> None:
    unit = {"layers.1.attn", "layers.1.mlp"}
    all_fqns = unit | {"layers.0.attn", "layers.2.mlp"}
    assert MARGINAL.targets(unit, all_fqns) == all_fqns - unit
    assert MARGINAL.score(0.40, 0.66) == pytest.approx(0.26)
    assert MARGINAL.score(0.70, 0.66) == 0.0  # recovery clamps at 0
    assert "marginal" in STRATEGIES


def _us(unit: str, s: float) -> UnitSensitivity:
    return UnitSensitivity(unit=unit, sensitivity=s, raw_divergence=s, n_linears=7)


def test_sensitivity_result_ranks_and_suggests() -> None:
    result = SensitivityResult(
        model_id="m", scheme="int4wo", strategy="isolated", granularity="layer",
        full_quant_divergence=0.7,
        units=[_us("layer.0", 0.05), _us("layer.5", 0.30), _us("layer.2", 0.12)],
    )
    assert [x.unit for x in result.ranked] == ["layer.5", "layer.2", "layer.0"]
    assert result.keep_high_precision(2) == ["layer.5", "layer.2"]


def test_render_sensitivity_headline_and_ranking() -> None:
    from firefly.report import render_sensitivity

    result = SensitivityResult(
        model_id="m", scheme="w8a8", strategy="isolated", granularity="layer",
        full_quant_divergence=0.66,
        units=[_us("layer.29", 0.40), _us("layer.1", 0.02)],
    )
    out = render_sensitivity(result, keep_k=1)
    assert "layer.29" in out
    assert "66.00%" in out
    assert "keep-in-high-precision (top 1): layer.29" in out


def test_render_sensitivity_linear_granularity_wording() -> None:
    from firefly.report import render_sensitivity

    result = SensitivityResult(
        model_id="m", scheme="w8a8", strategy="isolated", granularity="linear",
        full_quant_divergence=0.5,
        units=[_us("model.layers.28.mlp.down_proj", 0.3)],
    )
    out = render_sensitivity(result, keep_k=1)
    assert "model.layers.28.mlp.down_proj" in out
    assert "Linears quantized" in out  # granularity-aware noun


def test_recovery_fraction() -> None:
    assert _recovery(0.66, 0.10) == pytest.approx((0.66 - 0.10) / 0.66)
    assert _recovery(0.66, 0.66) == 0.0
    assert _recovery(0.0, 0.0) == 1.0
    assert _recovery(0.5, 0.8) == 0.0


def test_recommend_k_smallest_meeting_target() -> None:
    curve = [
        RecipePoint(k=1, kept_units=["layer.28"], output_divergence=0.20, recovery=0.70),
        RecipePoint(k=4, kept_units=["layer.28", "layer.29"], output_divergence=0.07, recovery=0.92),
        RecipePoint(k=8, kept_units=[f"layer.{i}" for i in range(8)], output_divergence=0.03, recovery=0.96),
    ]
    assert _recommend_k(curve, target=0.9) == 4
    assert _recommend_k(curve, target=0.5) == 1
    assert _recommend_k(curve, target=0.99) == 8


def test_render_recipe_curve_and_recommendation() -> None:
    from firefly.report import render_recipe

    sens = SensitivityResult(
        model_id="m", scheme="int4wo", strategy="isolated", granularity="layer",
        full_quant_divergence=0.66, units=[_us("layer.28", 0.58)],
    )
    result = RecipeResult(
        sensitivity=sens,
        curve=[
            RecipePoint(k=1, kept_units=["layer.28"], output_divergence=0.20, recovery=0.70),
            RecipePoint(k=4, kept_units=["layer.28", "layer.29", "layer.11", "layer.27"],
                        output_divergence=0.07, recovery=0.92),
        ],
        recommended_k=4, recovery_target=0.9,
    )
    out = render_recipe(result)
    assert "66.00%" in out
    assert "Mixed-precision recipe curve" in out
    assert "recommended: keep 4 layers" in out
    assert "92%" in out


def test_greedy_select_picks_highest_impact_first() -> None:
    from firefly.quant.sensitivity import _greedy_select

    units = {"a": ["l0"], "b": ["l1"], "c": ["l2"]}
    all_fqns = {"l0", "l1", "l2"}
    # Synthetic oracle: divergence = sum of penalties for the QUANTIZED fqns.
    penalty = {"l0": 0.1, "l1": 0.2, "l2": 0.5}

    def measure(targets: set[str]) -> float:
        return sum(penalty[f] for f in targets)

    order = _greedy_select(units, all_fqns, measure, max_k=3)
    # Greedy keeps the unit whose fp-keeping helps most first (c → b → a).
    assert [name for name, _ in order] == ["c", "b", "a"]
    divs = [d for _, d in order]
    assert divs == sorted(divs, reverse=True)
    assert divs[-1] == pytest.approx(0.0)


def test_greedy_is_a_recipe_strategy_not_a_score_strategy() -> None:
    from firefly.quant.sensitivity import GREEDY, RECIPE_STRATEGIES

    assert GREEDY in RECIPE_STRATEGIES
    assert GREEDY not in STRATEGIES


@pytest.mark.slow
def test_compute_recipe_smollm_recovers_fidelity() -> None:
    pytest.importorskip("torchao", reason="quant recipe needs the torchao extra")
    import tempfile

    from firefly.quant.sensitivity import compute_recipe

    inputs = Path(tempfile.mkdtemp()) / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox"], "max_length": 12}))

    result = compute_recipe(
        "HuggingFaceTB/SmolLM-135M", inputs, device="cpu", scheme="w8a8", k_values=[1, 2, 4, 8]
    )
    assert len(result.curve) == 4
    assert all(0.0 <= p.recovery <= 1.0 for p in result.curve)
    by_k = sorted(result.curve, key=lambda p: p.k)
    assert by_k[-1].recovery >= by_k[0].recovery - 1e-6
    assert by_k[0].recovery > 0.1
    assert result.recommended_k in {1, 2, 4, 8}


@pytest.mark.slow
def test_compute_sensitivity_smollm_linear_granularity() -> None:
    pytest.importorskip("torchao", reason="quant sensitivity needs the torchao extra")
    import tempfile

    from firefly.quant.sensitivity import compute_sensitivity

    inputs = Path(tempfile.mkdtemp()) / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox"], "max_length": 12}))

    result = compute_sensitivity(
        "HuggingFaceTB/SmolLM-135M", inputs, device="cpu", scheme="w8a8", granularity="linear"
    )
    assert result.granularity == "linear"
    # 30 layers * 7 Linears each = 210 units; each unit is a single Linear.
    assert len(result.units) == 210
    assert all(u.n_linears == 1 for u in result.units)
    assert all(u.unit.startswith("model.layers.") for u in result.units)


@pytest.mark.slow
def test_compute_sensitivity_smollm_isolated() -> None:
    pytest.importorskip("torchao", reason="quant sensitivity needs the torchao extra")
    import tempfile

    from firefly.quant.sensitivity import compute_sensitivity

    inputs = Path(tempfile.mkdtemp()) / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox"], "max_length": 12}))

    result = compute_sensitivity(
        "HuggingFaceTB/SmolLM-135M", inputs, device="cpu", scheme="w8a8", strategy="isolated"
    )
    assert len(result.units) == 30
    assert result.full_quant_divergence > 0.0
    assert max(x.sensitivity for x in result.units) > 0.0
    assert result.keep_high_precision(3) == [x.unit for x in result.ranked[:3]]
