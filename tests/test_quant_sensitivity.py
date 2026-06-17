"""Per-layer quantization sensitivity (phase 2 P0).

Fast tests cover the pure parts (layer discovery, the strategy seam, the
renderer). The slow test runs the real isolated sweep on SmolLM-135M (needs
torchao); W8A8 runs on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch.nn as nn

from firefly.quant_sensitivity import (
    ISOLATED,
    STRATEGIES,
    LayerSensitivity,
    RecipePoint,
    RecipeResult,
    SensitivityResult,
    _recommend_k,
    _recovery,
    discover_layers,
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


def test_discover_layers_groups_and_excludes_non_layer() -> None:
    layers = discover_layers(_Tiny(3))
    assert sorted(layers) == [0, 1, 2]
    assert layers[0] == ["layers.0.attn", "layers.0.mlp"]
    # lm_head has no layer index → excluded.
    assert all("lm_head" not in fqn for fqns in layers.values() for fqn in fqns)


def test_isolated_strategy_targets_and_score() -> None:
    layer = {"layers.1.attn", "layers.1.mlp"}
    all_fqns = layer | {"layers.0.attn", "layers.2.mlp"}
    # isolated quantizes only this layer ...
    assert ISOLATED.targets(layer, all_fqns) == layer
    # ... and scores by the measured divergence directly (full-quant ignored).
    assert ISOLATED.score(0.42, 0.99) == 0.42
    assert "isolated" in STRATEGIES


def test_sensitivity_result_ranks_and_suggests() -> None:
    result = SensitivityResult(
        model_id="m", scheme="int4wo", strategy="isolated", full_quant_divergence=0.7,
        layers=[
            LayerSensitivity(layer=0, sensitivity=0.05, raw_divergence=0.05, n_linears=7),
            LayerSensitivity(layer=5, sensitivity=0.30, raw_divergence=0.30, n_linears=7),
            LayerSensitivity(layer=2, sensitivity=0.12, raw_divergence=0.12, n_linears=7),
        ],
    )
    assert [x.layer for x in result.ranked] == [5, 2, 0]
    assert result.keep_high_precision(2) == [5, 2]


def test_render_sensitivity_headline_and_ranking() -> None:
    from firefly.report import render_sensitivity

    result = SensitivityResult(
        model_id="m", scheme="w8a8", strategy="isolated", full_quant_divergence=0.66,
        layers=[
            LayerSensitivity(layer=29, sensitivity=0.40, raw_divergence=0.40, n_linears=7),
            LayerSensitivity(layer=1, sensitivity=0.02, raw_divergence=0.02, n_linears=7),
        ],
    )
    out = render_sensitivity(result, keep_k=1)
    assert "layer.29" in out
    assert "66.00%" in out  # full-quant output divergence headline
    assert "keep-in-high-precision (top 1): layer.29" in out


def test_recovery_fraction() -> None:
    assert _recovery(0.66, 0.10) == pytest.approx((0.66 - 0.10) / 0.66)
    assert _recovery(0.66, 0.66) == 0.0
    assert _recovery(0.0, 0.0) == 1.0  # no degradation to recover
    assert _recovery(0.5, 0.8) == 0.0  # worse than full-quant clamps to 0


def test_recommend_k_smallest_meeting_target() -> None:
    curve = [
        RecipePoint(k=1, kept_layers=[28], output_divergence=0.20, recovery=0.70),
        RecipePoint(k=4, kept_layers=[28, 29, 11, 27], output_divergence=0.07, recovery=0.92),
        RecipePoint(k=8, kept_layers=list(range(8)), output_divergence=0.03, recovery=0.96),
    ]
    assert _recommend_k(curve, target=0.9) == 4   # smallest clearing 90%
    assert _recommend_k(curve, target=0.5) == 1
    assert _recommend_k(curve, target=0.99) == 8  # none clear it → largest k


def test_render_recipe_curve_and_recommendation() -> None:
    from firefly.report import render_recipe

    sens = SensitivityResult(
        model_id="m", scheme="int4wo", strategy="isolated", full_quant_divergence=0.66,
        layers=[LayerSensitivity(layer=28, sensitivity=0.58, raw_divergence=0.58, n_linears=7)],
    )
    result = RecipeResult(
        sensitivity=sens,
        curve=[
            RecipePoint(k=1, kept_layers=[28], output_divergence=0.20, recovery=0.70),
            RecipePoint(k=4, kept_layers=[28, 29, 11, 27], output_divergence=0.07, recovery=0.92),
        ],
        recommended_k=4,
        recovery_target=0.9,
    )
    out = render_recipe(result)
    assert "66.00%" in out  # full-quant headline
    assert "Mixed-precision recipe curve" in out
    assert "recommended: keep 4 layers" in out
    assert "92%" in out


@pytest.mark.slow
def test_compute_recipe_smollm_recovers_fidelity() -> None:
    pytest.importorskip("torchao", reason="quant recipe needs the torchao extra")
    import tempfile

    from firefly.quant_sensitivity import compute_recipe

    inputs = Path(tempfile.mkdtemp()) / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox"], "max_length": 12}))

    result = compute_recipe(
        "HuggingFaceTB/SmolLM-135M", inputs, device="cpu", scheme="w8a8",
        k_values=[1, 2, 4, 8],
    )

    assert len(result.curve) == 4
    # Keeping any layers high-precision recovers some fidelity vs all-quantized.
    assert all(0.0 <= p.recovery <= 1.0 for p in result.curve)
    by_k = sorted(result.curve, key=lambda p: p.k)
    # More high-precision layers ⇒ at least as much recovery (allow fp noise).
    assert by_k[-1].recovery >= by_k[0].recovery - 1e-6
    # The strong isolated signal (layer.28) means even k=1 recovers meaningfully.
    assert by_k[0].recovery > 0.1
    assert result.recommended_k in {1, 2, 4, 8}


@pytest.mark.slow
def test_compute_sensitivity_smollm_isolated() -> None:
    pytest.importorskip("torchao", reason="quant sensitivity needs the torchao extra")
    import tempfile

    from firefly.quant_sensitivity import compute_sensitivity

    inputs = Path(tempfile.mkdtemp()) / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox"], "max_length": 12}))

    result = compute_sensitivity(
        "HuggingFaceTB/SmolLM-135M", inputs, device="cpu", scheme="w8a8", strategy="isolated"
    )

    assert len(result.layers) == 30  # SmolLM-135M decoder layers
    assert result.full_quant_divergence > 0.0
    # Every isolated measurement is a non-negative output divergence, and at
    # least one layer is meaningfully sensitive.
    assert all(x.sensitivity >= 0.0 for x in result.layers)
    assert max(x.sensitivity for x in result.layers) > 0.0
    # Ranking is sorted; the suggested keep-set is the top of it.
    ranked = result.ranked
    assert ranked == sorted(result.layers, key=lambda x: x.sensitivity, reverse=True)
    assert result.keep_high_precision(3) == [x.layer for x in ranked[:3]]
