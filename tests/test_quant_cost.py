"""Pure tests for the cost model + Pareto machinery (no model loads)."""

from __future__ import annotations

import pytest

from firefly.quant.cost import (
    BudgetExceededError,
    estimate_measurements,
    format_bytes,
    frontier_and_knee,
    knee_index,
    memory_envelope,
    pareto_indices,
    recipe_memory_bytes,
)

# --- memory cost ---------------------------------------------------------------

def test_recipe_memory_bytes_base_vs_scheme() -> None:
    numels = {"a": 100, "b": 200}  # 300 weights total
    # all fp16 (16 bits): 300 * 16 / 8 = 600 B
    assert recipe_memory_bytes(numels, set(), base_bits=16, scheme="w8a8", group_size=32) == 600
    # all w8a8 (8 bits): 300 * 8 / 8 = 300 B
    allq = recipe_memory_bytes(numels, {"a", "b"}, base_bits=16, scheme="w8a8", group_size=32)
    assert allq == 300
    # mixed: keep 'a' fp (100*16/8=200), quantize 'b' (200*8/8=200) → 400 B
    mixed = recipe_memory_bytes(numels, {"b"}, base_bits=16, scheme="w8a8", group_size=32)
    assert mixed == 400


def test_int4wo_adds_group_scale_overhead() -> None:
    numels = {"a": 1024}
    # 4-bit weights: 1024 * 4 / 8 = 512 B; + group scales: (1024/32)*2 = 64 B → 576 B
    cost = recipe_memory_bytes(numels, {"a"}, base_bits=16, scheme="int4wo", group_size=32)
    assert cost == 512 + 64


def test_memory_envelope_compression() -> None:
    env = memory_envelope({"a": 100, "b": 200}, base_bits=16, scheme="w8a8", group_size=32)
    assert env.all_fp_bytes == 600
    assert env.all_quant_bytes == 300
    assert env.compression(300) == pytest.approx(2.0)
    assert env.compression(0) == 1.0  # guard


# --- pareto --------------------------------------------------------------------

def test_pareto_drops_dominated_lower_is_better() -> None:
    # quality lower-better (divergence). D=(2.5, 0.4) is dominated by B=(2, 0.3):
    # cheaper-or-equal AND better-or-equal, strict on both.
    costs = [1.0, 2.0, 3.0, 2.5]
    quals = [0.5, 0.3, 0.1, 0.4]
    idx = pareto_indices(costs, quals, quality_higher_is_better=False)
    assert idx == [0, 1, 2]  # sorted by cost; D (index 3) dropped


def test_pareto_higher_is_better() -> None:
    # quality higher-better (accuracy). low-cost low-acc, high-cost high-acc are
    # both frontier; a high-cost low-acc point is dominated.
    costs = [1.0, 3.0, 3.0]
    quals = [0.8, 0.95, 0.80]  # index 2: expensive AND worse than index 0
    idx = pareto_indices(costs, quals, quality_higher_is_better=True)
    assert set(idx) == {0, 1}


def test_pareto_keeps_ties() -> None:
    costs = [1.0, 1.0]
    quals = [0.5, 0.5]
    assert pareto_indices(costs, quals, quality_higher_is_better=False) == [0, 1]


def test_pareto_full_fp_dominated_by_nonmonotonic_quality() -> None:
    # The real SmolLM case: full-fp (most expensive) has WORSE perplexity than a
    # cheaper partial recipe (mild quant regularization) → must be dropped.
    costs = [106.0, 265.0, 425.0]  # k=0, k=15, k=30 (MB)
    ppl = [47.0, 38.6, 41.4]       # lower better; k=30 worse than k=15 AND pricier
    idx = pareto_indices(costs, ppl, quality_higher_is_better=False)
    assert 2 not in idx  # full-fp dominated


# --- knee ----------------------------------------------------------------------

def test_knee_picks_the_elbow() -> None:
    # A sharp elbow at the middle point.
    costs = [0.0, 1.0, 10.0]
    quals = [10.0, 1.0, 0.0]  # big quality gain cheap, then flat
    assert knee_index(costs, quals) == 1


def test_knee_degenerate_small_inputs() -> None:
    assert knee_index([], []) == -1
    assert knee_index([1.0], [1.0]) == 0
    assert knee_index([1.0, 2.0], [1.0, 2.0]) == 1


def test_frontier_and_knee_roundtrip() -> None:
    costs = [1.0, 2.0, 3.0, 2.5]
    quals = [0.5, 0.3, 0.1, 0.4]
    frontier, knee = frontier_and_knee(costs, quals, quality_higher_is_better=False)
    assert frontier == [0, 1, 2]
    assert knee in frontier


# --- measurement budget --------------------------------------------------------

def test_estimate_isolated_and_bar() -> None:
    # ranking = all-quant baseline (1) + one per unit (30) = 31, then the candidates
    assert estimate_measurements(30, "isolated", [1, 2, 4, 8]) == 31 + 4
    # bar: ranking (31) + 2 anchors + binary search (ceil(log2 30)=5)
    assert estimate_measurements(30, "isolated", [], bar=True) == 31 + 7


def test_estimate_greedy_is_quadratic_ish() -> None:
    # 1 + K*n - K(K-1)/2 for K=max(k)
    assert estimate_measurements(30, "greedy", [1, 2, 4, 8]) == 1 + 8 * 30 - 8 * 7 // 2


def test_guard_budget_raises_when_over() -> None:
    from firefly.quant.sensitivity import _guard_budget

    # linear-granularity greedy on 210 units blows any small budget.
    with pytest.raises(BudgetExceededError) as ei:
        _guard_budget(210, "greedy", [8], 50, bar=False)
    assert ei.value.estimate > 50
    assert ei.value.budget == 50


def test_guard_budget_noop_when_under_or_off() -> None:
    from firefly.quant.sensitivity import _guard_budget

    _guard_budget(30, "isolated", [1, 2, 4, 8], None, bar=False)  # off
    _guard_budget(30, "isolated", [1, 2, 4, 8], 1000, bar=False)  # under


def test_format_bytes() -> None:
    assert format_bytes(600) == "600 B"
    assert format_bytes(1500) == "1.5 KB"
    assert format_bytes(1.5e6) == "1.5 MB"
    assert format_bytes(1.5e9) == "1.5 GB"
