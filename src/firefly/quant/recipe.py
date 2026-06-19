"""Mixed-precision recipe curves: rank units, sweep keep-k, verify recovery.

The recipe layer on top of :mod:`firefly.quant.sensitivity` — it owns the
``RecipeResult`` curve, the ``greedy`` forward-selection search, and the
Pareto/cost annotation. Eval-bar optimization lives next door in
:mod:`firefly.quant.bar`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from firefly.quant.cost import (
    dtype_bits,
    frontier_and_knee,
    linear_numels,
    memory_envelope,
    recipe_memory_bytes,
)
from firefly.quant.sensitivity import (
    GRANULARITIES,
    STRATEGIES,
    SensitivityResult,
    UnitSensitivity,
    _Ctx,
    _guard_budget,
    _run_sensitivity,
    _setup,
    _smoothquant_pre_transforms,
)


@dataclass
class RecipePoint:
    """One mixed-precision recipe: keep the top-``k`` sensitive units in high
    precision, quantize the rest, and the output fidelity it achieves."""

    k: int
    kept_units: list[str]
    output_divergence: float
    recovery: float
    """Fraction of the all-quantized degradation recovered: (full - this) / full."""
    memory_bytes: float = 0.0
    """Weight footprint of the quantizable Linears under this recipe."""


@dataclass
class RecipeResult:
    sensitivity: SensitivityResult
    curve: list[RecipePoint]
    recommended_k: int
    recovery_target: float
    all_fp_bytes: float = 0.0
    all_quant_bytes: float = 0.0
    unit_fqns: dict[str, list[str]] = field(default_factory=dict)
    """unit name → its Linear FQNs — lets recipe export resolve exact quantized
    / kept FQNs without reloading the model."""

    @property
    def recommended_point(self) -> RecipePoint | None:
        return next((p for p in self.curve if p.k == self.recommended_k), None)

    def frontier_knee_ks(self) -> tuple[set[int], int | None]:
        """``(ks on the Pareto frontier, knee k)`` over (memory, divergence) —
        both lower-better, so the frontier minimizes size and divergence."""
        pts = sorted(self.curve, key=lambda p: p.k)
        idx, knee = frontier_and_knee(
            [p.memory_bytes for p in pts],
            [p.output_divergence for p in pts],
            quality_higher_is_better=False,
        )
        return {pts[i].k for i in idx}, (pts[knee].k if knee is not None else None)


def _recovery(full_div: float, recipe_div: float) -> float:
    if full_div <= 0:
        return 1.0
    return max(0.0, (full_div - recipe_div) / full_div)


def _recommend_k(curve: list[RecipePoint], target: float) -> int:
    """Smallest swept k whose recovery clears ``target``; else the largest k."""
    for p in sorted(curve, key=lambda p: p.k):
        if p.recovery >= target:
            return p.k
    return max((p.k for p in curve), default=0)


#: ``greedy`` is a recipe SEARCH, not a per-unit score — it doesn't fit the
#: STRATEGIES (score-based) seam, so it's a recipe-only strategy.
GREEDY = "greedy"
RECIPE_STRATEGIES = (*sorted(STRATEGIES), GREEDY)


def _greedy_select(
    units: dict[str, list[str]],
    all_fqns: set[str],
    measure: Callable[[set[str]], float],
    max_k: int,
) -> list[tuple[str, float]]:
    """Sequential forward selection: repeatedly add to the high-precision set
    the unit that most reduces output divergence *given what's already kept*,
    re-measuring each step. Returns ``(unit, divergence_after_adding)`` in
    selection order. Accounts for interactions, unlike a single-pass ranking."""
    kept: list[str] = []
    kept_fqns: set[str] = set()
    order: list[tuple[str, float]] = []
    while len(kept) < max_k:
        best: tuple[float, str] | None = None  # (divergence, unit)
        for name, fqns in units.items():
            if name in kept:
                continue
            div = measure(all_fqns - (kept_fqns | set(fqns)))  # keep kept+candidate fp
            if best is None or div < best[0]:
                best = (div, name)
        assert best is not None
        div, name = best
        kept.append(name)
        kept_fqns |= set(units[name])
        order.append((name, div))
    return order


def _greedy_curve(ctx: _Ctx, k_values: list[int], full_div: float) -> tuple[list[RecipePoint], list[UnitSensitivity]]:
    wanted = sorted({k for k in k_values if 0 < k < len(ctx.units)})
    order = _greedy_select(ctx.units, ctx.all_fqns, ctx.measure, max(wanted, default=0))

    by_k: dict[int, tuple[list[str], float]] = {}
    gains: dict[str, float] = {}
    prev = full_div
    for step, (name, div) in enumerate(order, 1):
        by_k[step] = ([o[0] for o in order[:step]], div)
        gains[name] = max(0.0, prev - div)  # marginal divergence reduction at selection time
        prev = div

    curve = [
        RecipePoint(k, by_k[k][0], by_k[k][1], _recovery(full_div, by_k[k][1])) for k in wanted
    ]
    # Per-unit "sensitivity" for greedy = its marginal gain when selected (0 if
    # never selected). Keeps SensitivityResult/render_recipe working uniformly.
    units = [
        UnitSensitivity(name, gains.get(name, 0.0), raw_divergence=0.0, n_linears=len(fqns))
        for name, fqns in ctx.units.items()
    ]
    return curve, units


def compute_recipe(
    model_id: str,
    inputs_path: Path,
    device: str = "cpu",
    dtype: str = "float32",
    scheme: str = "w8a8",
    group_size: int = 32,
    strategy: str = "isolated",
    granularity: str = "layer",
    k_values: list[int] | None = None,
    recovery_target: float = 0.9,
    max_measurements: int | None = None,
    smoothquant: bool = False,
) -> RecipeResult:
    """Build + **verify** mixed-precision recipes: for each k, keep the chosen
    units in high precision, quantize the rest, and measure the recovered output
    fidelity. ``isolated``/``marginal`` keep the top-k by a single-pass per-unit
    score; ``greedy`` builds the keep-set by sequential forward selection (more
    measurements, accounts for interactions). The curve both delivers the recipe
    and verifies the strategy.

    Cost: each measurement is a fresh ``deepcopy(fp_model)`` + ``quantize_`` +
    full forward (deepcopy is required because torchao's ``quantize_`` mutates
    in place). ``isolated``/``marginal`` cost ~N forwards (one per unit);
    ``greedy`` costs ~N + N·max(k) ≈ O(N·k). ``--granularity linear`` multiplies
    the unit count ~7× over ``layer`` (q/k/v/o + gate/up/down per block), so a
    30-layer model is ~30 units at layer granularity, ~210 at linear. Budget
    accordingly on a real model; this is wall-clock-heavy, not just "more
    compute." (See review #5 in project_product_portfolio for the planned
    shortlist/budget knobs.)"""
    if strategy not in RECIPE_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {list(RECIPE_STRATEGIES)}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}; choose from {GRANULARITIES}")
    ctx = _setup(
        model_id, inputs_path, device, dtype, scheme, group_size, granularity,
        pre_transforms=_smoothquant_pre_transforms(smoothquant),
    )
    n_units = len(ctx.units)
    if k_values is None:
        k_values = [k for k in (1, 2, 4, 8, 16) if k < n_units]
    _guard_budget(n_units, strategy, k_values, max_measurements, bar=False)

    if strategy == GREEDY:
        full_div = ctx.measure(ctx.all_fqns)
        curve, units = _greedy_curve(ctx, k_values, full_div)
        sens = SensitivityResult(
            ctx.model_id, ctx.scheme, GREEDY, ctx.granularity, full_div, units
        )
    else:
        sens = _run_sensitivity(ctx, strategy)
        full_div = sens.full_quant_divergence
        curve = []
        for k in sorted({k for k in k_values if 0 < k < n_units}):
            keep = sens.keep_high_precision(k)
            keep_fqns = {fqn for name in keep for fqn in ctx.units[name]}
            div = ctx.measure(ctx.all_fqns - keep_fqns)
            curve.append(RecipePoint(k, keep, div, _recovery(full_div, div)))

    numels = linear_numels(ctx.fp_model, ctx.all_fqns)
    base_bits = dtype_bits(dtype)
    for p in curve:
        kept_fqns = {fqn for name in p.kept_units for fqn in ctx.units[name]}
        p.memory_bytes = recipe_memory_bytes(
            numels, ctx.all_fqns - kept_fqns,
            base_bits=base_bits, scheme=scheme, group_size=group_size,
        )
    env = memory_envelope(numels, base_bits=base_bits, scheme=scheme, group_size=group_size)
    return RecipeResult(
        sens, curve, _recommend_k(curve, recovery_target), recovery_target,
        all_fp_bytes=env.all_fp_bytes, all_quant_bytes=env.all_quant_bytes,
        unit_fqns=dict(ctx.units),
    )
