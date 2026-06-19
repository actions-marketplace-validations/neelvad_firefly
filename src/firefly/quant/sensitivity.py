"""Per-unit quantization sensitivity — the attribution that guides mixed precision.

torchao's autoquant searches blindly and can't tell you *why* a config is bad.
This measures, causally, how much *each unit's* quantization contributes to
**output** divergence, so the worst offenders can be kept in higher precision.
The diagnosis is the product.

A *unit* is the atomic keep-or-quantize granularity:

* ``layer`` (default) — a whole decoder layer (its 7 Linears together). Coarse,
  interpretable, deployable.
* ``linear`` — each ``nn.Linear`` on its own (e.g. keep just one ``down_proj``
  in fp). Finer recipes, but ~7x more units → much bigger search.

Strategies (how a unit's sensitivity is scored / the keep-set is searched) are
pluggable so the user trades compute for resolution:

* ``isolated`` — quantize *only* this unit; sensitivity = the divergence alone.
* ``marginal`` — quantize *all but* this unit; sensitivity = fidelity recovered.
* ``greedy`` (recipe-only) — sequential forward selection over units.

The output metric is the relative divergence at the ``final_norm`` tap (the
network's output proxy) vs the fp baseline, captured on the same golden inputs.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from firefly.capture import (
    load_golden_inputs,
    load_model_and_tokenizer,
    parse_dtype,
    run_capture,
)
from firefly.determinism import set_deterministic
from firefly.quant.cost import (
    BudgetExceededError,
    dtype_bits,
    estimate_measurements,
    frontier_and_knee,
    linear_numels,
    memory_envelope,
    recipe_memory_bytes,
)
from firefly.quant.evaluate import AccuracyBar, Evaluator
from firefly.quant.intervention import Pipeline, PrecisionPolicy
from firefly.quant.torchao import rel_l1

#: Matches a decoder-layer index in a module FQN, with or without a leading
#: prefix (``model.layers.5.mlp...`` and a bare ``layers.5...``).
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_OUTPUT_TAP = "final_norm"

GRANULARITIES = ("layer", "linear")


@dataclass
class UnitSensitivity:
    """One unit's contribution to output divergence under a strategy."""

    unit: str
    """The unit name — ``"layer.28"`` (layer granularity) or a Linear FQN like
    ``"model.layers.28.mlp.down_proj"`` (linear granularity)."""
    sensitivity: float
    """Higher = more important to keep in high precision. For ``isolated`` it's
    the output divergence this unit's quantization causes alone; for ``marginal``
    it's the output fidelity recovered by keeping this unit fp."""
    raw_divergence: float
    """The measured ``final_norm`` divergence for this unit's perturbation."""
    n_linears: int


@dataclass
class SensitivityResult:
    model_id: str
    scheme: str
    strategy: str
    granularity: str
    full_quant_divergence: float
    """Output divergence with *every* unit quantized — the thing we decompose."""
    units: list[UnitSensitivity] = field(default_factory=list)
    output_tap: str = _OUTPUT_TAP

    @property
    def ranked(self) -> list[UnitSensitivity]:
        return sorted(self.units, key=lambda x: x.sensitivity, reverse=True)

    def keep_high_precision(self, k: int) -> list[str]:
        """The top-``k`` most-sensitive unit names — the suggested set to keep in
        higher precision. The recipe sweep verifies it recovers fidelity."""
        return [x.unit for x in self.ranked[:k]]


@dataclass(frozen=True)
class Strategy:
    """A way to score per-unit sensitivity. ``targets`` picks which units to
    quantize when measuring a unit; ``score`` turns the measured output
    divergence into a sensitivity (higher = keep in high precision)."""

    name: str
    describe: str
    targets: Callable[[set[str], set[str]], set[str]]  # (unit_fqns, all_fqns) -> to-quantize
    score: Callable[[float, float], float]  # (measured_div, full_quant_div) -> sensitivity


ISOLATED = Strategy(
    name="isolated",
    describe="quantize only this unit; sensitivity = the output divergence it causes alone",
    targets=lambda unit_fqns, _all: set(unit_fqns),
    score=lambda measured, _full: measured,
)

MARGINAL = Strategy(
    name="marginal",
    describe="quantize all but this unit; sensitivity = the output fidelity recovered by keeping it fp",
    targets=lambda unit_fqns, all_fqns: all_fqns - set(unit_fqns),
    # measured = divergence with every unit but this one quantized; full - measured
    # is how much keeping this unit fp recovers vs all-quantized. Measures marginal
    # contribution directly, yet still builds worse recipes than isolated when one
    # unit dominates (see the P1 finding) — interactions aren't additive.
    score=lambda measured, full: max(0.0, full - measured),
)

STRATEGIES: dict[str, Strategy] = {ISOLATED.name: ISOLATED, MARGINAL.name: MARGINAL}


def discover_units(model: nn.Module, granularity: str = "layer") -> dict[str, list[str]]:
    """Map unit name -> its ``nn.Linear`` FQNs, in forward order. Both
    granularities operate on the same set of decoder-layer Linears (so the
    all-quantized baseline matches); Linears with no layer index (e.g.
    ``lm_head``) are excluded. ``layer`` groups a decoder layer's Linears under
    ``"layer.N"``; ``linear`` makes each Linear its own unit (FQN as the name)."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}; choose from {GRANULARITIES}")
    by_layer: dict[int, list[str]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            m = _LAYER_RE.search(name)
            if m:
                by_layer.setdefault(int(m.group(1)), []).append(name)
    if granularity == "layer":
        return {f"layer.{i}": by_layer[i] for i in sorted(by_layer)}
    return {fqn: [fqn] for i in sorted(by_layer) for fqn in by_layer[i]}


def _guard_budget(
    n_units: int, strategy: str, k_values: list[int], max_measurements: int | None, *, bar: bool
) -> None:
    """Abort before spending any measurements if the a-priori count is over the
    budget. The count is known from n_units + strategy + k_values alone."""
    if max_measurements is None or max_measurements <= 0:
        return
    est = estimate_measurements(n_units, strategy, k_values, bar=bar)
    if est > max_measurements:
        raise BudgetExceededError(est, max_measurements, n_units)


def _fresh_copy(fp_model: nn.Module) -> nn.Module:
    """A pristine fp copy to quantize (quantize_ mutates in place). deepcopy
    avoids re-reading weights from disk per unit."""
    return copy.deepcopy(fp_model)


def _apply_policy(
    fp_model: nn.Module, scheme: str, group_size: int, quantize_fqns: set[str],
    pre_transforms: list | None = None, calib: object | None = None,
) -> nn.Module:
    """Fresh fp copy → the intervention pipeline (``pre_transforms`` then RTN)
    under a :class:`PrecisionPolicy`. The single choke point both the divergence
    sweep and the eval loop quantize through; the agent hands richer pipelines
    here. ``calib`` feeds any pre-transform that needs activation stats."""
    policy = PrecisionPolicy(scheme=scheme, group_size=group_size, quantize=set(quantize_fqns))
    pipeline = Pipeline(pre_transforms=list(pre_transforms or []))
    return pipeline.run(_fresh_copy(fp_model), policy, calib)


def _smoothquant_pre_transforms(smoothquant: bool) -> list:
    """[SmoothQuant()] when enabled (scope=None → smooths whatever the policy
    quantizes), else []. Lazy import keeps the optional adapter off the hot path."""
    if not smoothquant:
        return []
    from firefly.quant.smoothquant import SmoothQuant

    return [SmoothQuant()]


def _measure(
    fp_model: nn.Module,
    batch: dict,
    ref_output: torch.Tensor,
    targets: set[str],
    scheme: str,
    group_size: int,
    pre_transforms: list | None = None,
    calib: object | None = None,
) -> float:
    """Output (``final_norm``) relative divergence with ``targets`` quantized."""
    model = _apply_policy(fp_model, scheme, group_size, targets, pre_transforms, calib)
    caps = run_capture(model, batch)
    return rel_l1(ref_output, caps[_OUTPUT_TAP])


@dataclass
class _Ctx:
    """Shared measurement context — one fp model load reused across all sweeps."""

    model_id: str
    fp_model: nn.Module
    tokenizer: object
    batch: dict
    ref_output: torch.Tensor
    units: dict[str, list[str]]
    all_fqns: set[str]
    scheme: str
    group_size: int
    granularity: str
    pre_transforms: list = field(default_factory=list)
    """Stage.PRE_TRANSFORM interventions (e.g. SmoothQuant) applied before the
    quantizer on every measurement; calibrated on ``batch``."""

    def measure(self, targets: set[str]) -> float:
        return _measure(
            self.fp_model, self.batch, self.ref_output, targets, self.scheme,
            self.group_size, self.pre_transforms, self.batch,
        )


def _setup(
    model_id: str, inputs_path: Path, device: str, dtype: str,
    scheme: str, group_size: int, granularity: str, pre_transforms: list | None = None,
) -> _Ctx:
    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(model_id, device=device, dtype=parse_dtype(dtype))
    batch = load_golden_inputs(inputs_path, tok, device)
    ref_output = run_capture(fp_model, batch)[_OUTPUT_TAP]
    units = discover_units(fp_model, granularity)
    all_fqns = {fqn for fqns in units.values() for fqn in fqns}
    return _Ctx(
        model_id, fp_model, tok, batch, ref_output, units, all_fqns,
        scheme, group_size, granularity, pre_transforms or [],
    )


def _run_sensitivity(ctx: _Ctx, strategy: str) -> SensitivityResult:
    strat = STRATEGIES[strategy]
    full_div = ctx.measure(ctx.all_fqns)
    out: list[UnitSensitivity] = []
    for name, fqns in ctx.units.items():
        measured = ctx.measure(strat.targets(set(fqns), ctx.all_fqns))
        out.append(
            UnitSensitivity(
                unit=name,
                sensitivity=strat.score(measured, full_div),
                raw_divergence=measured,
                n_linears=len(fqns),
            )
        )
    return SensitivityResult(
        model_id=ctx.model_id, scheme=ctx.scheme, strategy=strategy,
        granularity=ctx.granularity, full_quant_divergence=full_div, units=out,
    )


def compute_sensitivity(
    model_id: str,
    inputs_path: Path,
    device: str = "cpu",
    dtype: str = "float32",
    scheme: str = "w8a8",
    group_size: int = 32,
    strategy: str = "isolated",
    granularity: str = "layer",
) -> SensitivityResult:
    """Run the per-unit sensitivity sweep. Captures its own fp baseline, the
    all-quantized baseline, then one measurement per unit."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}; choose from {GRANULARITIES}")
    return _run_sensitivity(
        _setup(model_id, inputs_path, device, dtype, scheme, group_size, granularity), strategy
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
    )


# ---------------------------------------------------------------------------
# Optimize to an accuracy bar: rank by the cheap proxy (the filter), then gate
# candidate recipes on a real eval metric (the wrapper) and return the smallest
# keep-set that clears the bar. See firefly.quant.evaluate.
# ---------------------------------------------------------------------------


@dataclass
class BarPoint:
    """One evaluated recipe: top-``k`` units kept fp, the rest quantized, and
    the real metric it achieved (with whether that clears the bar)."""

    k: int
    kept_units: list[str]
    metric: float
    passes: bool
    memory_bytes: float = 0.0


@dataclass
class BarRecipeResult:
    model_id: str
    scheme: str
    granularity: str
    strategy: str
    metric_name: str
    higher_is_better: bool
    bar: AccuracyBar
    baseline_metric: float
    """The fp model's metric — the bar is measured relative to this."""
    full_quant_metric: float
    """The all-quantized metric — the floor the recipe climbs back from."""
    threshold: float
    n_units: int
    chosen_k: int
    chosen_kept_units: list[str]
    chosen_metric: float
    evaluated: list[BarPoint]
    evals_used: int
    chosen_memory_bytes: float = 0.0
    all_fp_bytes: float = 0.0
    all_quant_bytes: float = 0.0

    def frontier_knee_ks(self) -> tuple[set[int], int | None]:
        """``(ks on the Pareto frontier, knee k)`` over (memory, metric), with
        the metric's own direction; cost is always lower-better."""
        pts = sorted(self.evaluated, key=lambda p: p.k)
        idx, knee = frontier_and_knee(
            [p.memory_bytes for p in pts],
            [p.metric for p in pts],
            quality_higher_is_better=self.higher_is_better,
        )
        return {pts[i].k for i in idx}, (pts[knee].k if knee is not None else None)


def _bar_search(n: int, passes_at: Callable[[int], bool]) -> int:
    """Smallest ``k`` in ``[0, n]`` with ``passes_at(k)`` True.

    Assumes ``passes_at`` is monotonic non-decreasing in ``k`` (keeping more
    units in high precision never hurts the metric) and that ``passes_at(n)``
    holds (the fp baseline always meets its own bar). Binary search → ~log2(n)
    calls; ``passes_at`` should be memoized so a repeated ``k`` is free.

    The monotonicity assumption is what makes this cheap; quantization
    interactions can dent it slightly, so the result is the smallest *confirmed*
    passing k under that assumption, not a proof of global minimality.
    """
    if passes_at(0):
        return 0
    lo, hi = 0, n  # invariant: not passes_at(lo); passes_at(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes_at(mid):
            hi = mid
        else:
            lo = mid
    return hi


def optimize_to_bar(
    model_id: str,
    inputs_path: Path,
    evaluator: Evaluator,
    bar: AccuracyBar,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    scheme: str = "w8a8",
    group_size: int = 32,
    strategy: str = "isolated",
    granularity: str = "layer",
    max_measurements: int | None = None,
    smoothquant: bool = False,
) -> BarRecipeResult:
    """Find the smallest mixed-precision recipe that clears an accuracy bar.

    Two tiers: rank units by the cheap output-divergence proxy on the
    calibration prompts (one ``_run_sensitivity`` pass — the filter), then
    binary-search ``k`` and spend a *real* ``evaluator`` call on the held-out
    eval set only at each probed recipe (the wrapper). So the expensive evals
    cost ~log2(N) + 2 (baseline + all-quantized floor), not O(N).

    ``strategy`` is the ranking only (``isolated`` / ``marginal``); the bar
    decides how many of that ranking to keep.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"optimize_to_bar ranks with {sorted(STRATEGIES)}, got {strategy!r} "
            "(greedy is a recipe search, not a ranking)"
        )
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}; choose from {GRANULARITIES}")

    ctx = _setup(
        model_id, inputs_path, device, dtype, scheme, group_size, granularity,
        pre_transforms=_smoothquant_pre_transforms(smoothquant),
    )
    n = len(ctx.units)
    _guard_budget(n, strategy, [], max_measurements, bar=True)
    ranked = _run_sensitivity(ctx, strategy).keep_high_precision(n)  # most-sensitive first

    evaluated: dict[int, float] = {}

    def metric_at(k: int) -> float:
        if k not in evaluated:
            kept_fqns = {fqn for name in ranked[:k] for fqn in ctx.units[name]}
            model = _apply_policy(
                ctx.fp_model, scheme, group_size, ctx.all_fqns - kept_fqns,
                ctx.pre_transforms, ctx.batch,
            )
            evaluated[k] = evaluator(model, ctx.tokenizer)
        return evaluated[k]

    hib = evaluator.higher_is_better
    baseline = metric_at(n)        # all fp — the reference the bar is relative to
    full_quant = metric_at(0)      # all quantized — the floor

    chosen_k = _bar_search(n, lambda k: bar.passes(metric_at(k), baseline, hib))

    numels = linear_numels(ctx.fp_model, ctx.all_fqns)
    base_bits = dtype_bits(dtype)

    def cost_at(k: int) -> float:
        kept_fqns = {fqn for name in ranked[:k] for fqn in ctx.units[name]}
        return recipe_memory_bytes(
            numels, ctx.all_fqns - kept_fqns,
            base_bits=base_bits, scheme=scheme, group_size=group_size,
        )

    points = [
        BarPoint(k, ranked[:k], evaluated[k], bar.passes(evaluated[k], baseline, hib), cost_at(k))
        for k in sorted(evaluated)
    ]
    env = memory_envelope(numels, base_bits=base_bits, scheme=scheme, group_size=group_size)
    return BarRecipeResult(
        model_id=ctx.model_id,
        scheme=ctx.scheme,
        granularity=ctx.granularity,
        strategy=strategy,
        metric_name=evaluator.name,
        higher_is_better=hib,
        bar=bar,
        baseline_metric=baseline,
        full_quant_metric=full_quant,
        threshold=bar.threshold(baseline, hib),
        n_units=n,
        chosen_k=chosen_k,
        chosen_kept_units=ranked[:chosen_k],
        chosen_metric=metric_at(chosen_k),
        evaluated=points,
        evals_used=len(evaluated),
        chosen_memory_bytes=cost_at(chosen_k),
        all_fp_bytes=env.all_fp_bytes,
        all_quant_bytes=env.all_quant_bytes,
    )
