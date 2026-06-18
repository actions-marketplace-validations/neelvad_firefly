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
from firefly.quant.torchao import quantize_model, rel_l1

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


def _fresh_copy(fp_model: nn.Module) -> nn.Module:
    """A pristine fp copy to quantize (quantize_ mutates in place). deepcopy
    avoids re-reading weights from disk per unit."""
    return copy.deepcopy(fp_model)


def _measure(
    fp_model: nn.Module,
    batch: dict,
    ref_output: torch.Tensor,
    targets: set[str],
    scheme: str,
    group_size: int,
) -> float:
    """Output (``final_norm``) relative divergence with ``targets`` quantized."""
    model = _fresh_copy(fp_model)
    if targets:
        quantize_model(
            model, scheme=scheme, group_size=group_size,
            module_filter=lambda _mod, fqn: fqn in targets,
        )
    caps = run_capture(model, batch)
    return rel_l1(ref_output, caps[_OUTPUT_TAP])


@dataclass
class _Ctx:
    """Shared measurement context — one fp model load reused across all sweeps."""

    model_id: str
    fp_model: nn.Module
    batch: dict
    ref_output: torch.Tensor
    units: dict[str, list[str]]
    all_fqns: set[str]
    scheme: str
    group_size: int
    granularity: str

    def measure(self, targets: set[str]) -> float:
        return _measure(
            self.fp_model, self.batch, self.ref_output, targets, self.scheme, self.group_size
        )


def _setup(
    model_id: str, inputs_path: Path, device: str, dtype: str,
    scheme: str, group_size: int, granularity: str,
) -> _Ctx:
    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(model_id, device=device, dtype=parse_dtype(dtype))
    batch = load_golden_inputs(inputs_path, tok, device)
    ref_output = run_capture(fp_model, batch)[_OUTPUT_TAP]
    units = discover_units(fp_model, granularity)
    all_fqns = {fqn for fqns in units.values() for fqn in fqns}
    return _Ctx(model_id, fp_model, batch, ref_output, units, all_fqns, scheme, group_size, granularity)


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


@dataclass
class RecipeResult:
    sensitivity: SensitivityResult
    curve: list[RecipePoint]
    recommended_k: int
    recovery_target: float

    @property
    def recommended_point(self) -> RecipePoint | None:
        return next((p for p in self.curve if p.k == self.recommended_k), None)


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
    ctx = _setup(model_id, inputs_path, device, dtype, scheme, group_size, granularity)
    n_units = len(ctx.units)
    if k_values is None:
        k_values = [k for k in (1, 2, 4, 8, 16) if k < n_units]

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

    return RecipeResult(sens, curve, _recommend_k(curve, recovery_target), recovery_target)
