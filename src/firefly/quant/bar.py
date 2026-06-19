"""Eval-bar optimization: the smallest recipe whose real eval metric clears a bar.

Two tiers on top of :mod:`firefly.quant.sensitivity`: the cheap divergence proxy
ranks units (the filter), then a real :class:`~firefly.quant.evaluate.Evaluator`
gates candidate recipes and a binary search finds the smallest passing keep-set
(the wrapper). See :mod:`firefly.quant.evaluate` for the metric + accuracy bar.
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
from firefly.quant.evaluate import AccuracyBar, Evaluator
from firefly.quant.sensitivity import (
    GRANULARITIES,
    STRATEGIES,
    _apply_policy,
    _guard_budget,
    _run_sensitivity,
    _setup,
    _smoothquant_pre_transforms,
)


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
    unit_fqns: dict[str, list[str]] = field(default_factory=dict)
    """unit name → its Linear FQNs (for recipe export)."""

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
        unit_fqns=dict(ctx.units),
    )
