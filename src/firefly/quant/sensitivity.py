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
from firefly.quant.cost import BudgetExceededError, estimate_measurements
from firefly.quant.intervention import Pipeline, PrecisionPolicy
from firefly.quant.torchao import rel_l1
from firefly.tap_points import find_decoder_layers_path

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
    ``"layer.N"``; ``linear`` makes each Linear its own unit (FQN as the name).

    The decoder-layer path is resolved with the shared
    :func:`firefly.tap_points.find_decoder_layers_path` (handles ``model.layers``
    / ``transformer.h`` / ``layers``), which **raises** on an unrecognized
    layout rather than silently finding zero units; a recognized layout with no
    ``nn.Linear`` (e.g. GPT-2's ``Conv1D``) also fails loudly here."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}; choose from {GRANULARITIES}")
    layers_path = find_decoder_layers_path(model)  # raises on unknown layout
    idx_re = re.compile(re.escape(layers_path) + r"\.(\d+)\.")
    by_layer: dict[int, list[str]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            m = idx_re.search(name)
            if m:
                by_layer.setdefault(int(m.group(1)), []).append(name)
    if not by_layer:
        raise ValueError(
            f"No quantizable nn.Linear found under decoder layers {layers_path!r} "
            f"({type(model).__name__}). The layout may use a non-Linear projection "
            "(e.g. GPT-2 Conv1D), which the recipe path does not support."
        )
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

