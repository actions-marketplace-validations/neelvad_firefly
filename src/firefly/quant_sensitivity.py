"""Per-layer quantization sensitivity — the attribution that guides mixed precision.

torchao's autoquant searches blindly and can't tell you *why* a config is bad.
This measures, causally, how much *each decoder layer's* quantization contributes
to **output** divergence, so the worst offenders can be kept in higher precision.
The diagnosis is the product.

Strategies are pluggable so the user trades compute for resolution:

* ``isolated`` (default) — quantize *only* this layer (everything else fp) and
  measure the output divergence it causes alone. Cheap, interpretable; N+1
  forwards for N layers.
* ``marginal`` (later) — quantize *all but* this layer and measure how much
  keeping it fp *recovers*. More decision-relevant, same forward count but each
  pass quantizes N-1 layers.

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
from firefly.quant_validate import quantize_model, rel_l1

#: Matches a decoder-layer index in a module FQN, with or without a leading
#: prefix (``model.layers.5.mlp...`` and a bare ``layers.5...``).
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_OUTPUT_TAP = "final_norm"


@dataclass
class LayerSensitivity:
    """One decoder layer's contribution to output divergence under a strategy."""

    layer: int
    sensitivity: float
    """Higher = more important to keep in high precision. For ``isolated`` it's
    the output divergence this layer's quantization causes alone; for
    ``marginal`` it's the output fidelity recovered by keeping this layer fp."""
    raw_divergence: float
    """The measured ``final_norm`` divergence for this layer's perturbation."""
    n_linears: int


@dataclass
class SensitivityResult:
    model_id: str
    scheme: str
    strategy: str
    full_quant_divergence: float
    """Output divergence with *every* layer quantized — the thing we decompose."""
    layers: list[LayerSensitivity] = field(default_factory=list)
    output_tap: str = _OUTPUT_TAP

    @property
    def ranked(self) -> list[LayerSensitivity]:
        return sorted(self.layers, key=lambda x: x.sensitivity, reverse=True)

    def keep_high_precision(self, k: int) -> list[int]:
        """The top-``k`` most-sensitive layer indices — the suggested set to
        keep in higher precision. P1 verifies that this recipe recovers fidelity."""
        return [x.layer for x in self.ranked[:k]]


@dataclass(frozen=True)
class Strategy:
    """A way to score per-layer sensitivity. ``targets`` picks which layers to
    quantize when measuring layer i; ``score`` turns the measured output
    divergence into a sensitivity (higher = keep in high precision)."""

    name: str
    describe: str
    targets: Callable[[set[str], set[str]], set[str]]  # (layer_fqns, all_fqns) -> to-quantize
    score: Callable[[float, float], float]  # (measured_div, full_quant_div) -> sensitivity


ISOLATED = Strategy(
    name="isolated",
    describe="quantize only this layer; sensitivity = the output divergence it causes alone",
    targets=lambda layer_fqns, _all: set(layer_fqns),
    score=lambda measured, _full: measured,
)

# Next strategy plugs in here (same shape), e.g. marginal:
#   targets=lambda layer_fqns, all_fqns: all_fqns - set(layer_fqns)
#   score=lambda measured, full: full - measured   # recovery from keeping layer fp
STRATEGIES: dict[str, Strategy] = {ISOLATED.name: ISOLATED}


def discover_layers(model: nn.Module) -> dict[int, list[str]]:
    """Map decoder-layer index -> its ``nn.Linear`` FQNs, in order. Linears with
    no layer index (e.g. ``lm_head``) are excluded — sensitivity is per layer."""
    layers: dict[int, list[str]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            m = _LAYER_RE.search(name)
            if m:
                layers.setdefault(int(m.group(1)), []).append(name)
    return {i: layers[i] for i in sorted(layers)}


def _fresh_copy(fp_model: nn.Module) -> nn.Module:
    """A pristine fp copy to quantize (quantize_ mutates in place). deepcopy
    avoids re-reading weights from disk per layer."""
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
    layers: dict[int, list[str]]
    all_fqns: set[str]
    scheme: str
    group_size: int

    def measure(self, targets: set[str]) -> float:
        return _measure(
            self.fp_model, self.batch, self.ref_output, targets, self.scheme, self.group_size
        )


def _setup(
    model_id: str, inputs_path: Path, device: str, dtype: str, scheme: str, group_size: int
) -> _Ctx:
    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(model_id, device=device, dtype=parse_dtype(dtype))
    batch = load_golden_inputs(inputs_path, tok, device)
    ref_output = run_capture(fp_model, batch)[_OUTPUT_TAP]
    layers = discover_layers(fp_model)
    all_fqns = {fqn for fqns in layers.values() for fqn in fqns}
    return _Ctx(model_id, fp_model, batch, ref_output, layers, all_fqns, scheme, group_size)


def _run_sensitivity(ctx: _Ctx, strategy: str) -> SensitivityResult:
    strat = STRATEGIES[strategy]
    full_div = ctx.measure(ctx.all_fqns)
    out: list[LayerSensitivity] = []
    for idx, fqns in ctx.layers.items():
        measured = ctx.measure(strat.targets(set(fqns), ctx.all_fqns))
        out.append(
            LayerSensitivity(
                layer=idx,
                sensitivity=strat.score(measured, full_div),
                raw_divergence=measured,
                n_linears=len(fqns),
            )
        )
    return SensitivityResult(
        model_id=ctx.model_id, scheme=ctx.scheme, strategy=strategy,
        full_quant_divergence=full_div, layers=out,
    )


def compute_sensitivity(
    model_id: str,
    inputs_path: Path,
    device: str = "cpu",
    dtype: str = "float32",
    scheme: str = "w8a8",
    group_size: int = 32,
    strategy: str = "isolated",
) -> SensitivityResult:
    """Run the per-layer sensitivity sweep. Captures its own fp baseline, the
    all-quantized baseline, then one measurement per decoder layer."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    return _run_sensitivity(
        _setup(model_id, inputs_path, device, dtype, scheme, group_size), strategy
    )


@dataclass
class RecipePoint:
    """One mixed-precision recipe: keep the top-``k`` sensitive layers in high
    precision, quantize the rest, and the output fidelity it achieves."""

    k: int
    kept_layers: list[int]
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


def compute_recipe(
    model_id: str,
    inputs_path: Path,
    device: str = "cpu",
    dtype: str = "float32",
    scheme: str = "w8a8",
    group_size: int = 32,
    strategy: str = "isolated",
    k_values: list[int] | None = None,
    recovery_target: float = 0.9,
) -> RecipeResult:
    """Rank layers, then build + **verify** mixed-precision recipes: for each k,
    keep the top-k sensitive layers in high precision, quantize the rest, and
    measure the recovered output fidelity. The curve both delivers the recipe
    and verifies the ranking actually predicts good recipes."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    ctx = _setup(model_id, inputs_path, device, dtype, scheme, group_size)
    sens = _run_sensitivity(ctx, strategy)
    full_div = sens.full_quant_divergence
    n_layers = len(ctx.layers)

    if k_values is None:
        k_values = [k for k in (1, 2, 4, 8, 16) if k < n_layers]
    curve: list[RecipePoint] = []
    for k in sorted({k for k in k_values if 0 < k < n_layers}):
        keep = sens.keep_high_precision(k)
        keep_fqns = {fqn for idx in keep for fqn in ctx.layers[idx]}
        div = ctx.measure(ctx.all_fqns - keep_fqns)  # quantize all EXCEPT kept layers
        curve.append(RecipePoint(k, keep, div, _recovery(full_div, div)))

    return RecipeResult(sens, curve, _recommend_k(curve, recovery_target), recovery_target)
