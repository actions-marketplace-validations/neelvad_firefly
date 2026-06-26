"""Memory cost of a recipe + the Pareto machinery to compare recipes.

A recipe trades *quality* against *cost*. Quality is measured elsewhere (output
divergence or a real eval metric); this module supplies the other axis — the
weight memory a recipe actually costs — and the tools to turn a set of
(cost, quality) recipes into a decision: the non-dominated **Pareto frontier**
(the recipes nobody beats on both axes) and its **knee** (best quality per byte).

Cost is the *variable* part the recipe controls: the quantizable decoder
Linears. Embeddings / norms / lm_head are constant across recipes, so they'd
only shift the whole frontier, never reorder it — excluding them keeps the
numbers about the decision. Everything here is pure (param shapes + bit-widths,
no forward pass).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import nn

#: Effective *weight* bits per scheme (w8a8 also quantizes activations, but
#: that's runtime, not weight footprint).
SCHEME_WEIGHT_BITS = {"w8a8": 8, "int8wo": 8, "int4wo": 4}

_DTYPE_BITS = {"float32": 32, "fp32": 32, "bfloat16": 16, "bf16": 16, "float16": 16, "fp16": 16}


def dtype_bits(dtype: str) -> int:
    try:
        return _DTYPE_BITS[dtype]
    except KeyError as e:
        raise ValueError(f"unknown dtype {dtype!r} for cost; known: {sorted(_DTYPE_BITS)}") from e


def linear_numels(model: nn.Module, fqns: set[str]) -> dict[str, int]:
    """``{fqn: weight element count}`` for the given Linear FQNs."""
    out: dict[str, int] = {}
    for fqn in fqns:
        mod = model.get_submodule(fqn)
        out[fqn] = mod.weight.numel()
    return out


def _quantized_bytes(numel: int, scheme: str, group_size: int) -> float:
    """Bytes for one quantized Linear weight: nominal bits + (int4wo only) the
    fp16 group-scale overhead, which is real and non-trivial at small groups."""
    bits = SCHEME_WEIGHT_BITS[scheme]
    weight_bytes = numel * bits / 8
    if scheme == "int4wo" and group_size > 0:
        weight_bytes += (numel / group_size) * 2  # one fp16 scale per group
    return weight_bytes


def recipe_memory_bytes(
    numels: dict[str, int],
    quantized_fqns: set[str],
    *,
    base_bits: int,
    scheme: str,
    group_size: int,
) -> float:
    """Total weight bytes of the quantizable Linears under a recipe: the ones in
    ``quantized_fqns`` at the scheme's precision, the rest at ``base_bits``."""
    total = 0.0
    for fqn, numel in numels.items():
        if fqn in quantized_fqns:
            total += _quantized_bytes(numel, scheme, group_size)
        else:
            total += numel * base_bits / 8
    return total


@dataclass
class MemoryEnvelope:
    """The two anchors a recipe sits between."""

    all_fp_bytes: float       # nothing quantized — the ceiling cost / best quality
    all_quant_bytes: float    # everything quantized — the floor cost / worst quality

    def compression(self, recipe_bytes: float) -> float:
        """all-fp size / recipe size — ``1.0`` = no savings, ``4.0`` = 4× smaller."""
        return self.all_fp_bytes / recipe_bytes if recipe_bytes > 0 else 1.0


def memory_envelope(
    numels: dict[str, int], *, base_bits: int, scheme: str, group_size: int
) -> MemoryEnvelope:
    all_fqns = set(numels)
    return MemoryEnvelope(
        all_fp_bytes=recipe_memory_bytes(
            numels, set(), base_bits=base_bits, scheme=scheme, group_size=group_size
        ),
        all_quant_bytes=recipe_memory_bytes(
            numels, all_fqns, base_bits=base_bits, scheme=scheme, group_size=group_size
        ),
    )


def pareto_indices(
    costs: list[float], qualities: list[float], *, quality_higher_is_better: bool
) -> list[int]:
    """Indices of the non-dominated points, sorted by ascending cost.

    Cost is always lower-better. A point is dominated if another is no worse on
    cost AND no worse on quality, and strictly better on at least one. Ties
    (identical cost+quality) are both kept.
    """
    n = len(costs)
    keep: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i:
                continue
            cost_ok = costs[j] <= costs[i]
            if quality_higher_is_better:
                qual_ok = qualities[j] >= qualities[i]
                strict = costs[j] < costs[i] or qualities[j] > qualities[i]
            else:
                qual_ok = qualities[j] <= qualities[i]
                strict = costs[j] < costs[i] or qualities[j] < qualities[i]
            if cost_ok and qual_ok and strict:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: (costs[i], qualities[i]))


def knee_index(costs: list[float], qualities: list[float]) -> int:
    """Index (into the given frontier-ordered lists) of the knee — the point
    farthest from the chord joining the cheapest and most-expensive frontier
    points, in min-max-normalized space (the classic elbow). With <3 points
    there's no interior knee; returns the last (highest-cost / best-quality) one.
    """
    m = len(costs)
    if m == 0:
        return -1
    if m <= 2:
        return m - 1

    def norm(vals: list[float]) -> list[float]:
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals]

    cs, qs = norm(costs), norm(qualities)
    x0, y0, x1, y1 = cs[0], qs[0], cs[-1], qs[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = math.hypot(dx, dy) or 1.0
    best_i, best_d = 0, -1.0
    for i in range(m):
        d = abs(dy * (cs[i] - x0) - dx * (qs[i] - y0)) / denom
        if d > best_d:
            best_d, best_i = d, i
    return best_i


def frontier_and_knee(
    costs: list[float], qualities: list[float], *, quality_higher_is_better: bool
) -> tuple[list[int], int | None]:
    """``(frontier_positions, knee_position)`` as indices into the inputs. The
    knee is one of the frontier positions (or ``None`` if there are no points).
    """
    idx = pareto_indices(costs, qualities, quality_higher_is_better=quality_higher_is_better)
    if not idx:
        return [], None
    ki = knee_index([costs[i] for i in idx], [qualities[i] for i in idx])
    return idx, (idx[ki] if ki >= 0 else None)


class BudgetExceededError(RuntimeError):
    """Raised before any measurement when the a-priori count exceeds the budget,
    so an O(N·k) greedy / linear-granularity run doesn't start unannounced."""

    def __init__(self, estimate: int, budget: int, n_units: int) -> None:
        self.estimate, self.budget, self.n_units = estimate, budget, n_units
        super().__init__(
            f"estimated {estimate} measurements (over {n_units} units) exceeds "
            f"--max-measurements {budget}"
        )


def estimate_measurements(n_units: int, strategy: str, k_values: list[int], *, bar: bool = False) -> int:
    """A-priori count of model runs a recipe will cost — known before running,
    so it can be shown up front and capped with ``--max-measurements``."""
    valid_ks = [k for k in k_values if 0 < k < n_units]
    if strategy == "greedy":
        K = max(valid_ks, default=0)
        # 1 (all-quant baseline) + sequential selection: (n) + (n-1) + ... + (n-K+1)
        return 1 + K * n_units - K * (K - 1) // 2
    # isolated / marginal: _run_sensitivity measures the all-quant baseline once
    # (+1) then one per unit (n) — that ranking pass is n + 1, not n.
    ranking = n_units + 1
    if bar:
        search = (math.ceil(math.log2(n_units)) + 2) if n_units > 1 else 1  # 2 anchors + binary search
        return ranking + search
    return ranking + len(set(valid_ks))


def format_bytes(n: float) -> str:
    """Human bytes (decimal MB/GB, matching how model sizes are usually quoted)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1000
    return f"{n:.1f} GB"
