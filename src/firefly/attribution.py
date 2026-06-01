"""First-divergence-layer attribution.

The user-facing value of Firefly is not "outputs differ" but "divergence
first appears at model.layers.7.mlp, everything upstream matches." That's
actionable. This module is the small, pure function that picks the first
flagged tap from a forward-ordered divergence list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from firefly.compare import TapDivergence


@dataclass
class AttributionResult:
    first_divergent_tap: str | None
    any_exceeded: bool
    divergences: list[TapDivergence] = field(default_factory=list)


def attribute_first_divergence(divergences: list[TapDivergence]) -> AttributionResult:
    """Walk divergences in forward order; return the first tap that exceeds tolerance."""
    first = next((d.tap_name for d in divergences if d.exceeds_tolerance), None)
    return AttributionResult(
        first_divergent_tap=first,
        any_exceeded=first is not None,
        divergences=divergences,
    )
