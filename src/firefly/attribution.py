"""First-divergence-layer attribution.

The user-facing value: not "outputs differ" but "divergence first appears at
model.layers.7.mlp.down_proj, everything upstream matches." That's actionable.
"""

from __future__ import annotations

from dataclasses import dataclass

from firefly.compare import TapDivergence


@dataclass
class AttributionResult:
    first_divergent_tap: str | None
    divergences: list[TapDivergence]
    any_exceeded: bool


def attribute_first_divergence(divergences: list[TapDivergence]) -> AttributionResult:
    """Walk divergences in forward order, return the first tap that exceeds tolerance."""
    raise NotImplementedError
