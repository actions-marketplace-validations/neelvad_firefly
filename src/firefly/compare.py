"""Per-tap-point comparison of candidate vs reference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TapDivergence:
    tap_name: str
    max_abs_diff: float
    mean_abs_diff: float
    tolerance: float
    exceeds_tolerance: bool


def compare_to_reference(
    reference_dir,
    candidate_model_id: str,
    inputs_path,
    tolerances: dict[str, float] | None = None,
) -> list[TapDivergence]:
    """Run candidate, diff against reference, return per-tap divergences in forward order."""
    raise NotImplementedError
