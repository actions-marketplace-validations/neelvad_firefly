"""Per-layer tolerance calibration from an empirical determinism baseline.

Runs the reference against itself N times under the target nondeterminism,
measures per-tap-point noise floor, derives per-tap tolerances.

This is the moat. A flat atol/rtol is wrong because error accumulates with
depth — last-bit differences at layer 0 compound through 32 layers, so a
tolerance that's right for layer 1 screams false alarms by layer 30. The
calibrated per-layer tolerance is what makes divergence flags signal, not noise.
"""

from __future__ import annotations

from pathlib import Path


def calibrate(reference_dir: Path, runs: int = 16, sigma_multiplier: float = 6.0) -> dict[str, float]:
    """Run reference forward N times, return per-tap-point tolerance map.

    Tolerance for tap_i = sigma_multiplier * std(max_abs_diff_across_runs at tap_i).
    """
    raise NotImplementedError
