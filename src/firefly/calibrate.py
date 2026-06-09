"""Per-layer tolerance calibration from an empirical noise baseline.

The flat-default ``atol = 1e-5`` is fine for CPU+fp32 deterministic regimes
where divergence is exactly zero or clearly real. It is wildly wrong in any
regime with real nondeterminism: error accumulates with depth, so a
tolerance that's right for layer 1 is too tight by layer 30 and floods
``check`` with false positives.

The fix is empirical: run the reference's forward pass N times, measure
how much each run's per-tap activations deviate from the captured reference
tensors, and set ``atol = safety_factor × observed_max_deviation`` per tap.
That's per-layer calibration: each tap gets a tolerance shaped by its own
position in the network's noise-amplification profile.

CPU+fp32 normally produces zero noise floor (the regression-test mode); to
exercise the methodology, callers can pass a :class:`NoiseSpec` that injects
controllable synthetic noise — see ``firefly.noise``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from firefly.capture import (
    load_golden_inputs,
    load_model_and_tokenizer,
    parse_dtype,
    run_capture_repeated,
)
from firefly.compare import DEFAULT_TOLERANCE, TapTolerance, write_tolerances
from firefly.determinism import set_deterministic, set_hardware_noise_baseline
from firefly.noise import NoiseSpec
from firefly.reference import read_reference


def derive_tolerances(
    reference_tensors: dict[str, torch.Tensor],
    captures: dict[str, list[torch.Tensor]],
    safety_factor: float,
    n_runs: int,
) -> dict[str, TapTolerance]:
    """Pure function: given the reference tensors and N per-tap calibration
    captures, compute per-tap ``TapTolerance`` with
    ``atol = max(safety_factor × max|run - reference|, DEFAULT_TOLERANCE)``.

    The ``DEFAULT_TOLERANCE`` floor handles the zero-noise case (CPU+fp32 with
    ``mode='none'``): without it, ``atol = 0`` would flag any divergence,
    making ``check`` useless after a clean calibration.
    """
    tolerances: dict[str, TapTolerance] = {}
    for tap_name, tensors in captures.items():
        if tap_name not in reference_tensors:
            continue
        ref = reference_tensors[tap_name]

        per_run_max_dev = [
            float((t.float() - ref.float()).abs().max().item()) for t in tensors
        ]
        noise_floor = max(per_run_max_dev) if per_run_max_dev else 0.0
        atol = max(safety_factor * noise_floor, DEFAULT_TOLERANCE)

        tolerances[tap_name] = TapTolerance(
            atol=atol,
            source="calibrated",
            noise_floor=noise_floor,
            n_calibration_runs=n_runs,
        )
    return tolerances


def calibrate(
    reference_dir: Path,
    inputs_path: Path,
    runs: int = 16,
    safety_factor: float = 6.0,
    noise: NoiseSpec | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, TapTolerance]:
    """Calibrate per-tap tolerances by running the reference model N times.

    Loads the reference's model_id from the manifest, runs the same golden
    inputs through it ``runs`` times (optionally with synthetic noise via
    ``noise``), measures per-tap deviation from the captured reference,
    derives ``atol = safety_factor × noise_floor`` per tap with a
    ``DEFAULT_TOLERANCE`` floor.

    Writes ``tolerances.json`` into ``reference_dir`` and returns the
    tolerances dict so callers can introspect without reloading.
    """
    manifest, reference_tensors = read_reference(reference_dir)
    model_dtype = parse_dtype(manifest.dtype)

    if noise is not None and noise.mode == "hardware":
        set_hardware_noise_baseline(seed=seed, allow_tf32=noise.allow_tf32)
    else:
        set_deterministic(seed=seed)
    model, tokenizer = load_model_and_tokenizer(
        manifest.model_id, device=device, dtype=model_dtype,
    )
    batch = load_golden_inputs(inputs_path, tokenizer, device)

    captures = run_capture_repeated(
        model,
        batch,
        runs=runs,
        domain=manifest.domain,
        noise=noise,
        # Match the reference: if it has per-head taps, calibrate them too so
        # they get an empirical tolerance instead of falling to the flat default.
        per_head=bool(manifest.head_counts),
    )

    tolerances = derive_tolerances(
        reference_tensors=reference_tensors,
        captures=captures,
        safety_factor=safety_factor,
        n_runs=runs,
    )
    write_tolerances(reference_dir, tolerances)
    return tolerances
