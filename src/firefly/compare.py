"""Per-tap-point comparison of candidate vs reference.

Two layers, matching the capture module:

* :func:`diff_captures` is the pure, testable diff. Takes two dicts of
  ``{tap_name: tensor}`` and a tap-order list, returns ``TapDivergence`` per
  tap in forward order. No I/O, no HF.
* :func:`compare_to_reference` is the orchestrator: loads the candidate,
  tokenizes the golden inputs, runs capture against the candidate, calls
  ``diff_captures``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from firefly.reference import read_reference

DEFAULT_TOLERANCE = 1e-5
TOLERANCES_FILE = "tolerances.json"

ToleranceSource = Literal["default", "manual", "calibrated"]


class FingerprintMismatchError(ValueError):
    """Candidate's actual weights don't match what the reference recorded.

    This usually means the model behind ``candidate_model_id`` has been
    retrained, fine-tuned, or swapped since the reference was captured.
    Continuing would compare against an unintended model and silently
    produce wrong results, so we raise unless the caller opts in.
    """


@dataclass
class TapTolerance:
    """The threshold applied at a single tap, plus the provenance metadata
    Phase 2 calibration will fill in.

    ``source`` distinguishes a default-flat tolerance from one the user
    supplied by hand from one derived empirically by ``firefly calibrate``.
    The numerical fields are populated for ``source="calibrated"``.
    """

    atol: float
    source: ToleranceSource = "default"
    noise_floor: float | None = None
    n_calibration_runs: int | None = None


def _default_tolerance() -> TapTolerance:
    return TapTolerance(atol=DEFAULT_TOLERANCE, source="default")


def write_tolerances(ref_dir: Path, tolerances: dict[str, TapTolerance]) -> None:
    """Persist ``tolerances`` to ``<ref_dir>/tolerances.json``.

    Written by ``firefly calibrate``; auto-loaded by ``firefly check``.
    Users can also hand-edit the file to tune individual atols.
    """
    ref_dir.mkdir(parents=True, exist_ok=True)
    payload = {"tolerances": {name: asdict(t) for name, t in tolerances.items()}}
    with (ref_dir / TOLERANCES_FILE).open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_tolerances(ref_dir: Path) -> dict[str, TapTolerance] | None:
    """Load tolerances from ``<ref_dir>/tolerances.json``, or ``None`` if absent."""
    path = ref_dir / TOLERANCES_FILE
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    raw = data.get("tolerances", {})
    return {name: TapTolerance(**fields) for name, fields in raw.items()}


@dataclass
class TapDivergence:
    tap_name: str
    max_abs_diff: float
    mean_abs_diff: float
    tolerance: TapTolerance
    exceeds_tolerance: bool
    effective_atol: float = 0.0  # the threshold actually applied (atol or rel-error ceiling)


def diff_captures(
    reference_tensors: dict[str, torch.Tensor],
    candidate_tensors: dict[str, torch.Tensor],
    tap_order: list[str],
    tolerances: dict[str, TapTolerance] | None = None,
    max_rel_error: float | None = None,
) -> list[TapDivergence]:
    """Diff two captures in the given tap order.

    ``max_rel_error`` is an optional global ceiling on top of per-tap
    calibrated tolerances. If set, a tap with very tight calibration (say
    1e-6 atol on a tensor whose max abs is 10) effectively gets the looser
    of ``tol.atol`` and ``max_rel_error × max|ref|`` — i.e., the user is
    saying "I don't care about sub-X% drift anywhere, even if calibration
    says it's outside the noise floor." Defaults off; calibrated tolerances
    are the sole gate.

    Raises ``ValueError`` on missing tap or shape mismatch — those are
    structural problems with the candidate (or the reference), not numerical
    divergence, and conflating them would hide real bugs from the user.
    """
    tolerances = tolerances or {}
    divergences: list[TapDivergence] = []

    for tap_name in tap_order:
        if tap_name not in reference_tensors:
            raise ValueError(f"Reference is missing tap {tap_name!r}")
        if tap_name not in candidate_tensors:
            raise ValueError(f"Candidate is missing tap {tap_name!r}")

        ref_t = reference_tensors[tap_name]
        cand_t = candidate_tensors[tap_name]
        if ref_t.shape != cand_t.shape:
            raise ValueError(
                f"Shape mismatch at {tap_name!r}: "
                f"reference {tuple(ref_t.shape)} vs candidate {tuple(cand_t.shape)}"
            )

        diff = (cand_t.float() - ref_t.float()).abs()
        max_d = float(diff.max().item())
        mean_d = float(diff.mean().item())
        tol = tolerances.get(tap_name, _default_tolerance())
        effective_atol = tol.atol
        if max_rel_error is not None and max_rel_error > 0:
            ref_max = float(ref_t.float().abs().max().item())
            effective_atol = max(effective_atol, max_rel_error * ref_max)
        divergences.append(
            TapDivergence(
                tap_name=tap_name,
                max_abs_diff=max_d,
                mean_abs_diff=mean_d,
                tolerance=tol,
                exceeds_tolerance=max_d > effective_atol,
                effective_atol=effective_atol,
            )
        )

    return divergences


def _run_candidate(
    reference_dir: Path,
    candidate_model_id: str,
    inputs_path: Path,
    device: str,
    seed: int,
    tolerances: dict[str, TapTolerance] | None,
    allow_fingerprint_mismatch: bool,
    candidate_dtype: str | None = None,
    runner: object | None = None,
):
    """Load the reference, capture the candidate via a Runner, fingerprint-check.

    Returns ``(manifest, ref_tensors, candidate_tensors, tolerances)``. Shared
    by :func:`compare_to_reference` and :func:`compare_to_reference_per_head`
    so the candidate forward pass runs exactly once per check.

    The candidate is captured to mirror the reference: same dtype
    (``manifest.dtype`` — comparing a bf16 reference against an fp32-loaded
    candidate would report the dtype gap as divergence, not a real
    regression), same domain, and per-head taps iff the reference has them.
    ``candidate_dtype`` overrides the dtype for a deliberate cross-dtype run.
    """
    from firefly.runners import get_runner  # local import: avoids an import cycle

    manifest, ref_tensors = read_reference(reference_dir)

    if tolerances is None:
        tolerances = read_tolerances(reference_dir)

    active_runner = runner if runner is not None else get_runner("hf")
    result = active_runner.capture(
        candidate_model_id,
        inputs_path,
        device=device,
        seed=seed,
        domain=manifest.domain,
        dtype=candidate_dtype or manifest.dtype,
        per_head=bool(manifest.head_counts),
    )

    if result.fingerprint != manifest.model_fingerprint and not allow_fingerprint_mismatch:
        raise FingerprintMismatchError(
            f"Candidate fingerprint differs from reference manifest.\n"
            f"  reference model_id:    {manifest.model_id}\n"
            f"  reference fingerprint: {manifest.model_fingerprint}\n"
            f"  candidate model_id:    {candidate_model_id}\n"
            f"  candidate fingerprint: {result.fingerprint}\n"
            f"\n"
            f"The reference was captured against a different version of this model. "
            f"Either re-capture with `firefly capture`, or pass "
            f"--allow-fingerprint-mismatch to proceed anyway."
        )

    return manifest, ref_tensors, result.tensors, tolerances


def compare_to_reference(
    reference_dir: Path,
    candidate_model_id: str,
    inputs_path: Path,
    device: str = "cpu",
    seed: int = 0,
    tolerances: dict[str, TapTolerance] | None = None,
    allow_fingerprint_mismatch: bool = False,
    max_rel_error: float | None = None,
    candidate_dtype: str | None = None,
    runner: object | None = None,
) -> list[TapDivergence]:
    """Run candidate, diff against reference, return per-tap divergences in forward order.

    Raises :class:`FingerprintMismatchError` if the candidate's fingerprint
    doesn't match the reference manifest, unless ``allow_fingerprint_mismatch``
    is set. The candidate loads at the reference's dtype unless
    ``candidate_dtype`` overrides it. ``runner`` selects the capture backend
    (defaults to the HF runner).
    """
    manifest, ref_tensors, candidate_tensors, tolerances = _run_candidate(
        reference_dir,
        candidate_model_id,
        inputs_path,
        device,
        seed,
        tolerances,
        allow_fingerprint_mismatch,
        candidate_dtype,
        runner,
    )
    return diff_captures(
        reference_tensors=ref_tensors,
        candidate_tensors=candidate_tensors,
        tap_order=manifest.tap_points,
        tolerances=tolerances,
        max_rel_error=max_rel_error,
    )


def compare_to_reference_per_head(
    reference_dir: Path,
    candidate_model_id: str,
    inputs_path: Path,
    device: str = "cpu",
    seed: int = 0,
    tolerances: dict[str, TapTolerance] | None = None,
    allow_fingerprint_mismatch: bool = False,
    max_rel_error: float | None = None,
    candidate_dtype: str | None = None,
    runner: object | None = None,
):
    """Like :func:`compare_to_reference` but also returns per-head attribution.

    Returns ``(divergences, per_head_attributions)``. The candidate runs once
    and both the per-tap diff (the gate) and the per-head drill-down (purely
    diagnostic) are computed from the same capture. ``per_head_attributions``
    is empty when the reference has no per-head taps.
    """
    from firefly.head_attribution import attribute_divergent_heads

    manifest, ref_tensors, candidate_tensors, tolerances = _run_candidate(
        reference_dir,
        candidate_model_id,
        inputs_path,
        device,
        seed,
        tolerances,
        allow_fingerprint_mismatch,
        candidate_dtype,
        runner,
    )
    divergences = diff_captures(
        reference_tensors=ref_tensors,
        candidate_tensors=candidate_tensors,
        tap_order=manifest.tap_points,
        tolerances=tolerances,
        max_rel_error=max_rel_error,
    )
    # Order the per-head taps by the manifest's forward-ordered tap list, not
    # by head_counts dict order (which is lexical after JSON sort_keys). Keeps
    # the per-head table consistent with the forward-order divergence table.
    ordered_head_counts = {
        name: manifest.head_counts[name]
        for name in manifest.tap_points
        if name in manifest.head_counts
    }
    per_head = attribute_divergent_heads(
        ref_tensors, candidate_tensors, ordered_head_counts
    )
    return divergences, per_head
