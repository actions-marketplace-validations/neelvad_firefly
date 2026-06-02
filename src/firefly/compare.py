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

from dataclasses import dataclass
from pathlib import Path

import torch

from firefly.capture import (
    fingerprint_model,
    load_golden_inputs,
    load_model_and_tokenizer,
    run_capture,
)
from firefly.determinism import set_deterministic
from firefly.reference import read_reference

DEFAULT_TOLERANCE = 1e-5


class FingerprintMismatchError(ValueError):
    """Candidate's actual weights don't match what the reference recorded.

    This usually means the model behind ``candidate_model_id`` has been
    retrained, fine-tuned, or swapped since the reference was captured.
    Continuing would compare against an unintended model and silently
    produce wrong results, so we raise unless the caller opts in.
    """


@dataclass
class TapDivergence:
    tap_name: str
    max_abs_diff: float
    mean_abs_diff: float
    tolerance: float
    exceeds_tolerance: bool


def diff_captures(
    reference_tensors: dict[str, torch.Tensor],
    candidate_tensors: dict[str, torch.Tensor],
    tap_order: list[str],
    tolerances: dict[str, float] | None = None,
) -> list[TapDivergence]:
    """Diff two captures in the given tap order.

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
        tol = tolerances.get(tap_name, DEFAULT_TOLERANCE)
        divergences.append(
            TapDivergence(
                tap_name=tap_name,
                max_abs_diff=max_d,
                mean_abs_diff=mean_d,
                tolerance=tol,
                exceeds_tolerance=max_d > tol,
            )
        )

    return divergences


def compare_to_reference(
    reference_dir: Path,
    candidate_model_id: str,
    inputs_path: Path,
    device: str = "cpu",
    seed: int = 0,
    tolerances: dict[str, float] | None = None,
    allow_fingerprint_mismatch: bool = False,
) -> list[TapDivergence]:
    """Run candidate, diff against reference, return per-tap divergences in forward order.

    Raises :class:`FingerprintMismatchError` if the candidate's fingerprint
    doesn't match the reference manifest, unless ``allow_fingerprint_mismatch``
    is set.
    """
    manifest, ref_tensors = read_reference(reference_dir)

    set_deterministic(seed=seed)
    candidate, tokenizer = load_model_and_tokenizer(candidate_model_id, device=device)

    candidate_fp = fingerprint_model(candidate)
    if candidate_fp != manifest.model_fingerprint and not allow_fingerprint_mismatch:
        raise FingerprintMismatchError(
            f"Candidate fingerprint differs from reference manifest.\n"
            f"  reference model_id:    {manifest.model_id}\n"
            f"  reference fingerprint: {manifest.model_fingerprint}\n"
            f"  candidate model_id:    {candidate_model_id}\n"
            f"  candidate fingerprint: {candidate_fp}\n"
            f"\n"
            f"The reference was captured against a different version of this model. "
            f"Either re-capture with `firefly capture`, or pass "
            f"--allow-fingerprint-mismatch to proceed anyway."
        )

    batch = load_golden_inputs(inputs_path, tokenizer, device)
    candidate_tensors = run_capture(candidate, batch, domain=manifest.domain)

    return diff_captures(
        reference_tensors=ref_tensors,
        candidate_tensors=candidate_tensors,
        tap_order=manifest.tap_points,
        tolerances=tolerances,
    )
