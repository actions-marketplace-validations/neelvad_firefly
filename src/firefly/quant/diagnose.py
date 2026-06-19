"""Diagnosis: the sensor that emits failure-mode signatures from measurements.

This is the half that makes the loop real rather than theater — it connects the
*measurement* substrate (quant-risk, sensitivity) to the *actuator* (the
:class:`~firefly.quant.intervention.Intervention` ``treats`` vocabulary). The
agent's v1 is deterministic: diagnose → route to the matching intervention →
verify with ``optimize_to_bar`` → explain from the measured before/after. No LLM,
fully measured.

**Honest coverage.** Only the signatures with a real detector on Firefly's
activation-capture substrate are emitted:

* ``ACTIVATION_OUTLIERS`` — from quant-risk's ``channel_concentration`` + the
  measured per-tensor→per-channel error rescue (the Dettmers outlier signal).
* ``SINGLE_UNIT_DOMINANCE`` — from a sensitivity sweep where one unit's quant
  sensitivity dwarfs the rest.

AWQ's salient-weight-channel signal needs a new weight-side sensor (|W|·|X| per
channel — buildable, not built); GPTQ's diffuse-weight-loss case is justified in
weight-space (Hessian), which a forward-pass capture can't measure. We do **not**
emit labels for detectors that don't exist — see ``intervention.py``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import torch

from firefly.quant.intervention import ACTIVATION_OUTLIERS, SINGLE_UNIT_DOMINANCE
from firefly.quant.risk import analyze_quant_risk

#: Which intervention treats each detectable signature (the routing table).
SIGNATURE_TREATMENTS: dict[str, str] = {
    ACTIVATION_OUTLIERS: "smoothquant",
    SINGLE_UNIT_DOMINANCE: "mixed-precision",
}


@dataclass
class Finding:
    """One diagnosed failure mode: where it is, the measured evidence, the
    intervention it routes to, and a causal explanation built from the numbers."""

    signature: str
    location: str
    recommend: str
    evidence: dict
    explanation: str


@dataclass
class Diagnosis:
    findings: list[Finding] = field(default_factory=list)

    def by_signature(self, signature: str) -> list[Finding]:
        return [f for f in self.findings if f.signature == signature]


def diagnose_activation_outliers(
    tensors: dict[str, torch.Tensor],
    tap_order: list[str],
    *,
    bits: int = 8,
    concentration_threshold: float = 8.0,
    per_tensor_err_threshold: float = 0.1,
) -> list[Finding]:
    """ACTIVATION_OUTLIERS from stored activations (no model run): taps whose
    per-tensor int{bits} error is large *and* concentrated in a few channels —
    the case SmoothQuant fixes. Evidence is quant-risk's measured per-tensor vs
    per-channel error, so the explanation is causal, not asserted."""
    findings: list[Finding] = []
    for r in analyze_quant_risk(tensors, tap_order, bits=bits):
        if r.channel_concentration < concentration_threshold or r.per_tensor_rel_err < per_tensor_err_threshold:
            continue
        findings.append(
            Finding(
                signature=ACTIVATION_OUTLIERS,
                location=r.tap_name,
                recommend=SIGNATURE_TREATMENTS[ACTIVATION_OUTLIERS],
                evidence={
                    "channel_concentration": r.channel_concentration,
                    "per_tensor_rel_err": r.per_tensor_rel_err,
                    "per_channel_rel_err": r.per_channel_rel_err,
                    "mitigation_gain": r.mitigation_gain,
                    "bits": bits,
                },
                explanation=(
                    f"{r.tap_name}: int{bits} per-tensor error {r.per_tensor_rel_err:.0%} is "
                    f"dominated by outlier channels ({r.channel_concentration:.0f}x concentration) "
                    f"— per-channel rescues it to {r.per_channel_rel_err:.1%} "
                    f"({r.mitigation_gain:.0f}x). SmoothQuant migrates those outliers into the "
                    f"weights so per-token activation quant stops crushing the rest; "
                    f"apply --smoothquant and verify against an --accuracy-bar."
                ),
            )
        )
    return findings


def diagnose_single_unit_dominance(sensitivity, *, ratio_threshold: float = 5.0) -> list[Finding]:
    """SINGLE_UNIT_DOMINANCE from a sensitivity sweep: one unit's quant
    sensitivity dwarfs the median — keeping just it in high precision recovers
    most of the loss (mixed precision). ``sensitivity`` is a SensitivityResult."""
    ranked = sensitivity.ranked
    if len(ranked) < 2:
        return []
    top = ranked[0].sensitivity
    median = statistics.median(u.sensitivity for u in ranked)
    if top <= 0 or median <= 0 or top / median < ratio_threshold:
        return []
    ratio = top / median
    return [
        Finding(
            signature=SINGLE_UNIT_DOMINANCE,
            location=ranked[0].unit,
            recommend=SIGNATURE_TREATMENTS[SINGLE_UNIT_DOMINANCE],
            evidence={"top_sensitivity": top, "median_sensitivity": median, "ratio": ratio},
            explanation=(
                f"{ranked[0].unit} dominates quant sensitivity ({top:.1%} vs median "
                f"{median:.1%}, {ratio:.0f}x) — keep it in high precision (mixed precision); "
                f"verify the recipe with --accuracy-bar."
            ),
        )
    ]


def diagnose(
    tensors: dict[str, torch.Tensor],
    tap_order: list[str],
    *,
    sensitivity=None,
    bits: int = 8,
    concentration_threshold: float = 8.0,
    per_tensor_err_threshold: float = 0.1,
    ratio_threshold: float = 5.0,
) -> Diagnosis:
    """Emit all detectable signatures. ACTIVATION_OUTLIERS comes from the stored
    activations (cheap, no model run); SINGLE_UNIT_DOMINANCE is added when a
    sensitivity sweep result is supplied."""
    findings = diagnose_activation_outliers(
        tensors, tap_order, bits=bits,
        concentration_threshold=concentration_threshold,
        per_tensor_err_threshold=per_tensor_err_threshold,
    )
    if sensitivity is not None:
        findings += diagnose_single_unit_dominance(sensitivity, ratio_threshold=ratio_threshold)
    return Diagnosis(findings)
