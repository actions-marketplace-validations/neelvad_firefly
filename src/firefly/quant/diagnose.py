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
* ``SALIENT_WEIGHT_CHANNELS`` — from the weight-salience sensor (|W|·|X| per
  channel concentration); routes to AWQ. Added once both its detector
  (firefly.quant.salience) and treatment (firefly.quant.awq) existed.

GPTQ's diffuse-weight-loss case is still out: justified in weight-space (Hessian),
which a forward-pass capture can't measure. We do **not** emit labels for
detectors/treatments that don't exist — see ``intervention.py``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import torch

from firefly.quant.intervention import (
    ACTIVATION_OUTLIERS,
    SALIENT_WEIGHT_CHANNELS,
    SINGLE_UNIT_DOMINANCE,
)
from firefly.quant.risk import analyze_quant_risk

#: Which intervention treats each detectable signature (the routing table).
SIGNATURE_TREATMENTS: dict[str, str] = {
    ACTIVATION_OUTLIERS: "smoothquant",
    SINGLE_UNIT_DOMINANCE: "mixed-precision",
    SALIENT_WEIGHT_CHANNELS: "awq",
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


def diagnose_salient_weight_channels(
    saliences, *, concentration_threshold: float = 50.0, top_n: int = 5
) -> list[Finding]:
    """SALIENT_WEIGHT_CHANNELS from the weight-salience sensor: Linears whose
    per-input-channel salience (|W|·|X|) is concentrated in a few channels — int4
    round-to-nearest blurs those salient weights, and AWQ protects them. ``saliences``
    is a list of :class:`firefly.quant.salience.LinearSalience` (ranked).

    The threshold is a heuristic cut on a ranking signal (concentration spans
    orders of magnitude); the raw values are the real signal an agent reads.
    """
    flagged = [s for s in saliences if s.salience_concentration >= concentration_threshold][:top_n]
    return [
        Finding(
            signature=SALIENT_WEIGHT_CHANNELS,
            location=s.fqn,
            recommend=SIGNATURE_TREATMENTS[SALIENT_WEIGHT_CHANNELS],
            evidence={"salience_concentration": s.salience_concentration, "n_channels": s.n_channels},
            explanation=(
                f"{s.fqn}: weight salience is concentrated in a few input channels "
                f"({s.salience_concentration:.0f}x max/median) — int4 round-to-nearest "
                f"blurs those salient weights. AWQ protects them via per-channel scaling; "
                f"route to the AWQ quantizer (int4)."
            ),
        )
        for s in flagged
    ]


def diagnose(
    tensors: dict[str, torch.Tensor],
    tap_order: list[str],
    *,
    sensitivity=None,
    salience=None,
    bits: int = 8,
    concentration_threshold: float = 8.0,
    per_tensor_err_threshold: float = 0.1,
    ratio_threshold: float = 5.0,
    salience_threshold: float = 50.0,
) -> Diagnosis:
    """Emit all detectable signatures. ACTIVATION_OUTLIERS comes from the stored
    activations (cheap, no model run); SINGLE_UNIT_DOMINANCE is added when a
    sensitivity sweep result is supplied; SALIENT_WEIGHT_CHANNELS (→ AWQ) when a
    weight-salience result is supplied."""
    findings = diagnose_activation_outliers(
        tensors, tap_order, bits=bits,
        concentration_threshold=concentration_threshold,
        per_tensor_err_threshold=per_tensor_err_threshold,
    )
    if sensitivity is not None:
        findings += diagnose_single_unit_dominance(sensitivity, ratio_threshold=ratio_threshold)
    if salience is not None:
        findings += diagnose_salient_weight_channels(salience, concentration_threshold=salience_threshold)
    return Diagnosis(findings)
