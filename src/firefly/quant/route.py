"""Deterministic router: a diagnosis → a concrete recipe.

The non-LLM "agent". It sends each detected failure-mode signature to the
intervention that treats it (the ``diagnose.SIGNATURE_TREATMENTS`` table made
concrete as a recipe). Reproducible, no hallucination surface — and it occupies
the same ``(diagnosis) -> recipe`` slot an LLM proposer would later plug into, so
the harness around it is agent-agnostic.

Routing rules (each gated on the scheme the technique applies to):
  * SINGLE_UNIT_DOMINANCE  → keep that unit in fp (mixed precision).
  * ACTIVATION_OUTLIERS    → SmoothQuant pre-transform   (w8a8 only — it treats
                             activation quant; no-op for weight-only int4).
  * SALIENT_WEIGHT_CHANNELS → AWQ quantizer              (int4 only).
Anything not triggered falls back to plain RTN at the target scheme.
"""

from __future__ import annotations

from firefly.quant.awq import AWQQuantizer
from firefly.quant.diagnose import Diagnosis
from firefly.quant.intervention import (
    ACTIVATION_OUTLIERS,
    SALIENT_WEIGHT_CHANNELS,
    SINGLE_UNIT_DOMINANCE,
    RTNQuantizer,
)
from firefly.quant.recipe_io import Recipe, build_recipe
from firefly.quant.smoothquant import SmoothQuant


def route_recipe(
    diagnosis: Diagnosis,
    *,
    model_id: str,
    scheme: str,
    group_size: int,
    all_fqns: set[str],
    unit_fqns: dict[str, list[str]],
    inputs_path,
    dtype: str = "float32",
    device: str = "cpu",
) -> Recipe:
    """Turn a :class:`Diagnosis` into a deployable :class:`Recipe`."""
    keep_fp: set[str] = set()
    rationale: list[str] = []

    for f in diagnosis.by_signature(SINGLE_UNIT_DOMINANCE):
        keep_fp.update(unit_fqns.get(f.location, []))
        rationale.append(f"keep {f.location} fp (single_unit_dominance)")

    pre_transforms: list = []
    quantizer = RTNQuantizer()

    if scheme == "w8a8" and diagnosis.by_signature(ACTIVATION_OUTLIERS):
        pre_transforms = [SmoothQuant()]
        rationale.append("SmoothQuant pre-transform (activation_outliers)")

    if scheme == "int4wo" and diagnosis.by_signature(SALIENT_WEIGHT_CHANNELS):
        quantizer = AWQQuantizer(group_size=group_size)
        rationale.append("AWQ quantizer (salient_weight_channels)")

    if not rationale:
        rationale.append(f"no actionable signature for {scheme}; plain RTN")

    return build_recipe(
        model_id=model_id, scheme=scheme, group_size=group_size, granularity="layer",
        quantize_fqns=set(all_fqns) - keep_fp, kept_fp_fqns=keep_fp,
        pre_transforms=pre_transforms, quantizer=quantizer,
        dtype=dtype, device=device, inputs_path=inputs_path,
        result={"routing": rationale},
    )
