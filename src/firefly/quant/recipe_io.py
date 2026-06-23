"""Deployable recipe artifacts: serialize a chosen recipe, re-apply it.

A recipe *is* a ``(PrecisionPolicy, [Intervention])`` pair plus provenance, so it
serializes cleanly: the policy is primitives, each intervention round-trips
through its ``name`` + ``config()`` via a small registry here (kept out of the
seam — serialization is this module's concern, not the interventions'). The
artifact records the *technique* (e.g. SmoothQuant α/scope), not baked weights;
``apply_recipe`` re-runs it, re-calibrating any pre-transform from the supplied
inputs (whose hash is recorded so a mismatch can be flagged). The output of
``apply`` is the quantized model — the artifact stays small and portable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from torch import nn

from firefly.quant.awq import AWQQuantizer
from firefly.quant.intervention import Intervention, Pipeline, PrecisionPolicy, RTNQuantizer
from firefly.quant.smoothquant import SmoothQuant

#: name -> class for deserialization. New adapters register here.
_REGISTRY: dict[str, type] = {"rtn": RTNQuantizer, "smoothquant": SmoothQuant, "awq": AWQQuantizer}

RECIPE_VERSION = 1


def serialize_intervention(it: Intervention) -> dict:
    return {"name": it.name, "params": it.config()}


def deserialize_intervention(d: dict) -> Intervention:
    try:
        cls = _REGISTRY[d["name"]]
    except KeyError as e:
        raise ValueError(f"unknown intervention {d['name']!r}; known: {sorted(_REGISTRY)}") from e
    return cls(**d.get("params", {}))


@dataclass
class Recipe:
    """A reproducible mixed-precision recipe + provenance."""

    model_id: str
    scheme: str
    group_size: int
    granularity: str
    quantize_fqns: list[str]
    kept_fp_fqns: list[str]
    pre_transforms: list[dict] = field(default_factory=list)
    quantizer: dict = field(default_factory=lambda: serialize_intervention(RTNQuantizer()))
    provenance: dict = field(default_factory=dict)
    result: dict | None = None
    recipe_version: int = RECIPE_VERSION

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> Recipe:
        data = json.loads(Path(path).read_text())
        v = data.get("recipe_version")
        if v != RECIPE_VERSION:
            raise ValueError(f"recipe_version {v} unsupported (this build reads {RECIPE_VERSION})")
        return cls(**data)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build_recipe(
    *,
    model_id: str,
    scheme: str,
    group_size: int,
    granularity: str,
    quantize_fqns: set[str],
    kept_fp_fqns: set[str],
    pre_transforms: list[Intervention],
    dtype: str,
    device: str,
    inputs_path: Path,
    quantizer: Intervention | None = None,
    result: dict | None = None,
) -> Recipe:
    """Assemble a :class:`Recipe` from a chosen recipe + its run context.
    ``quantizer`` defaults to RTN; the router passes AWQ when it routes there."""
    try:
        from importlib.metadata import version

        ver = version("firefly")
    except Exception:  # noqa: BLE001 — provenance is best-effort
        ver = "unknown"
    return Recipe(
        model_id=model_id,
        scheme=scheme,
        group_size=group_size,
        granularity=granularity,
        quantize_fqns=sorted(quantize_fqns),
        kept_fp_fqns=sorted(kept_fp_fqns),
        pre_transforms=[serialize_intervention(it) for it in pre_transforms],
        quantizer=serialize_intervention(quantizer or RTNQuantizer()),
        provenance={
            "dtype": dtype,
            "device": device,
            "inputs": str(inputs_path),
            "inputs_sha256": file_sha256(inputs_path),
            "firefly_version": ver,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        result=result,
    )


def apply_recipe(recipe: Recipe, model: nn.Module, calib: object | None = None) -> nn.Module:
    """Reconstruct the recipe's pipeline and run it on ``model`` (mutated in
    place — pass a fresh copy). ``calib`` feeds any pre-transform that needs
    activation stats (e.g. SmoothQuant); pass the same inputs the recipe was
    built from (its hash is in ``provenance``)."""
    policy = PrecisionPolicy(
        scheme=recipe.scheme, group_size=recipe.group_size, quantize=set(recipe.quantize_fqns)
    )
    pipeline = Pipeline(
        pre_transforms=[deserialize_intervention(d) for d in recipe.pre_transforms],
        quantizer=deserialize_intervention(recipe.quantizer),
    )
    return pipeline.run(model, policy, calib)
