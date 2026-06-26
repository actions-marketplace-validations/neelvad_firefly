"""Close the loop: a chosen recipe → a portable checkpoint a serving engine loads.

The compression loop produces a :class:`~firefly.quant.recipe_io.Recipe` (the
right *internal* object) plus measured numbers. This module turns the deployable
ones into the thing a user actually serves: a **compressed-tensors** checkpoint
directory + a ``vllm serve`` command + a provenance/serving manifest, so the
flow is apply → save → ``vllm serve`` → (re-)benchmark, not "here's a recipe,
good luck."

**Why compressed-tensors, not torchao.** torchao is Firefly's *measurement*
backend (it has the per-layer filter and the activation hooks the diagnosis
loop needs). But torchao's quantized tensor subclasses don't serialize through
transformers' ``save_pretrained`` in the current stack (GPU-confirmed: both the
w8a8 ``LinearActivationQuantizedTensor`` and the plain weight-only
``AffineQuantizedTensor`` raise "Unsupported tensor type" on save). So
*deployment* goes through `llm-compressor`, which writes the vLLM-native
compressed-tensors format — portable, and it serializes w8a8 / int8 / int4
cleanly. A recipe is backend-agnostic (a scheme + which layers); torchao
measures it, compressed-tensors serves it. The two are the *same scheme*, not
bit-identical implementations — so the exported artifact is **re-benchmarked**
(and should be re-evaluated for quality), never assumed to transfer.

**Honesty boundary.** A *uniform* RTN scheme (w8a8 / int8wo / int4wo, every
quantizable Linear at one precision) maps directly to a compressed-tensors
preset and serves. SmoothQuant (a torchao runtime pre-transform), per-layer
mixed precision, and AWQ each need more wiring (llm-compressor has its own
SmoothQuant/AWQ/ignore-list machinery — a follow-up); they're reported, not
faked. :func:`classify_recipe` names the bucket; :func:`export_deployable`
exports the deployable one and refuses the rest with guidance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from firefly.quant.recipe_io import Recipe

#: Recipe deployability buckets.
DIRECTLY_DEPLOYABLE = "directly_deployable"  # uniform RTN → compressed-tensors → vLLM
NEEDS_FOLDING = "needs_folding"              # SmoothQuant: fold scale first
NOT_YET = "not_yet"                          # mixed precision / AWQ: more wiring

#: Firefly scheme → compressed-tensors preset. W8A16 / W4A16 are weight-only
#: (RTN, no calibration); W8A8 adds dynamic per-token activation quant (also no
#: calibration). All three serialize to compressed-tensors and load in vLLM.
_COMPRESSED_TENSORS_SCHEME = {
    "w8a8": "W8A8",
    "int8wo": "W8A16",
    "int4wo": "W4A16",
}


class DeployabilityError(RuntimeError):
    """A recipe that can't be exported to a directly-loadable checkpoint yet,
    raised with the reason + what would make it deployable."""


def classify_recipe(recipe: Recipe) -> tuple[str, str]:
    """``(status, human reason)`` — which deployability bucket the recipe is in.

    Order matters: a recipe can trip several conditions; we report the one the
    user must resolve first (pre-transform folding before mixed-precision, etc.).
    """
    pre_names = [p.get("name") for p in recipe.pre_transforms]
    if "smoothquant" in pre_names:
        return NEEDS_FOLDING, (
            "recipe uses SmoothQuant (a torchao runtime pre-transform); to serve, "
            "use llm-compressor's own SmoothQuant modifier in the export — not yet "
            "wired. The plain quantized recipe is deployable."
        )
    quant_name = recipe.quantizer.get("name")
    if quant_name == "awq":
        return NOT_YET, (
            "AWQ needs llm-compressor's AWQModifier in the export path (a separate "
            "wiring); the path here covers uniform RTN w8a8 / int8wo / int4wo."
        )
    if recipe.kept_fp_fqns:
        return NOT_YET, (
            f"recipe keeps {len(recipe.kept_fp_fqns)} Linear(s) in fp (per-layer "
            "mixed precision); express this as a compressed-tensors ignore-list in "
            "the export — not yet wired. Serve a uniform scheme for now."
        )
    if recipe.scheme in _COMPRESSED_TENSORS_SCHEME and quant_name == "rtn":
        return DIRECTLY_DEPLOYABLE, (
            f"uniform {recipe.scheme} (RTN) → compressed-tensors "
            f"{_COMPRESSED_TENSORS_SCHEME[recipe.scheme]}, loads in vLLM."
        )
    return NOT_YET, (
        f"no deployable path for scheme={recipe.scheme!r} quantizer={quant_name!r}."
    )


@dataclass
class DeployArtifact:
    """A served-ready quantized checkpoint and how to serve it."""

    path: Path
    scheme: str
    compressed_tensors_scheme: str
    serve_command: str
    manifest: dict = field(default_factory=dict)


def serve_command(path: Path, *, max_model_len: int | None = None) -> str:
    """The ``vllm serve`` invocation for a compressed-tensors checkpoint.

    No ``--quantization`` flag: vLLM auto-detects compressed-tensors from the
    checkpoint's ``config.json`` quantization_config.
    """
    parts = [f"vllm serve {path}"]
    if max_model_len is not None:
        parts.append(f"--max-model-len {max_model_len}")
    return " ".join(parts)


def export_deployable(
    recipe: Recipe,
    out_dir: str | Path,
    *,
    max_model_len: int | None = None,
    measured: dict | None = None,
) -> DeployArtifact:
    """Export ``recipe`` to a vLLM-loadable compressed-tensors checkpoint.

    Only the :data:`DIRECTLY_DEPLOYABLE` bucket is exported; anything else raises
    :class:`DeployabilityError` with the reason from :func:`classify_recipe`
    (rather than writing a checkpoint that serves something we didn't measure).

    Quantization is one-shot RTN via llm-compressor's ``QuantizationModifier``
    (weight-only W8A16/W4A16, or W8A8 with dynamic activations) — no calibration
    set required for these presets. ``lm_head`` is left in fp (standard). The
    output dir holds the packed safetensors + a ``firefly_serving.json`` manifest
    recording the scheme, the serve command, recipe provenance, and any
    ``measured`` numbers.
    """
    status, reason = classify_recipe(recipe)
    if status != DIRECTLY_DEPLOYABLE:
        raise DeployabilityError(f"{recipe.model_id}: {reason}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ct_scheme = _COMPRESSED_TENSORS_SCHEME[recipe.scheme]

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    modifier = QuantizationModifier(targets="Linear", scheme=ct_scheme, ignore=["lm_head"])
    oneshot(model=recipe.model_id, recipe=modifier, output_dir=str(out))

    cmd = serve_command(out, max_model_len=max_model_len)
    manifest = {
        "firefly_serving_version": 1,
        "model_id": recipe.model_id,
        "scheme": recipe.scheme,
        "compressed_tensors_scheme": ct_scheme,
        "quantization_backend": "llm-compressor / compressed-tensors",
        "measurement_backend": "torchao",
        "serve_command": cmd,
        "recipe_provenance": recipe.provenance,
        "measured": measured or {},
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "firefly_serving.json").write_text(json.dumps(manifest, indent=2))
    return DeployArtifact(
        path=out, scheme=recipe.scheme, compressed_tensors_scheme=ct_scheme,
        serve_command=cmd, manifest=manifest,
    )
