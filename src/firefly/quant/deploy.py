"""Close the loop: a chosen recipe → a checkpoint a serving engine can load.

The compression loop produces a :class:`~firefly.quant.recipe_io.Recipe` (the
right *internal* object) plus measured numbers. This module turns the
deployable ones into the thing a user actually serves: a quantized checkpoint
directory + a ``vllm serve`` command + a provenance/serving manifest, so the
flow is apply → save → ``vllm serve`` → (re-)benchmark, not "here's a recipe,
good luck."

**Honesty boundary.** Not every recipe is loadable by a serving engine, and
faking a checkpoint that silently serves something other than what we measured
would betray the whole thesis. A *uniform weight-only* torchao checkpoint
(int8wo / int4wo, every quantizable Linear at one precision) loads via vLLM's
``quantization="torchao"`` and is faithful to our RTN quantizer (same torchao
call). But:

* **w8a8** (int8 *dynamic-activation*) produces a ``LinearActivationQuantizedTensor``
  that transformers' ``save_pretrained`` can't serialize in the current
  torchao/transformers stack (GPU-confirmed). It stays a measurement/recovery
  scheme; for *serving*, deploy a weight-only scheme (this is also where the
  decode-throughput win lives — weight quant helps the memory-bound regime).
* **SmoothQuant** applies a runtime input scale (a forward hook) — to serve, its
  scale must be *folded* into the producing layer so the checkpoint is a plain
  rescaled model. Not yet automated → reported, not faked.
* **Per-layer mixed precision** (some Linears kept fp) isn't something vLLM's
  torchao loader reconstructs.
* **AWQ** needs the AutoAWQ checkpoint format, a separate exporter.

:func:`classify_recipe` names which bucket a recipe is in; :func:`export_deployable`
exports the deployable bucket and refuses the rest with guidance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from firefly.quant.recipe_io import Recipe

#: Recipe deployability buckets.
DIRECTLY_DEPLOYABLE = "directly_deployable"  # uniform torchao → vLLM today
NEEDS_FOLDING = "needs_folding"              # SmoothQuant: fold scale, then deployable
NOT_YET = "not_yet"                          # mixed precision / AWQ: separate exporter

#: Weight-only schemes that serialize through save_pretrained AND load in vLLM.
#: w8a8 (dynamic-activation) is deliberately absent — its tensor subclass won't
#: serialize (GPU-confirmed), so it can't be exported to a servable checkpoint.
#: Values are the transformers TorchAoConfig string quant_type fallback names.
_TORCHAO_QUANT_TYPE = {
    "int8wo": "int8_weight_only",
    "int4wo": "int4_weight_only",
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
            "recipe uses SmoothQuant, whose runtime input scale must be folded "
            "into the producing layer before it can serve as a plain checkpoint "
            "(scale-folding export is not yet wired)."
        )
    quant_name = recipe.quantizer.get("name")
    if quant_name == "awq":
        return NOT_YET, (
            "AWQ checkpoints need the AutoAWQ export format, a separate exporter "
            "(the torchao→vLLM path here covers uniform RTN int8wo / int4wo)."
        )
    if recipe.kept_fp_fqns:
        return NOT_YET, (
            f"recipe keeps {len(recipe.kept_fp_fqns)} Linear(s) in fp (per-layer "
            "mixed precision), which vLLM's torchao loader doesn't reconstruct; "
            "serve a uniform scheme, or wait for the mixed-precision exporter."
        )
    if recipe.scheme == "w8a8":
        return NOT_YET, (
            "w8a8 (int8 dynamic-activation) produces a LinearActivationQuantizedTensor "
            "that save_pretrained can't serialize in this torchao/transformers stack; "
            "deploy a weight-only scheme for serving (int8wo for ~2x robust, int4wo "
            "for ~4x) — w8a8 stays available for measurement/recovery analysis."
        )
    if recipe.scheme in _TORCHAO_QUANT_TYPE and quant_name == "rtn":
        return DIRECTLY_DEPLOYABLE, (
            f"uniform {recipe.scheme} weight-only (RTN) — loads via vLLM quantization=torchao."
        )
    return NOT_YET, (
        f"no deployable path for scheme={recipe.scheme!r} quantizer={quant_name!r}."
    )


@dataclass
class DeployArtifact:
    """A served-ready quantized checkpoint and how to serve it."""

    path: Path
    scheme: str
    group_size: int
    serve_command: str
    manifest: dict = field(default_factory=dict)


def _build_torchao_config(scheme: str, group_size: int):
    """A transformers ``TorchAoConfig`` for ``scheme``.

    Prefer wrapping Firefly's own torchao config object (so int4 keeps the
    ``tile_packed_to_4d`` packing that actually runs on our GPUs — the plain
    packing needs the 'mslk' kernel lib we don't ship); fall back to the string
    quant_type on transformers versions that only accept a name.
    """
    from transformers import TorchAoConfig

    from firefly.quant.torchao import _quant_config

    try:
        return TorchAoConfig(quant_type=_quant_config(scheme, group_size=group_size))
    except (TypeError, ValueError):
        qt = _TORCHAO_QUANT_TYPE[scheme]
        if scheme == "int4wo":
            return TorchAoConfig(quant_type=qt, group_size=group_size)
        return TorchAoConfig(quant_type=qt)


def serve_command(path: Path, *, max_model_len: int | None = None) -> str:
    """The ``vllm serve`` invocation for a torchao checkpoint."""
    parts = [f"vllm serve {path}", "--quantization torchao"]
    if max_model_len is not None:
        parts.append(f"--max-model-len {max_model_len}")
    return " ".join(parts)


def export_deployable(
    recipe: Recipe,
    out_dir: str | Path,
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
    max_model_len: int | None = None,
    measured: dict | None = None,
) -> DeployArtifact:
    """Export ``recipe`` to a vLLM-loadable quantized checkpoint at ``out_dir``.

    Only the :data:`DIRECTLY_DEPLOYABLE` bucket is exported; anything else raises
    :class:`DeployabilityError` with the reason from :func:`classify_recipe`
    (rather than writing a checkpoint that serves something we didn't measure).

    The export goes through transformers' ``TorchAoConfig`` quantizer so the
    saved directory is the canonical torchao format vLLM reconstructs — faithful
    to our RTN quantizer for a uniform scheme (same torchao call). ``measured``
    (e.g. a benchmark result dict) is recorded in the manifest for provenance.
    """
    status, reason = classify_recipe(recipe)
    if status != DIRECTLY_DEPLOYABLE:
        raise DeployabilityError(f"{recipe.model_id}: {reason}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from firefly.capture import parse_dtype

    quant_config = _build_torchao_config(recipe.scheme, recipe.group_size)
    model = AutoModelForCausalLM.from_pretrained(
        recipe.model_id,
        torch_dtype=parse_dtype(dtype),
        device_map=device,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    # torchao tensor subclasses don't go through safetensors — save as .bin.
    model.save_pretrained(out, safe_serialization=False)
    AutoTokenizer.from_pretrained(recipe.model_id, trust_remote_code=True).save_pretrained(out)

    cmd = serve_command(out, max_model_len=max_model_len)
    manifest = {
        "firefly_serving_version": 1,
        "model_id": recipe.model_id,
        "scheme": recipe.scheme,
        "group_size": recipe.group_size,
        "quantization": "torchao",
        "serve_command": cmd,
        "dtype": dtype,
        "recipe_provenance": recipe.provenance,
        "measured": measured or {},
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "firefly_serving.json").write_text(json.dumps(manifest, indent=2))
    return DeployArtifact(
        path=out, scheme=recipe.scheme, group_size=recipe.group_size,
        serve_command=cmd, manifest=manifest,
    )
