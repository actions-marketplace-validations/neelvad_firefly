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
DIRECTLY_DEPLOYABLE = "directly_deployable"  # RTN (+ SmoothQuant / mixed-precision) → compressed-tensors → vLLM
NOT_YET = "not_yet"                          # AWQ / unmapped pre-transforms: more wiring

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


def export_method(recipe: Recipe) -> str:
    """The llm-compressor weight-quant method this recipe maps to.

    GPU-validated recovery picture (Qwen2.5-1.5B): plain int4 RTN is wrecked
    (+113% perplexity) but **GPTQ recovers it ~96% when served** (AWQ ~94%); int8
    schemes serve fine on RTN; and SmoothQuant is a *no-op* for compressed-tensors
    W8A8 (its recovery was a torchao-measurement artifact). So int4 always uses a
    calibration-based method (AWQ if the recipe routed to it, else GPTQ); int8
    uses RTN; SmoothQuant pre-transforms are dropped (no servable effect).
    """
    if recipe.scheme == "int4wo":
        return "awq" if recipe.quantizer.get("name") == "awq" else "gptq"
    return "rtn"


def classify_recipe(recipe: Recipe) -> tuple[str, str]:
    """``(status, human reason)`` — which deployability bucket the recipe is in.

    Deployable via llm-compressor: int8wo / w8a8 (RTN) and int4wo (GPTQ, or AWQ
    when the recipe routed there), each optionally with kept-fp Linears (an
    ignore-list = mixed precision). int4 needs a calibration set. AWQ is wired for
    int4 only. SmoothQuant pre-transforms are accepted but *dropped* (proven a
    no-op for serving), so they never block or drive deployability.
    """
    quant_name = recipe.quantizer.get("name")
    if quant_name == "awq" and recipe.scheme != "int4wo":
        return NOT_YET, (
            f"AWQ recovery is wired for int4 (W4A16) only, not scheme={recipe.scheme!r}."
        )
    if recipe.scheme not in _COMPRESSED_TENSORS_SCHEME or quant_name not in ("rtn", "awq"):
        return NOT_YET, (
            f"no deployable path for scheme={recipe.scheme!r} quantizer={quant_name!r}."
        )
    method = export_method(recipe).upper()
    extras = []
    if recipe.kept_fp_fqns:
        extras.append(f"{len(recipe.kept_fp_fqns)} fp-kept")
    detail = f" + {', '.join(extras)}" if extras else ""
    return DIRECTLY_DEPLOYABLE, (
        f"{recipe.scheme} ({method}){detail} → compressed-tensors "
        f"{_COMPRESSED_TENSORS_SCHEME[recipe.scheme]}, loads in vLLM."
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


def _build_calib_dataset(model_id: str, texts: list[str], max_length: int):
    """A pre-tokenized HF dataset for llm-compressor's calibration forward pass.

    GPTQ/AWQ derive their corrections from activation stats, so they need
    calibration data — we feed the *same* texts the recipe was diagnosed on
    (faithful to what we measured), pre-tokenized so oneshot doesn't have to
    guess a text column.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    rows = [tok(t, truncation=True, max_length=max_length) for t in texts]
    return Dataset.from_list(
        [{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows]
    )


def _quant_modifier(recipe: Recipe, method: str, ct_scheme: str, ignore: list[str]):
    """The llm-compressor weight-quant modifier for the recipe's method."""
    if method == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier

        return GPTQModifier(targets="Linear", scheme=ct_scheme, ignore=ignore)
    if method == "awq":
        from llmcompressor.modifiers.awq import AWQModifier

        return AWQModifier(targets="Linear", scheme=ct_scheme, ignore=ignore)
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(targets="Linear", scheme=ct_scheme, ignore=ignore)


def export_deployable(
    recipe: Recipe,
    out_dir: str | Path,
    *,
    max_model_len: int | None = None,
    measured: dict | None = None,
    calib_texts: list[str] | None = None,
    calib_max_length: int = 64,
) -> DeployArtifact:
    """Export ``recipe`` to a vLLM-loadable compressed-tensors checkpoint.

    Only the :data:`DIRECTLY_DEPLOYABLE` bucket is exported; anything else raises
    :class:`DeployabilityError` with the reason from :func:`classify_recipe`
    (rather than writing a checkpoint that serves something we didn't measure).

    The weight-quant method is chosen by :func:`export_method` from the
    GPU-validated recovery picture: **int4 → GPTQ** (or AWQ when the recipe routed
    there) — RTN int4 serves at +113% perplexity but GPTQ recovers it ~96%;
    **int8 (w8a8 / int8wo) → RTN** (serves fine). ``lm_head`` plus any kept-fp
    Linears go in the ignore-list (that ignore-list *is* mixed precision). int4
    (GPTQ/AWQ) needs ``calib_texts``. SmoothQuant pre-transforms are **dropped** —
    they're a no-op for compressed-tensors serving (its recovery was a
    torchao-measurement artifact). The output dir holds the packed safetensors +
    a ``firefly_serving.json`` manifest recording the method, serve command, and
    ``measured``.
    """
    status, reason = classify_recipe(recipe)
    if status != DIRECTLY_DEPLOYABLE:
        raise DeployabilityError(f"{recipe.model_id}: {reason}")

    method = export_method(recipe)
    needs_calib = method in ("gptq", "awq")
    if needs_calib and not calib_texts:
        raise DeployabilityError(
            f"{recipe.model_id}: int4 {method.upper()} export needs calibration data — "
            "pass calib_texts (the same set the recipe was diagnosed on)."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ct_scheme = _COMPRESSED_TENSORS_SCHEME[recipe.scheme]
    ignore = ["lm_head", *sorted(recipe.kept_fp_fqns)]
    dropped = [p["name"] for p in recipe.pre_transforms]  # SmoothQuant etc.: no servable effect

    from llmcompressor import oneshot

    pipeline = [_quant_modifier(recipe, method, ct_scheme, ignore)]
    if needs_calib:
        ds = _build_calib_dataset(recipe.model_id, calib_texts, calib_max_length)
        oneshot(
            model=recipe.model_id, recipe=pipeline, dataset=ds,
            num_calibration_samples=len(calib_texts), max_seq_length=calib_max_length,
            output_dir=str(out),
        )
    else:
        oneshot(model=recipe.model_id, recipe=pipeline, output_dir=str(out))

    cmd = serve_command(out, max_model_len=max_model_len)
    treatments = [method] + (["mixed-precision"] if recipe.kept_fp_fqns else [])
    manifest = {
        "dropped_pre_transforms": dropped,
        "firefly_serving_version": 1,
        "model_id": recipe.model_id,
        "scheme": recipe.scheme,
        "compressed_tensors_scheme": ct_scheme,
        "method": method,
        "treatments": treatments,
        "kept_fp_count": len(recipe.kept_fp_fqns),
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


def evaluate_deployed(
    checkpoint_dir: str | Path,
    eval_texts: list[str],
    *,
    max_length: int = 64,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> float:
    """Perplexity of the *served* compressed-tensors checkpoint, via the same
    evaluator selection used on the torchao model.

    The point is the cross-backend check: selection measures quality on a
    *torchao*-quantized model, but we ship a *compressed-tensors* one — the same
    scheme, not a bit-identical implementation. Loading the exported checkpoint
    in transformers (which auto-applies the compressed-tensors config) and
    running the identical perplexity evaluator isolates exactly that handoff, so
    the quality we *report* is the quality we *ship*, not a proxy from a different
    backend.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from firefly.capture import parse_dtype
    from firefly.quant.evaluate import perplexity_evaluator

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir), torch_dtype=parse_dtype(dtype), device_map=device,
    )
    tok = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    return perplexity_evaluator(eval_texts, max_length=max_length)(model, tok)
