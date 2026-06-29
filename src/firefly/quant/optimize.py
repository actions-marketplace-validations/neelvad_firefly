"""The end-to-end `optimize`: model + bar → a faster servable model + evidence.

Gap 3 of the adoption arc — the one call that ties the whole pipeline together
so a stranger can run it: **select → export → benchmark → report**.

* **select** — :func:`firefly.quant.auto.auto_quant` diagnoses the model, routes
  the diagnosis to a recipe, and measurement-gates it against plain quant.
* **choose what to ship** — the deployable recipe. Today that's the uniform
  scheme (compressed-tensors serves it). If the *quality-optimal* routed recipe
  (e.g. SmoothQuant) isn't directly servable yet, we ship the uniform scheme and
  report the routed recipe's extra recovery as **headroom**, honestly — never a
  recipe we can't actually serve.
* **export** — :func:`firefly.quant.deploy.export_deployable` writes the portable
  compressed-tensors checkpoint + serve command.
* **benchmark** — optionally measure the *served artifact*'s real QPS/memory
  (the cost the analytic model can't predict) and fold it into the manifest.

Thin on purpose: it composes existing library functions and makes the
ship/quality-bar/headroom decisions. The heavy steps (load, quantize, serve)
each need their own resources; this orchestrates them and frees between.
"""

from __future__ import annotations

from pathlib import Path

from firefly.quant.auto import auto_quant
from firefly.quant.cost import SCHEME_WEIGHT_BITS, dtype_bits
from firefly.quant.deploy import (
    DIRECTLY_DEPLOYABLE,
    classify_recipe,
    evaluate_deployed,
    export_deployable,
)
from firefly.quant.intervention import RTNQuantizer
from firefly.quant.recipe_io import Recipe, serialize_intervention


def _uniform_recipe(model_id: str, scheme: str, group_size: int) -> Recipe:
    """The plain uniform-RTN recipe for ``scheme`` — always deployable
    (compressed-tensors quantizes every Linear via targets='Linear', so the
    empty fqn lists are fine; classification only inspects scheme/quantizer)."""
    return Recipe(
        model_id=model_id, scheme=scheme, group_size=group_size, granularity="layer",
        quantize_fqns=[], kept_fp_fqns=[], pre_transforms=[],
        quantizer=serialize_intervention(RTNQuantizer()),
    )


def _free_accelerator() -> None:
    """Release GPU memory between heavy stages (auto_quant's model, the export
    model, the vLLM engine each claim the device)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def choose_ship_recipe(auto: dict, model_id: str, scheme: str, group_size: int) -> tuple[Recipe, str, float]:
    """Pick the recipe to actually deploy from an auto_quant result.

    Ship the routed recipe only if it both *won* the gate AND is directly
    deployable; otherwise ship the uniform scheme. Returns
    ``(recipe, kind, shipped_perplexity)`` where kind is 'routed' | 'plain'.
    """
    routed = auto["recipe_obj"]
    routed_status, _ = classify_recipe(routed)
    if auto["accepted"] and routed_status == DIRECTLY_DEPLOYABLE:
        return routed, "routed", auto["perplexity"]["routed"]
    return _uniform_recipe(model_id, scheme, group_size), "plain", auto["perplexity"]["plain"]


def _headroom(auto: dict) -> dict | None:
    """If a better recipe won the gate but isn't servable yet, describe the
    recovery the user is leaving on the table (and why) — honest, not hidden."""
    routed = auto["recipe_obj"]
    status, reason = classify_recipe(routed)
    if auto["accepted"] and status != DIRECTLY_DEPLOYABLE:
        return {
            "recipe": auto["recipe"]["quantizer"],
            "pre_transforms": auto["recipe"]["pre_transforms"],
            "perplexity": auto["perplexity"]["routed"],
            "recovery": auto["recovery"],
            "blocked_by": reason,
        }
    return None


def optimize(
    model_id: str,
    inputs_path,
    eval_texts: list[str],
    *,
    scheme: str = "w8a8",
    group_size: int = 128,
    device: str = "cpu",
    dtype: str = "float32",
    max_length: int = 64,
    with_sensitivity: bool = False,
    quality_bar: float | None = None,
    out_dir: str | Path | None = None,
    reeval_quality: bool = False,
    benchmark: bool = False,
    bench_config=None,
) -> dict:
    """Run the full select → export → (re-eval) → benchmark loop and return the verdict.

    ``quality_bar`` is the max tolerated relative perplexity increase vs fp
    (``0.05`` = ship must be within 5% of fp). ``out_dir`` exports the deployable
    checkpoint; ``reeval_quality`` re-measures the *served* (compressed-tensors)
    checkpoint's perplexity — the quality we actually ship, vs the torchao one
    selection used — and checks the bar against it; ``benchmark`` additionally
    measures the served artifact's QPS/memory (needs a GPU + vLLM). With none of
    them it's a measured *plan*: the chosen recipe, its quality, the estimated cost.
    """
    auto = auto_quant(
        model_id, inputs_path, eval_texts, scheme=scheme, group_size=group_size,
        device=device, dtype=dtype, max_length=max_length, with_sensitivity=with_sensitivity,
    )
    ship_recipe, ship_kind, ship_ppl = choose_ship_recipe(auto, model_id, scheme, group_size)
    ppl_fp = auto["perplexity"]["fp"]
    rel_to_fp = (ship_ppl - ppl_fp) / ppl_fp if ppl_fp > 0 else 0.0
    meets_bar = quality_bar is None or rel_to_fp <= quality_bar

    # Bits-based weight-compression estimate (uniform scheme); the benchmark, if
    # run, reports the real served weight bytes.
    compression_estimate = dtype_bits(dtype) / SCHEME_WEIGHT_BITS[scheme]

    result = {
        "model": model_id,
        "scheme": scheme,
        "ship": ship_kind,
        # "shipped"/"rel_to_fp" = the torchao (selection) backend; "served*" is
        # filled by the cross-backend re-eval below (the compressed-tensors model
        # we actually deploy). The bar tracks served when we have it.
        "quality": {
            "fp": ppl_fp, "shipped": ship_ppl, "rel_to_fp": rel_to_fp,
            "served": None, "served_rel_to_fp": None, "backend_delta": None,
        },
        "quality_bar": quality_bar,
        "meets_bar": meets_bar,
        "bar_basis": "selection",  # 'served' once re-eval runs
        "compression_estimate": compression_estimate,
        "diagnosis_summary": auto.get("diagnosis_summary", {}),
        "headroom": _headroom(auto),
        "artifact": None,
        "measured": None,
        "auto": auto,
    }

    if out_dir is None:
        return result

    _free_accelerator()
    max_model_len = None if bench_config is None else (bench_config.input_len + bench_config.output_len + 16)
    # A SmoothQuant recipe needs the calibration set at export time (it re-derives
    # the smoothing scales) — feed the same texts the recipe was diagnosed on.
    calib_texts = None
    if any(p.get("name") == "smoothquant" for p in ship_recipe.pre_transforms):
        import json
        from pathlib import Path as _Path

        calib_texts = json.loads(_Path(inputs_path).read_text()).get("texts")
    artifact = export_deployable(
        ship_recipe, out_dir, max_model_len=max_model_len,
        calib_texts=calib_texts, calib_max_length=max_length,
    )
    result["artifact"] = {
        "path": str(artifact.path),
        "compressed_tensors_scheme": artifact.compressed_tensors_scheme,
        "serve_command": artifact.serve_command,
        "treatments": artifact.manifest.get("treatments"),
    }

    if reeval_quality:
        _free_accelerator()
        served = evaluate_deployed(
            artifact.path, eval_texts, max_length=max_length, device=device, dtype=dtype
        )
        served_rel = (served - ppl_fp) / ppl_fp if ppl_fp > 0 else 0.0
        result["quality"].update({
            "served": served,
            "served_rel_to_fp": served_rel,
            "backend_delta": served - ship_ppl,  # compressed-tensors − torchao, same scheme
        })
        # The honest bar: check the model we actually ship.
        result["meets_bar"] = quality_bar is None or served_rel <= quality_bar
        result["bar_basis"] = "served"

    if benchmark:
        _free_accelerator()
        from firefly.bench import get_benchmarker

        b = get_benchmarker("vllm").benchmark(str(artifact.path), bench_config, dtype=dtype, quantization=None)
        measured = {
            "decode_tok_s": b.decode_throughput_tok_s,
            "prefill_tok_s": b.prefill_throughput_tok_s,
            "ttft_ms": b.ttft_ms,
            "weight_mb": (b.weight_memory_bytes / 1e6) if b.weight_memory_bytes else None,
        }
        result["measured"] = measured
        # Fold the measurement into the shipped manifest so the artifact records it.
        import json

        manifest = dict(artifact.manifest)
        manifest["measured"] = measured
        (artifact.path / "firefly_serving.json").write_text(json.dumps(manifest, indent=2))

    return result
