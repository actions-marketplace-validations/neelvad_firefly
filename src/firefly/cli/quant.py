"""Quantization commands: quant-risk, quant-diff, quant-sensitivity, quant-recipe."""

from __future__ import annotations

from pathlib import Path

import typer

from firefly.cli._app import _resolve_or_exit, app


@app.command("quant-risk")
def quant_risk(
    reference: str = typer.Option(
        ...,
        "--reference",
        "-r",
        help=(
            "Reference artifact directory (local path or hf://org/repo). "
            "Analysis uses the activations already stored in the artifact — "
            "no model run needed."
        ),
    ),
    bits: int = typer.Option(
        8,
        "--bits",
        help="Quantization bit-width to simulate (8 for int8, 4 for int4).",
    ),
    threshold: float = typer.Option(
        0.01,
        "--threshold",
        help=(
            "Flag taps whose simulated per-tensor relative error exceeds this "
            "fraction (default 1%). Purely diagnostic — exit code is always 0."
        ),
    ),
    report_json: Path | None = typer.Option(
        None, "--report-json", help="Write the structured per-tap report to this path."
    ),
) -> None:
    """Flag outlier-feature layers sensitive to per-tensor quantization, from stored activations.

    Simulates symmetric round-to-nearest quantization of each tap's captured
    activation (per-tensor and per-channel) and reports the taps where
    per-tensor quantization breaks down — typically outlier-feature layers
    whose few extreme channels force a scale that crushes the rest. This is a
    cheap heuristic diagnostic (no model run); it flags candidates, it does not
    reliably rank real per-layer quant error across models.

    To *measure and attribute* what real torchao quantization actually does to
    a model (vs predict it), use ``firefly quant-diff``.
    """
    import json
    from dataclasses import asdict

    from firefly.quant.risk import analyze_quant_risk
    from firefly.reference import read_reference
    from firefly.report import render_quant_risk

    resolved_reference = _resolve_or_exit(reference)
    manifest, tensors = read_reference(resolved_reference)
    risks = analyze_quant_risk(tensors, manifest.tap_points, bits=bits)
    typer.echo(render_quant_risk(risks, bits=bits, threshold=threshold))

    if report_json is not None:
        payload = {
            "bits": bits,
            "threshold": threshold,
            "taps": [
                {**asdict(r), "mitigation_gain": r.mitigation_gain} for r in risks
            ],
        }
        with report_json.open("w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Wrote quant-risk report to {report_json}")


@app.command("quant-diff")
def quant_diff(
    reference: str = typer.Option(
        ..., "--reference", "-r",
        help="Reference artifact directory (the fp baseline to diff against).",
    ),
    inputs: Path = typer.Option(
        ..., "--inputs", "-i", help="Same golden-inputs JSON used at capture time."
    ),
    candidate: str | None = typer.Option(
        None, "--candidate", "-c",
        help="Model to quantize and diff. Defaults to the reference's model_id "
        "(quantization is a transform of the same model).",
    ),
    scheme: str = typer.Option(
        "w8a8", "--scheme",
        help="torchao quant scheme: w8a8 (int8 dynamic act + int8 weight) or "
        "int4wo (int4 weight-only; needs CUDA).",
    ),
    group_size: int = typer.Option(32, "--group-size", help="int4wo group size."),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward pass."),
    seed: int = typer.Option(0, "--seed", help="Determinism seed."),
    candidate_dtype: str | None = typer.Option(
        None, "--candidate-dtype",
        help="Base dtype to quantize from. Defaults to the reference's dtype, so "
        "the only delta is the quantization (recommended).",
    ),
    top_n: int = typer.Option(15, "--top-n", help="How many worst taps to show."),
    rel_threshold: float = typer.Option(
        0.0, "--rel-threshold",
        help="Exit non-zero if any tap's mean relative divergence exceeds this "
        "(e.g. 0.05 = 5%). 0 = report only, always exit 0.",
    ),
    report_json: Path | None = typer.Option(
        None, "--report-json", help="Write the structured divergence report here."
    ),
    ci_format: str = typer.Option(
        "human", "--ci-format",
        help="Output format: 'human' (rich terminal table) or 'markdown' "
        "(PR-comment-friendly, for $GITHUB_STEP_SUMMARY / gh pr comment).",
    ),
) -> None:
    """Diff a torchao-quantized model against its fp baseline, ranked by divergence.

    Quantization is run as a candidate through the standard capture/compare
    pipeline, then divergences are ranked by *relative* magnitude so the layers
    the quant scheme damaged most surface first — with per-head attribution when
    the reference has per-head taps. This measures (and attributes) what
    quantization did; it does not predict it.
    """
    from firefly.attribution import attribute_first_divergence
    from firefly.compare import compare_to_reference, compare_to_reference_per_head
    from firefly.quant.torchao import (
        QUANT_SCHEMES,
        QuantCompatibilityError,
        quant_preflight,
    )
    from firefly.reference import read_manifest
    from firefly.report import render_quant_diff, render_quant_diff_markdown, write_json
    from firefly.runners import get_runner

    if scheme not in QUANT_SCHEMES:
        raise typer.BadParameter(
            f"--scheme must be one of {QUANT_SCHEMES}, got {scheme!r}",
            param_hint="--scheme",
        )
    if ci_format not in {"human", "markdown"}:
        raise typer.BadParameter(
            f"--ci-format must be 'human' or 'markdown', got {ci_format!r}",
            param_hint="--ci-format",
        )
    # Fast-fail known-incompatible combos (e.g. int4wo on CPU) before the load.
    try:
        quant_preflight(scheme, device)
    except QuantCompatibilityError as e:
        typer.echo(f"Incompatible quantization config: {e}", err=True)
        raise typer.Exit(2) from e

    resolved_reference = _resolve_or_exit(reference)
    manifest = read_manifest(resolved_reference)
    candidate_id = candidate or manifest.model_id
    options = {"quantize": scheme, "group_size": str(group_size)}
    threshold = rel_threshold if rel_threshold > 0 else None

    common = dict(
        reference_dir=resolved_reference,
        candidate_model_id=candidate_id,
        inputs_path=inputs,
        device=device,
        seed=seed,
        allow_fingerprint_mismatch=False,  # pre-quant fingerprint matches baseline
        candidate_dtype=candidate_dtype,
        runner=get_runner("hf"),
        options=options,
    )
    try:
        if manifest.head_counts:
            divergences, per_head = compare_to_reference_per_head(**common)
        else:
            divergences, per_head = compare_to_reference(**common), []
    except (ImportError, QuantCompatibilityError) as e:  # missing extra / bad config
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e

    result = attribute_first_divergence(divergences)
    render = render_quant_diff_markdown if ci_format == "markdown" else render_quant_diff
    typer.echo(
        render(
            result, scheme=scheme, top_n=top_n, per_head=per_head,
            rel_threshold=threshold,
        )
    )
    if report_json is not None:
        write_json(result, report_json, per_head=per_head)

    if threshold is not None and any(d.rel_mean > threshold for d in divergences):
        raise typer.Exit(1)


@app.command("quant-sensitivity")
def quant_sensitivity(
    model: str = typer.Option(..., "--model", "-m", help="HF model ID or checkpoint path."),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Golden-inputs JSON."),
    scheme: str = typer.Option(
        "w8a8", "--scheme", help="torchao quant scheme: w8a8 or int4wo (needs CUDA)."
    ),
    group_size: int = typer.Option(32, "--group-size", help="int4wo group size."),
    strategy: str = typer.Option(
        "isolated", "--strategy",
        help="Sensitivity strategy. 'isolated' (default): quantize one layer at a "
        "time and measure its output divergence alone. More strategies (marginal) "
        "trade compute for resolution.",
    ),
    granularity: str = typer.Option(
        "layer", "--granularity",
        help="Unit of analysis: 'layer' (a decoder layer's Linears together) or "
        "'linear' (each Linear separately — finer recipes, ~7x more units/compute).",
    ),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward passes."),
    dtype: str = typer.Option("float32", "--dtype", help="Base dtype to quantize from."),
    top_n: int = typer.Option(15, "--top-n", help="How many most-sensitive units to show."),
    keep_k: int = typer.Option(4, "--keep-k", help="Suggest keeping this many units high-precision."),
    report_json: Path | None = typer.Option(
        None, "--report-json", help="Write the structured sensitivity report here."
    ),
) -> None:
    """Rank units by how much their quantization hurts the model output.

    Quantizes one unit at a time (the 'isolated' strategy) and measures the
    resulting divergence at the output, so you can see *which* units to keep in
    higher precision — the attribution that guides mixed-precision quantization.
    Runs N+1 forwards for N units; use a small model or --device cuda (and
    --granularity linear multiplies the unit count ~7x).
    """
    from dataclasses import asdict

    from firefly.quant.sensitivity import GRANULARITIES, STRATEGIES, compute_sensitivity
    from firefly.quant.torchao import QuantCompatibilityError, quant_preflight
    from firefly.report import render_sensitivity

    if strategy not in STRATEGIES:
        raise typer.BadParameter(
            f"--strategy must be one of {sorted(STRATEGIES)}, got {strategy!r}",
            param_hint="--strategy",
        )
    if granularity not in GRANULARITIES:
        raise typer.BadParameter(
            f"--granularity must be one of {list(GRANULARITIES)}, got {granularity!r}",
            param_hint="--granularity",
        )
    try:
        quant_preflight(scheme, device)
    except QuantCompatibilityError as e:
        typer.echo(f"Incompatible quantization config: {e}", err=True)
        raise typer.Exit(2) from e

    try:
        result = compute_sensitivity(
            model, inputs, device=device, dtype=dtype, scheme=scheme,
            group_size=group_size, strategy=strategy, granularity=granularity,
        )
    except (ImportError, QuantCompatibilityError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e

    typer.echo(render_sensitivity(result, top_n=top_n, keep_k=keep_k))

    if report_json is not None:
        import json

        payload = {
            "model_id": result.model_id,
            "scheme": result.scheme,
            "strategy": result.strategy,
            "granularity": result.granularity,
            "full_quant_divergence": result.full_quant_divergence,
            "output_tap": result.output_tap,
            "keep_high_precision": result.keep_high_precision(keep_k),
            "units": [asdict(x) for x in result.ranked],
        }
        with report_json.open("w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Wrote sensitivity report to {report_json}")


@app.command("quant-recipe")
def quant_recipe(
    model: str = typer.Option(..., "--model", "-m", help="HF model ID or checkpoint path."),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Golden-inputs JSON."),
    scheme: str = typer.Option("w8a8", "--scheme", help="torchao scheme: w8a8 or int4wo."),
    group_size: int = typer.Option(32, "--group-size", help="int4wo group size."),
    strategy: str = typer.Option(
        "isolated", "--strategy",
        help="Recipe strategy: isolated/marginal (single-pass per-layer ranking) "
        "or greedy (sequential forward selection — more compute, accounts for "
        "layer interactions).",
    ),
    granularity: str = typer.Option(
        "layer", "--granularity",
        help="Unit of analysis: 'layer' or 'linear' (finer recipes, ~7x more "
        "units — greedy gets ~7^2x slower).",
    ),
    k_values: str = typer.Option(
        "1,2,4,8", "--k-values",
        help="Comma-separated keep-high-precision counts to evaluate (the curve).",
    ),
    recovery_target: float = typer.Option(
        0.9, "--recovery-target",
        help="Recommend the smallest k recovering at least this fraction of the "
        "all-quantized degradation (default 0.9 = 90%%).",
    ),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward passes."),
    dtype: str = typer.Option("float32", "--dtype", help="Base dtype to quantize from."),
    report_json: Path | None = typer.Option(
        None, "--report-json", help="Write the structured recipe report here."
    ),
) -> None:
    """Build and VERIFY a mixed-precision recipe: keep the most quant-sensitive
    units in high precision, quantize the rest, and measure the recovered output
    fidelity. The curve shows how few high-precision units recover most of the
    fidelity — the attribution-guided answer torchao autoquant can't explain.
    """
    from firefly.quant.sensitivity import GRANULARITIES, RECIPE_STRATEGIES, compute_recipe
    from firefly.quant.torchao import QuantCompatibilityError, quant_preflight
    from firefly.report import render_recipe

    if strategy not in RECIPE_STRATEGIES:
        raise typer.BadParameter(
            f"--strategy must be one of {list(RECIPE_STRATEGIES)}, got {strategy!r}",
            param_hint="--strategy",
        )
    if granularity not in GRANULARITIES:
        raise typer.BadParameter(
            f"--granularity must be one of {list(GRANULARITIES)}, got {granularity!r}",
            param_hint="--granularity",
        )
    try:
        ks = [int(x) for x in k_values.split(",") if x.strip()]
    except ValueError as e:
        raise typer.BadParameter("--k-values must be comma-separated integers", param_hint="--k-values") from e
    try:
        quant_preflight(scheme, device)
    except QuantCompatibilityError as e:
        typer.echo(f"Incompatible quantization config: {e}", err=True)
        raise typer.Exit(2) from e

    try:
        result = compute_recipe(
            model, inputs, device=device, dtype=dtype, scheme=scheme,
            group_size=group_size, strategy=strategy, granularity=granularity,
            k_values=ks, recovery_target=recovery_target,
        )
    except (ImportError, QuantCompatibilityError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e

    typer.echo(render_recipe(result))

    if report_json is not None:
        import json
        from dataclasses import asdict

        payload = {
            "model_id": result.sensitivity.model_id,
            "scheme": result.sensitivity.scheme,
            "strategy": result.sensitivity.strategy,
            "granularity": result.sensitivity.granularity,
            "full_quant_divergence": result.sensitivity.full_quant_divergence,
            "recovery_target": result.recovery_target,
            "recommended_k": result.recommended_k,
            "curve": [asdict(p) for p in sorted(result.curve, key=lambda p: p.k)],
        }
        with report_json.open("w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Wrote recipe report to {report_json}")


