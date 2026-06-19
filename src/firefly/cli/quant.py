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


@app.command("quant-diagnose")
def quant_diagnose(
    reference: str = typer.Option(
        ..., "--reference", "-r",
        help="Reference artifact (local path or hf://org/repo). Diagnosis runs on "
        "the stored activations — no model run.",
    ),
    bits: int = typer.Option(8, "--bits", help="Quant bit-width to diagnose (8 / 4)."),
    concentration: float = typer.Option(
        8.0, "--concentration",
        help="Min channel-concentration to flag a tap as ACTIVATION_OUTLIERS.",
    ),
    report_json: Path | None = typer.Option(
        None, "--report-json", help="Write the structured findings here."
    ),
) -> None:
    """Diagnose quant failure-mode signatures from stored activations, and route
    each to the intervention that treats it.

    The sensor half of the loop: connects measurements (quant-risk's
    channel_concentration) to the intervention vocabulary, emitting
    ACTIVATION_OUTLIERS findings with a measured, causal explanation and the
    verify command to run next. Coverage is deliberately honest — it only emits
    signatures it can actually detect (see `firefly.quant.diagnose`).
    """
    from dataclasses import asdict

    from firefly.quant.diagnose import diagnose
    from firefly.reference import read_reference
    from firefly.report import render_diagnosis

    resolved_reference = _resolve_or_exit(reference)
    manifest, tensors = read_reference(resolved_reference)
    diagnosis = diagnose(
        tensors, manifest.tap_points, bits=bits, concentration_threshold=concentration
    )
    typer.echo(render_diagnosis(diagnosis, model_id=manifest.model_id))

    if report_json is not None:
        import json

        with report_json.open("w") as f:
            json.dump({"findings": [asdict(x) for x in diagnosis.findings]}, f, indent=2)
        typer.echo(f"Wrote diagnosis to {report_json}")


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
    accuracy_bar: str | None = typer.Option(
        None, "--accuracy-bar",
        help="Switch to the eval-gated path: find the smallest recipe whose REAL "
        "metric (see --eval/--metric) stays within this bar of the fp baseline. "
        "'rel:0.01' = within 1%%; 'abs:0.5' = within 0.5 metric units. Ranks units "
        "by the cheap proxy, spends ~log2(N) real evals to gate acceptance.",
    ),
    eval_set: Path | None = typer.Option(
        None, "--eval",
        help="Held-out eval set for --accuracy-bar (JSON {\"texts\":[...]} or JSONL). "
        "Required for --metric perplexity; callable metrics supply their own data.",
    ),
    metric: str = typer.Option(
        "perplexity", "--metric",
        help="Eval metric for --accuracy-bar: 'perplexity' or a 'module:function' "
        "callable taking (model, tokenizer) → float.",
    ),
    eval_max_length: int = typer.Option(
        512, "--eval-max-length", help="Max tokens per eval text (perplexity)."
    ),
    max_measurements: int = typer.Option(
        0, "--max-measurements",
        help="Budget guard: abort before running if the a-priori measurement count "
        "(known from #units × strategy × k) exceeds this. 0 = no cap. Catches an "
        "accidental O(N·k) greedy/linear run on a big model.",
    ),
    smoothquant: bool = typer.Option(
        False, "--smoothquant/--no-smoothquant",
        help="Apply SmoothQuant (a PRE_TRANSFORM) before quantizing: migrates "
        "activation outliers into the weights so per-token activation quant stops "
        "crushing the other channels. Calibrated on --inputs.",
    ),
    export: Path | None = typer.Option(
        None, "--export",
        help="Write the chosen recipe (exact FQNs, interventions, provenance) to "
        "this recipe.json — reproducible via `firefly quant-apply`.",
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
    from firefly.quant.recipe import RECIPE_STRATEGIES, compute_recipe
    from firefly.quant.sensitivity import GRANULARITIES
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

    budget = max_measurements if max_measurements > 0 else None

    if accuracy_bar is not None:
        _run_accuracy_bar(
            model, inputs, accuracy_bar, eval_set, metric, eval_max_length,
            scheme, group_size, strategy, granularity, device, dtype, budget,
            smoothquant, report_json, export,
        )
        return

    from firefly.quant.cost import BudgetExceededError

    try:
        result = compute_recipe(
            model, inputs, device=device, dtype=dtype, scheme=scheme,
            group_size=group_size, strategy=strategy, granularity=granularity,
            k_values=ks, recovery_target=recovery_target, max_measurements=budget,
            smoothquant=smoothquant,
        )
    except BudgetExceededError as e:
        typer.echo(
            f"{e}. Use a coarser --granularity, fewer --k-values, or raise "
            f"--max-measurements.", err=True,
        )
        raise typer.Exit(2) from e
    except (ImportError, QuantCompatibilityError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e

    typer.echo(render_recipe(result))

    if report_json is not None:
        import json
        from dataclasses import asdict

        frontier, knee = result.frontier_knee_ks()
        payload = {
            "model_id": result.sensitivity.model_id,
            "scheme": result.sensitivity.scheme,
            "strategy": result.sensitivity.strategy,
            "granularity": result.sensitivity.granularity,
            "full_quant_divergence": result.sensitivity.full_quant_divergence,
            "recovery_target": result.recovery_target,
            "recommended_k": result.recommended_k,
            "all_fp_bytes": result.all_fp_bytes,
            "all_quant_bytes": result.all_quant_bytes,
            "pareto_frontier_ks": sorted(frontier),
            "knee_k": knee,
            "curve": [asdict(p) for p in sorted(result.curve, key=lambda p: p.k)],
        }
        with report_json.open("w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Wrote recipe report to {report_json}")

    if export is not None:
        rec = result.recommended_point
        result_dict = (
            {"divergence": rec.output_divergence, "recovery": rec.recovery,
             "memory_bytes": rec.memory_bytes}
            if rec else None
        )
        _export_chosen_recipe(
            kept_units=rec.kept_units if rec else [], unit_fqns=result.unit_fqns,
            model=model, scheme=scheme, group_size=group_size, granularity=granularity,
            dtype=dtype, device=device, inputs=inputs, smoothquant=smoothquant,
            result_dict=result_dict, path=export,
        )


def _export_chosen_recipe(
    *, kept_units: list[str], unit_fqns: dict[str, list[str]], model: str, scheme: str,
    group_size: int, granularity: str, dtype: str, device: str, inputs: Path,
    smoothquant: bool, result_dict: dict | None, path: Path,
) -> None:
    """Build + write the chosen recipe (kept units → exact FQNs) to recipe.json."""
    from firefly.quant.recipe_io import build_recipe
    from firefly.quant.smoothquant import SmoothQuant

    all_fqns = {f for fqns in unit_fqns.values() for f in fqns}
    kept_fqns = {f for u in kept_units for f in unit_fqns.get(u, [])}
    recipe = build_recipe(
        model_id=model, scheme=scheme, group_size=group_size, granularity=granularity,
        quantize_fqns=all_fqns - kept_fqns, kept_fp_fqns=kept_fqns,
        pre_transforms=[SmoothQuant()] if smoothquant else [],
        dtype=dtype, device=device, inputs_path=inputs, result=result_dict,
    )
    recipe.to_json(path)
    typer.echo(f"Wrote recipe to {path} (apply with: firefly quant-apply --recipe {path} ...)")


def _run_accuracy_bar(
    model: str, inputs: Path, accuracy_bar: str, eval_set: Path | None, metric: str,
    eval_max_length: int, scheme: str, group_size: int, strategy: str, granularity: str,
    device: str, dtype: str, max_measurements: int | None, smoothquant: bool,
    report_json: Path | None, export: Path | None,
) -> None:
    """The eval-gated recipe path (``--accuracy-bar``): real metric decides the
    smallest passing recipe. Ranking is single-pass, so greedy doesn't apply."""
    from firefly.quant.bar import optimize_to_bar
    from firefly.quant.cost import BudgetExceededError
    from firefly.quant.evaluate import AccuracyBar, resolve_evaluator
    from firefly.quant.sensitivity import STRATEGIES
    from firefly.quant.torchao import QuantCompatibilityError
    from firefly.report import render_bar_recipe

    if strategy not in STRATEGIES:
        raise typer.BadParameter(
            f"--accuracy-bar ranks with {sorted(STRATEGIES)} (greedy is a search, "
            f"not a ranking); got {strategy!r}",
            param_hint="--strategy",
        )
    try:
        bar = AccuracyBar.parse(accuracy_bar)
        evaluator = resolve_evaluator(metric, eval_set, max_length=eval_max_length)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    try:
        result = optimize_to_bar(
            model, inputs, evaluator, bar, device=device, dtype=dtype, scheme=scheme,
            group_size=group_size, strategy=strategy, granularity=granularity,
            max_measurements=max_measurements, smoothquant=smoothquant,
        )
    except BudgetExceededError as e:
        typer.echo(
            f"{e}. Use a coarser --granularity or raise --max-measurements.", err=True
        )
        raise typer.Exit(2) from e
    except (ImportError, QuantCompatibilityError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e

    typer.echo(render_bar_recipe(result))

    if report_json is not None:
        import json
        from dataclasses import asdict

        frontier, knee = result.frontier_knee_ks()
        payload = {
            "model_id": result.model_id,
            "scheme": result.scheme,
            "granularity": result.granularity,
            "strategy": result.strategy,
            "metric": result.metric_name,
            "higher_is_better": result.higher_is_better,
            "bar": {"mode": result.bar.mode, "value": result.bar.value},
            "baseline_metric": result.baseline_metric,
            "full_quant_metric": result.full_quant_metric,
            "threshold": result.threshold,
            "n_units": result.n_units,
            "chosen_k": result.chosen_k,
            "chosen_metric": result.chosen_metric,
            "chosen_kept_units": result.chosen_kept_units,
            "chosen_memory_bytes": result.chosen_memory_bytes,
            "all_fp_bytes": result.all_fp_bytes,
            "all_quant_bytes": result.all_quant_bytes,
            "pareto_frontier_ks": sorted(frontier),
            "knee_k": knee,
            "evals_used": result.evals_used,
            "evaluated": [asdict(p) for p in result.evaluated],
        }
        with report_json.open("w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Wrote recipe report to {report_json}")

    if export is not None:
        _export_chosen_recipe(
            kept_units=result.chosen_kept_units, unit_fqns=result.unit_fqns,
            model=model, scheme=scheme, group_size=group_size, granularity=granularity,
            dtype=dtype, device=device, inputs=inputs, smoothquant=smoothquant,
            result_dict={
                "metric": result.metric_name, "value": result.chosen_metric,
                "threshold": result.threshold, "memory_bytes": result.chosen_memory_bytes,
            },
            path=export,
        )


@app.command("quant-apply")
def quant_apply(
    recipe_path: Path = typer.Option(..., "--recipe", help="recipe.json from `quant-recipe --export`."),
    model: str = typer.Option(..., "--model", "-m", help="HF model ID or checkpoint path."),
    inputs: Path = typer.Option(
        ..., "--inputs", "-i",
        help="Calibration inputs for re-applying pre-transforms (SmoothQuant). Use the "
        "same set the recipe was built from — a mismatch is warned (its hash is recorded).",
    ),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward passes."),
    out: Path | None = typer.Option(
        None, "--out",
        help="Save the quantized state_dict here (best-effort; load it with torchao imported).",
    ),
) -> None:
    """Reconstruct a recipe and apply it to a model: rebuilds the pipeline
    (pre-transforms + quantizer) from recipe.json, runs it, and **reproduces** the
    recipe's recorded output divergence as a verification. Optionally saves the
    quantized weights."""
    import copy

    from firefly.capture import (
        load_golden_inputs,
        load_model_and_tokenizer,
        parse_dtype,
        run_capture,
    )
    from firefly.determinism import set_deterministic
    from firefly.quant.recipe_io import Recipe, apply_recipe, file_sha256
    from firefly.quant.torchao import QuantCompatibilityError, quant_preflight, rel_l1

    rec = Recipe.from_json(recipe_path)
    if file_sha256(inputs) != rec.provenance.get("inputs_sha256"):
        typer.echo(
            "warning: --inputs differs from the recipe's calibration set; any "
            "pre-transform (SmoothQuant) scales will differ from the original.",
            err=True,
        )
    try:
        quant_preflight(rec.scheme, device)
    except QuantCompatibilityError as e:
        typer.echo(f"Incompatible quantization config: {e}", err=True)
        raise typer.Exit(2) from e

    set_deterministic()
    dtype = rec.provenance.get("dtype", "float32")
    fp_model, tok = load_model_and_tokenizer(model, device=device, dtype=parse_dtype(dtype))
    batch = load_golden_inputs(inputs, tok, device)
    ref = run_capture(fp_model, batch)["final_norm"]

    quantized = apply_recipe(rec, copy.deepcopy(fp_model), batch)
    div = rel_l1(ref, run_capture(quantized, batch)["final_norm"])

    typer.echo(
        f"applied recipe: {len(rec.quantize_fqns)} FQNs quantized ({rec.scheme}), "
        f"{len(rec.kept_fp_fqns)} kept fp"
        + (f", pre-transforms: {[p['name'] for p in rec.pre_transforms]}" if rec.pre_transforms else "")
    )
    line = f"reproduction: output divergence vs fp = {div:.4%}"
    if rec.result and "divergence" in rec.result:
        line += f"  (recipe recorded {rec.result['divergence']:.4%})"
    typer.echo(line)

    if out is not None:
        import torch

        torch.save(quantized.state_dict(), out)
        typer.echo(f"saved quantized state_dict to {out} (load with torchao imported)")
