"""Parity-gate commands: capture, publish, calibrate, check."""

from __future__ import annotations

from pathlib import Path

import typer

from firefly.cli._app import (
    _parse_runner_opts,
    _publish_or_exit,
    _resolve_or_exit,
    app,
)


@app.command()
def capture(
    model: str = typer.Option(..., "--model", "-m", help="HF model ID or path to a checkpoint."),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Path to a golden-inputs JSON file."),
    out: Path = typer.Option(..., "--out", "-o", help="Directory to write the reference artifact."),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward pass."),
    seed: int = typer.Option(0, "--seed", help="Determinism seed."),
    dtype: str = typer.Option("fp32", "--dtype", help="Model dtype: fp32, bf16, or fp16."),
    per_head: bool = typer.Option(
        False,
        "--per-head",
        help=(
            "Also capture per-head attention taps (the input to each layer's "
            "attention output projection). Enables `firefly check` to attribute "
            "divergence to individual attention heads."
        ),
    ),
    runner: str = typer.Option(
        "hf",
        "--runner",
        help=(
            "Capture backend: 'hf' (transformers, eager hooks, default), "
            "'vllm', or 'sglang' (in-process serving engines; each needs its "
            "extra — `pip install firefly[vllm|sglang]` — and a CUDA GPU). "
            "A reference and its candidates must use the same runner."
        ),
    ),
    runner_opt: list[str] = typer.Option(
        [],
        "--runner-opt",
        help=(
            "Engine-specific knob as key=value (repeatable). HF: "
            "quantize (w8a8/int4wo — apply real torchao quant, needs "
            "firefly[torchao]), group_size (int4 only). vLLM: "
            "attention_backend, engine (v0/v1), max_seq_len, "
            "gpu_memory_utilization, capture_decode, speculative_tokens. "
            "E.g. --runner-opt attention_backend=FLASH_ATTN."
        ),
    ),
    push: str | None = typer.Option(
        None,
        "--push",
        help=(
            "If set, publish the artifact to this URI after capture "
            "(hf://org/repo, s3://bucket/prefix, gs://bucket/prefix, or "
            "az://account/container/prefix). Equivalent to running "
            "`firefly publish` immediately after capture."
        ),
    ),
) -> None:
    """Capture a reference artifact from a model + golden inputs."""
    from firefly.capture import capture_reference, parse_dtype
    from firefly.runners import get_runner

    options = _parse_runner_opts(runner_opt)
    try:
        active_runner = get_runner(runner)
    except (ValueError, NotImplementedError) as e:
        raise typer.BadParameter(str(e), param_hint="--runner") from e

    typer.echo(
        f"Capturing reference: model={model} runner={runner} device={device} "
        f"dtype={dtype} per_head={per_head}"
    )
    capture_reference(
        model_id=model,
        inputs_path=inputs,
        out_dir=out,
        device=device,
        seed=seed,
        dtype=parse_dtype(dtype),
        per_head=per_head,
        runner=active_runner,
        options=options,
    )
    typer.echo(f"Wrote reference artifact to {out}")

    if push is not None:
        typer.echo(f"Publishing {out} → {push}")
        _publish_or_exit(out, push, commit_message=f"Firefly reference capture: {model}")
        typer.echo(f"Published to {push}")


@app.command()
def publish(
    reference: Path = typer.Option(
        ...,
        "--reference",
        "-r",
        help="Local reference artifact directory to upload.",
    ),
    to: str = typer.Option(
        ...,
        "--to",
        "-t",
        help=(
            "Destination URI. Supported: hf://org/repo (optionally with "
            "@revision and /subpath), s3://bucket/prefix, gs://bucket/prefix, "
            "or az://account/container/prefix. HF Hub creates the repo if it "
            "doesn't exist; all backends use ambient credentials (HF_TOKEN for "
            "HF, and each cloud SDK's default credential chain otherwise)."
        ),
    ),
    message: str = typer.Option(
        "Firefly reference upload",
        "--message",
        "-m",
        help="Commit message (HF Hub only; ignored for S3).",
    ),
) -> None:
    """Upload a local reference artifact to HF Hub or S3."""
    typer.echo(f"Publishing {reference} → {to}")
    _publish_or_exit(reference, to, commit_message=message)
    typer.echo(f"Published to {to}")


@app.command()
def calibrate(
    reference: str = typer.Option(
        ...,
        "--reference",
        "-r",
        help=(
            "Reference artifact directory. Accepts a local path or an "
            "hf://org/repo (optionally with @revision and /subpath) "
            "for HF Hub-hosted references, or a cloud URI: s3://bucket/"
            "prefix, gs://bucket/prefix, or az://account/container/prefix "
            "(install the matching extra: firefly[s3|gcs|azure])."
        ),
    ),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Path to the same golden-inputs JSON used at capture time."),
    runs: int = typer.Option(16, "--runs", "-n", help="Number of self-runs for the noise baseline. Non-HF runners (vllm/sglang) reload the engine once per run, so high values get expensive against serving engines."),
    safety_factor: float = typer.Option(6.0, "--safety-factor", help="atol = safety_factor × observed noise floor."),
    noise_mode: str = typer.Option("none", "--noise-mode", help="'none' (deterministic), 'synthetic' (Gaussian injection), or 'hardware' (real hardware noise)."),
    noise_sigma: float = typer.Option(0.0, "--noise-sigma", help="Standard deviation of injected noise (synthetic mode)."),
    noise_inject_at: str | None = typer.Option(None, "--noise-inject-at", help="Tap name to inject noise at (e.g. layer.0)."),
    noise_base_seed: int = typer.Option(0, "--noise-base-seed", help="Base seed for noise injection (run i uses base_seed+i)."),
    allow_tf32: bool = typer.Option(False, "--allow-tf32", help="Allow TF32 matmul (hardware mode only)."),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward pass."),
    seed: int = typer.Option(0, "--seed", help="Determinism seed."),
    push: str | None = typer.Option(
        None,
        "--push",
        help=(
            "If set, publish the reference dir (including the freshly "
            "written tolerances.json) to this URI after calibration "
            "(hf://org/repo or s3://bucket/prefix). Useful when the "
            "reference was originally pulled from the same URI: calibrate "
            "in-place, then push the calibrated artifact back."
        ),
    ),
) -> None:
    """Calibrate per-tap-point tolerances by running the reference against itself."""
    from firefly.calibrate import calibrate as run_calibrate
    from firefly.noise import NoiseSpec

    resolved_reference = _resolve_or_exit(reference)

    if noise_mode not in ("none", "synthetic", "hardware"):
        raise typer.BadParameter(
            f"--noise-mode must be 'none', 'synthetic', or 'hardware', got {noise_mode!r}",
            param_hint="--noise-mode",
        )
    if noise_mode == "synthetic":
        if noise_sigma <= 0:
            raise typer.BadParameter(
                "--noise-sigma must be > 0 when --noise-mode=synthetic",
                param_hint="--noise-sigma",
            )
        if noise_inject_at is None:
            raise typer.BadParameter(
                "--noise-inject-at is required when --noise-mode=synthetic",
                param_hint="--noise-inject-at",
            )

    noise = NoiseSpec(
        mode=noise_mode,  # type: ignore[arg-type]
        sigma=noise_sigma,
        inject_at=noise_inject_at,
        base_seed=noise_base_seed,
        allow_tf32=allow_tf32,
    )

    typer.echo(
        f"Calibrating: reference={reference} (resolved to {resolved_reference}) "
        f"runs={runs} noise_mode={noise_mode}"
    )
    tolerances = run_calibrate(
        reference_dir=resolved_reference,
        inputs_path=inputs,
        runs=runs,
        safety_factor=safety_factor,
        noise=noise,
        device=device,
        seed=seed,
    )

    n_above_floor = sum(1 for t in tolerances.values() if t.noise_floor > 0)
    max_floor = max((t.noise_floor for t in tolerances.values()), default=0.0)
    typer.echo(
        f"Calibrated {len(tolerances)} taps "
        f"({n_above_floor} above zero, max noise_floor={max_floor:.3e})"
    )
    typer.echo(f"Wrote tolerances.json to {resolved_reference}")

    if push is not None:
        typer.echo(f"Publishing {resolved_reference} → {push}")
        _publish_or_exit(
            resolved_reference, push, commit_message="Firefly calibration update"
        )
        typer.echo(f"Published to {push}")


@app.command()
def check(
    reference: str = typer.Option(
        ...,
        "--reference",
        "-r",
        help=(
            "Reference artifact directory. Accepts a local path or an "
            "hf://org/repo (optionally with @revision and /subpath) "
            "for HF Hub-hosted references, or a cloud URI: s3://bucket/"
            "prefix, gs://bucket/prefix, or az://account/container/prefix "
            "(install the matching extra: firefly[s3|gcs|azure])."
        ),
    ),
    candidate: str = typer.Option(..., "--candidate", "-c", help="Candidate HF model ID or checkpoint path."),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Path to the same golden-inputs JSON used at capture time."),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward pass."),
    seed: int = typer.Option(0, "--seed", help="Determinism seed."),
    report_json: Path | None = typer.Option(None, "--report-json", help="Write structured report to this path."),
    allow_fingerprint_mismatch: bool = typer.Option(
        False,
        "--allow-fingerprint-mismatch",
        help="Proceed even if the candidate's weights have changed since reference was captured.",
    ),
    allow_default_tolerances: bool = typer.Option(
        False,
        "--allow-default-tolerances",
        help=(
            "Proceed without calibrated tolerances (uses a flat default atol). "
            "Useful for testing or one-off comparisons; not recommended for CI."
        ),
    ),
    candidate_dtype: str | None = typer.Option(
        None,
        "--candidate-dtype",
        help=(
            "Dtype to load the candidate at (fp32, bf16, fp16). Defaults to "
            "the reference's recorded dtype — matching it is almost always "
            "what you want, since a dtype mismatch shows up as divergence. "
            "Override only for a deliberate cross-dtype comparison."
        ),
    ),
    runner: str | None = typer.Option(
        None,
        "--runner",
        help=(
            "Capture backend for the candidate: 'hf', 'vllm', or 'sglang'. "
            "Defaults to the runner the reference was captured with (recorded in "
            "the manifest) — comparing across engines surfaces engine artifacts "
            "as divergence, so override only deliberately."
        ),
    ),
    runner_opt: list[str] = typer.Option(
        [],
        "--runner-opt",
        help=(
            "Engine-specific knob as key=value (repeatable); see "
            "`firefly capture --help`. E.g. --runner-opt attention_backend=XFORMERS, "
            "or --runner-opt quantize=w8a8 to diff a torchao-quantized model "
            "against an fp baseline."
        ),
    ),
    jitter_floor: float = typer.Option(
        0.0,
        "--jitter-floor",
        help=(
            "Floor on TOLERATED relative drift (absorbs cross-platform FP "
            "jitter). atol becomes max(calibrated, jitter_floor × max|ref|) "
            "per tap, so this can only LOOSEN the gate — it is NOT a ceiling "
            "that fails when drift exceeds it. 0 = off (calibrated tolerances "
            "alone gate the check)."
        ),
    ),
    ci_format: str = typer.Option(
        "human",
        "--ci-format",
        help=(
            "Output format. 'human' (default) prints a rich-terminal table; "
            "'markdown' prints a PR-comment-friendly summary suitable for "
            "$GITHUB_STEP_SUMMARY or `gh pr comment --body-file -`."
        ),
    ),
) -> None:
    """Check a candidate against a reference. Exits non-zero if divergence exceeds tolerance."""
    from firefly.attribution import attribute_first_divergence
    from firefly.compare import (
        TOLERANCES_FILE,
        compare_to_reference,
        compare_to_reference_per_head,
    )
    from firefly.reference import read_manifest
    from firefly.report import render_human, render_markdown, write_json
    from firefly.runners import get_runner

    resolved_reference = _resolve_or_exit(reference)

    if ci_format not in {"human", "markdown"}:
        raise typer.BadParameter(
            f"--ci-format must be 'human' or 'markdown', got {ci_format!r}",
            param_hint="--ci-format",
        )

    # Enforce calibration: refuse to gate without empirically derived tolerances.
    # The flat 1e-5 default is almost certainly wrong for any non-FP32-deterministic
    # setup and would either spam false positives or silently miss real regressions.
    tolerances_path = resolved_reference / TOLERANCES_FILE
    if not tolerances_path.exists() and not allow_default_tolerances:
        typer.echo(
            f"ERROR: {tolerances_path} not found.\n"
            f"\n"
            f"Firefly requires calibrated per-tap tolerances to know what counts as a\n"
            f"real divergence vs expected noise. Run this once to set them up:\n"
            f"\n"
            f"  firefly calibrate --reference {reference} --inputs {inputs} --runs 8\n"
            f"\n"
            f"This re-runs your reference model and records the empirical noise floor\n"
            f"per tap. Takes ~3 minutes on CPU, ~30s on GPU. Commit the resulting\n"
            f"{tolerances_path.name} to your repo so CI picks it up.\n"
            f"\n"
            f"For one-off comparisons (not CI), pass --allow-default-tolerances to\n"
            f"proceed with a flat {1e-5:.0e} atol everywhere.",
            err=True,
        )
        raise typer.Exit(code=2)

    manifest = read_manifest(resolved_reference)

    # Default the runner (and its knobs) to what the reference was captured
    # with — recorded in the manifest — so check re-runs through the same
    # engine. Warn if the user deliberately overrides to a different engine.
    effective_runner = runner if runner is not None else manifest.runner
    if runner is not None and runner != manifest.runner:
        typer.echo(
            f"WARNING: checking with runner '{runner}' but the reference was "
            f"captured with '{manifest.runner}'. Cross-engine differences will "
            f"show up as divergence.",
            err=True,
        )
    # Manifest knobs first, user --runner-opt overrides on top.
    runner_options = {**manifest.runner_options, **_parse_runner_opts(runner_opt)}
    try:
        active_runner = get_runner(effective_runner)
    except (ValueError, NotImplementedError) as e:
        raise typer.BadParameter(str(e), param_hint="--runner") from e

    # If the reference carries per-head taps, run the per-head attribution
    # path so the report can name which attention head diverged. Both paths
    # run the candidate exactly once.
    per_head_taps = bool(manifest.head_counts)
    floor = jitter_floor if jitter_floor > 0 else None
    if per_head_taps:
        divergences, per_head = compare_to_reference_per_head(
            reference_dir=resolved_reference,
            candidate_model_id=candidate,
            inputs_path=inputs,
            device=device,
            seed=seed,
            allow_fingerprint_mismatch=allow_fingerprint_mismatch,
            jitter_floor=floor,
            candidate_dtype=candidate_dtype,
            runner=active_runner,
            options=runner_options,
        )
    else:
        divergences = compare_to_reference(
            reference_dir=resolved_reference,
            candidate_model_id=candidate,
            inputs_path=inputs,
            device=device,
            seed=seed,
            allow_fingerprint_mismatch=allow_fingerprint_mismatch,
            jitter_floor=floor,
            candidate_dtype=candidate_dtype,
            runner=active_runner,
            options=runner_options,
        )
        per_head = []
    result = attribute_first_divergence(divergences)

    if ci_format == "markdown":
        typer.echo(render_markdown(result, per_head=per_head))
    else:
        typer.echo(render_human(result, per_head=per_head))

    if report_json is not None:
        write_json(result, report_json, per_head=per_head)

    if result.any_exceeded:
        raise typer.Exit(code=1)


