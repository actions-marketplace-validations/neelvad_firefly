"""Firefly CLI: capture, calibrate, check."""

from __future__ import annotations

from pathlib import Path

import typer

from firefly.storage import publish_reference, resolve_reference

app = typer.Typer(
    name="firefly",
    help="Numerical-parity CI gate for ML model deployments.",
    no_args_is_help=True,
)


def _resolve_or_exit(reference: str) -> Path:
    """Resolve a reference URI to a local path, exiting cleanly on errors."""
    try:
        return resolve_reference(reference)
    except NotImplementedError as e:
        raise typer.BadParameter(str(e), param_hint="--reference") from e
    except (RuntimeError, ValueError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2) from e


def _publish_or_exit(local_path: Path, uri: str, *, commit_message: str) -> None:
    """Publish a reference dir to a URI, exiting cleanly on errors."""
    try:
        publish_reference(local_path, uri, commit_message=commit_message)
    except NotImplementedError as e:
        raise typer.BadParameter(str(e), param_hint="--to") from e
    except (ImportError, RuntimeError, ValueError, FileNotFoundError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2) from e


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
    push: str | None = typer.Option(
        None,
        "--push",
        help=(
            "If set, publish the artifact to this URI after capture "
            "(hf://org/repo or s3://bucket/prefix). Equivalent to running "
            "`firefly publish` immediately after capture."
        ),
    ),
) -> None:
    """Capture a reference artifact from a model + golden inputs."""
    from firefly.capture import capture_reference, parse_dtype

    typer.echo(
        f"Capturing reference: model={model} device={device} dtype={dtype} "
        f"per_head={per_head}"
    )
    capture_reference(
        model_id=model,
        inputs_path=inputs,
        out_dir=out,
        device=device,
        seed=seed,
        dtype=parse_dtype(dtype),
        per_head=per_head,
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
            "@revision and /subpath) or s3://bucket/prefix. HF Hub creates "
            "the repo if it doesn't exist; both backends use ambient "
            "credentials (HF_TOKEN for HF, AWS default credential chain for S3)."
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
            "for HF Hub-hosted references. S3/GCS/Azure are stubbed "
            "and return a clear planned-for-vN error."
        ),
    ),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Path to the same golden-inputs JSON used at capture time."),
    runs: int = typer.Option(16, "--runs", "-n", help="Number of self-runs for the noise baseline."),
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
            "for HF Hub-hosted references. S3/GCS/Azure are stubbed "
            "and return a clear planned-for-vN error."
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
    max_rel_error: float = typer.Option(
        0.0,
        "--max-rel-error",
        help=(
            "Optional global ceiling on relative error. atol becomes "
            "max(calibrated, max_rel_error × max|ref|) per tap. 0 = no ceiling "
            "(calibrated tolerances alone gate the check)."
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

    # If the reference carries per-head taps, run the per-head attribution
    # path so the report can name which attention head diverged. Both paths
    # run the candidate exactly once.
    per_head_taps = bool(read_manifest(resolved_reference).head_counts)
    max_rel = max_rel_error if max_rel_error > 0 else None
    if per_head_taps:
        divergences, per_head = compare_to_reference_per_head(
            reference_dir=resolved_reference,
            candidate_model_id=candidate,
            inputs_path=inputs,
            device=device,
            seed=seed,
            allow_fingerprint_mismatch=allow_fingerprint_mismatch,
            max_rel_error=max_rel,
        )
    else:
        divergences = compare_to_reference(
            reference_dir=resolved_reference,
            candidate_model_id=candidate,
            inputs_path=inputs,
            device=device,
            seed=seed,
            allow_fingerprint_mismatch=allow_fingerprint_mismatch,
            max_rel_error=max_rel,
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
    """Predict which layers will degrade under quantization, from stored activations.

    Simulates symmetric round-to-nearest quantization of each tap's captured
    activation (per-tensor and per-channel) and reports the taps where
    per-tensor quantization breaks down — typically outlier-feature layers
    whose few extreme channels force a scale that crushes the rest.
    """
    import json
    from dataclasses import asdict

    from firefly.quant_risk import analyze_quant_risk
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


if __name__ == "__main__":
    app()
