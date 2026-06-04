"""Firefly CLI: capture, calibrate, check."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="firefly",
    help="Numerical-parity CI gate for ML model deployments.",
    no_args_is_help=True,
)


@app.command()
def capture(
    model: str = typer.Option(..., "--model", "-m", help="HF model ID or path to a checkpoint."),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Path to a golden-inputs JSON file."),
    out: Path = typer.Option(..., "--out", "-o", help="Directory to write the reference artifact."),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward pass."),
    seed: int = typer.Option(0, "--seed", help="Determinism seed."),
    dtype: str = typer.Option("fp32", "--dtype", help="Model dtype: fp32, bf16, or fp16."),
) -> None:
    """Capture a reference artifact from a model + golden inputs."""
    from firefly.capture import capture_reference, parse_dtype

    typer.echo(f"Capturing reference: model={model} device={device} dtype={dtype}")
    capture_reference(
        model_id=model,
        inputs_path=inputs,
        out_dir=out,
        device=device,
        seed=seed,
        dtype=parse_dtype(dtype),
    )
    typer.echo(f"Wrote reference artifact to {out}")


@app.command()
def calibrate(
    reference: Path = typer.Option(..., "--reference", "-r", help="Reference artifact directory."),
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
) -> None:
    """Calibrate per-tap-point tolerances by running the reference against itself."""
    from firefly.calibrate import calibrate as run_calibrate
    from firefly.noise import NoiseSpec

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

    typer.echo(f"Calibrating: reference={reference} runs={runs} noise_mode={noise_mode}")
    tolerances = run_calibrate(
        reference_dir=reference,
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
    typer.echo(f"Wrote tolerances.json to {reference}")


@app.command()
def check(
    reference: Path = typer.Option(..., "--reference", "-r", help="Reference artifact directory."),
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
    from firefly.compare import TOLERANCES_FILE, compare_to_reference
    from firefly.report import render_human, render_markdown, write_json

    if ci_format not in {"human", "markdown"}:
        raise typer.BadParameter(
            f"--ci-format must be 'human' or 'markdown', got {ci_format!r}",
            param_hint="--ci-format",
        )

    # Enforce calibration: refuse to gate without empirically derived tolerances.
    # The flat 1e-5 default is almost certainly wrong for any non-FP32-deterministic
    # setup and would either spam false positives or silently miss real regressions.
    tolerances_path = reference / TOLERANCES_FILE
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

    divergences = compare_to_reference(
        reference_dir=reference,
        candidate_model_id=candidate,
        inputs_path=inputs,
        device=device,
        seed=seed,
        allow_fingerprint_mismatch=allow_fingerprint_mismatch,
        max_rel_error=(max_rel_error if max_rel_error > 0 else None),
    )
    result = attribute_first_divergence(divergences)

    if ci_format == "markdown":
        typer.echo(render_markdown(result))
    else:
        typer.echo(render_human(result))

    if report_json is not None:
        write_json(result, report_json)

    if result.any_exceeded:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
