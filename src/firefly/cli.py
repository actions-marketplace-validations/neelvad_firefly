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
) -> None:
    """Capture a reference artifact from a model + golden inputs."""
    from firefly.capture import capture_reference

    typer.echo(f"Capturing reference: model={model} device={device}")
    capture_reference(
        model_id=model,
        inputs_path=inputs,
        out_dir=out,
        device=device,
        seed=seed,
    )
    typer.echo(f"Wrote reference artifact to {out}")


@app.command()
def calibrate(
    reference: Path = typer.Option(..., "--reference", "-r", help="Reference artifact directory."),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Path to the same golden-inputs JSON used at capture time."),
    runs: int = typer.Option(16, "--runs", "-n", help="Number of self-runs for the noise baseline."),
    safety_factor: float = typer.Option(6.0, "--safety-factor", help="atol = safety_factor × observed noise floor."),
    noise_mode: str = typer.Option("none", "--noise-mode", help="'none' (deterministic) or 'synthetic' (Gaussian injection)."),
    noise_sigma: float = typer.Option(0.0, "--noise-sigma", help="Standard deviation of injected noise (synthetic mode)."),
    noise_inject_at: str | None = typer.Option(None, "--noise-inject-at", help="Tap name to inject noise at (e.g. layer.0)."),
    noise_base_seed: int = typer.Option(0, "--noise-base-seed", help="Base seed for noise injection (run i uses base_seed+i)."),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward pass."),
    seed: int = typer.Option(0, "--seed", help="Determinism seed."),
) -> None:
    """Calibrate per-tap-point tolerances by running the reference against itself."""
    from firefly.calibrate import calibrate as run_calibrate
    from firefly.noise import NoiseSpec

    if noise_mode not in ("none", "synthetic"):
        raise typer.BadParameter(
            f"--noise-mode must be 'none' or 'synthetic', got {noise_mode!r}",
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
) -> None:
    """Check a candidate against a reference. Exits non-zero if divergence exceeds tolerance."""
    from firefly.attribution import attribute_first_divergence
    from firefly.compare import compare_to_reference
    from firefly.report import render_human, write_json

    divergences = compare_to_reference(
        reference_dir=reference,
        candidate_model_id=candidate,
        inputs_path=inputs,
        device=device,
        seed=seed,
        allow_fingerprint_mismatch=allow_fingerprint_mismatch,
    )
    result = attribute_first_divergence(divergences)

    typer.echo(render_human(result))

    if report_json is not None:
        write_json(result, report_json)

    if result.any_exceeded:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
