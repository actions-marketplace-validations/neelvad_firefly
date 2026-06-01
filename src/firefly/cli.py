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
    runs: int = typer.Option(16, "--runs", "-n", help="Number of self-runs for the noise baseline."),
) -> None:
    """Calibrate per-tap-point tolerances by running the reference against itself."""
    typer.echo(f"[stub] calibrate reference={reference} runs={runs}")


@app.command()
def check(
    reference: Path = typer.Option(..., "--reference", "-r", help="Reference artifact directory."),
    candidate: str = typer.Option(..., "--candidate", "-c", help="Candidate HF model ID or checkpoint path."),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Path to the same golden-inputs JSON used at capture time."),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward pass."),
    seed: int = typer.Option(0, "--seed", help="Determinism seed."),
    report_json: Path | None = typer.Option(None, "--report-json", help="Write structured report to this path."),
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
    )
    result = attribute_first_divergence(divergences)

    typer.echo(render_human(result))

    if report_json is not None:
        write_json(result, report_json)

    if result.any_exceeded:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
