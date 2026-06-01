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
) -> None:
    """Capture a reference artifact from a model + golden inputs."""
    typer.echo(f"[stub] capture model={model} inputs={inputs} out={out}")


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
    report_json: Path | None = typer.Option(None, "--report-json", help="Write structured report to this path."),
) -> None:
    """Check a candidate against a reference. Exits non-zero if divergence exceeds tolerance."""
    typer.echo(f"[stub] check reference={reference} candidate={candidate}")


if __name__ == "__main__":
    app()
