"""Op-level drill-down command: op-diff."""

from __future__ import annotations

from pathlib import Path

import typer

from firefly.cli._app import app


@app.command("op-diff")
def op_diff(
    model: str = typer.Option(..., "--model", "-m", help="HF model ID or checkpoint path."),
    inputs: Path = typer.Option(..., "--inputs", "-i", help="Golden-inputs JSON."),
    module: str = typer.Option(
        ..., "--module",
        help="FQN of the module to drill into, e.g. model.layers.7.self_attn "
        "(the module the parity gate flagged).",
    ),
    ref_dtype: str = typer.Option("float32", "--ref-dtype", help="Reference dtype."),
    cand_dtype: str = typer.Option("bfloat16", "--cand-dtype", help="Candidate dtype."),
    tol: float = typer.Option(
        0.01, "--tol", help="Flag the first op whose relative divergence exceeds this."
    ),
    top_n: int = typer.Option(25, "--top-n", help="How many ops to show."),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for the forward passes."),
) -> None:
    """Drill into one module to the ATen op where two executions first diverge.

    The op-level rung below the module gate: it re-runs the model at two dtypes
    (or precisions), scopes a TorchDispatchMode to ``--module`` via forward hooks,
    and diffs the ops in execution order. Eager/CI-only and Python-per-op slow,
    so it's a targeted drill-down into a flagged module, not a default scan.
    """
    from firefly.op_drill import op_diff_dtypes
    from firefly.report import render_op_diff

    try:
        result = op_diff_dtypes(
            model, inputs, module, ref_dtype=ref_dtype, cand_dtype=cand_dtype,
            device=device, tol=tol,
        )
    except AttributeError as e:
        raise typer.BadParameter(
            f"module {module!r} not found in {model}", param_hint="--module"
        ) from e

    typer.echo(render_op_diff(result, top_n=top_n))
    if result.any_exceeded:
        raise typer.Exit(1)


