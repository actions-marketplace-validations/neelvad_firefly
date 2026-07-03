"""Diagnose per-layer activation magnitudes from a reference capture.

Tests the SwiGLU saturation hypothesis: if the layer-11 TF32 noise jump is
driven by activation magnitude growth, we should see ``max(|activation|)``
ramping into layer 11 in the captured reference — at the same layer where
the noise floor in scripts/results/ jumps by ~250×.

Usage:
    uv run python experiments/analyze.py                            # capture fresh, analyze
    uv run python experiments/analyze.py --reference ref/           # analyze an existing reference
    uv run python experiments/analyze.py --model HuggingFaceTB/SmolLM-360M
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import typer

app = typer.Typer(add_completion=False, no_args_is_help=False)


_LAYER_RE = re.compile(r"layer\.(\d+)(?:\.(self_attn|mlp))?$")


@app.command()
def main(
    reference: Path | None = typer.Option(
        None, "--reference", "-r", help="Existing reference directory (skips capture)."
    ),
    model: str = typer.Option(
        "HuggingFaceTB/SmolLM-135M", "--model", "-m", help="HF model ID for fresh capture."
    ),
    max_length: int = typer.Option(16, "--max-length", help="Token length for fresh capture."),
) -> None:
    """Print per-layer activation magnitudes from a Firefly reference capture."""
    from firefly.capture import capture_reference
    from firefly.reference import read_reference

    if reference is not None:
        manifest, tensors = read_reference(reference)
    else:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs_path = tmp_path / "golden.json"
            inputs_path.write_text(
                json.dumps(
                    {
                        "texts": [
                            "the quick brown fox jumps over the lazy dog",
                            "to be or not to be",
                        ],
                        "max_length": max_length,
                    }
                )
            )
            ref_dir = tmp_path / "reference"
            typer.echo(f"Capturing {model} on CPU…")
            capture_reference(model, inputs_path, ref_dir, device="cpu")
            manifest, tensors = read_reference(ref_dir)

    typer.echo(f"\nReference: {manifest.model_id}")
    typer.echo(f"Capture device: {manifest.env.get('device', 'unknown')}")
    typer.echo(f"Total tap points: {len(tensors)}")

    # Bucket taps by layer index (and sub-block) so we can render a depth profile.
    depth: dict[int, dict[str, float]] = {}
    other: dict[str, float] = {}
    for name, t in tensors.items():
        m = _LAYER_RE.match(name)
        if m is None:
            other[name] = float(t.abs().max().item())
            continue
        i = int(m.group(1))
        sub = m.group(2) or "residual"
        depth.setdefault(i, {})[sub] = float(t.abs().max().item())

    typer.echo("\n=== max(|activation|) per layer ===")
    typer.echo(
        f"{'layer':<10} {'self_attn':>13} {'mlp':>13} {'residual':>13} "
        f"{'mlp Δ':>10} {'residual Δ':>12}"
    )

    prev_mlp: float | None = None
    prev_resid: float | None = None
    transitions: list[tuple[int, float]] = []

    for i in sorted(depth):
        d = depth[i]
        sa = d.get("self_attn", float("nan"))
        ml = d.get("mlp", float("nan"))
        rs = d.get("residual", float("nan"))

        def _ratio(curr: float, prev: float | None) -> str:
            if prev is None or prev == 0 or curr != curr:
                return "—"
            return f"{curr / prev:5.2f}×"

        mlp_ratio_s = _ratio(ml, prev_mlp)
        resid_ratio_s = _ratio(rs, prev_resid)
        if prev_mlp and prev_mlp > 0 and ml / prev_mlp > 0:
            transitions.append((i, ml / prev_mlp))

        typer.echo(
            f"layer.{i:<4} "
            f"{sa:>13.4e} {ml:>13.4e} {rs:>13.4e} {mlp_ratio_s:>10} {resid_ratio_s:>12}"
        )
        prev_mlp, prev_resid = ml, rs

    if other:
        typer.echo("\n=== Other taps ===")
        for name, val in sorted(other.items()):
            typer.echo(f"  {name:<24} {val:>13.4e}")

    if transitions:
        transitions.sort(key=lambda x: x[1], reverse=True)
        typer.echo("\n=== Top 5 layer-over-layer MLP-output magnitude jumps ===")
        for i, r in transitions[:5]:
            typer.echo(f"  layer.{i}.mlp: {r:.2f}× the magnitude of layer.{i - 1}.mlp")

        typer.echo("\n=== Interpretation hint ===")
        typer.echo(
            "If the layer.{i}.mlp where the magnitude jumps matches the layer where "
            "your TF32 noise floor jumped (scripts/results/), the SwiGLU saturation "
            "hypothesis is supported. If the magnitude is flat through that layer, "
            "the noise is driven by something else (specific weights, kernel selection)."
        )


if __name__ == "__main__":
    app()
