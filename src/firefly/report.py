"""Report formatting: rich tables for humans, structured JSON for machines.

The CLI prints the human report and optionally writes the JSON report. The
GitHub Action wrapper (later) consumes the JSON for PR annotations.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from firefly.attribution import AttributionResult


def render_human(result: AttributionResult, console: Console | None = None) -> str:
    """Render a per-tap divergence table + a one-line attribution summary.

    Returns the rendered text so callers (CLI, tests) can capture it.
    """
    console = console or Console(record=True, width=100)

    table = Table(
        title="Firefly divergence report",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Tap", no_wrap=True)
    table.add_column("max |Δ|", justify="right")
    table.add_column("mean |Δ|", justify="right")
    table.add_column("Tolerance", justify="right")
    table.add_column("Status", justify="center")

    for d in result.divergences:
        status = "[red]✗[/]" if d.exceeds_tolerance else "[green]✓[/]"
        row_style = "bold red" if d.tap_name == result.first_divergent_tap else None
        table.add_row(
            d.tap_name,
            f"{d.max_abs_diff:.3e}",
            f"{d.mean_abs_diff:.3e}",
            f"{d.tolerance.atol:.0e}",
            status,
            style=row_style,
        )

    console.print(table)
    if result.first_divergent_tap:
        console.print(f"[bold red]First divergence:[/] {result.first_divergent_tap}")
    else:
        console.print("[bold green]No divergence detected.[/]")

    return console.export_text()


def write_json(result: AttributionResult, path: Path) -> None:
    """Structured report — the machine-readable consumer of attribution results."""
    payload = {
        "first_divergent_tap": result.first_divergent_tap,
        "any_exceeded": result.any_exceeded,
        "divergences": [asdict(d) for d in result.divergences],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
