"""Report formatting: rich tables for humans, structured JSON for machines.

The CLI prints the human report and optionally writes the JSON report. The
GitHub Action wrapper (later) consumes the JSON for PR annotations.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from firefly.attribution import AttributionResult

if TYPE_CHECKING:
    from firefly.head_attribution import PerHeadAttribution


def render_human(
    result: AttributionResult,
    console: Console | None = None,
    per_head: list[PerHeadAttribution] | None = None,
) -> str:
    """Render a per-tap divergence table + a one-line attribution summary.

    When ``per_head`` is supplied, append a per-head attribution table that
    names the worst attention head per ``attn_heads`` tap and how concentrated
    the divergence is (worst / median head).

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

    if per_head:
        head_table = Table(
            title="Per-head attention attribution",
            show_header=True,
            header_style="bold",
        )
        head_table.add_column("Tap", no_wrap=True)
        head_table.add_column("Worst head", justify="right")
        head_table.add_column("max |Δ|", justify="right")
        head_table.add_column("median head", justify="right")
        head_table.add_column("concentration", justify="right")
        for ph in per_head:
            head_table.add_row(
                ph.tap_name,
                f"{ph.worst_head} / {ph.n_heads}",
                f"{ph.worst_max_abs_diff:.3e}",
                f"{ph.median_max_abs_diff:.3e}",
                f"{ph.concentration:.1f}×",
            )
        console.print(head_table)

    return console.export_text()


def write_json(
    result: AttributionResult,
    path: Path,
    per_head: list[PerHeadAttribution] | None = None,
) -> None:
    """Structured report — the machine-readable consumer of attribution results."""
    payload = {
        "first_divergent_tap": result.first_divergent_tap,
        "any_exceeded": result.any_exceeded,
        "divergences": [asdict(d) for d in result.divergences],
    }
    if per_head:
        payload["per_head"] = [
            {**asdict(ph), "concentration": ph.concentration} for ph in per_head
        ]
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def render_markdown(
    result: AttributionResult,
    max_rows: int = 10,
    per_head: list[PerHeadAttribution] | None = None,
) -> str:
    """Render a PR-comment-friendly markdown summary.

    Designed for GitHub PR comments and ``$GITHUB_STEP_SUMMARY``:
    one-line headline so reviewers can read the verdict at a glance,
    then a compact table of the first ``max_rows`` divergent taps.
    Passing taps are omitted from the table (the count is in the footer).

    When ``per_head`` is supplied, a per-head attribution table follows so
    reviewers can see which attention head carries the divergence.
    """
    n_total = len(result.divergences)
    exceeded = [d for d in result.divergences if d.exceeds_tolerance]
    n_exceeded = len(exceeded)

    lines: list[str] = []
    if result.first_divergent_tap is None:
        lines.append(f"## ✅ Firefly: no divergence ({n_total} taps within tolerance)")
        _append_per_head_markdown(lines, per_head, max_rows)
        return "\n".join(lines) + "\n"

    lines.append(
        f"## ❌ Firefly: divergence at `{result.first_divergent_tap}`"
    )
    lines.append("")
    lines.append(
        f"**{n_exceeded} of {n_total}** taps exceeded tolerance. "
        f"First divergent tap: **`{result.first_divergent_tap}`**."
    )
    lines.append("")
    lines.append("| Tap | max \\|Δ\\| | mean \\|Δ\\| | atol applied |")
    lines.append("| --- | ---: | ---: | ---: |")
    for d in exceeded[:max_rows]:
        marker = " (first)" if d.tap_name == result.first_divergent_tap else ""
        atol = d.effective_atol if d.effective_atol else d.tolerance.atol
        lines.append(
            f"| `{d.tap_name}`{marker} | "
            f"{d.max_abs_diff:.3e} | "
            f"{d.mean_abs_diff:.3e} | "
            f"{atol:.3e} |"
        )
    if n_exceeded > max_rows:
        lines.append(f"| _… and {n_exceeded - max_rows} more_ | | | |")
    lines.append("")
    lines.append(
        "_See the JSON report or run `firefly check` locally for the full per-tap table._"
    )
    _append_per_head_markdown(lines, per_head, max_rows)
    return "\n".join(lines) + "\n"


def _append_per_head_markdown(
    lines: list[str],
    per_head: list[PerHeadAttribution] | None,
    max_rows: int,
) -> None:
    """Append a per-head attribution table to ``lines`` (in place), if any."""
    if not per_head:
        return
    lines.append("")
    lines.append("### Per-head attention attribution")
    lines.append("")
    lines.append("| Tap | worst head | max \\|Δ\\| | median head | concentration |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for ph in per_head[:max_rows]:
        lines.append(
            f"| `{ph.tap_name}` | {ph.worst_head} / {ph.n_heads} | "
            f"{ph.worst_max_abs_diff:.3e} | {ph.median_max_abs_diff:.3e} | "
            f"{ph.concentration:.1f}× |"
        )
    if len(per_head) > max_rows:
        lines.append(f"| _… and {len(per_head) - max_rows} more_ | | | | |")
