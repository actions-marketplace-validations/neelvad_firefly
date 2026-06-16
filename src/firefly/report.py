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
    from firefly.quant_risk import TapQuantRisk
    from firefly.quant_validate import TorchaoValidationResult


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


def render_quant_risk(
    risks: list[TapQuantRisk],
    bits: int,
    threshold: float,
    console: Console | None = None,
) -> str:
    """Render the quantization-risk table in forward order.

    Rows whose simulated per-tensor relative error exceeds ``threshold``
    are flagged; the worst tap is highlighted. Returns the rendered text.
    """
    console = console or Console(record=True, width=120)

    table = Table(
        title=f"Firefly quantization-risk report (int{bits}, symmetric)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Tap", no_wrap=True)
    table.add_column("abs max", justify="right")
    table.add_column("outlier ratio", justify="right")
    table.add_column("channel conc.", justify="right")
    table.add_column("per-tensor err", justify="right")
    table.add_column("per-channel err", justify="right")
    table.add_column("mitigation gain", justify="right")
    table.add_column("Status", justify="center")

    worst = max(risks, key=lambda r: r.per_tensor_rel_err, default=None)
    n_flagged = 0
    for r in risks:
        flagged = r.per_tensor_rel_err > threshold
        n_flagged += flagged
        status = "[red]⚠[/]" if flagged else "[green]✓[/]"
        row_style = "bold red" if (worst is not None and r is worst and flagged) else None
        table.add_row(
            r.tap_name,
            f"{r.abs_max:.3e}",
            f"{r.outlier_ratio:.1f}×",
            f"{r.channel_concentration:.1f}×",
            f"{r.per_tensor_rel_err:.2%}",
            f"{r.per_channel_rel_err:.2%}",
            f"{r.mitigation_gain:.1f}×",
            status,
            style=row_style,
        )

    console.print(table)
    if worst is not None and worst.per_tensor_rel_err > threshold:
        console.print(
            f"[bold red]{n_flagged} of {len(risks)} taps above {threshold:.1%} "
            f"simulated int{bits} error.[/] Worst: {worst.tap_name} "
            f"({worst.per_tensor_rel_err:.2%} per-tensor; per-channel scaling "
            f"reduces it {worst.mitigation_gain:.1f}× to {worst.per_channel_rel_err:.2%})."
        )
    else:
        console.print(
            f"[bold green]All {len(risks)} taps within {threshold:.1%} "
            f"simulated int{bits} error.[/]"
        )

    return console.export_text()


def render_torchao_validation(
    result: TorchaoValidationResult,
    top_n: int = 12,
    console: Console | None = None,
) -> str:
    """Render the torchao-validation verdict: Spearman correlations + top layers.

    Confirms (or refutes) that quant-risk's per-input prediction ranks the
    Linear layers where real torchao W8A8 actually diverges most.
    """
    from firefly.quant_validate import PASS_THRESHOLD

    console = console or Console(record=True, width=120)

    table = Table(
        title=f"Firefly quant-risk validation vs real torchao W8A8 (int{result.bits})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Predictor (on Linear input)", no_wrap=True)
    table.add_column("Spearman vs local torchao err", justify="right")
    for label, rho in (
        ("channel concentration", result.spearman_concentration),
        ("per-tensor err", result.spearman_per_tensor),
        ("mitigation gain", result.spearman_mitigation_gain),
    ):
        strong = rho > PASS_THRESHOLD
        table.add_row(label, f"[{'green' if strong else 'yellow'}]{rho:+.3f}[/]")
    console.print(table)

    top = sorted(result.records, key=lambda r: r.actual_local_err, reverse=True)[:top_n]
    detail = Table(
        title=f"Top {len(top)} layers by ACTUAL local torchao divergence",
        show_header=True,
        header_style="bold",
    )
    detail.add_column("Linear", no_wrap=True)
    detail.add_column("channel conc.", justify="right")
    detail.add_column("pred per-tensor err", justify="right")
    detail.add_column("actual local err", justify="right")
    for r in top:
        detail.add_row(
            r.name,
            f"{r.channel_concentration:.1f}×",
            f"{r.per_tensor_rel_err:.1%}",
            f"{r.actual_local_err:.2%}",
        )
    console.print(detail)

    if result.passed:
        console.print(
            f"[bold green]PASS[/] ({len(result.records)} Linears): quant-risk's per-input "
            f"ranking predicts where real int{result.bits} W8A8 hurts locally "
            f"(best Spearman {result.best_spearman:+.3f} > {PASS_THRESHOLD:.2f})."
        )
    else:
        console.print(
            f"[bold red]WEAK[/] ({len(result.records)} Linears): quant-risk's ranking does "
            f"NOT clearly predict real int{result.bits} W8A8 divergence "
            f"(best Spearman {result.best_spearman:+.3f} ≤ {PASS_THRESHOLD:.2f})."
        )
    return console.export_text()


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
