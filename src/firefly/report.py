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
    from firefly.op_drill import OpDiffResult
    from firefly.quant.recipe import RecipeResult
    from firefly.quant.risk import TapQuantRisk
    from firefly.quant.sensitivity import SensitivityResult


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


def render_quant_diff(
    result: AttributionResult,
    scheme: str | None = None,
    top_n: int = 15,
    per_head: list[PerHeadAttribution] | None = None,
    rel_threshold: float | None = None,
    console: Console | None = None,
) -> str:
    """Magnitude-ranked divergence report for quantization diffs.

    :func:`render_human` is tuned to find the *first* tap to exceed tolerance —
    uninformative for quantization, which perturbs every tap from layer 0. This
    ranks taps by *relative* divergence (``rel_mean``) so the layers a quant
    scheme actually damaged most surface first, and reports the accumulated
    divergence at the network output.
    """
    console = console or Console(record=True, width=100)
    divs = result.divergences
    if not divs:
        console.print("[yellow]No taps to compare.[/]")
        return console.export_text()

    ranked = sorted(divs, key=lambda d: d.rel_mean, reverse=True)
    worst = ranked[0]
    output_tap = divs[-1]  # forward-order last tap ≈ the network output

    title = "Firefly quantization diff" + (f" — {scheme}" if scheme else "")
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Tap", no_wrap=True)
    table.add_column("rel mean", justify="right")
    table.add_column("rel max", justify="right")
    table.add_column("mean |Δ|", justify="right")
    for i, d in enumerate(ranked[:top_n], 1):
        over = rel_threshold is not None and d.rel_mean > rel_threshold
        style = "bold red" if d is worst else ("red" if over else None)
        table.add_row(
            str(i), d.tap_name,
            f"{d.rel_mean:.2%}", f"{d.rel_max:.2%}", f"{d.mean_abs_diff:.3e}",
            style=style,
        )
    console.print(table)
    if len(ranked) > top_n:
        console.print(
            f"[dim]… {len(ranked) - top_n} more taps "
            f"(showing top {top_n} by relative divergence)[/]"
        )

    console.print(
        f"[bold]worst layer:[/] {worst.tap_name} "
        f"({worst.rel_mean:.2%} mean relative divergence)"
    )
    console.print(
        f"[bold]accumulated at output[/] ({output_tap.tap_name}): "
        f"{output_tap.rel_mean:.2%}"
    )

    if per_head:
        wh = max(per_head, key=lambda ph: ph.worst_max_abs_diff)
        console.print(
            f"[bold]worst head:[/] {wh.tap_name} head {wh.worst_head}/{wh.n_heads} "
            f"({wh.concentration:.1f}× concentration)"
        )

    if rel_threshold is not None:
        n_over = sum(1 for d in divs if d.rel_mean > rel_threshold)
        if n_over:
            console.print(
                f"[bold red]{n_over} tap(s) exceed {rel_threshold:.1%} "
                f"relative divergence.[/]"
            )
        else:
            console.print(
                f"[bold green]All taps within {rel_threshold:.1%} "
                f"relative divergence.[/]"
            )
    return console.export_text()


def render_quant_diff_markdown(
    result: AttributionResult,
    scheme: str | None = None,
    top_n: int = 10,
    per_head: list[PerHeadAttribution] | None = None,
    rel_threshold: float | None = None,
) -> str:
    """PR-comment-friendly markdown for a quantization diff.

    Markdown counterpart of :func:`render_quant_diff`: a one-line headline
    (worst layer + accumulated output divergence) then a compact table of the
    top taps by *relative* divergence. For ``$GITHUB_STEP_SUMMARY`` / PR comments.
    """
    divs = result.divergences
    label = f" — `{scheme}`" if scheme else ""
    if not divs:
        return f"## Firefly quant-diff{label}: no taps to compare\n"

    ranked = sorted(divs, key=lambda d: d.rel_mean, reverse=True)
    worst = ranked[0]
    output_tap = divs[-1]

    lines = [
        f"## Firefly quant-diff{label}",
        "",
        f"Worst layer **`{worst.tap_name}`** ({worst.rel_mean:.2%} mean rel divergence); "
        f"accumulated at output (`{output_tap.tap_name}`): **{output_tap.rel_mean:.2%}**.",
    ]
    if rel_threshold is not None:
        n_over = sum(1 for d in divs if d.rel_mean > rel_threshold)
        lines += [
            "",
            f"⚠️ {n_over} of {len(divs)} taps exceed {rel_threshold:.1%} relative divergence."
            if n_over
            else f"✅ all {len(divs)} taps within {rel_threshold:.1%} relative divergence.",
        ]
    lines += [
        "",
        "| # | Tap | rel mean | rel max | mean \\|Δ\\| |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for i, d in enumerate(ranked[:top_n], 1):
        lines.append(
            f"| {i} | `{d.tap_name}` | {d.rel_mean:.2%} | {d.rel_max:.2%} | {d.mean_abs_diff:.3e} |"
        )
    if len(ranked) > top_n:
        lines.append(f"| | _… and {len(ranked) - top_n} more_ | | | |")
    _append_per_head_markdown(lines, per_head, top_n)
    return "\n".join(lines) + "\n"


def render_sensitivity(
    result: SensitivityResult,
    top_n: int = 15,
    keep_k: int = 4,
    console: Console | None = None,
) -> str:
    """Render per-unit quantization sensitivity, ranked most-sensitive first.

    The headline is the all-quantized output divergence we're decomposing; the
    table ranks units by how much keeping them in high precision matters, and
    the footer suggests the top-``keep_k`` to keep high-precision (a recipe that
    ``firefly quant-recipe`` will verify).
    """
    console = console or Console(record=True, width=100)
    ranked = result.ranked
    noun = "Linears" if result.granularity == "linear" else "layers"

    table = Table(
        title=f"Firefly quant sensitivity — {result.scheme} "
        f"({result.strategy} strategy, {result.granularity} granularity)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", justify="right")
    table.add_column("Unit", no_wrap=True)
    table.add_column("sensitivity", justify="right")
    table.add_column("Linears", justify="right")
    for i, ls in enumerate(ranked[:top_n], 1):
        style = "bold red" if i == 1 else None
        table.add_row(str(i), ls.unit, f"{ls.sensitivity:.2%}", str(ls.n_linears), style=style)
    console.print(table)
    if len(ranked) > top_n:
        console.print(f"[dim]… {len(ranked) - top_n} more {noun} (showing top {top_n})[/]")

    console.print(
        f"[bold]all {len(result.units)} {noun} quantized[/] ({result.scheme}) → "
        f"{result.full_quant_divergence:.2%} output divergence at {result.output_tap}"
    )
    keep = result.keep_high_precision(keep_k)
    console.print(
        f"[bold]suggested keep-in-high-precision[/] (top {keep_k}): {', '.join(keep)}"
    )
    return console.export_text()


def render_recipe(
    result: RecipeResult,
    console: Console | None = None,
) -> str:
    """Render the verified mixed-precision recipe curve.

    Shows, for each k, the divergence achieved by keeping the top-k sensitive
    layers in high precision and quantizing the rest — and how much of the
    all-quantized degradation that recovers. The recommendation is the smallest
    k that clears the recovery target.
    """
    from firefly.quant.cost import format_bytes

    console = console or Console(record=True, width=100)
    sens = result.sensitivity
    noun = "Linears" if sens.granularity == "linear" else "layers"
    frontier, knee = result.frontier_knee_ks()

    console.print(
        f"[bold]all {len(sens.units)} {noun} quantized[/] ({sens.scheme}) → "
        f"{sens.full_quant_divergence:.2%} output divergence "
        f"[dim](strategy: {sens.strategy}, granularity: {sens.granularity})[/]"
    )
    if result.all_fp_bytes:
        console.print(
            f"[dim]weight footprint: all-fp {format_bytes(result.all_fp_bytes)} → "
            f"all-{sens.scheme} {format_bytes(result.all_quant_bytes)} "
            f"({result.all_fp_bytes / result.all_quant_bytes:.1f}× smaller)[/]"
        )

    table = Table(
        title="Mixed-precision recipe curve (verified)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("keep hi-prec", justify="right")
    table.add_column("units kept", no_wrap=False)
    table.add_column("output Δ", justify="right")
    table.add_column("recovery", justify="right")
    table.add_column("memory", justify="right")
    table.add_column("Pareto", justify="center")
    for p in sorted(result.curve, key=lambda p: p.k):
        kept = ", ".join(p.kept_units)
        mark = "knee" if p.k == knee else ("frontier" if p.k in frontier else "")
        if p.k == result.recommended_k:
            style = "bold green"
        elif p.k in frontier:
            style = None
        else:
            style = "dim"  # dominated — strictly beaten on both size and quality
        table.add_row(
            str(p.k), kept, f"{p.output_divergence:.2%}", f"{p.recovery:.1%}",
            format_bytes(p.memory_bytes), mark, style=style,
        )
    console.print(table)

    if knee is not None:
        kp = next(p for p in result.curve if p.k == knee)
        console.print(
            f"[cyan]Pareto knee:[/] keep {knee} {noun} → {format_bytes(kp.memory_bytes)}, "
            f"{kp.output_divergence:.2%} divergence — best quality-per-byte before "
            f"diminishing returns."
        )

    rec = result.recommended_point
    if rec is not None:
        console.print(
            f"[bold green]recommended (recovery target):[/] keep {rec.k} {noun} in high "
            f"precision ({', '.join(rec.kept_units)}) → {rec.output_divergence:.2%} "
            f"divergence, {rec.recovery:.0%} of the degradation recovered "
            f"(target {result.recovery_target:.0%}), {format_bytes(rec.memory_bytes)}."
        )
    else:
        console.print("[yellow]No recipe points evaluated.[/]")
    return console.export_text()


def render_search(result: dict, console: Console | None = None) -> str:
    """Render an LLM min-memory-at-bar search: the proposal trajectory and the
    lowest-memory recipe that cleared the bar."""
    console = console or Console(record=True, width=100)
    r = result
    console.print(
        f"[bold]LLM agent: min memory @ perplexity bar[/]  {r['model']} ({r['scheme']})\n"
        f"[dim]fp {r['perplexity_fp']:.2f}  bar ≤ {r['threshold']:.2f}  "
        f"plain-quant {r['perplexity_plain']:.2f}  "
        f"(all-fp {r['all_fp_mb']:.0f} MB → all-quant {r['all_quant_mb']:.0f} MB)[/]"
    )
    table = Table(title="proposal trajectory", show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("recipe", no_wrap=False)
    table.add_column("perplexity", justify="right")
    table.add_column("memory", justify="right")
    table.add_column("bar?", justify="center")
    best_step = r["best"]["step"] if r["best"] else None
    for h in r["history"]:
        a = h["action"]
        recipe = f"{a.get('quantizer', '?')}"
        if a.get("keep_fp_units"):
            recipe += f" + fp[{','.join(a['keep_fp_units'])}]"
        mark = "[green]yes[/]" if h["passed_bar"] else "[red]no[/]"
        style = "bold green" if h["step"] == best_step else None
        table.add_row(str(h["step"]), recipe, f"{h['perplexity']:.2f}", f"{h['memory_mb']:.0f} MB", mark, style=style)
    console.print(table)

    if r["best"]:
        b = r["best"]
        shrink = r["all_fp_mb"] / (b["memory_bytes"] / 1e6)
        kept = b["action"].get("keep_fp_units", [])
        console.print(
            f"[bold green]best:[/] perplexity {b['perplexity']:.2f} (≤ bar) at "
            f"[bold]{b['memory_bytes'] / 1e6:.0f} MB[/] ({shrink:.1f}× smaller than fp) — "
            f"{b['action'].get('quantizer')} + keep {kept or '[]'} fp.  "
            f"[dim]{b['action'].get('rationale', '')}[/]"
        )
    else:
        console.print("[yellow]no proposal cleared the bar within budget.[/]")
    return console.export_text()


def render_auto(result: dict, console: Console | None = None) -> str:
    """Render a deterministic auto-quant run: diagnosis → routed recipe →
    verified recovery vs the plain-quant baseline → residual attribution."""
    console = console or Console(record=True, width=100)
    r = result
    console.print(f"[bold]auto-quant[/] {r['model']}  (scheme={r['scheme']})")

    summary = r.get("diagnosis_summary", {})
    if summary:
        console.print(
            "[bold]diagnosis:[/] "
            + ", ".join(f"[yellow]{sig}[/] ×{n}" for sig, n in summary.items())
        )
    console.print("[bold]routed recipe:[/] " + "; ".join(r["routing"]))
    rec = r["recipe"]
    console.print(
        f"  quantizer=[bold]{rec['quantizer']}[/]  pre-transforms={rec['pre_transforms'] or '[]'}  "
        f"keep-fp={rec['kept_fp_fqns']}/{rec['kept_fp_fqns'] + rec['quantize_fqns']} Linears"
    )

    p = r["perplexity"]
    console.print(
        f"[bold]verified:[/] perplexity fp {p['fp']:.2f} → plain {r['scheme']} {p['plain']:.2f} "
        f"→ routed {p['routed']:.2f}"
    )
    if r["accepted"]:
        console.print(
            f"[bold green]ACCEPTED[/] — routed recipe recovers "
            f"[green]{r['recovery']:.0%}[/] of the degradation; ship it."
        )
    else:
        console.print(
            f"[bold red]REJECTED[/] by verification — the routed recipe didn't beat plain "
            f"{r['scheme']} (would ship plain). The measurement caught it: the proposed "
            f"intervention doesn't help this model."
        )
    if r["attribution_worst_taps"]:
        residual = ", ".join(f"{t} ({v:.1%})" for t, v in r["attribution_worst_taps"][:3])
        console.print(f"[dim]residual divergence concentrated at: {residual}[/]")
    return console.export_text()


def render_optimize(result: dict, console: Console | None = None) -> str:
    """Render the end-to-end optimize verdict: shipped recipe → quality vs the
    bar → cost (estimated + measured) → the deployable artifact + serve command,
    and any headroom a not-yet-servable recipe leaves on the table."""
    console = console or Console(record=True, width=100)
    r = result
    console.print(f"[bold]optimize[/] {r['model']}  (scheme={r['scheme']})")

    summary = r.get("diagnosis_summary", {})
    if summary:
        console.print(
            "[bold]diagnosis:[/] " + ", ".join(f"[yellow]{s}[/] ×{n}" for s, n in summary.items())
        )

    q = r["quality"]
    console.print(
        f"[bold]ship:[/] [cyan]{r['ship']}[/] {r['scheme']}  —  perplexity fp {q['fp']:.2f} → "
        f"shipped {q['shipped']:.2f} ([yellow]{q['rel_to_fp']:+.1%}[/] vs fp, torchao)"
    )
    if q.get("served") is not None:
        delta = q["backend_delta"]
        flag = "" if abs(delta) < 0.05 * q["fp"] else "  [yellow](backends disagree)[/]"
        console.print(
            f"[bold]served:[/] compressed-tensors perplexity {q['served']:.2f} "
            f"([yellow]{q['served_rel_to_fp']:+.1%}[/] vs fp)  "
            f"[dim]Δ {delta:+.2f} vs torchao[/]{flag}"
        )
    # The bar is checked against the served quality when we have it, else selection.
    basis_rel = q["served_rel_to_fp"] if q.get("served") is not None else q["rel_to_fp"]
    if r["quality_bar"] is not None:
        if r["meets_bar"]:
            console.print(f"[bold green]MEETS BAR[/] (≤ {r['quality_bar']:.1%} vs fp, {r['bar_basis']})")
        else:
            console.print(
                f"[bold red]MISSES BAR[/] ({basis_rel:+.1%} > {r['quality_bar']:.1%} vs fp, "
                f"{r['bar_basis']}) — try a milder scheme or relax the bar."
            )

    cost = f"[bold]cost:[/] ~{r['compression_estimate']:.1f}× smaller weights (est)"
    if r.get("measured"):
        m = r["measured"]
        wt = f", {m['weight_mb']:.0f} MB weights" if m.get("weight_mb") else ""
        cost += (
            f"  •  [bold]measured:[/] decode [green]{m['decode_tok_s']:.0f}[/] tok/s, "
            f"prefill {m['prefill_tok_s']:.0f} tok/s, TTFT {m['ttft_ms']:.0f} ms{wt}"
        )
    console.print(cost)

    art = r.get("artifact")
    if art:
        console.print(
            f"[bold]deployable:[/] {art['path']} ([dim]{art['compressed_tensors_scheme']}[/])\n"
            f"  [bold]serve:[/] [cyan]{art['serve_command']}[/]"
        )
    else:
        console.print("[dim]plan only — pass an output dir to export the servable checkpoint.[/]")

    h = r.get("headroom")
    if h:
        pre = "+".join(h["pre_transforms"]) if h["pre_transforms"] else h["recipe"]
        console.print(
            f"[dim]headroom: a [italic]{pre}[/] recipe recovers {h['recovery']:.0%} more "
            f"(perplexity {h['perplexity']:.2f}) but isn't directly servable yet "
            f"({h['blocked_by']})[/]"
        )
    return console.export_text()


def render_optimize_schemes(result: dict, console: Console | None = None) -> str:
    """Render a multi-scheme search: the per-scheme served-quality-vs-bar table,
    which scheme was chosen (most compression that meets the bar), then the
    winner's full optimize detail."""
    console = console or Console(record=True, width=100)
    r = result
    console.print(
        f"[bold]optimize (multi-scheme)[/] {r['model']}  —  bar ≤ "
        f"[cyan]{r['quality_bar']:.1%}[/] vs fp"
    )
    for s in r["per_scheme"]:
        rel = s["served_rel_to_fp"]
        rel_str = f"{rel:+.1%}" if rel is not None else "n/a"
        mark = "[green]✓[/]" if s["meets_bar"] else "[red]✗[/]"
        chosen = "  [bold cyan]← chosen[/]" if s["scheme"] == r["chosen_scheme"] else ""
        console.print(
            f"  {mark} [bold]{s['scheme']:8s}[/]  ~{s['compression']:.1f}× smaller  "
            f"served {rel_str} vs fp{chosen}"
        )
    if r["met_bar"]:
        console.print(
            f"[bold green]chosen {r['chosen_scheme']}[/] — the most compression that meets the bar"
        )
    else:
        console.print(
            f"[bold red]no scheme meets the bar[/] — shipping the best served quality "
            f"([cyan]{r['chosen_scheme']}[/])"
        )
    console.print("")
    console.print(render_optimize(r["winner"]))
    return console.export_text()


def render_salience(saliences, top_n: int = 15, console: Console | None = None) -> str:
    """Render the weight-salience (AWQ signal) sensor: Linears ranked by how
    concentrated their per-input-channel salience is. High = a few channels carry
    the weight, the case AWQ protects; ~1 = uniform."""
    console = console or Console(record=True, width=100)
    console.print(
        "[bold]weight salience[/] (AWQ signal: mean|X|·max|W| per input channel) "
        "[dim]— concentration = max / median[/]"
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("Linear", no_wrap=False)
    table.add_column("salience concentration", justify="right")
    table.add_column("channels", justify="right")
    for s in saliences[:top_n]:
        style = "bold yellow" if s.salience_concentration >= 100.0 else None
        table.add_row(s.fqn, f"{s.salience_concentration:.1f}x", str(s.n_channels), style=style)
    console.print(table)
    top = saliences[0].salience_concentration if saliences else 0.0
    console.print(
        f"[dim]A ranking signal, not pass/fail: a few Linears (outlier-feature layers) "
        f"dominate (top {top:.0f}x). AWQ protects those channels — route to the "
        f"AWQQuantizer (validated: ~91% int4 perplexity-gap recovery on Qwen2.5-7B).[/]"
    )
    return console.export_text()


def render_diagnosis(diagnosis, model_id: str = "<model>", console: Console | None = None) -> str:
    """Render a quant diagnosis: each detected failure-mode signature, where it
    is, the intervention it routes to, and the measured causal explanation —
    plus the verify command to run next. Honest by construction: it only shows
    signatures a real detector emitted."""
    console = console or Console(record=True, width=100)
    findings = diagnosis.findings
    if not findings:
        console.print(
            "[green]No quant failure-mode signatures detected[/] "
            "(no high-concentration activation-outlier taps)."
        )
        return console.export_text()

    console.print(f"[bold]Firefly quant diagnosis[/] — {len(findings)} finding(s)")
    routes: set[str] = set()
    for f in findings:
        routes.add(f.recommend)
        console.print(f"  [bold yellow]{f.signature}[/] @ [cyan]{f.location}[/] → {f.recommend}")
        console.print(f"    {f.explanation}")

    if "smoothquant" in routes:
        console.print(
            "\n[bold green]verify:[/] firefly quant-recipe -m "
            f"{model_id} -i <inputs> --smoothquant "
            "--accuracy-bar rel:0.01 --eval <eval> --metric perplexity"
        )
    if "mixed-precision" in routes:
        console.print(
            "[bold green]verify:[/] firefly quant-recipe -m "
            f"{model_id} -i <inputs> --accuracy-bar rel:0.01 --eval <eval> --metric perplexity"
        )
    return console.export_text()


def render_bar_recipe(result, console: Console | None = None) -> str:
    """Render the accuracy-bar recipe: the smallest keep-set that clears a real
    eval metric, with the candidates that were actually evaluated.

    The headline is the chosen recipe — keep k units in high precision and the
    quantized model stays inside the bar on the held-out eval. The table shows
    the binary-search probes (real evals), not a dense curve, so it doubles as a
    receipt of how few evals it took.
    """
    from firefly.quant.cost import format_bytes

    console = console or Console(record=True, width=100)
    noun = "Linears" if result.granularity == "linear" else "layers"
    metric = result.metric_name
    direction = "↑ higher better" if result.higher_is_better else "↓ lower better"
    bar_str = (
        f"{result.bar.value:.1%} rel" if result.bar.mode == "rel" else f"{result.bar.value:g} abs"
    )
    frontier, knee = result.frontier_knee_ks()

    console.print(
        f"[bold]{metric}[/] ({direction})  fp baseline {result.baseline_metric:.4g} → "
        f"all-{result.scheme} {result.full_quant_metric:.4g}   "
        f"[dim](bar {bar_str} → threshold {result.threshold:.4g}, "
        f"rank: {result.strategy}, {result.granularity})[/]"
    )
    if result.all_fp_bytes:
        console.print(
            f"[dim]weight footprint: all-fp {format_bytes(result.all_fp_bytes)} → "
            f"all-{result.scheme} {format_bytes(result.all_quant_bytes)} "
            f"({result.all_fp_bytes / result.all_quant_bytes:.1f}× smaller)[/]"
        )

    table = Table(
        title="Accuracy-bar recipe — evaluated candidates",
        show_header=True,
        header_style="bold",
    )
    table.add_column("keep hi-prec", justify="right")
    table.add_column(metric, justify="right")
    table.add_column("memory", justify="right")
    table.add_column("within bar?", justify="center")
    table.add_column("Pareto", justify="center")
    for p in result.evaluated:
        bar_mark = "[green]yes[/]" if p.passes else "[red]no[/]"
        pareto = "knee" if p.k == knee else ("frontier" if p.k in frontier else "")
        style = "bold green" if p.k == result.chosen_k else None
        table.add_row(
            f"{p.k}/{result.n_units}", f"{p.metric:.4g}", format_bytes(p.memory_bytes),
            bar_mark, pareto, style=style,
        )
    console.print(table)

    kept = ", ".join(result.chosen_kept_units) or "(none — fully quantized clears the bar)"
    compression = (
        result.all_fp_bytes / result.chosen_memory_bytes
        if result.chosen_memory_bytes else 1.0
    )
    console.print(
        f"[bold green]recipe:[/] keep [bold]{result.chosen_k}/{result.n_units}[/] {noun} "
        f"in high precision → {metric} {result.chosen_metric:.4g} "
        f"(threshold {result.threshold:.4g}), {format_bytes(result.chosen_memory_bytes)} "
        f"({compression:.1f}× smaller than fp).  kept: {kept}"
    )
    console.print(f"[dim]{result.evals_used} real evals spent (binary search + baseline + floor).[/]")
    return console.export_text()


def render_op_diff(result: OpDiffResult, top_n: int = 25, console: Console | None = None) -> str:
    """Render the op-by-op drill-down inside a module, in execution order.

    The first structural (op-name) or numerical (rel > tol) divergence is the
    headline — the ATen op where the two executions part inside the module.
    """
    console = console or Console(record=True, width=100)
    first = result.first_divergent

    table = Table(
        title=f"Firefly op drill-down — {result.module} ({len(result.divergences)} ops)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", justify="right")
    table.add_column("ATen op", no_wrap=True)
    table.add_column("rel |Δ|", justify="right")
    table.add_column("", justify="center")
    for d in result.divergences[:top_n]:
        rel = "—" if d.rel is None else f"{d.rel:.2%}"
        mark = "[red]⚠[/]" if d.structural else ("[red]✗[/]" if d.exceeds else "[green]✓[/]")
        style = "bold red" if first is not None and d.index == first.index else None
        table.add_row(str(d.index), d.op, rel, mark, style=style)
    console.print(table)
    if len(result.divergences) > top_n:
        console.print(f"[dim]… {len(result.divergences) - top_n} more ops[/]")

    if first is None:
        console.print(
            f"[bold green]No op exceeds {result.tol:.1%} relative divergence[/] "
            f"inside {result.module}."
        )
    elif first.structural:
        console.print(
            f"[bold red]Structural divergence[/] at op #{first.index} (`{first.op}`): "
            f"the op graphs differ here (ref {result.n_ref_ops} ops, cand {result.n_cand_ops})."
        )
    else:
        console.print(
            f"[bold red]First divergence[/] at op #{first.index}: `{first.op}` "
            f"({first.rel:.2%} rel, tol {result.tol:.1%})."
        )
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
