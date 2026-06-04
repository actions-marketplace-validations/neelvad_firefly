"""Reproducible runner for the vLLM parity test matrix.

Reads ``scripts/vllm_test_suite.yml``, loads each declared pair of
reference dirs, computes the per-tap diff (via the same compare path
production CI uses), validates the result against the declared
expectation, and emits a markdown summary table.

Usage:
    uv run python scripts/run_vllm_suite.py
    uv run python scripts/run_vllm_suite.py --config scripts/vllm_test_suite.yml
    uv run python scripts/run_vllm_suite.py --out scripts/results/suite_summary.md

Exit code:
    0 — all tests passed their expectations
    1 — one or more tests failed (script still emits the full report)
    2 — config / loading error before any test could run

Supported expectation shapes:
    bit_equal
        Asserts max_abs_diff == 0 on every tap.

    first_divergence: <tap_name>
        Asserts the first divergent tap (in forward order) matches.

    n_divergent_min: <int>
        Asserts at least N taps exceed their tolerance.

These compose: a single ``expected`` block may set both
``first_divergence`` and ``n_divergent_min``.

Extending the matrix is just adding a YAML entry; no Python edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
import yaml

from firefly.attribution import attribute_first_divergence
from firefly.compare import TapTolerance, diff_captures
from firefly.reference import read_reference

app = typer.Typer(add_completion=False, no_args_is_help=False)


@dataclass
class TestResult:
    name: str
    description: str
    passed: bool
    actual: str         # short human summary, eg "bit_equal (0 diff taps)"
    expected: str       # rendered expectation for the report
    notes: list[str]    # any per-check failure messages


def _diff_pair(ref_a_dir: Path, ref_b_dir: Path):
    """Diff two reference dirs. Returns (divergences, attribution_result, total_taps)."""
    manifest_a, tensors_a = read_reference(ref_a_dir)
    _manifest_b, tensors_b = read_reference(ref_b_dir)
    # Use the reference's own tap_points list as the forward order; with
    # both refs sharing it, we get the same diff ordering production gets.
    # Calibration is *not* required here — we apply a tight default atol
    # (1e-6) so even small numerical differences are flagged. The matrix
    # is built around exact-equality and known-divergence expectations.
    tolerances = {name: TapTolerance(atol=1e-6) for name in manifest_a.tap_points}
    divergences = diff_captures(
        tensors_a, tensors_b, manifest_a.tap_points, tolerances=tolerances
    )
    attribution = attribute_first_divergence(divergences)
    return divergences, attribution, len(manifest_a.tap_points)


def _evaluate(expected, divergences, attribution) -> tuple[bool, list[str]]:
    """Return (passed, notes)."""
    notes: list[str] = []

    # bit_equal as a string expectation
    if expected == "bit_equal":
        nonzero = [d for d in divergences if d.max_abs_diff > 0]
        if nonzero:
            return False, [f"expected bit_equal; {len(nonzero)} tap(s) had nonzero diff"]
        return True, []

    if not isinstance(expected, dict):
        return False, [f"unrecognized expected: {expected!r}"]

    passed = True
    fd = expected.get("first_divergence")
    if fd is not None and attribution.first_divergent_tap != fd:
        passed = False
        notes.append(
            f"expected first_divergence={fd!r}, got {attribution.first_divergent_tap!r}"
        )

    n_min = expected.get("n_divergent_min")
    if n_min is not None:
        n_actual = sum(1 for d in divergences if d.exceeds_tolerance)
        if n_actual < n_min:
            passed = False
            notes.append(f"expected n_divergent >= {n_min}, got {n_actual}")

    return passed, notes


def _summarize_actual(expected, divergences, attribution, total) -> str:
    n_nonzero = sum(1 for d in divergences if d.max_abs_diff > 0)
    if expected == "bit_equal":
        return "bit_equal" if n_nonzero == 0 else f"{n_nonzero}/{total} diff"
    parts = []
    if attribution.first_divergent_tap:
        parts.append(f"first={attribution.first_divergent_tap}")
    parts.append(f"diff={sum(1 for d in divergences if d.exceeds_tolerance)}/{total}")
    return ", ".join(parts)


def _render_expected(expected) -> str:
    if isinstance(expected, str):
        return expected
    parts = []
    if "first_divergence" in expected:
        parts.append(f"first={expected['first_divergence']}")
    if "n_divergent_min" in expected:
        parts.append(f"n_diff≥{expected['n_divergent_min']}")
    return ", ".join(parts)


def _run_one_test(test: dict, project_root: Path) -> TestResult:
    name = test["name"]
    description = test.get("description", "")
    expected = test["expected"]
    ref_a = project_root / test["reference_a"]
    ref_b = project_root / test["reference_b"]

    if not ref_a.exists() or not (ref_a / "manifest.json").exists():
        return TestResult(
            name=name, description=description,
            passed=False,
            actual="MISSING reference_a",
            expected=_render_expected(expected),
            notes=[f"reference_a not found: {ref_a}"],
        )
    if not ref_b.exists() or not (ref_b / "manifest.json").exists():
        return TestResult(
            name=name, description=description,
            passed=False,
            actual="MISSING reference_b",
            expected=_render_expected(expected),
            notes=[f"reference_b not found: {ref_b}"],
        )

    try:
        divergences, attribution, total = _diff_pair(ref_a, ref_b)
    except ValueError as e:
        # Shape mismatch or missing tap — itself a meaningful finding.
        # Common cause: V0 vs V1 batch differently (per-prompt vs packed).
        return TestResult(
            name=name, description=description,
            passed=(expected == "shape_mismatch"),
            actual=f"shape_mismatch: {e}",
            expected=_render_expected(expected),
            notes=[] if expected == "shape_mismatch" else [str(e)],
        )

    passed, notes = _evaluate(expected, divergences, attribution)
    actual = _summarize_actual(expected, divergences, attribution, total)

    return TestResult(
        name=name, description=description,
        passed=passed, actual=actual,
        expected=_render_expected(expected),
        notes=notes,
    )


def _render_markdown(suite_name: str, results: list[TestResult]) -> str:
    lines: list[str] = []
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    headline = "✅" if n_pass == n_total else "❌"
    lines.append(f"# {suite_name}")
    lines.append("")
    lines.append(f"{headline} **{n_pass} / {n_total}** tests passed.")
    lines.append("")
    lines.append("| Test | Result | Expected | Actual |")
    lines.append("| --- | :-: | --- | --- |")
    for r in results:
        marker = "✅" if r.passed else "❌"
        lines.append(f"| {r.name} | {marker} | `{r.expected}` | `{r.actual}` |")
    lines.append("")

    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(f"### {r.name}")
            if r.description:
                lines.append(f"_{r.description}_")
            for note in r.notes:
                lines.append(f"- {note}")
            lines.append("")

    return "\n".join(lines) + "\n"


@app.command()
def main(
    config: Path = typer.Option(
        Path("scripts/vllm_test_suite.yml"),
        "--config", "-c",
        help="Path to the suite YAML.",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Optional path to write the markdown report. Always also printed to stdout.",
    ),
    project_root: Path = typer.Option(
        Path.cwd(), "--project-root",
        help="Resolve relative reference paths against this directory.",
    ),
) -> None:
    """Run the vLLM parity test matrix and emit a markdown summary."""
    if not config.exists():
        typer.echo(f"ERROR: config not found: {config}", err=True)
        raise typer.Exit(code=2)
    with config.open() as f:
        suite = yaml.safe_load(f)

    suite_name = suite.get("suite_name", "vLLM Parity Suite")
    tests = suite.get("tests", [])
    if not tests:
        typer.echo("ERROR: no tests declared in suite config.", err=True)
        raise typer.Exit(code=2)

    results = [_run_one_test(t, project_root) for t in tests]
    report = _render_markdown(suite_name, results)
    typer.echo(report)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        typer.echo(f"Wrote report to {out}")

    if any(not r.passed for r in results):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
