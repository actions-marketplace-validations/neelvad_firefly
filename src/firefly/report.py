"""Report formatting: structured JSON for machines, rich tables for humans."""

from __future__ import annotations

from pathlib import Path

from firefly.attribution import AttributionResult


def render_human(result: AttributionResult) -> str:
    """Pretty terminal output via rich."""
    raise NotImplementedError


def write_json(result: AttributionResult, path: Path) -> None:
    """Structured report — what the GitHub Action consumes."""
    raise NotImplementedError
