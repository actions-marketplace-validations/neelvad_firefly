"""Smoke tests: the package imports and the CLI wires up."""

from __future__ import annotations

from typer.testing import CliRunner

import firefly
from firefly.cli import app

runner = CliRunner()


def test_package_imports() -> None:
    assert firefly.__version__


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "capture" in result.stdout
    assert "calibrate" in result.stdout
    assert "check" in result.stdout


def test_capture_stub() -> None:
    result = runner.invoke(
        app,
        ["capture", "--model", "HuggingFaceTB/SmolLM-135M", "--inputs", "x.json", "--out", "ref/"],
    )
    assert result.exit_code == 0
    assert "[stub] capture" in result.stdout
