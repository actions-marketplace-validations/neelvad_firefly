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


def test_capture_command_advertises_options() -> None:
    result = runner.invoke(app, ["capture", "--help"])
    assert result.exit_code == 0
    for option in ("--model", "--inputs", "--out", "--device", "--seed"):
        assert option in result.stdout


def test_capture_command_errors_on_missing_required_args() -> None:
    result = runner.invoke(app, ["capture"])
    assert result.exit_code != 0


def test_check_command_advertises_options() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    for option in ("--reference", "--candidate", "--inputs", "--device", "--report-json"):
        assert option in result.stdout


def test_check_command_errors_on_missing_required_args() -> None:
    result = runner.invoke(app, ["check"])
    assert result.exit_code != 0
