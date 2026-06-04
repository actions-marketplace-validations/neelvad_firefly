"""Smoke tests: the package imports and the CLI wires up."""

from __future__ import annotations

from pathlib import Path

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
    for option in (
        "--reference",
        "--candidate",
        "--inputs",
        "--device",
        "--report-json",
        "--allow-fingerprint",  # truncated by narrow help-text width in CliRunner
    ):
        assert option in result.stdout


def test_check_command_errors_on_missing_required_args() -> None:
    result = runner.invoke(app, ["check"])
    assert result.exit_code != 0


def test_check_advertises_allow_default_tolerances_flag() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    assert "--allow-default-tolerances" in result.stdout


def test_check_refuses_without_calibration(tmp_path: Path) -> None:
    """If tolerances.json doesn't exist in the reference dir, refuse the gate.

    We don't need a real reference here — the calibration check fires *before*
    any artifact loading.
    """
    ref = tmp_path / "ref"
    ref.mkdir()
    inputs = tmp_path / "x.json"
    inputs.write_text("{}")

    result = runner.invoke(
        app,
        [
            "check",
            "--reference", str(ref),
            "--candidate", "HuggingFaceTB/SmolLM-135M",
            "--inputs", str(inputs),
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "tolerances.json" in combined
    assert "firefly calibrate" in combined


def test_calibrate_command_advertises_options() -> None:
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    for option in (
        "--reference",
        "--inputs",
        "--runs",
        "--safety-factor",
        "--noise-mode",
        "--noise-sigma",
        "--noise-inject-at",
        "--noise-base-seed",
        "--allow-tf32",
        "--device",
    ):
        assert option in result.stdout


def test_calibrate_command_errors_on_missing_required_args() -> None:
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code != 0


def test_calibrate_rejects_synthetic_without_sigma(tmp_path: Path) -> None:
    """--noise-mode=synthetic requires --noise-sigma > 0 and --noise-inject-at."""
    ref = tmp_path / "ref"
    ref.mkdir()
    inputs = tmp_path / "x.json"
    inputs.write_text("{}")

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--reference", str(ref),
            "--inputs", str(inputs),
            "--noise-mode", "synthetic",
        ],
    )
    assert result.exit_code != 0
    assert "noise-sigma" in result.output.lower() or "noise-sigma" in result.stderr.lower()


def test_calibrate_rejects_unknown_noise_mode(tmp_path: Path) -> None:
    ref = tmp_path / "ref"
    ref.mkdir()
    inputs = tmp_path / "x.json"
    inputs.write_text("{}")

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--reference", str(ref),
            "--inputs", str(inputs),
            "--noise-mode", "magic",
        ],
    )
    assert result.exit_code != 0
