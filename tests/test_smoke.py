"""Smoke tests: the package imports and the CLI wires up."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

import firefly
from firefly.cli import app

runner = CliRunner()

# typer/rich sometimes emits ANSI color codes through CliRunner (the policy
# depends on the click/rich version + the TERM env in the runner). Colored
# output splits the dashes off ``--option`` into separate escape-wrapped
# spans, so a plain substring check for ``--option`` misses. Strip ANSI
# before any text-presence assertion to make the tests robust across
# environments.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(s: str) -> str:
    return _ANSI_RE.sub("", s)


def test_package_imports() -> None:
    assert firefly.__version__


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _plain(result.stdout)
    assert "capture" in out
    assert "calibrate" in out
    assert "check" in out
    assert "publish" in out


def test_publish_help_lists_to_flag() -> None:
    result = runner.invoke(app, ["publish", "--help"])
    assert result.exit_code == 0
    assert "--to" in _plain(result.stdout)


def test_capture_help_lists_push_flag() -> None:
    result = runner.invoke(app, ["capture", "--help"])
    assert result.exit_code == 0
    assert "--push" in _plain(result.stdout)


def test_calibrate_help_lists_push_flag() -> None:
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    assert "--push" in _plain(result.stdout)


def test_publish_malformed_uri_exits_cleanly(tmp_path: Path) -> None:
    """`firefly publish --to <malformed>` should exit non-zero with a clear message."""
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "manifest.json").write_text("{}")

    # az:// with only a container (missing account/) should fail the URI regex.
    result = runner.invoke(
        app,
        ["publish", "--reference", str(ref), "--to", "az://just-one-segment"],
    )
    assert result.exit_code != 0
    combined = _plain(result.output + (result.stderr or ""))
    assert "az://" in combined


def test_capture_command_advertises_options() -> None:
    result = runner.invoke(app, ["capture", "--help"])
    assert result.exit_code == 0
    out = _plain(result.stdout)
    for option in ("--model", "--inputs", "--out", "--device", "--seed"):
        assert option in out


def test_capture_command_errors_on_missing_required_args() -> None:
    result = runner.invoke(app, ["capture"])
    assert result.exit_code != 0


def test_check_command_advertises_options() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    out = _plain(result.stdout)
    for option in (
        "--reference",
        "--candidate",
        "--inputs",
        "--device",
        "--report-json",
        "--allow-fingerprint",  # truncated by narrow help-text width in CliRunner
    ):
        assert option in out


def test_check_command_errors_on_missing_required_args() -> None:
    result = runner.invoke(app, ["check"])
    assert result.exit_code != 0


def test_check_advertises_allow_default_tolerances_flag() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    assert "--allow-default-tolerances" in _plain(result.stdout)


def test_check_emits_clean_error_on_malformed_remote_uri(tmp_path: Path) -> None:
    """A malformed az:// URI should error before getting to the
    'tolerances.json not found' check."""
    inputs = tmp_path / "x.json"
    inputs.write_text("{}")

    result = runner.invoke(
        app,
        [
            "check",
            "--reference", "az://just-one-segment",
            "--candidate", "HuggingFaceTB/SmolLM-135M",
            "--inputs", str(inputs),
        ],
    )
    assert result.exit_code != 0
    combined = _plain(result.output + (result.stderr or ""))
    assert "az://" in combined


def test_calibrate_emits_clean_error_on_malformed_remote_uri(tmp_path: Path) -> None:
    """Same URI validation should fire on calibrate."""
    inputs = tmp_path / "x.json"
    inputs.write_text("{}")

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--reference", "az://just-one-segment",
            "--inputs", str(inputs),
        ],
    )
    assert result.exit_code != 0
    combined = _plain(result.output + (result.stderr or ""))
    assert "az://" in combined


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
    combined = _plain(result.output + (result.stderr or ""))
    assert "tolerances.json" in combined
    assert "firefly calibrate" in combined


def test_calibrate_command_advertises_options() -> None:
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    out = _plain(result.stdout)
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
        assert option in out


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
    combined = _plain(result.output + (result.stderr or "")).lower()
    assert "noise-sigma" in combined


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
