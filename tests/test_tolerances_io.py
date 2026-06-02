"""Tests for the on-disk tolerances.json artifact."""

from __future__ import annotations

import json
from pathlib import Path

from firefly.compare import (
    TOLERANCES_FILE,
    TapTolerance,
    read_tolerances,
    write_tolerances,
)


def _sample_tolerances() -> dict[str, TapTolerance]:
    return {
        "layer.0": TapTolerance(
            atol=1.5e-5,
            source="calibrated",
            noise_floor=2.1e-7,
            n_calibration_runs=16,
        ),
        "layer.1": TapTolerance(atol=1e-3, source="manual"),
        "layer.2": TapTolerance(atol=1e-5, source="default"),
    }


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    tolerances = _sample_tolerances()

    write_tolerances(tmp_path, tolerances)
    loaded = read_tolerances(tmp_path)

    assert loaded == tolerances


def test_read_returns_none_when_file_absent(tmp_path: Path) -> None:
    assert read_tolerances(tmp_path) is None


def test_artifact_is_human_readable(tmp_path: Path) -> None:
    write_tolerances(tmp_path, _sample_tolerances())

    with (tmp_path / TOLERANCES_FILE).open() as f:
        data = json.load(f)

    assert "tolerances" in data
    assert data["tolerances"]["layer.0"]["atol"] == 1.5e-5
    assert data["tolerances"]["layer.0"]["source"] == "calibrated"
    assert data["tolerances"]["layer.0"]["n_calibration_runs"] == 16


def test_write_creates_directory(tmp_path: Path) -> None:
    new_dir = tmp_path / "fresh"
    assert not new_dir.exists()

    write_tolerances(new_dir, _sample_tolerances())

    assert (new_dir / TOLERANCES_FILE).exists()
