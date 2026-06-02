"""Round-trip and validation tests for the on-disk reference artifact."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from firefly.reference import (
    ReferenceManifest,
    capture_env,
    read_reference,
    write_reference,
)


def _make_manifest() -> ReferenceManifest:
    return ReferenceManifest(
        model_id="HuggingFaceTB/SmolLM-135M",
        model_fingerprint="deadbeef",
        tap_points=["layer.0", "layer.1", "final_norm"],
        shapes={"layer.0": [1, 4, 8], "layer.1": [1, 4, 8], "final_norm": [1, 4, 8]},
        dtypes={"layer.0": "float32", "layer.1": "float32", "final_norm": "float32"},
        captured_at=datetime.now(UTC).isoformat(),
        env=capture_env(),
    )


def _make_tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "layer.0": torch.randn(1, 4, 8),
        "layer.1": torch.randn(1, 4, 8),
        "final_norm": torch.randn(1, 4, 8),
    }


def test_round_trip(tmp_path: Path) -> None:
    manifest = _make_manifest()
    tensors = _make_tensors()

    write_reference(tmp_path, manifest, tensors)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "weights.safetensors").exists()

    loaded_manifest, loaded_tensors = read_reference(tmp_path)

    assert loaded_manifest == manifest
    assert set(loaded_tensors.keys()) == set(tensors.keys())
    for name, tensor in tensors.items():
        assert torch.equal(loaded_tensors[name], tensor)


def test_manifest_is_human_readable_json(tmp_path: Path) -> None:
    """Manifest must be inspectable by a human without running code."""
    write_reference(tmp_path, _make_manifest(), _make_tensors())

    with (tmp_path / "manifest.json").open() as f:
        data = json.load(f)

    assert data["schema_version"] == 1
    assert data["model_id"] == "HuggingFaceTB/SmolLM-135M"
    assert "torch" in data["env"]


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    manifest = _make_manifest()
    write_reference(tmp_path, manifest, _make_tensors())

    with (tmp_path / "manifest.json").open() as f:
        data = json.load(f)
    data["schema_version"] = 999
    with (tmp_path / "manifest.json").open("w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Unsupported reference schema_version"):
        read_reference(tmp_path)


def test_missing_files_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No manifest"):
        read_reference(tmp_path)
