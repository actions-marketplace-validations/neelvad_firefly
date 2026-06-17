"""Round-trip and validation tests for the on-disk reference artifact."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from firefly.reference import (
    SCHEMA_VERSION,
    ReferenceManifest,
    capture_env,
    read_manifest,
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


def test_runner_metadata_round_trips(tmp_path: Path) -> None:
    manifest = _make_manifest()
    manifest.runner = "vllm"
    manifest.runner_options = {"attention_backend": "FLASH_ATTN"}
    write_reference(tmp_path, manifest, _make_tensors())

    loaded = read_manifest(tmp_path)
    assert loaded.runner == "vllm"
    assert loaded.runner_options == {"attention_backend": "FLASH_ATTN"}


def test_old_reference_defaults_runner_to_hf(tmp_path: Path) -> None:
    """A manifest written before the runner field existed reads back as 'hf'."""
    data = {
        "model_id": "m", "model_fingerprint": "x", "tap_points": ["final_norm"],
        "shapes": {"final_norm": [1, 4]}, "dtypes": {"final_norm": "float32"},
        "captured_at": datetime.now(UTC).isoformat(), "schema_version": SCHEMA_VERSION,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    loaded = read_manifest(tmp_path)
    assert loaded.runner == "hf"
    assert loaded.runner_options == {}


def test_read_manifest_tolerates_unknown_keys(tmp_path: Path) -> None:
    """A newer writer's additive field doesn't crash an older reader."""
    data = {
        "model_id": "m", "model_fingerprint": "x", "tap_points": ["final_norm"],
        "shapes": {"final_norm": [1, 4]}, "dtypes": {"final_norm": "float32"},
        "captured_at": datetime.now(UTC).isoformat(), "schema_version": SCHEMA_VERSION,
        "some_future_field": 42,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    loaded = read_manifest(tmp_path)  # must not raise
    assert loaded.model_id == "m"


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
