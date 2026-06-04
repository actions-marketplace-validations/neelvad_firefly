"""Tests for the reference-storage URI resolver."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from firefly.storage import resolve_reference


def test_local_path_passes_through(tmp_path: Path) -> None:
    assert resolve_reference(tmp_path) == tmp_path


def test_relative_path_passes_through() -> None:
    assert resolve_reference("./reference") == Path("./reference")


def test_planned_s3_scheme_raises_with_version() -> None:
    with pytest.raises(NotImplementedError, match="planned for v2"):
        resolve_reference("s3://my-bucket/ref")


def test_planned_gcs_scheme_raises_with_version() -> None:
    with pytest.raises(NotImplementedError, match="planned for v3"):
        resolve_reference("gs://my-bucket/ref")


def test_planned_azure_scheme_raises_with_version() -> None:
    with pytest.raises(NotImplementedError, match="planned for v3"):
        resolve_reference("az://my-container/ref")


def test_hf_uri_invokes_snapshot_download(tmp_path: Path) -> None:
    """An hf://org/repo URI should call snapshot_download with the parsed repo_id."""
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as mock:
        result = resolve_reference("hf://my-org/my-ref")
        mock.assert_called_once_with(
            repo_id="my-org/my-ref",
            revision=None,
            repo_type="model",
        )
        assert result == tmp_path


def test_hf_uri_with_revision(tmp_path: Path) -> None:
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as mock:
        resolve_reference("hf://my-org/my-ref@main")
        mock.assert_called_once_with(
            repo_id="my-org/my-ref",
            revision="main",
            repo_type="model",
        )


def test_hf_uri_with_subpath(tmp_path: Path) -> None:
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)):
        result = resolve_reference("hf://my-org/my-ref/nested/dir")
        assert result == tmp_path / "nested/dir"


def test_hf_uri_with_revision_and_subpath(tmp_path: Path) -> None:
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as mock:
        result = resolve_reference("hf://my-org/my-ref@v1.0/ref")
        mock.assert_called_once_with(
            repo_id="my-org/my-ref",
            revision="v1.0",
            repo_type="model",
        )
        assert result == tmp_path / "ref"


def test_huggingface_long_scheme_alias_works(tmp_path: Path) -> None:
    """Both ``hf://`` and ``huggingface://`` should resolve."""
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as mock:
        resolve_reference("huggingface://my-org/my-ref")
        mock.assert_called_once()


def test_malformed_hf_uri_raises_value_error() -> None:
    """Missing repo (only org, no /) should fail with a clear message."""
    with pytest.raises(ValueError, match="hf://"):
        resolve_reference("hf://just-an-org")


def test_windows_drive_letter_treated_as_path() -> None:
    """Single-letter 'schemes' like ``c:`` are Windows drive letters, not URIs."""
    # This shouldn't try to raise NotImplementedError or call HF.
    assert resolve_reference("c:/some/path") == Path("c:/some/path")
