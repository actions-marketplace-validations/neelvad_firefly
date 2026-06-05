"""Tests for the reference-storage URI resolver."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from firefly.storage import resolve_reference


def test_local_path_passes_through(tmp_path: Path) -> None:
    assert resolve_reference(tmp_path) == tmp_path


def test_relative_path_passes_through() -> None:
    assert resolve_reference("./reference") == Path("./reference")


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


# --- S3 backend -------------------------------------------------------------


def _fake_s3_client(objects: dict[str, tuple[str, bytes]]) -> MagicMock:
    """Build a MagicMock s3 client backed by an in-memory object map.

    ``objects`` maps key -> (etag, body). The fake supports paginated
    ``list_objects_v2`` and ``download_file`` (writes the body to disk).
    """
    client = MagicMock()

    def paginate(*, Bucket: str, Prefix: str = ""):
        contents = [
            {"Key": k, "ETag": etag}
            for k, (etag, _body) in objects.items()
            if k.startswith(Prefix)
        ]
        yield {"Contents": contents}

    client.get_paginator.return_value.paginate.side_effect = paginate

    def download_file(bucket: str, key: str, local_path: str) -> None:
        _etag, body = objects[key]
        Path(local_path).write_bytes(body)

    client.download_file.side_effect = download_file
    return client


def test_s3_uri_mirrors_prefix_to_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects = {
        "refs/v1/tap_index.json": ("etag-a", b'{"taps": []}'),
        "refs/v1/activations.safetensors": ("etag-b", b"binary-blob"),
        "refs/v1/nested/extra.json": ("etag-c", b"{}"),
    }
    client = _fake_s3_client(objects)

    with patch("boto3.client", return_value=client):
        result = resolve_reference("s3://my-bucket/refs/v1")

    assert (result / "tap_index.json").read_bytes() == b'{"taps": []}'
    assert (result / "activations.safetensors").read_bytes() == b"binary-blob"
    assert (result / "nested" / "extra.json").read_bytes() == b"{}"
    assert (result / "_manifest.json").exists()


def test_s3_uri_skips_redownload_when_etag_matches(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects = {"refs/v1/file.json": ("etag-a", b"{}")}
    client = _fake_s3_client(objects)

    with patch("boto3.client", return_value=client):
        resolve_reference("s3://my-bucket/refs/v1")
        first_calls = client.download_file.call_count
        resolve_reference("s3://my-bucket/refs/v1")
        second_calls = client.download_file.call_count

    assert first_calls == 1
    assert second_calls == 1, "ETag-matched object should not be re-downloaded"


def test_s3_uri_redownloads_when_etag_changes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects_v1 = {"refs/v1/file.json": ("etag-a", b'{"v": 1}')}
    objects_v2 = {"refs/v1/file.json": ("etag-b", b'{"v": 2}')}

    with patch("boto3.client", return_value=_fake_s3_client(objects_v1)):
        result = resolve_reference("s3://my-bucket/refs/v1")
        assert (result / "file.json").read_bytes() == b'{"v": 1}'

    with patch("boto3.client", return_value=_fake_s3_client(objects_v2)):
        result = resolve_reference("s3://my-bucket/refs/v1")
        assert (result / "file.json").read_bytes() == b'{"v": 2}'


def test_s3_uri_removes_objects_deleted_upstream(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects_v1 = {
        "refs/v1/keep.json": ("etag-a", b"a"),
        "refs/v1/gone.json": ("etag-b", b"b"),
    }
    objects_v2 = {"refs/v1/keep.json": ("etag-a", b"a")}

    with patch("boto3.client", return_value=_fake_s3_client(objects_v1)):
        result = resolve_reference("s3://my-bucket/refs/v1")
        assert (result / "gone.json").exists()

    with patch("boto3.client", return_value=_fake_s3_client(objects_v2)):
        result = resolve_reference("s3://my-bucket/refs/v1")
        assert (result / "keep.json").exists()
        assert not (result / "gone.json").exists()


def test_s3_uri_ignores_directory_marker_objects(
    tmp_path: Path, monkeypatch
) -> None:
    """0-byte objects with keys ending in ``/`` (S3 console folder markers) skipped."""
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects = {
        "refs/v1/": ("etag-marker", b""),
        "refs/v1/file.json": ("etag-a", b"{}"),
    }
    client = _fake_s3_client(objects)

    with patch("boto3.client", return_value=client):
        result = resolve_reference("s3://my-bucket/refs/v1")

    assert (result / "file.json").exists()
    # The marker should not have created an empty file at the root.
    assert not (result / "").is_file()


def test_s3_uri_root_prefix(tmp_path: Path, monkeypatch) -> None:
    """``s3://bucket`` (no prefix) mirrors the bucket root."""
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects = {"file.json": ("etag-a", b"{}")}
    client = _fake_s3_client(objects)

    with patch("boto3.client", return_value=client):
        result = resolve_reference("s3://my-bucket")

    assert (result / "file.json").exists()


def test_s3_missing_boto3_raises_import_error(monkeypatch) -> None:
    """Helpful error pointing at the install command when boto3 isn't installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="firefly\\[s3\\]"):
        resolve_reference("s3://my-bucket/refs/v1")
