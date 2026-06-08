"""Tests for the reference-storage URI resolver."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from firefly.storage import publish_reference, resolve_reference


def test_local_path_passes_through(tmp_path: Path) -> None:
    assert resolve_reference(tmp_path) == tmp_path


def test_relative_path_passes_through() -> None:
    assert resolve_reference("./reference") == Path("./reference")


def test_unknown_scheme_treated_as_path() -> None:
    """Unknown schemes pass through as local paths (matches Path's permissive behavior)."""
    # No more planned-vN stubs — all 4 cloud backends ship.
    # An obviously-not-a-real scheme like ``r2://`` is just a path.
    assert resolve_reference("r2://bucket/ref") == Path("r2://bucket/ref")


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


# --- publish_reference ------------------------------------------------------


def _make_reference_dir(root: Path) -> Path:
    """Build a minimal reference dir layout for publish tests."""
    ref = root / "reference"
    ref.mkdir()
    (ref / "manifest.json").write_text('{"taps": []}')
    (ref / "weights.safetensors").write_bytes(b"\x00\x01\x02")
    (ref / "tolerances.json").write_text("{}")
    return ref


def test_publish_local_path_rejected(tmp_path: Path) -> None:
    """Publishing to a plain local path should error with a hint."""
    ref = _make_reference_dir(tmp_path)
    with pytest.raises(ValueError, match="cp -r"):
        publish_reference(ref, str(tmp_path / "elsewhere"))


def test_publish_missing_reference_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        publish_reference(tmp_path / "nope", "hf://my-org/my-ref")


def test_publish_file_instead_of_dir_raises(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hi")
    with pytest.raises(ValueError, match="must be a directory"):
        publish_reference(f, "hf://my-org/my-ref")


def test_publish_unknown_scheme_treated_as_path(tmp_path: Path) -> None:
    """An unknown scheme (no planned-vN stubs left) should not be a remote target."""
    ref = _make_reference_dir(tmp_path)
    # Scheme dispatcher doesn't know r2://, so _extract_scheme returns "r2"
    # which isn't in any known set and isn't planned — raises ValueError.
    with pytest.raises(ValueError, match="Unknown URI scheme"):
        publish_reference(ref, "r2://my-bucket/ref")


def test_publish_hf_calls_upload_folder(tmp_path: Path) -> None:
    ref = _make_reference_dir(tmp_path)
    fake_api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        publish_reference(ref, "hf://my-org/my-ref")

    fake_api.create_repo.assert_called_once_with(
        repo_id="my-org/my-ref", repo_type="model", exist_ok=True
    )
    fake_api.upload_folder.assert_called_once()
    kwargs = fake_api.upload_folder.call_args.kwargs
    assert kwargs["folder_path"] == str(ref)
    assert kwargs["repo_id"] == "my-org/my-ref"
    assert kwargs["repo_type"] == "model"
    assert kwargs["revision"] is None
    assert kwargs["path_in_repo"] == ""
    assert kwargs["commit_message"] == "Firefly reference upload"


def test_publish_hf_passes_revision_and_subpath(tmp_path: Path) -> None:
    ref = _make_reference_dir(tmp_path)
    fake_api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        publish_reference(
            ref,
            "hf://my-org/my-ref@dev/nested/dir",
            commit_message="calibration v2",
        )
    kwargs = fake_api.upload_folder.call_args.kwargs
    assert kwargs["revision"] == "dev"
    assert kwargs["path_in_repo"] == "nested/dir"
    assert kwargs["commit_message"] == "calibration v2"


def test_publish_hf_wraps_errors_with_token_hint(tmp_path: Path) -> None:
    ref = _make_reference_dir(tmp_path)
    fake_api = MagicMock()
    fake_api.upload_folder.side_effect = RuntimeError("401 Unauthorized")
    with (
        patch("huggingface_hub.HfApi", return_value=fake_api),
        pytest.raises(RuntimeError, match="HF_TOKEN"),
    ):
        publish_reference(ref, "hf://my-org/my-ref")


def test_publish_s3_uploads_each_file(tmp_path: Path) -> None:
    ref = _make_reference_dir(tmp_path)
    client = MagicMock()
    with patch("boto3.client", return_value=client):
        publish_reference(ref, "s3://my-bucket/refs/v1")

    # 3 files in the reference dir → 3 upload_file calls.
    assert client.upload_file.call_count == 3
    keys = sorted(call.args[2] for call in client.upload_file.call_args_list)
    assert keys == [
        "refs/v1/manifest.json",
        "refs/v1/tolerances.json",
        "refs/v1/weights.safetensors",
    ]


def test_publish_s3_root_prefix(tmp_path: Path) -> None:
    """``s3://bucket`` (no prefix) uploads to bucket root."""
    ref = _make_reference_dir(tmp_path)
    client = MagicMock()
    with patch("boto3.client", return_value=client):
        publish_reference(ref, "s3://my-bucket")
    keys = sorted(call.args[2] for call in client.upload_file.call_args_list)
    assert keys == ["manifest.json", "tolerances.json", "weights.safetensors"]


# --- GCS backend ------------------------------------------------------------


def _fake_gcs_client(objects: dict[str, tuple[str, bytes]]) -> MagicMock:
    """Build a MagicMock GCS client backed by an in-memory object map.

    ``objects`` maps blob_name -> (etag, body). Tracks two counters on
    the client itself for tests to inspect:

    * ``client.downloads`` — list of (key, local_path) tuples written
      by ``download_to_filename``.
    * ``client.uploads`` — list of (key, local_path) tuples written by
      ``bucket(...).blob(key).upload_from_filename``.
    """
    client = MagicMock()
    client.downloads = []
    client.uploads = []

    def make_blob(name: str, etag: str, body: bytes) -> MagicMock:
        blob = MagicMock()
        blob.name = name
        blob.etag = etag

        def download_to_filename(local_path: str) -> None:
            Path(local_path).write_bytes(body)
            client.downloads.append((name, local_path))

        blob.download_to_filename.side_effect = download_to_filename
        return blob

    def list_blobs(_bucket, prefix: str = ""):
        return [
            make_blob(name, etag, body)
            for name, (etag, body) in objects.items()
            if name.startswith(prefix)
        ]

    client.list_blobs.side_effect = list_blobs

    bucket_mock = MagicMock()

    def blob(key: str) -> MagicMock:
        bl = MagicMock()

        def upload_from_filename(local_path: str) -> None:
            client.uploads.append((key, local_path))

        bl.upload_from_filename.side_effect = upload_from_filename
        return bl

    bucket_mock.blob.side_effect = blob
    client.bucket.return_value = bucket_mock
    return client


def test_gcs_uri_mirrors_prefix_to_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects = {
        "refs/v1/tap_index.json": ("etag-a", b'{"taps": []}'),
        "refs/v1/activations.safetensors": ("etag-b", b"binary-blob"),
        "refs/v1/nested/extra.json": ("etag-c", b"{}"),
    }
    client = _fake_gcs_client(objects)

    with patch("google.cloud.storage.Client", return_value=client):
        result = resolve_reference("gs://my-bucket/refs/v1")

    assert (result / "tap_index.json").read_bytes() == b'{"taps": []}'
    assert (result / "activations.safetensors").read_bytes() == b"binary-blob"
    assert (result / "nested" / "extra.json").read_bytes() == b"{}"
    assert (result / "_manifest.json").exists()


def test_gcs_uri_skips_redownload_when_etag_matches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects = {"refs/v1/file.json": ("etag-a", b"{}")}

    client1 = _fake_gcs_client(objects)
    with patch("google.cloud.storage.Client", return_value=client1):
        resolve_reference("gs://my-bucket/refs/v1")

    client2 = _fake_gcs_client(objects)
    with patch("google.cloud.storage.Client", return_value=client2):
        resolve_reference("gs://my-bucket/refs/v1")

    assert len(client1.downloads) == 1, "first resolve should download once"
    assert len(client2.downloads) == 0, "second resolve should skip download (ETag match)"


def test_gcs_uri_redownloads_when_etag_changes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    with patch(
        "google.cloud.storage.Client",
        return_value=_fake_gcs_client({"refs/v1/file.json": ("etag-a", b'{"v": 1}')}),
    ):
        result = resolve_reference("gs://my-bucket/refs/v1")
        assert (result / "file.json").read_bytes() == b'{"v": 1}'

    with patch(
        "google.cloud.storage.Client",
        return_value=_fake_gcs_client({"refs/v1/file.json": ("etag-b", b'{"v": 2}')}),
    ):
        result = resolve_reference("gs://my-bucket/refs/v1")
        assert (result / "file.json").read_bytes() == b'{"v": 2}'


def test_gcs_uri_alias_long_scheme_works(tmp_path: Path, monkeypatch) -> None:
    """Both ``gs://`` and ``gcs://`` should resolve."""
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    objects = {"file.json": ("etag-a", b"{}")}
    client = _fake_gcs_client(objects)
    with patch("google.cloud.storage.Client", return_value=client):
        resolve_reference("gcs://my-bucket")
    client.list_blobs.assert_called()


def test_publish_gcs_uploads_each_file(tmp_path: Path) -> None:
    ref = _make_reference_dir(tmp_path)
    client = _fake_gcs_client({})
    with patch("google.cloud.storage.Client", return_value=client):
        publish_reference(ref, "gs://my-bucket/refs/v1")

    # 3 files in the reference dir → 3 uploads under the prefix.
    keys = sorted(key for key, _local in client.uploads)
    assert keys == [
        "refs/v1/manifest.json",
        "refs/v1/tolerances.json",
        "refs/v1/weights.safetensors",
    ]
    client.bucket.assert_called_once_with("my-bucket")


def test_gcs_missing_library_raises_import_error(monkeypatch) -> None:
    """Helpful error pointing at the install command when the library isn't installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("google.cloud") or name.startswith("google.api_core"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"firefly\[gcs\]"):
        resolve_reference("gs://my-bucket/refs/v1")


# --- Azure backend ----------------------------------------------------------


def _fake_azure_client(objects: dict[str, tuple[str, bytes]]) -> MagicMock:
    """Build a MagicMock BlobServiceClient backed by an in-memory object map.

    Tracks ``client.downloads`` and ``client.uploads`` (lists of (key, local_path))
    for tests to inspect, mirroring the GCS fake.
    """
    client = MagicMock()
    client.downloads = []
    client.uploads = []

    def make_blob(name: str, etag: str) -> MagicMock:
        blob = MagicMock()
        blob.name = name
        blob.etag = etag  # Azure ETags often come quoted; storage.py strips them.
        return blob

    container_client = MagicMock()

    def list_blobs(name_starts_with: str = ""):
        return [
            make_blob(name, etag)
            for name, (etag, _body) in objects.items()
            if name.startswith(name_starts_with)
        ]

    container_client.list_blobs.side_effect = list_blobs

    def get_blob_client(blob_name: str) -> MagicMock:
        bc = MagicMock()

        def download_blob():
            dl = MagicMock()
            _etag, body = objects[blob_name]

            def readall():
                client.downloads.append((blob_name, "<readall>"))
                return body

            dl.readall.side_effect = readall
            return dl

        bc.download_blob.side_effect = download_blob
        return bc

    container_client.get_blob_client.side_effect = get_blob_client

    def upload_blob(name: str, data, overwrite: bool = False) -> None:  # noqa: ARG001
        # We track the upload but don't actually consume the file handle.
        client.uploads.append((name, "<upload>"))

    container_client.upload_blob.side_effect = upload_blob

    client.get_container_client.return_value = container_client
    return client


def _patch_azure(client: MagicMock):
    """Patch the BlobServiceClient + DefaultAzureCredential at the right import sites.

    storage.py imports these inside _azure_client. We patch both the
    connection-string and the credential-auth paths since the env var
    determines which one runs.
    """
    return patch.multiple(
        "azure.storage.blob",
        BlobServiceClient=MagicMock(
            from_connection_string=MagicMock(return_value=client),
            return_value=client,
        ),
    )


def test_azure_uri_mirrors_prefix_to_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=fake==;EndpointSuffix=core.windows.net",
    )
    objects = {
        "refs/v1/tap_index.json": ("etag-a", b'{"taps": []}'),
        "refs/v1/weights.safetensors": ("etag-b", b"binary"),
    }
    client = _fake_azure_client(objects)

    fake_bs_class = MagicMock(
        from_connection_string=MagicMock(return_value=client),
    )
    with patch("azure.storage.blob.BlobServiceClient", fake_bs_class):
        result = resolve_reference("az://myaccount/mycontainer/refs/v1")

    # We don't actually write files to disk for the Azure fake — we just
    # verify the right blob-client paths were taken.
    assert result == Path(tmp_path / "cache") / "azure" / "myaccount" / "mycontainer" / "refs_v1"
    keys = sorted(name for name, _path in client.downloads)
    assert keys == ["refs/v1/tap_index.json", "refs/v1/weights.safetensors"]


def test_azure_uri_skips_redownload_when_etag_matches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=fake==;EndpointSuffix=core.windows.net",
    )
    objects = {"refs/v1/file.json": ("etag-a", b"{}")}

    client1 = _fake_azure_client(objects)
    fake_bs_class_1 = MagicMock(from_connection_string=MagicMock(return_value=client1))
    with patch("azure.storage.blob.BlobServiceClient", fake_bs_class_1):
        resolve_reference("az://myaccount/mycontainer/refs/v1")

    # Touch the cache file so the "ETag matches AND local file exists" branch fires.
    cache_dir = Path(tmp_path / "cache") / "azure" / "myaccount" / "mycontainer" / "refs_v1"
    (cache_dir / "file.json").parent.mkdir(parents=True, exist_ok=True)
    (cache_dir / "file.json").write_text("{}")

    client2 = _fake_azure_client(objects)
    fake_bs_class_2 = MagicMock(from_connection_string=MagicMock(return_value=client2))
    with patch("azure.storage.blob.BlobServiceClient", fake_bs_class_2):
        resolve_reference("az://myaccount/mycontainer/refs/v1")

    assert len(client1.downloads) == 1, "first resolve should download once"
    assert len(client2.downloads) == 0, "second resolve should skip download (ETag match)"


def test_azure_etag_quotes_stripped(tmp_path: Path, monkeypatch) -> None:
    """Azure ETags sometimes come back with surrounding double-quotes;
    storage._sync_azure_prefix strips them so cache comparisons work."""
    monkeypatch.setenv("FIREFLY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=fake==;EndpointSuffix=core.windows.net",
    )
    # Object with a quoted ETag.
    objects = {"file.json": ('"etag-quoted"', b"{}")}

    client1 = _fake_azure_client(objects)
    fake_bs_class_1 = MagicMock(from_connection_string=MagicMock(return_value=client1))
    with patch("azure.storage.blob.BlobServiceClient", fake_bs_class_1):
        result = resolve_reference("az://myaccount/mycontainer")

    # Touch cache file so the ETag-match branch sees it.
    (result / "file.json").parent.mkdir(parents=True, exist_ok=True)
    (result / "file.json").write_text("{}")
    manifest = json.loads((result / "_manifest.json").read_text())
    assert manifest["file.json"] == "etag-quoted", "quotes should be stripped from ETag"


def test_publish_azure_uploads_each_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=fake==;EndpointSuffix=core.windows.net",
    )
    ref = _make_reference_dir(tmp_path)
    client = _fake_azure_client({})
    fake_bs_class = MagicMock(from_connection_string=MagicMock(return_value=client))
    with patch("azure.storage.blob.BlobServiceClient", fake_bs_class):
        publish_reference(ref, "az://myaccount/mycontainer/refs/v1")

    keys = sorted(name for name, _path in client.uploads)
    assert keys == [
        "refs/v1/manifest.json",
        "refs/v1/tolerances.json",
        "refs/v1/weights.safetensors",
    ]


def test_malformed_azure_uri_raises_value_error() -> None:
    """az:// with only one segment (missing container/) should error clearly."""
    with pytest.raises(ValueError, match="Azure URI"):
        resolve_reference("az://just-one-segment")


def test_azure_missing_library_raises_import_error(monkeypatch) -> None:
    """Helpful error pointing at the install command when the library isn't installed."""
    import builtins
    import json as _json

    _ = _json  # silence ruff unused-warning if any

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("azure"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"firefly\[azure\]"):
        resolve_reference("az://myaccount/mycontainer/refs/v1")
