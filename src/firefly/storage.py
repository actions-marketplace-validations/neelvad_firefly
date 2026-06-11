"""Reference-artifact storage backends.

Resolves a reference URI (local path or remote scheme) into a local
filesystem path that the rest of Firefly can read normally.

Supported:
- Local paths: ``/some/path`` or ``./reference``
- HuggingFace Hub: ``hf://<org>/<repo>[@<revision>][/<subpath>]``
- S3: ``s3://<bucket>/<prefix>/`` (requires ``boto3``; uses default
  credential chain — env vars, ``~/.aws/credentials``, IAM roles)
- GCS: ``gs://<bucket>/<prefix>/`` (requires ``google-cloud-storage``;
  uses Application Default Credentials — ``GOOGLE_APPLICATION_CREDENTIALS``,
  ``gcloud auth application-default login``, or GCE/GKE metadata server)
- Azure Blob: ``az://<account>/<container>/<prefix>/`` (requires
  ``azure-storage-blob``; uses ``AZURE_STORAGE_CONNECTION_STRING`` if
  set, otherwise ``DefaultAzureCredential`` — managed identity, az CLI,
  env-var-credentials, etc.)

Unrecognized schemes are treated as local paths to match ``Path``'s
permissive behavior; ill-formed URIs in a known scheme raise
``ValueError`` with a clear hint.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# All four storage schemes (local, hf, s3, gs, az) are now supported.
# Keep the planned-backends map as the place to stub future schemes if
# they get added (e.g., r2:// for Cloudflare R2 with a non-S3 client).
_PLANNED_BACKENDS: dict[str, str] = {}

# Internal sidecar the cloud-resolve cache writes alongside the mirrored
# reference (ETag bookkeeping for incremental sync). It is NOT part of the
# reference artifact, so publish flows must skip it — otherwise a
# resolve-then-publish round-trip would leak cache metadata into the bucket.
_CACHE_MANIFEST_NAME = "_manifest.json"


_HF_REGEX = re.compile(
    r"^(?:hf|huggingface)://"
    r"(?P<repo_id>[^/@]+/[^/@]+)"
    r"(?:@(?P<revision>[^/]+))?"
    r"(?:/(?P<subpath>.+))?$",
    re.IGNORECASE,
)


_S3_REGEX = re.compile(
    r"^s3://"
    r"(?P<bucket>[^/]+)"
    r"(?:/(?P<prefix>.*))?$",
    re.IGNORECASE,
)


_GCS_REGEX = re.compile(
    r"^(?:gs|gcs)://"
    r"(?P<bucket>[^/]+)"
    r"(?:/(?P<prefix>.*))?$",
    re.IGNORECASE,
)


# az://<account>/<container>/<prefix>. Account explicit in the URI;
# alternatives (env var, connection string with embedded account) get
# complicated quickly and we'd rather have the URI itself be
# unambiguously routable.
_AZURE_REGEX = re.compile(
    r"^(?:az|azure)://"
    r"(?P<account>[^/]+)/"
    r"(?P<container>[^/]+)"
    r"(?:/(?P<prefix>.*))?$",
    re.IGNORECASE,
)


def resolve_reference(uri: str | Path) -> Path:
    """Resolve a reference URI to a local filesystem path.

    Local paths pass through unchanged. ``hf://`` URIs are downloaded
    via ``huggingface_hub.snapshot_download``. ``s3://``, ``gs://``, and
    ``az://`` URIs are mirrored into a local cache via boto3 /
    google-cloud-storage / azure-storage-blob. In remote cases the local
    path is returned and cached on subsequent calls.

    Raises ``NotImplementedError`` only for recognized-but-unimplemented
    schemes registered in ``_PLANNED_BACKENDS`` (none at present).
    """
    raw = str(uri)

    scheme = _extract_scheme(raw)
    if scheme is None:
        return Path(raw)

    if scheme in ("hf", "huggingface"):
        return _resolve_hf(raw)

    if scheme == "s3":
        return _resolve_s3(raw)

    if scheme in ("gs", "gcs"):
        return _resolve_gcs(raw)

    if scheme in ("az", "azure"):
        return _resolve_azure(raw)

    planned = _PLANNED_BACKENDS.get(scheme)
    if planned is not None:
        raise NotImplementedError(
            f"Reference scheme {scheme!r} is not yet supported "
            f"(planned for {planned}). Use a local path, hf://, s3://, "
            f"gs://, or az://."
        )

    # Unknown scheme — treat as a path (matches Path's permissive behavior).
    return Path(raw)


def _extract_scheme(raw: str) -> str | None:
    """Return the URI scheme prefix, or None for paths.

    Single-letter schemes (Windows drive letters like ``c:``) are
    intentionally treated as paths, not URIs.
    """
    m = re.match(r"^([A-Za-z][A-Za-z0-9+\-.]*):", raw)
    if not m:
        return None
    scheme = m.group(1).lower()
    if len(scheme) == 1:
        return None
    return scheme


def _resolve_hf(uri: str) -> Path:
    """Download an HF Hub reference repo, return the local snapshot path."""
    m = _HF_REGEX.match(uri)
    if not m:
        raise ValueError(
            f"Invalid HF Hub URI {uri!r}. Expected format: "
            f"hf://<org>/<repo>[@<revision>][/<subpath>]"
        )
    repo_id = m.group("repo_id")
    revision = m.group("revision")
    subpath = m.group("subpath")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for hf:// references but isn't "
            "installed. Install with: pip install huggingface_hub"
        ) from e

    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                # References are stored as 'model' repo type by convention;
                # users uploading should `huggingface-cli upload <repo>
                # --repo-type model`.
                repo_type="model",
            )
        )
    except Exception as e:  # noqa: BLE001 — HF raises a wide variety of types
        # Re-raise as RuntimeError with a Firefly-flavored hint. The most
        # common failures are repo-not-found and unauthorized access; both
        # benefit from explicitly pointing at HF_TOKEN.
        raise RuntimeError(
            f"Failed to download HF Hub reference {uri!r}: {e}\n"
            f"Check that the repo exists and is accessible. For private "
            f"repos, set HF_TOKEN in your environment (or in the action "
            f"workflow's `env:` block)."
        ) from e
    return snapshot_path / subpath if subpath else snapshot_path


def _resolve_s3(uri: str) -> Path:
    """Mirror an S3 prefix to a local cache, return the cache path.

    ETag-based incremental sync: on repeat calls, objects whose ETags
    match the local manifest are not re-downloaded. CI runners that
    persist ``$FIREFLY_CACHE_DIR`` between jobs get fast no-op syncs
    after the first run.
    """
    m = _S3_REGEX.match(uri)
    if not m:
        raise ValueError(
            f"Invalid S3 URI {uri!r}. Expected format: "
            f"s3://<bucket>/<prefix>"
        )
    bucket = m.group("bucket")
    raw_prefix = (m.group("prefix") or "").strip("/")
    # Normalize to directory semantics: list_objects_v2 with this prefix
    # matches everything under the directory, including nested objects.
    prefix = f"{raw_prefix}/" if raw_prefix else ""

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as e:
        raise ImportError(
            "boto3 is required for s3:// references but isn't installed. "
            "Install with: pip install 'firefly[s3]' (or pip install boto3)."
        ) from e

    cache_dir = _s3_cache_dir(bucket, raw_prefix)
    client = boto3.client("s3")

    try:
        _sync_s3_prefix(client, bucket, prefix, cache_dir)
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(
            f"Failed to sync S3 reference {uri!r}: {e}\n"
            f"Check that the bucket exists and your credentials are set "
            f"(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, "
            f"or an IAM role for instance/CI runners)."
        ) from e

    return cache_dir


def _cache_root() -> Path:
    """Root cache directory. Respects ``$FIREFLY_CACHE_DIR``."""
    override = os.environ.get("FIREFLY_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "firefly"


def _s3_cache_dir(bucket: str, prefix: str) -> Path:
    """Cache directory for a given (bucket, prefix) pair."""
    safe_prefix = prefix.replace("/", "_") if prefix else "_root"
    return _cache_root() / "s3" / bucket / safe_prefix


def _resolve_gcs(uri: str) -> Path:
    """Mirror a GCS prefix to a local cache, return the cache path.

    ETag-based incremental sync, same shape as :func:`_resolve_s3`. Uses
    Application Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS``,
    ``gcloud auth application-default login``, or GCE/GKE metadata).
    """
    m = _GCS_REGEX.match(uri)
    if not m:
        raise ValueError(
            f"Invalid GCS URI {uri!r}. Expected format: "
            f"gs://<bucket>/<prefix>"
        )
    bucket_name = m.group("bucket")
    raw_prefix = (m.group("prefix") or "").strip("/")
    prefix = f"{raw_prefix}/" if raw_prefix else ""

    try:
        from google.api_core.exceptions import GoogleAPICallError
        from google.cloud import storage as gcs_storage
    except ImportError as e:
        raise ImportError(
            "google-cloud-storage is required for gs:// references but "
            "isn't installed. Install with: pip install 'firefly[gcs]' "
            "(or pip install google-cloud-storage)."
        ) from e

    cache_dir = _gcs_cache_dir(bucket_name, raw_prefix)
    client = gcs_storage.Client()

    try:
        _sync_gcs_prefix(client, bucket_name, prefix, cache_dir)
    except GoogleAPICallError as e:
        raise RuntimeError(
            f"Failed to sync GCS reference {uri!r}: {e}\n"
            f"Check that the bucket exists and your credentials are set "
            f"(GOOGLE_APPLICATION_CREDENTIALS, `gcloud auth application-default "
            f"login`, or a GCE/GKE service account)."
        ) from e

    return cache_dir


def _gcs_cache_dir(bucket: str, prefix: str) -> Path:
    """Cache directory for a given GCS (bucket, prefix) pair."""
    safe_prefix = prefix.replace("/", "_") if prefix else "_root"
    return _cache_root() / "gcs" / bucket / safe_prefix


def _azure_client(account: str):
    """Build a BlobServiceClient using env-var auth.

    Prefers ``AZURE_STORAGE_CONNECTION_STRING`` if set; otherwise uses
    ``DefaultAzureCredential`` (managed identity, az CLI, env vars).
    """
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)
    return BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )


def _resolve_azure(uri: str) -> Path:
    """Mirror an Azure Blob prefix to a local cache, return the cache path."""
    m = _AZURE_REGEX.match(uri)
    if not m:
        raise ValueError(
            f"Invalid Azure URI {uri!r}. Expected format: "
            f"az://<account>/<container>/<prefix>"
        )
    account = m.group("account")
    container_name = m.group("container")
    raw_prefix = (m.group("prefix") or "").strip("/")
    prefix = f"{raw_prefix}/" if raw_prefix else ""

    try:
        from azure.core.exceptions import AzureError
    except ImportError as e:
        raise ImportError(
            "azure-storage-blob and azure-identity are required for "
            "az:// references but aren't installed. Install with: "
            "pip install 'firefly[azure]'."
        ) from e

    cache_dir = _azure_cache_dir(account, container_name, raw_prefix)
    try:
        client = _azure_client(account)
    except ImportError as e:
        raise ImportError(
            "azure-storage-blob and azure-identity are required for "
            "az:// references but aren't installed. Install with: "
            "pip install 'firefly[azure]'."
        ) from e

    try:
        _sync_azure_prefix(client, container_name, prefix, cache_dir)
    except AzureError as e:
        raise RuntimeError(
            f"Failed to sync Azure reference {uri!r}: {e}\n"
            f"Check that the container exists and your credentials are "
            f"set (AZURE_STORAGE_CONNECTION_STRING, or DefaultAzureCredential "
            f"sources — managed identity, az CLI, env vars)."
        ) from e

    return cache_dir


def _azure_cache_dir(account: str, container: str, prefix: str) -> Path:
    """Cache directory for an Azure (account, container, prefix) tuple."""
    safe_prefix = prefix.replace("/", "_") if prefix else "_root"
    return _cache_root() / "azure" / account / container / safe_prefix


def _sync_azure_prefix(client, container_name: str, prefix: str, cache_dir: Path) -> None:
    """Mirror container/prefix into cache_dir, skipping unchanged blobs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / _CACHE_MANIFEST_NAME
    old_manifest: dict[str, str] = {}
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text())

    new_manifest: dict[str, str] = {}
    container_client = client.get_container_client(container_name)
    for blob in container_client.list_blobs(name_starts_with=prefix):
        key = blob.name
        # Azure ETags can be wrapped in double quotes; strip for stable comparison.
        etag = (blob.etag or "").strip('"')
        if key.endswith("/"):
            continue
        relpath = key[len(prefix):] if prefix and key.startswith(prefix) else key
        if not relpath:
            continue
        new_manifest[key] = etag
        local_path = cache_dir / relpath
        if old_manifest.get(key) == etag and local_path.exists():
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_client = container_client.get_blob_client(key)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

    for stale_key in set(old_manifest) - set(new_manifest):
        relpath = (
            stale_key[len(prefix):]
            if prefix and stale_key.startswith(prefix)
            else stale_key
        )
        stale_path = cache_dir / relpath
        if stale_path.exists():
            stale_path.unlink()

    manifest_path.write_text(json.dumps(new_manifest, indent=2, sort_keys=True))


def _sync_gcs_prefix(client, bucket_name: str, prefix: str, cache_dir: Path) -> None:
    """Mirror bucket/prefix into cache_dir, skipping unchanged objects."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / _CACHE_MANIFEST_NAME
    old_manifest: dict[str, str] = {}
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text())

    new_manifest: dict[str, str] = {}
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket, prefix=prefix):
        key = blob.name
        etag = blob.etag
        # GCS sometimes returns directory-marker blobs (0-byte, key ending in /).
        if key.endswith("/"):
            continue
        relpath = key[len(prefix):] if prefix and key.startswith(prefix) else key
        if not relpath:
            continue
        new_manifest[key] = etag
        local_path = cache_dir / relpath
        if old_manifest.get(key) == etag and local_path.exists():
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    # Remove objects that disappeared upstream.
    for stale_key in set(old_manifest) - set(new_manifest):
        relpath = (
            stale_key[len(prefix):]
            if prefix and stale_key.startswith(prefix)
            else stale_key
        )
        stale_path = cache_dir / relpath
        if stale_path.exists():
            stale_path.unlink()

    manifest_path.write_text(json.dumps(new_manifest, indent=2, sort_keys=True))


def publish_reference(
    local_path: Path,
    uri: str,
    *,
    commit_message: str = "Firefly reference upload",
) -> None:
    """Upload a local reference directory to a remote URI.

    Inverse of :func:`resolve_reference`. Supported destinations:

    * ``hf://<org>/<repo>[@<revision>][/<subpath>]`` — uploads via
      :class:`huggingface_hub.HfApi`. Creates the repo if it doesn't
      exist (``exist_ok=True``). ``commit_message`` is forwarded to
      the HF commit.
    * ``s3://<bucket>/<prefix>`` — uploads each file under
      ``local_path`` to ``<bucket>/<prefix>/<relpath>`` via boto3.
    * ``gs://<bucket>/<prefix>`` — uploads each file under
      ``local_path`` to ``<bucket>/<prefix>/<relpath>`` via
      google-cloud-storage.
    * ``az://<account>/<container>/<prefix>`` — uploads each file
      under ``local_path`` to ``<container>/<prefix>/<relpath>`` via
      azure-storage-blob.

    Local destinations are intentionally not supported — use
    ``cp -r`` instead.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Reference directory does not exist: {local_path}")
    if not local_path.is_dir():
        raise ValueError(f"Reference path must be a directory: {local_path}")

    scheme = _extract_scheme(str(uri))
    if scheme is None:
        raise ValueError(
            f"Publish target {uri!r} must be a remote URI "
            f"(e.g. hf://org/repo or s3://bucket/prefix). "
            f"For local copies, use `cp -r {local_path} {uri}`."
        )

    if scheme in ("hf", "huggingface"):
        return _publish_hf(local_path, uri, commit_message=commit_message)
    if scheme == "s3":
        return _publish_s3(local_path, uri)
    if scheme in ("gs", "gcs"):
        return _publish_gcs(local_path, uri)
    if scheme in ("az", "azure"):
        return _publish_azure(local_path, uri)

    planned = _PLANNED_BACKENDS.get(scheme)
    if planned is not None:
        raise NotImplementedError(
            f"Publishing to scheme {scheme!r} is not yet supported "
            f"(planned for {planned}). Use hf://, s3://, gs://, or az://."
        )
    raise ValueError(f"Unknown URI scheme {scheme!r}")


def _publish_hf(local_path: Path, uri: str, *, commit_message: str) -> None:
    """Upload a folder to an HF Hub repo (optionally to a path_in_repo)."""
    m = _HF_REGEX.match(uri)
    if not m:
        raise ValueError(
            f"Invalid HF Hub URI {uri!r}. Expected format: "
            f"hf://<org>/<repo>[@<revision>][/<subpath>]"
        )
    repo_id = m.group("repo_id")
    revision = m.group("revision")
    subpath = m.group("subpath") or ""

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for hf:// publish but isn't "
            "installed. Install with: pip install huggingface_hub"
        ) from e

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            path_in_repo=subpath,
            commit_message=commit_message,
        )
    except Exception as e:  # noqa: BLE001 — HF raises a wide variety of types
        raise RuntimeError(
            f"Failed to publish to HF Hub {uri!r}: {e}\n"
            f"Check that HF_TOKEN is set in your environment and has write "
            f"access to {repo_id!r}."
        ) from e


def _publish_s3(local_path: Path, uri: str) -> None:
    """Upload each file under local_path to s3://bucket/prefix/<relpath>."""
    m = _S3_REGEX.match(uri)
    if not m:
        raise ValueError(
            f"Invalid S3 URI {uri!r}. Expected format: "
            f"s3://<bucket>/<prefix>"
        )
    bucket = m.group("bucket")
    raw_prefix = (m.group("prefix") or "").strip("/")
    prefix = f"{raw_prefix}/" if raw_prefix else ""

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as e:
        raise ImportError(
            "boto3 is required for s3:// publish but isn't installed. "
            "Install with: pip install 'firefly[s3]' (or pip install boto3)."
        ) from e

    client = boto3.client("s3")
    try:
        for file in sorted(local_path.rglob("*")):
            if not file.is_file():
                continue
            if file.name == _CACHE_MANIFEST_NAME:
                continue  # cache bookkeeping, not part of the artifact
            relpath = file.relative_to(local_path).as_posix()
            key = f"{prefix}{relpath}"
            client.upload_file(str(file), bucket, key)
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(
            f"Failed to publish to S3 {uri!r}: {e}\n"
            f"Check that the bucket exists and your credentials have "
            f"PutObject permission."
        ) from e


def _publish_azure(local_path: Path, uri: str) -> None:
    """Upload each file under local_path to az://account/container/prefix/<relpath>."""
    m = _AZURE_REGEX.match(uri)
    if not m:
        raise ValueError(
            f"Invalid Azure URI {uri!r}. Expected format: "
            f"az://<account>/<container>/<prefix>"
        )
    account = m.group("account")
    container_name = m.group("container")
    raw_prefix = (m.group("prefix") or "").strip("/")
    prefix = f"{raw_prefix}/" if raw_prefix else ""

    try:
        from azure.core.exceptions import AzureError
    except ImportError as e:
        raise ImportError(
            "azure-storage-blob and azure-identity are required for "
            "az:// publish but aren't installed. Install with: "
            "pip install 'firefly[azure]'."
        ) from e

    try:
        client = _azure_client(account)
    except ImportError as e:
        raise ImportError(
            "azure-storage-blob and azure-identity are required for "
            "az:// publish but aren't installed. Install with: "
            "pip install 'firefly[azure]'."
        ) from e

    try:
        container_client = client.get_container_client(container_name)
        for file in sorted(local_path.rglob("*")):
            if not file.is_file():
                continue
            if file.name == _CACHE_MANIFEST_NAME:
                continue  # cache bookkeeping, not part of the artifact
            relpath = file.relative_to(local_path).as_posix()
            key = f"{prefix}{relpath}"
            with open(file, "rb") as f:
                container_client.upload_blob(name=key, data=f, overwrite=True)
    except AzureError as e:
        raise RuntimeError(
            f"Failed to publish to Azure {uri!r}: {e}\n"
            f"Check that the container exists and your credentials have "
            f"Blob Data Contributor permission."
        ) from e


def _publish_gcs(local_path: Path, uri: str) -> None:
    """Upload each file under local_path to gs://bucket/prefix/<relpath>."""
    m = _GCS_REGEX.match(uri)
    if not m:
        raise ValueError(
            f"Invalid GCS URI {uri!r}. Expected format: "
            f"gs://<bucket>/<prefix>"
        )
    bucket_name = m.group("bucket")
    raw_prefix = (m.group("prefix") or "").strip("/")
    prefix = f"{raw_prefix}/" if raw_prefix else ""

    try:
        from google.api_core.exceptions import GoogleAPICallError
        from google.cloud import storage as gcs_storage
    except ImportError as e:
        raise ImportError(
            "google-cloud-storage is required for gs:// publish but isn't "
            "installed. Install with: pip install 'firefly[gcs]' "
            "(or pip install google-cloud-storage)."
        ) from e

    client = gcs_storage.Client()
    try:
        bucket = client.bucket(bucket_name)
        for file in sorted(local_path.rglob("*")):
            if not file.is_file():
                continue
            if file.name == _CACHE_MANIFEST_NAME:
                continue  # cache bookkeeping, not part of the artifact
            relpath = file.relative_to(local_path).as_posix()
            key = f"{prefix}{relpath}"
            bucket.blob(key).upload_from_filename(str(file))
    except GoogleAPICallError as e:
        raise RuntimeError(
            f"Failed to publish to GCS {uri!r}: {e}\n"
            f"Check that the bucket exists and your service account has "
            f"storage.objects.create permission."
        ) from e


def _sync_s3_prefix(client, bucket: str, prefix: str, cache_dir: Path) -> None:
    """Mirror bucket/prefix into cache_dir, skipping unchanged objects."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / _CACHE_MANIFEST_NAME
    old_manifest: dict[str, str] = {}
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text())

    new_manifest: dict[str, str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            etag = obj["ETag"]
            # Skip "directory marker" objects (0-byte keys ending in /).
            if key.endswith("/"):
                continue
            relpath = key[len(prefix):] if prefix and key.startswith(prefix) else key
            if not relpath:
                continue
            new_manifest[key] = etag
            local_path = cache_dir / relpath
            if old_manifest.get(key) == etag and local_path.exists():
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))

    # Remove objects that disappeared upstream.
    for stale_key in set(old_manifest) - set(new_manifest):
        relpath = (
            stale_key[len(prefix):]
            if prefix and stale_key.startswith(prefix)
            else stale_key
        )
        stale_path = cache_dir / relpath
        if stale_path.exists():
            stale_path.unlink()

    manifest_path.write_text(json.dumps(new_manifest, indent=2, sort_keys=True))
