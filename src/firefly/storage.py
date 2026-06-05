"""Reference-artifact storage backends.

Resolves a reference URI (local path or remote scheme) into a local
filesystem path that the rest of Firefly can read normally.

Supported:
- Local paths: ``/some/path`` or ``./reference``
- HuggingFace Hub: ``hf://<org>/<repo>[@<revision>][/<subpath>]``
- S3: ``s3://<bucket>/<prefix>/`` (requires ``boto3``; uses default
  credential chain — env vars, ``~/.aws/credentials``, IAM roles)

Planned: GCS / Azure (v3). Each unsupported-but-recognized scheme
raises ``NotImplementedError`` with a clear message and the planned
version, so users hitting them get a useful pointer rather than a
generic "file not found" later in the pipeline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_PLANNED_BACKENDS: dict[str, str] = {
    "gs": "v3",
    "gcs": "v3",
    "az": "v3",
    "azure": "v3",
}


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


def resolve_reference(uri: str | Path) -> Path:
    """Resolve a reference URI to a local filesystem path.

    Local paths pass through unchanged. ``hf://`` URIs are downloaded
    via ``huggingface_hub.snapshot_download``. ``s3://`` URIs are
    mirrored into a local cache via boto3. In both remote cases the
    local path is returned and cached on subsequent calls.

    Raises ``NotImplementedError`` for recognized-but-unimplemented
    schemes (GCS, Azure) with the planned version.
    """
    raw = str(uri)

    scheme = _extract_scheme(raw)
    if scheme is None:
        return Path(raw)

    if scheme in ("hf", "huggingface"):
        return _resolve_hf(raw)

    if scheme == "s3":
        return _resolve_s3(raw)

    planned = _PLANNED_BACKENDS.get(scheme)
    if planned is not None:
        raise NotImplementedError(
            f"Reference scheme {scheme!r} is not yet supported "
            f"(planned for {planned}). Use a local path, hf://<org>/<repo>, "
            f"or s3://<bucket>/<prefix>."
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


def _sync_s3_prefix(client, bucket: str, prefix: str, cache_dir: Path) -> None:
    """Mirror bucket/prefix into cache_dir, skipping unchanged objects."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "_manifest.json"
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
