"""Reference-artifact storage backends.

Resolves a reference URI (local path or remote scheme) into a local
filesystem path that the rest of Firefly can read normally.

v1 supports:
- Local paths: ``/some/path`` or ``./reference``
- HuggingFace Hub: ``hf://<org>/<repo>[@<revision>][/<subpath>]``

Planned: S3 (v2), GCS / Azure (v3). Each unsupported-but-recognized
scheme raises ``NotImplementedError`` with a clear message and the
planned version, so users hitting them get a useful pointer rather
than a generic "file not found" later in the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

_PLANNED_BACKENDS: dict[str, str] = {
    "s3": "v2",
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


def resolve_reference(uri: str | Path) -> Path:
    """Resolve a reference URI to a local filesystem path.

    Local paths pass through unchanged. ``hf://`` URIs are downloaded
    via ``huggingface_hub.snapshot_download`` and the local snapshot
    path is returned (cached on subsequent calls).

    Raises ``NotImplementedError`` for recognized-but-unimplemented
    schemes (S3, GCS, Azure) with the planned version.
    """
    raw = str(uri)

    scheme = _extract_scheme(raw)
    if scheme is None:
        return Path(raw)

    if scheme in ("hf", "huggingface"):
        return _resolve_hf(raw)

    planned = _PLANNED_BACKENDS.get(scheme)
    if planned is not None:
        raise NotImplementedError(
            f"Reference scheme {scheme!r} is not yet supported "
            f"(planned for {planned}). Use a local path or hf://<org>/<repo>."
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
