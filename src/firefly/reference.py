"""The on-disk reference artifact.

Layout:
    reference/
        weights.safetensors     # captured per-tap-point tensors
        manifest.json           # tap points, shapes, dtypes, model fingerprint, env

This is a first-class file format. Inspectable, diffable, versionable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReferenceManifest:
    model_id: str
    model_fingerprint: str  # hash of weights or config
    tap_points: list[str]
    shapes: dict[str, tuple[int, ...]]
    dtypes: dict[str, str]
    captured_at: str  # ISO timestamp
    env: dict[str, str]  # torch version, device, platform


def write_reference(out_dir: Path, manifest: ReferenceManifest, tensors: dict) -> None:
    """Write the reference artifact to disk."""
    raise NotImplementedError


def read_reference(ref_dir: Path) -> tuple[ReferenceManifest, dict]:
    """Load a reference artifact from disk."""
    raise NotImplementedError
