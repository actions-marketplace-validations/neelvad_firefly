"""The on-disk reference artifact.

Layout::

    reference/
        weights.safetensors     # captured per-tap-point tensors, key = tap name
        manifest.json           # tap points, shapes, dtypes, fingerprint, env
        tolerances.json         # (optional) per-tap tolerances — see compare.py

This is a first-class file format. Inspectable, diffable, versionable — kept
deliberately simple so a reviewer can `unzip` it mentally without running code.
The optional ``tolerances.json`` is written by ``firefly calibrate`` and
auto-loaded by ``firefly check``; its read/write live in ``compare.py``
beside the ``TapTolerance`` type they serialize.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 1
_WEIGHTS_FILE = "weights.safetensors"
_MANIFEST_FILE = "manifest.json"


@dataclass
class ReferenceManifest:
    model_id: str
    model_fingerprint: str
    tap_points: list[str]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]
    captured_at: str  # ISO-8601 UTC
    env: dict[str, str] = field(default_factory=dict)
    domain: str = "llm"
    dtype: str = "float32"  # the model's storage dtype: float32 / bfloat16 / float16
    schema_version: int = SCHEMA_VERSION


def capture_env() -> dict[str, str]:
    """Snapshot of the environment that produced a reference."""
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    }


def write_reference(
    out_dir: Path,
    manifest: ReferenceManifest,
    tensors: dict[str, torch.Tensor],
) -> None:
    """Write the reference artifact to ``out_dir``.

    Tensors are detached and moved to CPU before serialization — references are
    portable across hardware, comparisons happen on the candidate's device.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_tensors = {name: t.detach().cpu().contiguous() for name, t in tensors.items()}
    save_file(cpu_tensors, str(out_dir / _WEIGHTS_FILE))

    with (out_dir / _MANIFEST_FILE).open("w") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)


def read_reference(ref_dir: Path) -> tuple[ReferenceManifest, dict[str, torch.Tensor]]:
    """Load a reference artifact from disk."""
    manifest_path = ref_dir / _MANIFEST_FILE
    weights_path = ref_dir / _WEIGHTS_FILE

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"No weights at {weights_path}")

    with manifest_path.open() as f:
        data = json.load(f)

    found_version = data.get("schema_version")
    if found_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported reference schema_version={found_version}; expected {SCHEMA_VERSION}"
        )

    manifest = ReferenceManifest(**data)
    tensors = load_file(str(weights_path))
    return manifest, tensors
