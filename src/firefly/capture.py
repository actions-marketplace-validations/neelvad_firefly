"""Forward-hook based activation capture.

Registers forward hooks at the chosen tap points, runs a forward pass on the
golden inputs, and serializes per-tap-point output tensors to a reference
artifact directory.
"""

from __future__ import annotations

from pathlib import Path


def capture_reference(
    model_id: str,
    inputs_path: Path,
    out_dir: Path,
) -> None:
    """Run a forward pass and write a reference artifact to out_dir."""
    raise NotImplementedError
