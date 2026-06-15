"""The Runner protocol, its result type, and the runner registry.

Kept dependency-light on purpose: this module imports no engine code, so the
heavy backends (transformers, vLLM) are imported lazily by :func:`get_runner`
only when a given runner is actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch


@dataclass
class CaptureResult:
    """Everything a Runner produces from one capture pass.

    The orchestrators (:func:`firefly.capture.capture_reference`,
    :func:`firefly.compare._run_candidate`) turn this into a reference
    artifact or feed ``tensors`` straight into the diff. ``shapes`` / ``dtypes``
    per tap are derived from ``tensors`` by the orchestrator, not here.
    """

    tensors: dict[str, torch.Tensor]
    fingerprint: str
    """Engine-specific model identity hash. HF hashes the nn.Module's params;
    vLLM uses a ``vllm-<ver>:<model>`` placeholder (its parallel-wrapped params
    don't hash like HF's). Compared against ``manifest.model_fingerprint``."""
    head_counts: dict[str, int] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    dtype: str = "float32"
    """Canonical name of the dtype actually used (float32 / bfloat16 / float16)."""


@runtime_checkable
class Runner(Protocol):
    """Capture backend. ``capture`` is the whole contract."""

    name: str

    def capture(
        self,
        model_id: str,
        inputs_path: Path,
        *,
        device: str = "cpu",
        seed: int = 0,
        domain: str = "llm",
        dtype: str = "float32",
        per_head: bool = False,
        options: dict[str, str] | None = None,
    ) -> CaptureResult:
        """Load ``model_id`` in this engine, run the golden inputs, capture taps.

        ``dtype`` accepts user-facing or canonical names (fp16 / float16, ...).
        ``options`` carries engine-specific knobs (e.g. vLLM's
        ``attention_backend``); runners ignore keys they don't recognize.
        """
        ...


def available_runners() -> list[str]:
    """Runner names ``get_runner`` accepts."""
    return ["hf"]  # "vllm" lands when the vLLM runner is extracted from scripts/


def get_runner(name: str) -> Runner:
    """Instantiate a runner by name. Engine modules are imported lazily."""
    if name == "hf":
        from firefly.runners.hf import HFRunner

        return HFRunner()
    if name == "vllm":
        raise NotImplementedError(
            "The vLLM runner is not wired into the CLI yet — the capture logic "
            "currently lives in scripts/capture_vllm.py. Use --runner hf for now."
        )
    raise ValueError(
        f"Unknown runner {name!r}. Available: {available_runners()}"
    )
