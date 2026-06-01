"""Architecturally stable tap points for transformer activation capture.

We deliberately tap at module boundaries that survive quantization and
torch.compile fusion: per-decoder-layer residual stream, attention output,
MLP output. This is what makes Firefly robust to candidates that differ
from the reference in graph structure (quantized, compiled, fused).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TapPoint:
    """A named hook site on a model."""

    name: str
    module_path: str  # dotted path into the model, e.g. "model.layers.7.mlp"


def select_default_tap_points(model_arch: str) -> list[TapPoint]:
    """Return the default tap-point set for a given HF architecture.

    v1 supports Llama-family decoder transformers (covers Qwen, SmolLM, Llama, Mistral).
    """
    raise NotImplementedError
