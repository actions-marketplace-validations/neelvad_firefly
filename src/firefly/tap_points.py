"""Architecturally stable tap points for transformer activation capture.

We deliberately tap at module boundaries that survive quantization and
torch.compile fusion: per-decoder-layer self-attn output, MLP output, and
the residual stream at layer end. This is what makes Firefly robust to
candidates that differ from the reference in graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

# Common paths under which HF models expose their decoder ModuleList.
_DECODER_LAYERS_PATHS = ("model.layers", "transformer.h", "layers")

# Common paths for the final pre-head normalization.
_FINAL_NORM_PATHS = ("model.norm", "transformer.ln_f", "norm")


@dataclass(frozen=True)
class TapPoint:
    """A named hook site on a model."""

    name: str
    module_path: str  # dotted attribute path from the model root


def _resolve(module: nn.Module, dotted_path: str) -> nn.Module:
    obj: object = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _try_resolve(module: nn.Module, dotted_path: str) -> nn.Module | None:
    try:
        return _resolve(module, dotted_path)
    except AttributeError:
        return None


def find_decoder_layers_path(model: nn.Module) -> str:
    """Return the dotted path to the decoder-layer ModuleList, or raise."""
    for path in _DECODER_LAYERS_PATHS:
        obj = _try_resolve(model, path)
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return path
    raise ValueError(
        "Could not locate decoder layers. Tried: " + ", ".join(_DECODER_LAYERS_PATHS)
    )


def select_default_tap_points(model: nn.Module) -> list[TapPoint]:
    """Walk a HF-style decoder transformer and return its stable tap points.

    Per decoder layer i, emits (in forward order):
        layer.{i}.self_attn  → output of the attention sub-block
        layer.{i}.mlp        → output of the MLP sub-block
        layer.{i}            → residual stream at end of layer

    Plus a single trailing `final_norm` tap when present.

    The forward-order ordering matters: attribution walks this list in order
    and reports the first tap that exceeds tolerance, which corresponds to
    the earliest point in the network where divergence appeared.
    """
    layers_path = find_decoder_layers_path(model)
    layers = _resolve(model, layers_path)

    taps: list[TapPoint] = []
    for i, layer in enumerate(layers):  # type: ignore[arg-type]
        if hasattr(layer, "self_attn"):
            taps.append(
                TapPoint(name=f"layer.{i}.self_attn", module_path=f"{layers_path}.{i}.self_attn")
            )
        if hasattr(layer, "mlp"):
            taps.append(TapPoint(name=f"layer.{i}.mlp", module_path=f"{layers_path}.{i}.mlp"))
        taps.append(TapPoint(name=f"layer.{i}", module_path=f"{layers_path}.{i}"))

    for norm_path in _FINAL_NORM_PATHS:
        if _try_resolve(model, norm_path) is not None:
            taps.append(TapPoint(name="final_norm", module_path=norm_path))
            break

    return taps
