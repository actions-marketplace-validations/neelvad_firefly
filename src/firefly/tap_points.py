"""Architecturally stable tap points for activation capture.

Tap-point selection is the only domain-specific module in Firefly: capture,
compare, attribute, calibrate, and report are all domain-agnostic. A new
domain (recsys, cv) plugs in by adding a `select_<domain>_tap_points`
function and a dispatch entry, without touching the rest of the pipeline.

For the LLM/transformer family we tap at module boundaries that survive
quantization and torch.compile fusion: per-decoder-layer self-attn output,
MLP output, and the residual stream at layer end.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

# Common paths under which HF models expose their decoder ModuleList.
_DECODER_LAYERS_PATHS = ("model.layers", "transformer.h", "layers")

# Common paths for the final pre-head normalization.
_FINAL_NORM_PATHS = ("model.norm", "transformer.ln_f", "norm")

# Attribute names under a decoder layer where the attention sub-module lives.
_ATTN_MODULE_NAMES = ("self_attn", "self_attention", "attention", "attn")

# Attribute names for the attention *output* projection within the attention
# sub-module. Llama/Mistral/Qwen use ``o_proj``; OPT uses ``out_proj``;
# GPT-NeoX/Falcon/BLOOM use ``dense``; GPT-2 uses ``c_proj``.
_ATTN_OUTPUT_PROJ_NAMES = ("o_proj", "out_proj", "dense", "c_proj")


@dataclass(frozen=True)
class TapPoint:
    """A named hook site on a model."""

    name: str
    module_path: str  # dotted attribute path from the model root
    capture_input: bool = False
    """If True, capture the *input* to this module rather than its output.

    Used for per-head attention taps: the attention output projection
    (``o_proj``) consumes the concatenated per-head outputs of shape
    ``(..., n_heads * head_dim)``. We tap its input because post-projection
    the heads are linearly mixed and no longer separable — only the input
    carries recoverable per-head structure.
    """


def resolve_module_path(module: nn.Module, dotted_path: str) -> nn.Module:
    """Walk a dotted attribute path from ``module`` and return the leaf module."""
    obj: object = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _try_resolve(module: nn.Module, dotted_path: str) -> nn.Module | None:
    try:
        return resolve_module_path(module, dotted_path)
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


def _find_attn_output_proj(layer: nn.Module) -> str | None:
    """Return the dotted sub-path (from the layer) to the attention output
    projection, e.g. ``"self_attn.o_proj"``, or ``None`` if not found.

    Probes the common attention-module names then the common projection
    names within. The first match wins, so the order of the name tuples
    matters only when a model has two plausible candidates (rare).
    """
    for attn_name in _ATTN_MODULE_NAMES:
        attn = getattr(layer, attn_name, None)
        if attn is None:
            continue
        for proj_name in _ATTN_OUTPUT_PROJ_NAMES:
            if hasattr(attn, proj_name):
                return f"{attn_name}.{proj_name}"
    return None


def select_llm_tap_points(model: nn.Module, per_head: bool = False) -> list[TapPoint]:
    """Walk a HF-style decoder transformer and return its stable tap points.

    Per decoder layer i, emits (in forward order):
        layer.{i}.self_attn  → output of the attention sub-block
        layer.{i}.attn_heads → input to the attention output projection
                               (per-head outputs); only when ``per_head``
        layer.{i}.mlp        → output of the MLP sub-block
        layer.{i}            → residual stream at end of layer

    Plus a single trailing `final_norm` tap when present.

    The forward-order ordering matters: attribution walks this list in order
    and reports the first tap that exceeds tolerance, which corresponds to
    the earliest point in the network where divergence appeared.

    When ``per_head`` is set, each layer additionally gets an
    ``attn_heads`` tap that captures the *input* to the attention output
    projection. That tensor is the concatenated per-head attention outputs,
    which :mod:`firefly.head_attribution` can split by head to attribute
    divergence to a specific attention head. The tap is placed right after
    ``self_attn`` in forward order since it observes an earlier point in the
    same sub-block.
    """
    layers_path = find_decoder_layers_path(model)
    layers = resolve_module_path(model, layers_path)

    taps: list[TapPoint] = []
    for i, layer in enumerate(layers):  # type: ignore[arg-type]
        if hasattr(layer, "self_attn"):
            taps.append(
                TapPoint(name=f"layer.{i}.self_attn", module_path=f"{layers_path}.{i}.self_attn")
            )
        if per_head:
            proj_subpath = _find_attn_output_proj(layer)
            if proj_subpath is not None:
                taps.append(
                    TapPoint(
                        name=f"layer.{i}.attn_heads",
                        module_path=f"{layers_path}.{i}.{proj_subpath}",
                        capture_input=True,
                    )
                )
        if hasattr(layer, "mlp"):
            taps.append(TapPoint(name=f"layer.{i}.mlp", module_path=f"{layers_path}.{i}.mlp"))
        taps.append(TapPoint(name=f"layer.{i}", module_path=f"{layers_path}.{i}"))

    for norm_path in _FINAL_NORM_PATHS:
        if _try_resolve(model, norm_path) is not None:
            taps.append(TapPoint(name="final_norm", module_path=norm_path))
            break

    return taps


_RECSYS_SPARSE_PATHS = (
    "sparse_arch",     # TorchRec
    "embedding_arch",  # TorchRec alternative
    "sparse",          # ad-hoc convention
    "embeddings",      # ad-hoc convention
)

_RECSYS_INTERACTION_PATHS = (
    "interaction",     # DLRM / generic
    "interactions",    # DLRM alternative
    "cross_net",       # DCN-v2
    "fm",              # FactorizationMachine
    "mask_block",      # MaskNet
)

_RECSYS_DENSE_PATHS = (
    "dense_arch",      # TorchRec input/bottom
    "bot_mlp",         # DLRM bottom MLP
    "bottom_mlp",      # DLRM alternative naming
    "dense",           # generic
)

_RECSYS_OVER_PATHS = (
    "over_arch",       # TorchRec output/top
    "top_mlp",         # DLRM top MLP
    "top",             # generic
    "head",            # generic
    "predictor",       # generic
    "classifier",      # generic
)


def select_recsys_tap_points(model: nn.Module, per_head: bool = False) -> list[TapPoint]:
    """Walk a recsys-style model and return its stable tap points.

    Recsys models lack the per-decoder-layer regularity that HF
    transformers have; instead we target the canonical four-stage
    decomposition that production recsys (TorchRec, DLRM, DCN-v2)
    converges on:

        sparse      → embedding lookups for categorical features
        bottom_mlp  → dense feature processing (optional)
        interaction → feature crossing (DLRM dot, DCN cross-net, FM, etc.)
        over_arch   → final tower MLPs and head

    Each stage's output is a tap. Forward order is preserved so the
    first-divergence attribution still names the earliest stage where
    behavior changes — embedding lookup, interaction, or tower.

    Naming conventions probed (in order, first match wins per stage):
      sparse:      sparse_arch / embedding_arch / sparse / embeddings
      bottom MLP:  dense_arch / bot_mlp / bottom_mlp / dense
      interaction: interaction / interactions / cross_net / fm / mask_block
      over_arch:   over_arch / top_mlp / top / head / predictor / classifier

    Raises ValueError if no recognized stage is present — likely a
    non-standard architecture that needs a hand-written tap config.
    """
    taps: list[TapPoint] = []

    # Sparse / embedding lookups (forward-order: first stage)
    for path in _RECSYS_SPARSE_PATHS:
        if _try_resolve(model, path) is not None:
            taps.append(TapPoint(name="sparse", module_path=path))
            break

    # Dense / bottom MLP (only matters when present; some architectures skip)
    for path in _RECSYS_DENSE_PATHS:
        if _try_resolve(model, path) is not None:
            taps.append(TapPoint(name="bottom_mlp", module_path=path))
            break

    # Interaction / feature crossing
    for path in _RECSYS_INTERACTION_PATHS:
        if _try_resolve(model, path) is not None:
            taps.append(TapPoint(name="interaction", module_path=path))
            break

    # Over-arch / top MLP / head
    for path in _RECSYS_OVER_PATHS:
        if _try_resolve(model, path) is not None:
            taps.append(TapPoint(name="over_arch", module_path=path))
            break

    if not taps:
        raise ValueError(
            "Could not locate any recsys tap points. Tried sparse, "
            "interaction, and dense paths under the standard TorchRec, "
            "DLRM, and DCN-v2 conventions. For non-standard architectures, "
            "consider exporting a hand-written tap config (planned)."
        )

    return taps


_TAP_SELECTORS = {
    "llm": select_llm_tap_points,
    "recsys": select_recsys_tap_points,
    # "cv":     select_cv_tap_points,      # planned v2
}


def select_tap_points(
    model: nn.Module, domain: str = "llm", per_head: bool = False
) -> list[TapPoint]:
    """Domain-aware tap-point selection. The dispatch seam for v2 domains.

    ``per_head`` adds per-head attention taps (LLM domain only; the recsys
    selector accepts and ignores it, since attention heads have no recsys
    analogue).
    """
    try:
        selector = _TAP_SELECTORS[domain]
    except KeyError as e:
        raise ValueError(
            f"Unsupported domain: {domain!r}. Available: {sorted(_TAP_SELECTORS)}"
        ) from e
    return selector(model, per_head=per_head)
