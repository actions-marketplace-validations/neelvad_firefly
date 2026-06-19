"""SmoothQuant — the first real PRE_TRANSFORM intervention.

The failure it treats (``ACTIVATION_OUTLIERS``): a few input-channel activations
have huge magnitude, so per-token activation quantization sets its scale from
those outliers and crushes every other channel to a couple of levels. SmoothQuant
uses the identity ``Y = X·Wᵀ = (X/s)·(s·W)ᵀ`` for a per-input-channel scale ``s``:
divide the activations by ``s`` (taming the outliers) and multiply the weight
columns by ``s`` (weights are smooth and absorb it). Same output, but now both
tensors are easy to quantize — the difficulty is *migrated* from activations into
weights.

``s_j = act_max_jᵅ / weight_max_j^(1-ᵅ)`` per input channel ``j`` (α = migration
strength, 0.5 by default). ``act_max`` comes from a calibration forward.

Why this is *implemented* rather than wrapped (cf. the "wrap, don't reimplement"
rule): SmoothQuant's core is a per-channel rescale — well-specified arithmetic,
not a hard algorithm like GPTQ's Hessian solve or AWQ's search. Owning ~50 lines
makes it a clean :class:`Stage.PRE_TRANSFORM` that hands a smoothed fp model to
whatever quantizer runs next; torchao's prototype ``SmoothQuantConfig`` is itself
a *quantizer*, which would conflate the two pipeline stages. GPTQ/AWQ, the genuine
algorithms, will be wrapped.

The activation scaling is applied as a forward pre-hook (×1/s) rather than folded
into the preceding LayerNorm — architecture-agnostic, and it survives torchao's
``quantize_`` (which swaps the weight to a tensor subclass but keeps the module
and its hooks). Cheap elementwise op; the point is that the *quantized* activation
it feeds is now well-conditioned.
"""

from __future__ import annotations

import torch
from torch import nn

from firefly.quant.intervention import ACTIVATION_OUTLIERS, PrecisionPolicy, Stage

_EPS = 1e-5


def smoothing_scales(act_max: torch.Tensor, weight_max: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-input-channel scale ``s = act_maxᵅ / weight_max^(1-ᵅ)``.

    Channels where either max is ~0 get ``s = 1`` (nothing to migrate). Pure."""
    act = act_max.clamp(min=_EPS).to(torch.float32)
    wts = weight_max.clamp(min=_EPS).to(torch.float32)
    s = act.pow(alpha) / wts.pow(1.0 - alpha)
    dead = (act_max <= _EPS) | (weight_max <= _EPS)
    return torch.where(dead, torch.ones_like(s), s)


def _collect_act_max(model: nn.Module, targets: dict[str, nn.Linear], calib) -> dict[str, torch.Tensor]:
    """Per-input-channel max |activation| at each target Linear, over the calib
    batch. Hooks capture each module's *input* (the tensor it quantizes)."""
    stats: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_mod, inputs):
            x = inputs[0].detach().to(torch.float32).abs()
            x = x.reshape(-1, x.shape[-1]).amax(dim=0)  # per input channel
            stats[name] = x if name not in stats else torch.maximum(stats[name], x)
        return hook

    for name, mod in targets.items():
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(**calib) if isinstance(calib, dict) else model(calib)
    finally:
        for h in handles:
            h.remove()
    return stats


def _install_input_scaling(module: nn.Linear, inv_s: torch.Tensor) -> None:
    """Pre-hook that scales the Linear's input by ``1/s`` per channel before it
    runs (and before its dynamic activation quant). Survives ``quantize_``."""

    def hook(_mod, inputs):
        x = inputs[0]
        return (x * inv_s.to(x.dtype),) + inputs[1:]

    module.register_forward_pre_hook(hook)


class SmoothQuant:
    """PRE_TRANSFORM that migrates activation outliers into weights.

    ``scope`` selects which Linear FQNs to smooth; ``None`` means "whatever the
    policy quantizes". ``apply`` calibrates on ``calib`` (required), scales the
    weights in place, and installs the input pre-hooks — returning a smoothed fp
    model for the quantizer stage to consume.
    """

    stage = Stage.PRE_TRANSFORM
    treats = frozenset({ACTIVATION_OUTLIERS})

    def __init__(self, scope: set[str] | None = None, alpha: float = 0.5, name: str = "smoothquant") -> None:
        self.scope = scope
        self.alpha = alpha
        self.name = name

    def apply(self, model: nn.Module, policy: PrecisionPolicy, calib: object | None = None) -> nn.Module:
        if calib is None:
            raise ValueError("SmoothQuant needs a calibration batch (calib=...) to measure activations")
        want = self.scope if self.scope is not None else policy.quantize
        targets = {
            name: mod
            for name, mod in model.named_modules()
            if name in want and isinstance(mod, nn.Linear)
        }
        if not targets:
            return model

        act_max = _collect_act_max(model, targets, calib)
        for name, mod in targets.items():
            if name not in act_max:
                continue  # module never fired on the calib batch
            w_max = mod.weight.detach().to(torch.float32).abs().amax(dim=0)  # per input channel
            s = smoothing_scales(act_max[name], w_max, self.alpha)
            with torch.no_grad():
                mod.weight.mul_(s.to(mod.weight.dtype))  # W[:, j] *= s_j
            _install_input_scaling(mod, 1.0 / s)
        return model
