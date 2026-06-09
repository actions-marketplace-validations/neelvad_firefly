"""Quantization-risk attribution from captured activations.

Predicts which layers will degrade under quantization *before* quantizing,
using only the activations already stored in a reference artifact — no model
run, no quantized weights needed.

The mechanism this targets is the outlier-feature phenomenon (Dettmers et
al. 2022, validated on SmolLM-135M in Firefly's Phase-2 experiments: the
residual stream jumps ~50 → ~30,000 at layer 11 and stays there). A handful
of channels carrying extreme magnitudes force a per-tensor quant scale sized
for the outliers, crushing resolution for every other channel. Per-channel
scaling rescues this — so the per-tensor / per-channel error *ratio* is the
actionable signal: high ratio means "this tap needs per-channel quant (or
outlier-aware handling) or it breaks."

Like the rest of Firefly's analysis layer, this module is pure: tensors in,
dataclasses out, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TapQuantRisk:
    """Quantization-risk metrics for one tap's captured activation."""

    tap_name: str
    abs_max: float
    abs_mean: float
    outlier_ratio: float
    """``abs_max / abs_mean`` — how heavy-tailed the activation distribution
    is. Gaussian activations sit around 3–5; outlier-feature layers reach
    hundreds."""
    channel_concentration: float
    """Max-channel abs-max / median-channel abs-max (channels = last dim).
    The Dettmers signature: a few channels dominating the dynamic range.
    ~1 means uniform channels; outlier-feature layers reach 10–100×."""
    per_tensor_rel_err: float
    """Simulated symmetric per-tensor quant: mean |q(x) - x| / mean |x|."""
    per_channel_rel_err: float
    """Same simulation with one scale per channel (last dim)."""

    @property
    def mitigation_gain(self) -> float:
        """``per_tensor_rel_err / per_channel_rel_err`` — how much per-channel
        scaling rescues this tap. ~1 means per-tensor is fine; high values
        mean the tap is outlier-dominated and per-tensor quant will hurt.
        Returns ``inf`` when per-channel error is 0 but per-tensor isn't."""
        if self.per_channel_rel_err > 0:
            return self.per_tensor_rel_err / self.per_channel_rel_err
        return float("inf") if self.per_tensor_rel_err > 0 else 1.0


def simulate_quant_error(
    t: torch.Tensor, bits: int = 8, per_channel: bool = False
) -> float:
    """Channel-mean relative error of symmetric round-to-nearest quantization.

    ``per_channel=False`` uses one scale for the whole tensor (sized by its
    global abs-max); ``per_channel=True`` uses one scale per slice of the
    last dimension.

    The error is computed *within* each channel — ``mean |q(x) - x| /
    mean |x|`` over that channel — then averaged across channels. This is
    deliberate: a globally magnitude-weighted error lets an outlier channel
    dominate numerator and denominator alike, hiding the real failure mode
    (per-tensor scaling rounds every *normal* channel to zero — 100% error
    on each — while the global average barely moves). Channel-mean error
    reports that destruction directly. All-zero channels contribute 0.
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    x = t.float().reshape(-1, t.shape[-1]) if t.ndim >= 1 else t.float().reshape(-1, 1)
    qmax = 2 ** (bits - 1) - 1

    if per_channel:
        scale = x.abs().amax(dim=0) / qmax  # one scale per channel
    else:
        scale = (x.abs().max() / qmax).expand(x.shape[-1])  # global, broadcast

    # Zero-scale channels (or an all-zero tensor) quantize exactly; guard
    # the division and zero the reconstruction wherever scale == 0.
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (x / safe_scale).round().clamp(-qmax, qmax) * safe_scale
    q = torch.where(scale > 0, q, torch.zeros_like(q))

    err_per_chan = (q - x).abs().mean(dim=0)
    mag_per_chan = x.abs().mean(dim=0)
    rel = torch.where(
        mag_per_chan > 0, err_per_chan / mag_per_chan, torch.zeros_like(mag_per_chan)
    )
    return float(rel.mean())


def tap_quant_risk(tap_name: str, t: torch.Tensor, bits: int = 8) -> TapQuantRisk:
    """Compute all quant-risk metrics for one captured activation tensor."""
    x = t.float()
    abs_max = float(x.abs().max()) if x.numel() else 0.0
    abs_mean = float(x.abs().mean()) if x.numel() else 0.0

    # Channel stats over the last dim — the feature axis for transformer
    # activations, matching where outlier features live.
    if x.ndim >= 1 and x.shape[-1] > 1:
        chan_max = x.abs().amax(dim=tuple(range(x.ndim - 1)))
        sorted_cm = chan_max.sort().values
        median_cm = float(sorted_cm[sorted_cm.numel() // 2])
        max_cm = float(sorted_cm[-1])
        if median_cm > 0:
            channel_concentration = max_cm / median_cm
        else:
            channel_concentration = float("inf") if max_cm > 0 else 1.0
    else:
        channel_concentration = 1.0

    return TapQuantRisk(
        tap_name=tap_name,
        abs_max=abs_max,
        abs_mean=abs_mean,
        outlier_ratio=(abs_max / abs_mean) if abs_mean > 0 else 1.0,
        channel_concentration=channel_concentration,
        per_tensor_rel_err=simulate_quant_error(x, bits=bits, per_channel=False),
        per_channel_rel_err=simulate_quant_error(x, bits=bits, per_channel=True),
    )


def analyze_quant_risk(
    tensors: dict[str, torch.Tensor],
    tap_order: list[str],
    bits: int = 8,
) -> list[TapQuantRisk]:
    """Run :func:`tap_quant_risk` for every tap, in forward order.

    Taps in ``tap_order`` but missing from ``tensors`` are skipped (the
    caller's manifest may list taps a partial capture didn't produce);
    taps in ``tensors`` but not in ``tap_order`` are ignored, preserving
    the forward-order contract the rest of Firefly relies on.
    """
    return [
        tap_quant_risk(name, tensors[name], bits=bits)
        for name in tap_order
        if name in tensors
    ]
