"""Weight-salience sensor — the AWQ signal, as a measurement.

AWQ protects the input channels whose weights carry the most signal, judged by
the activation magnitude flowing through them:

    salience_j ≈ mean|X[:, j]| · max|W[:, j]|   (per input channel j)

A few channels with salience far above the rest is the case AWQ's per-channel
weight scaling fixes; uniform salience means AWQ won't help. The summary signal
is ``salience_concentration`` = max / median of the per-channel salience (the
weight-side analogue of quant-risk's activation ``channel_concentration``).

This is the **detector** for the SALIENT_WEIGHT_CHANNELS signature; its treatment
is the AWQ quantizer (``firefly.quant.awq``). Both exist, so ``firefly.quant.
diagnose`` routes the signature to AWQ — we added the signal with its detector
and treatment, never a label without them.

Mechanism (answer to "do we hook the params?"): no. ``|W|`` is static, read
directly from ``module.weight``; only the activation side needs a forward
pre-hook on each target Linear to observe its *input* over a calibration pass —
the same pattern as SmoothQuant.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LinearSalience:
    fqn: str
    salience_concentration: float
    """max / median per-input-channel salience — high = a few channels dominate
    (AWQ-protectable); ~1 = uniform (AWQ won't help)."""
    n_channels: int


def _collect_input_mean_abs(
    model: nn.Module, targets: dict[str, nn.Linear], calib
) -> dict[str, torch.Tensor]:
    """Per-input-channel mean ``|X|`` at each target Linear over ``calib``."""
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_mod, inputs):
            x = inputs[0].detach().to(torch.float32).abs().reshape(-1, inputs[0].shape[-1])
            sums[name] = x.sum(0) if name not in sums else sums[name] + x.sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    for name, mod in targets.items():
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(**calib) if isinstance(calib, dict) else model(calib)
    finally:
        for h in handles:
            h.remove()
    return {name: sums[name] / counts[name] for name in sums}


def weight_salience(model: nn.Module, fqns, calib) -> list[LinearSalience]:
    """Per-Linear AWQ salience concentration, ranked. ``fqns`` selects which
    Linears; ``calib`` is the calibration batch (dict or tensor)."""
    want = set(fqns)
    targets = {
        name: mod
        for name, mod in model.named_modules()
        if name in want and isinstance(mod, nn.Linear)
    }
    act_mean = _collect_input_mean_abs(model, targets, calib)

    out: list[LinearSalience] = []
    for name, mod in targets.items():
        if name not in act_mean:
            continue
        w_mag = mod.weight.detach().to(torch.float32).abs().amax(dim=0)  # per input channel
        salience = (act_mean[name] * w_mag).tolist()
        median = statistics.median(salience)
        conc = (max(salience) / median) if median > 0 else float("inf")
        out.append(LinearSalience(name, conc, len(salience)))
    return sorted(out, key=lambda s: s.salience_concentration, reverse=True)
