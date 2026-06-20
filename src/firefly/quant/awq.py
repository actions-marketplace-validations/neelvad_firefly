"""AWQ as a QUANTIZER intervention — wraps torchao's prototype AWQ.

Unlike SmoothQuant (a quantizer-*agnostic* PRE_TRANSFORM), AWQ's per-channel
scaling is **searched to minimize the int4 quantization error**, so the scaling
and the int4 quant are coupled — AWQ is intrinsically a QUANTIZER (it *replaces*
RTN in the pipeline, it doesn't compose in front of it). The α search is
torchao's, exposed as ``scale_search_space_size``.

Wraps ``torchao.prototype.awq``'s observer-based flow: prepare (insert observers)
→ calibrate (one forward) → convert (search scales + int4-quantize). int4
weight-only and GPU-only, like ``int4wo`` (the packing format needs CUDA).

This is the (a)-path spike from the AWQ discussion: wrap the library rather than
reimplement the search. ``treats`` is left empty for now — the salient-weight
detector (``firefly.quant.salience``) exists, so re-adding the
SALIENT_WEIGHT_CHANNELS signature + a diagnose route is an honest follow-up once
this is validated, not part of the spike.
"""

from __future__ import annotations

import torch
from torch import nn

from firefly.quant.intervention import PrecisionPolicy, Stage, _TorchAOIntervention


class AWQQuantizer(_TorchAOIntervention):
    name = "awq"
    stage = Stage.QUANTIZER
    treats = frozenset()

    def __init__(self, group_size: int = 32, scale_search_space_size: int = 20) -> None:
        self.group_size = group_size
        self.scale_search_space_size = scale_search_space_size

    def config(self) -> dict:
        return {"group_size": self.group_size, "scale_search_space_size": self.scale_search_space_size}

    def apply(self, model: nn.Module, policy: PrecisionPolicy, calib: object | None = None) -> nn.Module:
        if calib is None:
            raise ValueError("AWQ needs a calibration batch (calib=...) to search the scales")
        if not policy.quantize:
            return model  # nothing to quantize → keep fp (mixed-precision extreme)

        from torchao.prototype.awq import AWQConfig
        from torchao.quantization import quantize_

        from firefly.quant.torchao import _quant_config

        base = _quant_config("int4wo", self.group_size)  # the GPU tile_packed int4 config
        filter_fn = policy.filter_fn()

        # prepare → calibrate → convert (the per-layer scale search runs at convert).
        quantize_(
            model,
            AWQConfig(base, step="prepare", scale_search_space_size=self.scale_search_space_size),
            filter_fn=filter_fn,
        )
        with torch.no_grad():
            model(**calib) if isinstance(calib, dict) else model(calib)
        quantize_(
            model,
            AWQConfig(base, step="convert", scale_search_space_size=self.scale_search_space_size),
            filter_fn=filter_fn,
        )
        return model
