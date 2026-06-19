"""SmoothQuant tests. Fast/deterministic: the scale math, output preservation,
channel-disparity reduction, and a synthetic int8 benefit on an outlier channel
(the mechanism proof). The real-model end-to-end is a slow test."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from firefly.quant.intervention import (
    ACTIVATION_OUTLIERS,
    Intervention,
    PrecisionPolicy,
    Stage,
)
from firefly.quant.smoothquant import SmoothQuant, smoothing_scales


def test_is_a_pre_transform_intervention() -> None:
    sq = SmoothQuant()
    assert isinstance(sq, Intervention)
    assert sq.stage == Stage.PRE_TRANSFORM
    assert ACTIVATION_OUTLIERS in sq.treats


def test_smoothing_scales_math() -> None:
    # s = act^0.5 / w^0.5; act=[100,1], w=[1,1] → [10, 1]
    s = smoothing_scales(torch.tensor([100.0, 1.0]), torch.tensor([1.0, 1.0]), alpha=0.5)
    assert torch.allclose(s, torch.tensor([10.0, 1.0]), atol=1e-4)


def test_smoothing_scales_dead_channel_is_one() -> None:
    # a channel with ~0 activation or weight has nothing to migrate → s=1
    s = smoothing_scales(torch.tensor([0.0, 5.0]), torch.tensor([2.0, 0.0]), alpha=0.5)
    assert s[0] == 1.0 and s[1] == 1.0


def test_output_preserved_before_quantization() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(8, 4)  # its own named_modules() yields ("", lin)
    x = torch.randn(3, 8)
    x[:, 2] *= 50.0  # an activation outlier channel
    y0 = lin(x).clone()
    SmoothQuant(scope={""}, alpha=0.5).apply(lin, PrecisionPolicy(quantize={""}), calib=x)
    assert torch.allclose(y0, lin(x), atol=1e-4)  # algebraic identity holds


def test_apply_requires_calib() -> None:
    with pytest.raises(ValueError, match="calibration batch"):
        SmoothQuant(scope={""}).apply(nn.Linear(4, 4), PrecisionPolicy(quantize={""}), calib=None)


def test_reduces_activation_channel_disparity() -> None:
    # smoothed activation max per channel = sqrt(act*w) — the outlier is pulled in
    act_max = torch.tensor([100.0, 1.0, 1.0])
    w_max = torch.tensor([1.0, 1.0, 1.0])
    s = smoothing_scales(act_max, w_max, alpha=0.5)
    smoothed_act_max = act_max / s
    assert smoothed_act_max.max() / smoothed_act_max.min() < act_max.max() / act_max.min()


def _fake_int8_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Per-token int8 activations, per-output-channel int8 weights (w8a8-style)."""
    a_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127
    xq = torch.round(x / a_scale).clamp(-127, 127) * a_scale
    w_scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127
    wq = torch.round(w / w_scale).clamp(-127, 127) * w_scale
    return xq @ wq.T


def test_synthetic_int8_benefit_on_outlier_channel() -> None:
    """The mechanism, unambiguously: with an activation outlier channel, the
    per-token int8 scale is dominated by it and crushes the rest. SmoothQuant
    must cut the matmul's quant error."""
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    x[:, 5] *= 60.0  # one outlier input channel
    w = torch.randn(8, 16)
    y = x @ w.T

    err_plain = (_fake_int8_matmul(x, w) - y).norm()

    s = smoothing_scales(x.abs().amax(0), w.abs().amax(0), alpha=0.5)
    err_smooth = (_fake_int8_matmul(x / s, w * s) - y).norm()

    assert err_smooth < err_plain
    assert err_smooth < 0.5 * err_plain  # a real cut, not a rounding coincidence


@pytest.mark.slow
def test_smoothquant_reduces_w8a8_divergence_smollm() -> None:
    """End-to-end through the pipeline on a real model (CPU w8a8): SmoothQuant as
    a PRE_TRANSFORM before RTN cuts the full-model output divergence sharply
    (measured ~66.7% → ~8.5%, an ~87% recovery)."""
    pytest.importorskip("torchao", reason="needs the torchao extra")
    import json
    import tempfile
    from pathlib import Path

    from firefly.capture import run_capture
    from firefly.quant.sensitivity import _apply_policy, _setup
    from firefly.quant.torchao import rel_l1

    inp = Path(tempfile.mkdtemp()) / "g.json"
    inp.write_text(json.dumps({"texts": ["the quick brown fox jumps over the lazy dog"], "max_length": 16}))
    ctx = _setup("HuggingFaceTB/SmolLM-135M", inp, "cpu", "float32", "w8a8", 32, "layer")
    allq = set(ctx.all_fqns)

    def divergence(model) -> float:
        return rel_l1(ctx.ref_output, run_capture(model, ctx.batch)["final_norm"])

    d_plain = divergence(_apply_policy(ctx.fp_model, "w8a8", 32, allq))
    d_smooth = divergence(
        _apply_policy(ctx.fp_model, "w8a8", 32, allq, [SmoothQuant()], ctx.batch)
    )
    assert d_plain > 0.1  # w8a8 really does hurt this model
    assert d_smooth < 0.5 * d_plain  # SmoothQuant recovers most of it
