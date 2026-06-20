"""Weight-salience (AWQ signal) sensor — fast, no torchao (pre-quant analysis)."""

from __future__ import annotations

import torch
import torch.nn as nn

from firefly.quant.salience import LinearSalience, weight_salience


def test_salient_channel_gives_high_concentration() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(16, 8, bias=False)  # root module name is ""
    with torch.no_grad():
        lin.weight[:, 5] *= 30.0  # weight magnitude concentrated in channel 5
    x = torch.randn(4, 16)
    x[:, 5] *= 40.0  # activation magnitude too → salient channel
    sal = weight_salience(lin, [""], x)
    assert len(sal) == 1 and sal[0].n_channels == 16
    assert sal[0].salience_concentration > 8.0  # AWQ-protectable


def test_uniform_weights_give_low_concentration() -> None:
    torch.manual_seed(1)
    lin = nn.Linear(16, 8, bias=False)
    sal = weight_salience(lin, [""], torch.randn(8, 16))
    assert sal[0].salience_concentration < 5.0  # no dominant channel


def test_salience_uses_both_activation_and_weight() -> None:
    # A big activation channel whose WEIGHT is tiny should NOT be salient
    # (salience = |X|·|W|), unlike SmoothQuant which keys on activation alone.
    torch.manual_seed(2)
    lin = nn.Linear(16, 8, bias=False)
    with torch.no_grad():
        lin.weight[:, 7] *= 1e-3  # near-zero weight in the big-activation channel
    x = torch.randn(4, 16)
    x[:, 7] *= 100.0
    sal = weight_salience(lin, [""], x)
    # channel 7 carries huge activation but ~no weight → not the concentration driver
    assert sal[0].salience_concentration < 50.0


def test_ranks_linears_by_concentration() -> None:
    torch.manual_seed(0)

    class _Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flat = nn.Linear(8, 8, bias=False)
            self.spiky = nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.spiky.weight[:, 2] *= 50.0

        def forward(self, x):
            a = self.flat(x)
            b = self.spiky(x.clone())
            b[:, 2] = b[:, 2]  # keep graph simple
            return a + b

    m = _Two()
    x = torch.randn(4, 8)
    x[:, 2] *= 50.0
    sal = weight_salience(m, ["flat", "spiky"], x)
    assert isinstance(sal[0], LinearSalience)
    assert sal[0].fqn == "spiky"  # most concentrated first
    assert sal[0].salience_concentration > sal[1].salience_concentration


def test_render_salience_smoke() -> None:
    from firefly.report import render_salience

    sal = [LinearSalience("model.layers.0.mlp.down_proj", 42.0, 768),
           LinearSalience("model.layers.1.mlp.up_proj", 1.5, 768)]
    out = render_salience(sal)
    assert "down_proj" in out and "42.0x" in out
    assert "AWQ" in out
