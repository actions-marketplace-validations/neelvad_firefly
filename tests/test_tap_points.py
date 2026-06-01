"""Unit tests for tap-point discovery on a fake llama-shaped module tree."""

from __future__ import annotations

import pytest
import torch.nn as nn

from firefly.tap_points import find_decoder_layers_path, select_default_tap_points


class _FakeAttn(nn.Module):
    def forward(self, x):  # noqa: D401
        return x


class _FakeMLP(nn.Module):
    def forward(self, x):
        return x


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttn()
        self.mlp = _FakeMLP()

    def forward(self, x):
        return x + self.mlp(self.self_attn(x))


class _FakeInner(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(n_layers)])
        self.norm = nn.LayerNorm(4)


class _FakeLlama(nn.Module):
    """Mimics the `model.layers[...]` + `model.norm` structure of HF llama-family."""

    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.model = _FakeInner(n_layers)


def test_find_decoder_layers_path() -> None:
    assert find_decoder_layers_path(_FakeLlama()) == "model.layers"


def test_select_default_tap_points_forward_order() -> None:
    taps = select_default_tap_points(_FakeLlama(n_layers=2))

    expected = [
        "layer.0.self_attn",
        "layer.0.mlp",
        "layer.0",
        "layer.1.self_attn",
        "layer.1.mlp",
        "layer.1",
        "final_norm",
    ]
    assert [t.name for t in taps] == expected

    assert taps[0].module_path == "model.layers.0.self_attn"
    assert taps[2].module_path == "model.layers.0"
    assert taps[-1].module_path == "model.norm"


def test_raises_when_no_decoder_layers_found() -> None:
    class _Empty(nn.Module):
        pass

    with pytest.raises(ValueError, match="Could not locate decoder layers"):
        find_decoder_layers_path(_Empty())


def test_handles_empty_module_list() -> None:
    class _ZeroLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([])

    with pytest.raises(ValueError):
        find_decoder_layers_path(_ZeroLayer())
