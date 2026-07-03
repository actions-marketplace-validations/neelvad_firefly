"""Unit tests for tap-point discovery on a fake llama-shaped module tree."""

from __future__ import annotations

import pytest
import torch.nn as nn

from firefly.tap_points import (
    find_decoder_layers_path,
    select_llm_tap_points,
    select_recsys_tap_points,
    select_tap_points,
)


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


def test_select_llm_tap_points_forward_order() -> None:
    taps = select_llm_tap_points(_FakeLlama(n_layers=2))

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


class _FakeMultimodalWrapper(nn.Module):
    """Mimics Gemma 3/4 unified checkpoints: the text decoder nests under
    `model.language_model` next to vision/audio towers."""

    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()

        class _Outer(nn.Module):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.language_model = _FakeInner(n_layers)
                inner_self.embed_vision = nn.Linear(4, 4)

        self.model = _Outer()


def test_select_llm_tap_points_multimodal_wrapper() -> None:
    taps = select_llm_tap_points(_FakeMultimodalWrapper(n_layers=2))

    assert [t.name for t in taps] == [
        "layer.0.self_attn",
        "layer.0.mlp",
        "layer.0",
        "layer.1.self_attn",
        "layer.1.mlp",
        "layer.1",
        "final_norm",
    ]
    assert taps[0].module_path == "model.language_model.layers.0.self_attn"
    assert taps[-1].module_path == "model.language_model.norm"


def test_raises_when_no_decoder_layers_found() -> None:
    class _Empty(nn.Module):
        pass

    with pytest.raises(ValueError, match="Could not locate decoder layers"):
        find_decoder_layers_path(_Empty())


def test_dispatcher_defaults_to_llm() -> None:
    via_dispatch = [t.name for t in select_tap_points(_FakeLlama(n_layers=2))]
    via_direct = [t.name for t in select_llm_tap_points(_FakeLlama(n_layers=2))]
    assert via_dispatch == via_direct


def test_dispatcher_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="Unsupported domain"):
        select_tap_points(_FakeLlama(), domain="finance")


def test_handles_empty_module_list() -> None:
    class _ZeroLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([])

    with pytest.raises(ValueError):
        find_decoder_layers_path(_ZeroLayer())


# ---- Recsys selector tests ----------------------------------------------


class _IdentityArch(nn.Module):
    """Stand-in for any arch sub-block; we only care about path resolution."""

    def forward(self, x):
        return x


def test_recsys_torchrec_convention() -> None:
    """A TorchRec-shaped model gets sparse / bottom_mlp / interaction / over_arch taps."""

    class _TorchRecModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sparse_arch = _IdentityArch()
            self.dense_arch = _IdentityArch()
            self.interaction = _IdentityArch()
            self.over_arch = _IdentityArch()

    taps = select_recsys_tap_points(_TorchRecModel())
    names = [t.name for t in taps]
    assert names == ["sparse", "bottom_mlp", "interaction", "over_arch"]
    paths = {t.name: t.module_path for t in taps}
    assert paths["sparse"] == "sparse_arch"
    assert paths["bottom_mlp"] == "dense_arch"
    assert paths["over_arch"] == "over_arch"


def test_recsys_dlrm_convention() -> None:
    """DLRM-style naming (bot_mlp / interactions / top_mlp) resolves to the same tap names."""

    class _DLRMModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = _IdentityArch()
            self.bot_mlp = _IdentityArch()
            self.interactions = _IdentityArch()
            self.top_mlp = _IdentityArch()

    taps = select_recsys_tap_points(_DLRMModel())
    assert [t.name for t in taps] == ["sparse", "bottom_mlp", "interaction", "over_arch"]


def test_recsys_dcn_convention() -> None:
    """DCN-v2 uses cross_net for interactions; no bottom MLP."""

    class _DCNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = _IdentityArch()
            self.cross_net = _IdentityArch()
            self.head = _IdentityArch()

    taps = select_recsys_tap_points(_DCNModel())
    # bottom_mlp is optional; gets omitted when not present
    assert [t.name for t in taps] == ["sparse", "interaction", "over_arch"]


def test_recsys_preserves_forward_order() -> None:
    """Forward order is sparse → bottom_mlp → interaction → over_arch
    regardless of attribute declaration order on the module."""

    class _ReorderedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Deliberately declare in reverse order
            self.over_arch = _IdentityArch()
            self.interaction = _IdentityArch()
            self.dense_arch = _IdentityArch()
            self.sparse_arch = _IdentityArch()

    taps = select_recsys_tap_points(_ReorderedModel())
    assert [t.name for t in taps] == ["sparse", "bottom_mlp", "interaction", "over_arch"]


def test_recsys_raises_on_unknown_architecture() -> None:
    """A model without any recognized recsys stage raises with a hint."""

    class _Empty(nn.Module):
        pass

    with pytest.raises(ValueError, match="Could not locate any recsys tap points"):
        select_recsys_tap_points(_Empty())


def test_recsys_via_dispatcher() -> None:
    """select_tap_points(domain='recsys') routes to the recsys selector."""

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sparse_arch = _IdentityArch()
            self.head = _IdentityArch()

    via_dispatch = [t.name for t in select_tap_points(_Model(), domain="recsys")]
    via_direct = [t.name for t in select_recsys_tap_points(_Model())]
    assert via_dispatch == via_direct
