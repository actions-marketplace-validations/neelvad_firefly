"""Tests for synthetic noise injection."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from firefly.capture import run_capture_repeated
from firefly.noise import NoiseSpec, _NoiseInjector, register_noise_hook


class _Submod(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _Layer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = _Submod(dim)
        self.mlp = _Submod(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class _Inner(nn.Module):
    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, dim)
        self.layers = nn.ModuleList([_Layer(dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)


class _FakeLM(nn.Module):
    def __init__(self, dim: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.model = _Inner(dim, n_layers)
        self.lm_head = nn.Linear(dim, 100)

    def forward(self, input_ids: torch.Tensor | None = None, **_kwargs) -> torch.Tensor:
        x = self.model.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return self.lm_head(x)


def _model_and_batch() -> tuple[_FakeLM, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = _FakeLM(dim=8, n_layers=2).eval()
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    return model, batch


# --- _NoiseInjector unit tests ------------------------------------------------


def test_noise_injector_preserves_shape_and_dtype() -> None:
    injector = _NoiseInjector(sigma=1e-3, base_seed=0)
    x = torch.zeros(2, 4, 8, dtype=torch.float32)
    out = injector(nn.Identity(), (x,), x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_noise_injector_handles_tuple_output() -> None:
    """HF self_attn returns (hidden, attn_weights, kv); only hidden is noised."""
    injector = _NoiseInjector(sigma=1.0, base_seed=0)
    primary = torch.zeros(1, 4)
    secondary = torch.tensor([42.0])
    out = injector(nn.Identity(), None, (primary, secondary))

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert not torch.equal(out[0], primary)
    assert torch.equal(out[1], secondary)


def test_noise_injector_is_deterministic_per_call_index() -> None:
    a = _NoiseInjector(sigma=1.0, base_seed=42)
    b = _NoiseInjector(sigma=1.0, base_seed=42)
    x = torch.zeros(8)

    for _ in range(3):
        out_a = a(nn.Identity(), None, x.clone())
        out_b = b(nn.Identity(), None, x.clone())
        assert torch.equal(out_a, out_b)


def test_noise_injector_varies_across_calls() -> None:
    """Successive calls on identical inputs must produce different outputs —
    that's the whole point: each forward gets a different noise realization.
    """
    injector = _NoiseInjector(sigma=1.0, base_seed=0)
    x = torch.zeros(8)

    out_1 = injector(nn.Identity(), None, x.clone())
    out_2 = injector(nn.Identity(), None, x.clone())

    assert not torch.equal(out_1, out_2)


# --- register_noise_hook tests ------------------------------------------------


def test_register_noise_hook_rejects_mode_none() -> None:
    model, _ = _model_and_batch()
    with pytest.raises(ValueError, match="mode='none'"):
        register_noise_hook(model, NoiseSpec(mode="none"))


def test_register_noise_hook_rejects_unknown_tap() -> None:
    model, _ = _model_and_batch()
    with pytest.raises(ValueError, match="doesn't match any tap"):
        register_noise_hook(
            model,
            NoiseSpec(mode="synthetic", sigma=1e-3, inject_at="bogus.tap"),
        )


def test_register_noise_hook_returns_removable_handle() -> None:
    model, _ = _model_and_batch()
    handle = register_noise_hook(
        model,
        NoiseSpec(mode="synthetic", sigma=1e-3, inject_at="layer.0"),
    )

    # Should be the only hook on the layer.0 module.
    target = model.model.layers[0]
    assert len(target._forward_hooks) == 1

    handle.remove()
    assert len(target._forward_hooks) == 0


# --- integration with run_capture_repeated -----------------------------------


def test_noise_only_affects_downstream_taps() -> None:
    """Injecting at layer.0 must leave upstream taps (layer.0.self_attn,
    layer.0.mlp) unaffected across runs, while layer.0 and downstream vary.
    """
    model, batch = _model_and_batch()
    spec = NoiseSpec(mode="synthetic", sigma=1e-2, inject_at="layer.0", base_seed=0)

    captures = run_capture_repeated(model, batch, runs=3, noise=spec)

    # Upstream of the injection point: identical across runs.
    for upstream in ("layer.0.self_attn", "layer.0.mlp"):
        first = captures[upstream][0]
        for other in captures[upstream][1:]:
            assert torch.equal(first, other), f"{upstream} unexpectedly varies"

    # Injection point and downstream: must vary across runs.
    for downstream in ("layer.0", "layer.1.self_attn", "layer.1", "final_norm"):
        first = captures[downstream][0]
        assert any(
            not torch.equal(first, other) for other in captures[downstream][1:]
        ), f"{downstream} should vary across runs"


def test_noise_mode_none_is_no_op() -> None:
    """noise=NoiseSpec(mode='none') must behave identically to noise=None."""
    model, batch = _model_and_batch()

    without_arg = run_capture_repeated(model, batch, runs=2)
    with_none = run_capture_repeated(
        model, batch, runs=2, noise=NoiseSpec(mode="none")
    )

    for tap in without_arg:
        for a, b in zip(without_arg[tap], with_none[tap], strict=True):
            assert torch.equal(a, b)


def test_noise_runs_are_reproducible_with_same_base_seed() -> None:
    """Two calibration runs with the same base_seed produce bit-identical captures."""
    spec = NoiseSpec(mode="synthetic", sigma=1e-3, inject_at="layer.0", base_seed=7)

    model_a, batch_a = _model_and_batch()
    model_b, batch_b = _model_and_batch()

    captures_a = run_capture_repeated(model_a, batch_a, runs=3, noise=spec)
    captures_b = run_capture_repeated(model_b, batch_b, runs=3, noise=spec)

    for tap in captures_a:
        for a, b in zip(captures_a[tap], captures_b[tap], strict=True):
            assert torch.equal(a, b)
