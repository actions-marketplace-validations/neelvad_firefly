"""Tests for the determinism / hardware-noise-baseline configuration.

These tests only verify the torch settings end up in the expected state;
actually proving hardware emits noise requires a GPU, which is the Modal
validation step.
"""

from __future__ import annotations

import torch

from firefly.determinism import set_deterministic, set_hardware_noise_baseline


def test_set_deterministic_locks_down_backends() -> None:
    set_deterministic(seed=0)

    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_set_hardware_noise_baseline_relaxes_backends() -> None:
    """Hardware-noise mode must invert the determinism settings, otherwise
    GPU runs won't produce the noise we're trying to measure."""
    set_hardware_noise_baseline(seed=0)

    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_set_hardware_noise_baseline_allow_tf32_toggle() -> None:
    set_hardware_noise_baseline(seed=0, allow_tf32=True)
    assert torch.backends.cuda.matmul.allow_tf32 is True

    set_hardware_noise_baseline(seed=0, allow_tf32=False)
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_set_deterministic_restores_after_hardware_baseline() -> None:
    """Switching back from hardware mode to deterministic must fully reset state."""
    set_hardware_noise_baseline(seed=0, allow_tf32=True)
    set_deterministic(seed=0)

    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cuda.matmul.allow_tf32 is False
