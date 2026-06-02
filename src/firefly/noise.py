"""Synthetic noise injection for calibration development.

CPU+fp32 forward passes are bit-deterministic, which is exactly what we want
for catching real divergences but useless for calibrating against a noise
floor — there is none. This module injects controllable Gaussian noise at
one chosen tap point, letting calibration develop on a MacBook before ever
touching a GPU.

The injection model deliberately mirrors what real hardware nondeterminism
looks like: noise originates at one early point (atomics, reduction order),
then propagates through the rest of a deterministic forward and accumulates
with depth. So we inject at *one* tap and let downstream taps amplify, rather
than adding iid noise at every tap — the latter would be both unrealistic
and would defeat the per-layer-tolerance methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from firefly.tap_points import resolve_module_path, select_tap_points

NoiseMode = Literal["none", "synthetic"]


@dataclass
class NoiseSpec:
    """Configuration for noise injection at a single tap.

    ``mode="none"`` is the no-op (pure determinism). ``mode="synthetic"`` injects
    Gaussian noise of standard deviation ``sigma`` at the module identified by
    ``inject_at`` (a tap name from ``select_tap_points``). Per-call seeding off
    ``base_seed`` makes successive forward passes produce *different* noise
    realizations while keeping the whole experiment reproducible.
    """

    mode: NoiseMode = "none"
    sigma: float = 0.0
    inject_at: str | None = None
    base_seed: int = 0


class _NoiseInjector:
    """Forward hook that adds Gaussian noise to a module's output.

    Each call uses ``base_seed + call_count`` so successive forward passes
    through the same hook produce N different noise realizations. The shape
    and dtype of the noise match the module's primary output tensor.
    """

    def __init__(self, sigma: float, base_seed: int) -> None:
        self.sigma = sigma
        self.base_seed = base_seed
        self.call_count = 0

    def __call__(self, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        primary = output[0] if isinstance(output, tuple) else output

        gen = torch.Generator(device=primary.device)
        gen.manual_seed(self.base_seed + self.call_count)
        self.call_count += 1

        noise = (
            torch.randn(
                primary.shape,
                generator=gen,
                device=primary.device,
                dtype=primary.dtype,
            )
            * self.sigma
        )
        noised = primary + noise

        if isinstance(output, tuple):
            return (noised,) + output[1:]
        return noised


def register_noise_hook(
    model: nn.Module,
    spec: NoiseSpec,
    domain: str = "llm",
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook on ``spec.inject_at`` that injects noise per ``spec``.

    Returns the handle so the caller can remove the hook when done.
    """
    if spec.mode == "none":
        raise ValueError("register_noise_hook called with mode='none'")
    if spec.mode != "synthetic":
        raise ValueError(f"Unknown noise mode: {spec.mode!r}")
    if spec.inject_at is None:
        raise ValueError("inject_at is required for noise injection")

    taps = select_tap_points(model, domain=domain)
    matching = next((t for t in taps if t.name == spec.inject_at), None)
    if matching is None:
        available = [t.name for t in taps]
        sample = ", ".join(available[:5]) + ("..." if len(available) > 5 else "")
        raise ValueError(
            f"inject_at={spec.inject_at!r} doesn't match any tap. Available: {sample}"
        )

    submod = resolve_module_path(model, matching.module_path)
    injector = _NoiseInjector(sigma=spec.sigma, base_seed=spec.base_seed)
    return submod.register_forward_hook(injector)
