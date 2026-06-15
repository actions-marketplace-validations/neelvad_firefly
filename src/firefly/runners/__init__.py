"""Pluggable capture backends ("runners") behind a common interface.

A :class:`Runner` owns the *capture* half of Firefly — loading a model into a
particular execution engine, running the golden inputs through it, and
returning per-tap activations plus the metadata the reference artifact needs.
Everything downstream (diff, attribution, report) consumes two
``{tap: tensor}`` dicts and is runner-agnostic, so adding an engine means
adding a Runner, not touching the comparison pipeline.

The default is :class:`~firefly.runners.hf.HFRunner` (HF transformers, eager
forward hooks). The vLLM runner (in-process, GPU-required) plugs in under the
same interface — Modal is only an experiment harness, never part of a
``firefly check --runner vllm`` product run.
"""

from __future__ import annotations

from firefly.runners.base import (
    CaptureResult,
    Runner,
    available_runners,
    get_runner,
)

__all__ = ["CaptureResult", "Runner", "available_runners", "get_runner"]
