"""Forward-hook based activation capture.

Two layers:

* :func:`run_capture` is the pure, testable core. Takes a model and a batch,
  registers forward hooks at the discovered tap points, runs the forward pass,
  returns ``{tap_name: tensor}``. Knows nothing about HF or disk.
* :func:`capture_reference` is the orchestrator the CLI calls. Loads the HF
  model + tokenizer, tokenizes a golden-inputs JSON file, invokes
  ``run_capture``, and writes the on-disk reference artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from firefly.determinism import set_deterministic
from firefly.noise import NoiseSpec, register_noise_hook
from firefly.reference import (
    ReferenceManifest,
    capture_env,
    write_reference,
)
from firefly.tap_points import resolve_module_path, select_tap_points

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def run_capture_repeated(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    runs: int = 1,
    domain: str = "llm",
    noise: NoiseSpec | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Register hooks once, run ``runs`` forward passes, return one tensor per
    run per tap. Used by calibration, which needs many self-runs to measure
    the per-tap noise floor without paying for repeated hook registration.

    If ``noise`` is supplied (and ``noise.mode != "none"``), a noise hook is
    registered at ``noise.inject_at`` *before* the capture hooks, so the
    captured tensors at and downstream of the injection point reflect the
    noised activations.

    Hooks handle tuple outputs (e.g., HF ``self_attn`` returns
    ``(hidden_states, attn_weights, past_kv)``) by capturing ``output[0]``.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    taps = select_tap_points(model, domain=domain)
    captures: dict[str, list[torch.Tensor]] = {tap.name: [] for tap in taps}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(tap_name: str):
        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            captures[tap_name].append(tensor.detach().cpu().contiguous())
        return _hook

    # Noise hook is registered FIRST so it fires before the capture hook on
    # the same module — the capture sees the noised output. Only synthetic
    # mode registers a hook; hardware mode relies on the hardware itself.
    if noise is not None and noise.mode == "synthetic":
        handles.append(register_noise_hook(model, noise, domain=domain))

    for tap in taps:
        submod = resolve_module_path(model, tap.module_path)
        handles.append(submod.register_forward_hook(_make_hook(tap.name)))

    try:
        with torch.inference_mode():
            for _ in range(runs):
                model(**batch)
    finally:
        for h in handles:
            h.remove()

    return captures


def run_capture(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    domain: str = "llm",
) -> dict[str, torch.Tensor]:
    """Single-run convenience wrapper around :func:`run_capture_repeated`."""
    repeated = run_capture_repeated(model, batch, runs=1, domain=domain)
    return {name: tensors[0] for name, tensors in repeated.items()}


def fingerprint_model(model: nn.Module) -> str:
    """Cheap, deterministic identity hash over parameter names + shapes + a
    sample of weight bytes. Catches "you loaded the wrong model" without
    paying for a full-weights hash.
    """
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        sample = p.detach().cpu().flatten()[:64].to(torch.float32).numpy().tobytes()
        h.update(sample)
    return h.hexdigest()[:16]


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, PreTrainedTokenizerBase]:
    """Load an HF causal-LM and its tokenizer onto ``device`` in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model.to(device).eval(), tokenizer


def load_golden_inputs(
    inputs_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> dict[str, torch.Tensor]:
    with inputs_path.open() as f:
        data = json.load(f)
    texts = data["texts"]
    max_length = data.get("max_length", 32)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items()}


def capture_reference(
    model_id: str,
    inputs_path: Path,
    out_dir: Path,
    device: str = "cpu",
    seed: int = 0,
    domain: str = "llm",
) -> None:
    """Load ``model_id``, run the golden inputs, write a reference artifact."""
    set_deterministic(seed=seed)
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    batch = load_golden_inputs(inputs_path, tokenizer, device)
    captured = run_capture(model, batch, domain=domain)

    manifest = ReferenceManifest(
        model_id=model_id,
        model_fingerprint=fingerprint_model(model),
        tap_points=list(captured.keys()),
        shapes={name: list(t.shape) for name, t in captured.items()},
        dtypes={name: _dtype_str(t.dtype) for name, t in captured.items()},
        captured_at=datetime.now(UTC).isoformat(),
        env=capture_env(),
        domain=domain,
    )
    write_reference(out_dir, manifest, captured)
