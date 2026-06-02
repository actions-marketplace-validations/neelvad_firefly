"""Tests for the activation-capture pipeline.

Unit tests use a hand-rolled fake model that mimics the llama-family module
layout — fast, offline, deterministic. The slow integration test exercises
the full pipeline against real SmolLM-135M weights from HF; run with
``pytest -m slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from firefly.capture import (
    dtype_to_name,
    fingerprint_model,
    parse_dtype,
    run_capture,
    run_capture_repeated,
)


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
    """Llama-shaped fake: model.layers[...] + model.norm + lm_head."""

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


def test_run_capture_captures_all_default_tap_points() -> None:
    torch.manual_seed(0)
    model = _FakeLM(dim=8, n_layers=2).eval()
    batch = {"input_ids": torch.randint(0, 100, (2, 4))}

    captured = run_capture(model, batch)

    expected_keys = {
        "layer.0.self_attn",
        "layer.0.mlp",
        "layer.0",
        "layer.1.self_attn",
        "layer.1.mlp",
        "layer.1",
        "final_norm",
    }
    assert set(captured.keys()) == expected_keys

    for tensor in captured.values():
        assert tensor.shape == (2, 4, 8)
        assert tensor.device.type == "cpu"


def test_run_capture_is_deterministic_on_cpu() -> None:
    """Two identical runs with the same seed produce bit-equal captures."""
    def _one_run() -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model = _FakeLM(dim=8, n_layers=2).eval()
        batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        return run_capture(model, batch)

    a = _one_run()
    b = _one_run()

    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert torch.equal(a[k], b[k]), f"Capture nondeterministic at {k}"


def test_run_capture_removes_hooks() -> None:
    """Hooks must be cleaned up even though we registered many."""
    torch.manual_seed(0)
    model = _FakeLM(dim=8, n_layers=2).eval()
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    run_capture(model, batch)

    leftover = sum(len(m._forward_hooks) for m in model.modules())
    assert leftover == 0


def test_fingerprint_is_stable_across_runs() -> None:
    torch.manual_seed(0)
    fp_a = fingerprint_model(_FakeLM(dim=8, n_layers=2))
    torch.manual_seed(0)
    fp_b = fingerprint_model(_FakeLM(dim=8, n_layers=2))
    assert fp_a == fp_b


def test_run_capture_repeated_returns_one_tensor_per_run() -> None:
    torch.manual_seed(0)
    model = _FakeLM(dim=8, n_layers=2).eval()
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    captures = run_capture_repeated(model, batch, runs=3)

    for name, tensors in captures.items():
        assert len(tensors) == 3, f"tap {name} has {len(tensors)} captures, expected 3"


def test_run_capture_repeated_is_deterministic_on_cpu() -> None:
    """All N runs must produce bit-equal tensors at every tap."""
    torch.manual_seed(0)
    model = _FakeLM(dim=8, n_layers=2).eval()
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    captures = run_capture_repeated(model, batch, runs=4)

    for tensors in captures.values():
        first = tensors[0]
        for other in tensors[1:]:
            assert torch.equal(first, other)


def test_run_capture_repeated_registers_hooks_once() -> None:
    """After the call, no leftover hooks remain — regardless of run count."""
    torch.manual_seed(0)
    model = _FakeLM(dim=8, n_layers=2).eval()
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    run_capture_repeated(model, batch, runs=5)

    leftover = sum(len(m._forward_hooks) for m in model.modules())
    assert leftover == 0


def test_run_capture_repeated_rejects_zero_runs() -> None:
    torch.manual_seed(0)
    model = _FakeLM(dim=8, n_layers=2).eval()
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    with pytest.raises(ValueError, match="runs must be >= 1"):
        run_capture_repeated(model, batch, runs=0)


def test_parse_dtype_accepts_short_and_long_names() -> None:
    assert parse_dtype("fp32") is torch.float32
    assert parse_dtype("float32") is torch.float32
    assert parse_dtype("bf16") is torch.bfloat16
    assert parse_dtype("bfloat16") is torch.bfloat16
    assert parse_dtype("fp16") is torch.float16


def test_parse_dtype_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        parse_dtype("int8")


def test_dtype_round_trip() -> None:
    for name in ("fp32", "bf16", "fp16"):
        dt = parse_dtype(name)
        canonical = dtype_to_name(dt)
        assert parse_dtype(canonical) is dt


def test_fingerprint_differs_for_different_weights() -> None:
    torch.manual_seed(0)
    fp_a = fingerprint_model(_FakeLM(dim=8, n_layers=2))
    torch.manual_seed(1)
    fp_b = fingerprint_model(_FakeLM(dim=8, n_layers=2))
    assert fp_a != fp_b


@pytest.mark.slow
def test_capture_reference_smollm_end_to_end(tmp_path: Path) -> None:
    """Real-model integration test: download SmolLM-135M, capture, verify artifact."""
    from firefly.capture import capture_reference
    from firefly.reference import read_reference

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    out_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, out_dir)

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "weights.safetensors").exists()

    manifest, tensors = read_reference(out_dir)
    assert manifest.model_id == "HuggingFaceTB/SmolLM-135M"
    assert manifest.model_fingerprint
    assert len(tensors) > 0
    assert "final_norm" in tensors


@pytest.mark.slow
def test_capture_via_cli(tmp_path: Path) -> None:
    """End-to-end through the typer CLI wiring."""
    from typer.testing import CliRunner

    from firefly.cli import app
    from firefly.reference import read_reference

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    out_dir = tmp_path / "reference"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "capture",
            "--model", "HuggingFaceTB/SmolLM-135M",
            "--inputs", str(inputs_path),
            "--out", str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Wrote reference artifact" in result.stdout
    manifest, _ = read_reference(out_dir)
    assert manifest.model_id == "HuggingFaceTB/SmolLM-135M"
