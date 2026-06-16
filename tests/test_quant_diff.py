"""Quantization-diff: torchao quant as a candidate in the capture pipeline.

The HF runner's ``quantize=<scheme>`` option applies real torchao quant after
load, so quantization becomes a candidate in the standard capture/compare/
attribute path. The fingerprint is taken *pre-quant*, so a quant candidate
still matches its fp baseline — the divergence is the quantization, not a model
swap.

Marked ``slow``: downloads SmolLM-135M and needs the optional ``torchao``
extra. W8A8 runs on CPU (int4wo needs CUDA), so this covers P0 end-to-end
without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torchao", reason="quant-diff needs the torchao extra")

import torch

from firefly.runners.hf import HFRunner

_MODEL = "HuggingFaceTB/SmolLM-135M"


def _golden(tmp_path: Path) -> Path:
    p = tmp_path / "golden.json"
    p.write_text(json.dumps({"texts": ["the quick brown fox jumps"], "max_length": 16}))
    return p


@pytest.mark.slow
def test_quant_candidate_diffs_against_fp_baseline(tmp_path: Path) -> None:
    inputs = _golden(tmp_path)
    runner = HFRunner()

    fp = runner.capture(_MODEL, inputs, device="cpu", dtype="fp32")
    q = runner.capture(_MODEL, inputs, device="cpu", dtype="fp32", options={"quantize": "w8a8"})

    # Same base model: quantization is a transform, not a different model, so
    # the pre-quant fingerprint matches the fp baseline (the existing check
    # fingerprint gate passes — no --allow-fingerprint-mismatch needed).
    assert q.fingerprint == fp.fingerprint

    # Same tap structure → drops straight into compare/attribute.
    assert set(q.tensors) == set(fp.tensors)
    assert "final_norm" in q.tensors

    # Quantization actually moved the activations: at least one tap diverges,
    # and the divergence has propagated to the final norm (the headline number).
    def rel(name: str) -> float:
        a, b = fp.tensors[name].float(), q.tensors[name].float()
        return (a - b).abs().mean().item() / (a.abs().mean().item() or 1.0)

    assert max(rel(n) for n in fp.tensors) > 0.01
    assert rel("final_norm") > 0.0


@pytest.mark.slow
def test_quant_diff_cli_end_to_end(tmp_path: Path) -> None:
    """`firefly capture` then `firefly quant-diff` through the CLI: the
    magnitude-ranked report renders and the --rel-threshold gate trips."""
    from typer.testing import CliRunner

    from firefly.cli import app

    inputs = _golden(tmp_path)
    baseline = tmp_path / "baseline"
    runner = CliRunner()

    r1 = runner.invoke(app, ["capture", "--model", _MODEL, "--inputs", str(inputs), "--out", str(baseline)])
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(
        app,
        ["quant-diff", "-r", str(baseline), "-i", str(inputs),
         "--scheme", "w8a8", "--rel-threshold", "0.05"],
    )
    # W8A8 perturbs many taps past 5% → gate trips (exit 1).
    assert r2.exit_code == 1, r2.output
    assert "quantization diff" in r2.output.lower()
    assert "worst layer" in r2.output
    assert "accumulated at output" in r2.output


@pytest.mark.slow
def test_quant_candidate_is_deterministic(tmp_path: Path) -> None:
    """Two quant captures of the same model agree bit-for-bit, so any diff a
    user sees is the quantization, not run-to-run noise."""
    inputs = _golden(tmp_path)
    runner = HFRunner()
    a = runner.capture(_MODEL, inputs, device="cpu", dtype="fp32", options={"quantize": "w8a8"})
    b = runner.capture(_MODEL, inputs, device="cpu", dtype="fp32", options={"quantize": "w8a8"})
    for name in a.tensors:
        assert torch.equal(a.tensors[name], b.tensors[name])
