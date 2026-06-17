"""Op-level drill-down (layer -> ... -> ATen op).

Fast tests cover the pure diff/alignment logic + the renderer with synthetic
op records. The slow test runs the real scoped TorchDispatchMode on SmolLM-135M
(fp32 vs bf16) inside one module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from firefly.op_drill import OpDiffResult, OpRecord, diff_op_sequences


def _rec(i: int, op: str, val: float | None) -> OpRecord:
    out = None if val is None else torch.full((4,), val)
    return OpRecord(index=i, op=op, output=out)


def test_diff_clean_when_within_tolerance() -> None:
    ref = [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::add", 2.0)]
    cand = [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::add", 2.0)]
    result = diff_op_sequences(ref, cand, tol=0.01)
    assert result.first_divergent is None
    assert not result.any_exceeded


def test_diff_flags_first_numerical_divergence() -> None:
    ref = [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::gelu", 1.0), _rec(2, "aten::add", 1.0)]
    cand = [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::gelu", 2.0), _rec(2, "aten::add", 5.0)]
    result = diff_op_sequences(ref, cand, tol=0.01)
    first = result.first_divergent
    assert first is not None
    assert first.index == 1  # gelu is the first op past tolerance
    assert first.op == "aten::gelu"
    assert not first.structural
    assert first.rel == pytest.approx(1.0)  # |1-2|/1


def test_diff_flags_structural_divergence_on_op_mismatch() -> None:
    ref = [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::gelu", 1.0)]
    cand = [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::silu", 1.0)]  # different op
    result = diff_op_sequences(ref, cand, tol=0.01)
    first = result.first_divergent
    assert first is not None
    assert first.index == 1
    assert first.structural
    assert "gelu" in first.op and "silu" in first.op


def test_diff_flags_length_mismatch_as_structural() -> None:
    ref = [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::add", 1.0)]
    cand = [_rec(0, "aten::addmm", 1.0)]  # candidate is shorter
    result = diff_op_sequences(ref, cand, tol=0.01)
    assert result.first_divergent is not None
    assert result.first_divergent.index == 1
    assert result.first_divergent.structural


def test_render_op_diff_headline() -> None:
    from firefly.report import render_op_diff

    result = OpDiffResult(
        module="model.layers.7.self_attn", n_ref_ops=2, n_cand_ops=2, tol=0.01,
        divergences=diff_op_sequences(
            [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::add", 1.0)],
            [_rec(0, "aten::addmm", 1.0), _rec(1, "aten::add", 2.0)],
            tol=0.01,
        ).divergences,
    )
    out = render_op_diff(result)
    assert "model.layers.7.self_attn" in out
    assert "First divergence" in out
    assert "aten::add" in out


@pytest.mark.slow
def test_op_diff_dtypes_smollm_scoped() -> None:
    import tempfile

    from firefly.op_drill import op_diff_dtypes

    inputs = Path(tempfile.mkdtemp()) / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox"], "max_length": 8}))

    result = op_diff_dtypes(
        "HuggingFaceTB/SmolLM-135M", inputs, "model.layers.0.mlp",
        ref_dtype="float32", cand_dtype="bfloat16", device="cpu", tol=0.0,
    )
    # The MLP block records its ATen ops (addmm for the projections, an
    # activation, a mul/add), and bf16 vs fp32 diverges somewhere within.
    assert len(result.divergences) > 0
    assert any("addmm" in d.op or "mm" in d.op for d in result.divergences)
    assert result.any_exceeded  # bf16 rounding > 0 with tol=0
    # Aligned (same architecture, same op graph) — no structural mismatch.
    assert all(not d.structural for d in result.divergences)
