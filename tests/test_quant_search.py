"""min_memory_search loop mechanics — slow, with a DETERMINISTIC stub proposer
(no LLM, no API). Validates setup → step → history → best-tracking → budget on
SmolLM w8a8 (CPU). The real LLM+GPU loop is the Modal demo."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow
def test_min_memory_search_loop_with_stub() -> None:
    pytest.importorskip("torchao", reason="needs the torchao extra")

    from firefly.quant.evaluate import AccuracyBar
    from firefly.quant.search import min_memory_search

    work = Path(tempfile.mkdtemp())
    inputs = work / "calib.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox", "a lazy dog"], "max_length": 16}))
    eval_texts = ["paris is in france", "water boils at one hundred degrees"]

    # Deterministic proposer: step 0 plain w8a8; step 1 keep two layers fp.
    seq = [
        {"quantizer": "rtn", "keep_fp_units": [], "rationale": "plain baseline"},
        {"quantizer": "rtn", "keep_fp_units": ["layer.0", "layer.1"], "rationale": "keep two fp"},
    ]

    def stub(bundle, history):
        return seq[len(history)] if len(history) < len(seq) else None

    res = min_memory_search(
        "HuggingFaceTB/SmolLM-135M", inputs, eval_texts, propose=stub, scheme="w8a8",
        bar=AccuracyBar("rel", 0.5), group_size=32, device="cpu", dtype="float32",
        max_length=16, budget=4,
    )

    # The loop ran both proposals and stopped when the stub returned None.
    assert len(res["history"]) == 2
    for h in res["history"]:
        assert "perplexity" in h and "memory_mb" in h and isinstance(h["passed_bar"], bool)
        assert h["residual_worst_taps"]  # attribution fed back
    # keeping two layers fp costs more memory than plain quant
    assert res["history"][1]["memory_mb"] > res["history"][0]["memory_mb"]
    # bundle carries the oracle context the agent reads
    assert "extra_mb_per_kept_fp_unit" in res["bundle"] and "salience_top_linears" in res["bundle"]
    # a loose bar → at least one passing recipe → best tracks the cheapest passing one
    if res["best"] is not None:
        passing = [h["memory_mb"] for h in res["history"] if h["passed_bar"]]
        assert res["best"]["memory_bytes"] / 1e6 == pytest.approx(min(passing), rel=1e-3)
