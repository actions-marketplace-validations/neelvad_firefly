"""The deterministic auto-quant harness — slow end-to-end on SmolLM (CPU w8a8).
Asserts the mechanics (diagnose → route → verify → structure + the measurement
gate), not a specific recovery magnitude (whether an intervention helps is
model-dependent — SmoothQuant helps a 7B but hurts SmolLM, which the gate
catches)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow
def test_auto_quant_smollm_w8a8() -> None:
    pytest.importorskip("torchao", reason="needs the torchao extra")

    from firefly.quant.auto import auto_quant

    work = Path(tempfile.mkdtemp())
    inputs = work / "calib.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox", "a lazy dog sleeps"], "max_length": 16}))
    eval_texts = ["paris is in france", "water boils at one hundred degrees"]

    res = auto_quant(
        "HuggingFaceTB/SmolLM-135M", inputs, eval_texts,
        scheme="w8a8", group_size=32, device="cpu", dtype="float32", max_length=16,
    )

    # diagnosis + routing: SmolLM has activation outliers → SmoothQuant routed.
    assert any(sig == "activation_outliers" for sig, _ in res["diagnosis"])
    assert res["recipe"]["pre_transforms"] == ["smoothquant"]
    assert res["recipe"]["quantizer"] == "rtn"
    # verify block is populated and the measurement gate decided.
    assert res["perplexity"]["fp"] > 0 and res["perplexity"]["plain"] > 0
    assert isinstance(res["accepted"], bool)
    assert res["chosen"] in {"routed", "plain"}
    assert res["recipe_obj"].quantizer["name"] == "rtn"
    assert len(res["attribution_worst_taps"]) >= 1
