"""Slow regression guard for the quant-risk → real-torchao claim.

`firefly.quant_validate` confronts quant-risk's *simulated* int8 prediction
with real torchao W8A8 kernels. The product claim is that quant-risk's
per-input ranking predicts where real int8 quantization hurts locally; this
test pins that claim so it can't silently rot.

Marked ``slow``: it downloads SmolLM-135M and needs the optional ``torchao``
extra. Run with ``pytest -m slow``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torchao", reason="quant-risk validation needs the torchao extra")

from firefly.quant_validate import PASS_THRESHOLD, validate_against_torchao


def test_quant_config_selects_scheme() -> None:
    """Scheme selection is CPU-fast (builds a config object, no quantization).
    Guards the W8A8 / int4-weight-only fork the breadth sweep relies on."""
    from firefly.quant_validate import QUANT_SCHEMES, _quant_config

    assert set(QUANT_SCHEMES) == {"w8a8", "int4wo"}
    assert _quant_config("w8a8") is not None
    assert _quant_config("int4wo", group_size=32) is not None
    with pytest.raises(ValueError, match="unknown quant scheme"):
        _quant_config("bogus")


@pytest.mark.slow
def test_quant_risk_ranking_validates_against_real_torchao() -> None:
    result = validate_against_torchao("HuggingFaceTB/SmolLM-135M", device="cpu")

    # A representative sample of Linears, all with a measured local error.
    assert len(result.records) > 100
    assert all(r.actual_local_err >= 0.0 for r in result.records)

    # The load-bearing claim: per-input prediction ranks real torchao pain.
    assert result.spearman_concentration > PASS_THRESHOLD
    assert result.passed

    # The mechanism: the worst real-divergence layer is an outlier-feature
    # layer (high channel concentration), not a flat one.
    worst = max(result.records, key=lambda r: r.actual_local_err)
    assert worst.channel_concentration > 5.0
