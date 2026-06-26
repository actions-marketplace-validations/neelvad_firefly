"""Smoke test for the torchao quant *measurement* utilities.

`firefly.quant.torchao` quantizes a model with real torchao kernels and
measures the per-Linear local divergence. The breadth sweep uses it as a
measurement tool. (The earlier "quant-risk predicts where int8 hurts" claim was
falsified by the breadth sweep — these utilities no longer assert a verdict, so
this no longer pins a PASS threshold.)

Marked ``slow``: it downloads SmolLM-135M and needs the optional ``torchao``
extra. Run with ``pytest -m slow``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torchao", reason="quant-risk validation needs the torchao extra")

from firefly.quant.torchao import validate_against_torchao


def test_quant_config_selects_scheme() -> None:
    """Scheme selection is CPU-fast (builds a config object, no quantization).
    Guards the W8A8 / int4-weight-only fork the breadth sweep relies on."""
    from firefly.quant.torchao import QUANT_SCHEMES, _quant_config

    assert set(QUANT_SCHEMES) == {"w8a8", "int8wo", "int4wo"}
    assert _quant_config("w8a8") is not None
    assert _quant_config("int8wo") is not None
    assert _quant_config("int4wo", group_size=32) is not None
    with pytest.raises(ValueError, match="unknown quant scheme"):
        _quant_config("bogus")


@pytest.mark.slow
def test_validate_against_torchao_measures_local_divergence() -> None:
    result = validate_against_torchao("HuggingFaceTB/SmolLM-135M", device="cpu")

    # A representative sample of Linears, each with a measured local error.
    assert len(result.records) > 100
    assert all(r.actual_local_err >= 0.0 for r in result.records)

    # The rank-correlation measurements are well-formed (in [-1, 1]); we do NOT
    # assert they clear any threshold — that was the falsified claim.
    assert -1.0 <= result.spearman_concentration <= 1.0
    assert -1.0 <= result.spearman_per_tensor <= 1.0
