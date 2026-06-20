"""AWQ intervention — protocol/registry/serialization (CPU). The int4 AWQ *run*
is GPU-only; it's validated on Modal (scripts/validate_awq_gpu.py)."""

from __future__ import annotations

import pytest
import torch.nn as nn

from firefly.quant.awq import AWQQuantizer
from firefly.quant.intervention import Intervention, PrecisionPolicy, Stage
from firefly.quant.recipe_io import _REGISTRY, deserialize_intervention, serialize_intervention


def test_awq_is_a_quantizer_intervention() -> None:
    a = AWQQuantizer()
    assert isinstance(a, Intervention)
    assert a.stage == Stage.QUANTIZER and a.name == "awq"


def test_awq_registered_and_round_trips() -> None:
    assert _REGISTRY["awq"] is AWQQuantizer
    a = AWQQuantizer(group_size=64, scale_search_space_size=10)
    b = deserialize_intervention(serialize_intervention(a))
    assert b.group_size == 64 and b.scale_search_space_size == 10


def test_awq_requires_calib() -> None:
    # the calib check fires before any torchao import → testable on CPU
    with pytest.raises(ValueError, match="calibration"):
        AWQQuantizer().apply(nn.Linear(8, 8), PrecisionPolicy(quantize={""}), calib=None)


def test_awq_empty_policy_is_noop() -> None:
    # nothing to quantize → return the model untouched, before touching torchao
    model = nn.Linear(8, 8)
    out = AWQQuantizer().apply(model, PrecisionPolicy(quantize=set()), calib="x")
    assert out is model
