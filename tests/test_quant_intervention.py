"""Tests for the intervention seam. The composition logic is exercised with
fake interventions (no torchao); the real RTN path is covered by the slow
recipe/bar tests in test_quant_sensitivity.py, which now route through it."""

from __future__ import annotations

import pytest
import torch.nn as nn

from firefly.quant.intervention import (
    Intervention,
    Pipeline,
    PrecisionPolicy,
    RTNQuantizer,
    Stage,
)


class _Fake:
    """A recording intervention that returns the model untouched."""

    def __init__(self, name: str, stage: Stage, log: list[str]) -> None:
        self.name = name
        self.stage = stage
        self.treats = frozenset()
        self._log = log

    def apply(self, model, policy, calib=None):
        self._log.append(self.name)
        return model


# --- PrecisionPolicy -----------------------------------------------------------

def test_precision_policy_filter_fn() -> None:
    f = PrecisionPolicy(quantize={"a", "b"}).filter_fn()
    assert f(None, "a") and f(None, "b")
    assert not f(None, "c")  # kept fp (mixed precision)


# --- protocol conformance ------------------------------------------------------

def test_rtn_is_an_intervention() -> None:
    rtn = RTNQuantizer()
    assert isinstance(rtn, Intervention)  # structural Protocol check
    assert rtn.stage == Stage.QUANTIZER and rtn.name == "rtn"


def test_rtn_empty_policy_is_noop() -> None:
    # Keeping everything fp must not touch torchao at all (would need a GPU/extra).
    model = nn.Linear(4, 4)
    out = RTNQuantizer().apply(model, PrecisionPolicy(quantize=set()))
    assert out is model


# --- pipeline composition ------------------------------------------------------

def test_pipeline_runs_pre_transforms_then_quantizer() -> None:
    log: list[str] = []
    pipe = Pipeline(
        pre_transforms=[_Fake("preA", Stage.PRE_TRANSFORM, log), _Fake("preB", Stage.PRE_TRANSFORM, log)],
        quantizer=_Fake("quant", Stage.QUANTIZER, log),
    )
    pipe.run(nn.Linear(4, 4), PrecisionPolicy(quantize={"x"}))
    assert log == ["preA", "preB", "quant"]  # pre-transforms in order, quantizer last


def test_pipeline_threads_the_model_through() -> None:
    seen: list[int] = []

    class _Track(_Fake):
        def apply(self, model, policy, calib=None):
            seen.append(id(model))
            return model

    m = nn.Linear(4, 4)
    Pipeline(
        pre_transforms=[_Track("p", Stage.PRE_TRANSFORM, [])],
        quantizer=_Track("q", Stage.QUANTIZER, []),
    ).run(m, PrecisionPolicy(quantize={"x"}))
    assert seen == [id(m), id(m)]  # same model object chained through every stage


def test_pipeline_default_quantizer_is_rtn() -> None:
    assert Pipeline().quantizer.name == "rtn"


def test_pipeline_rejects_misstaged_interventions() -> None:
    log: list[str] = []
    # a quantizer-staged op in the pre-transform slot
    with pytest.raises(ValueError, match="not a PRE_TRANSFORM"):
        Pipeline(pre_transforms=[_Fake("q", Stage.QUANTIZER, log)])
    # a pre-transform in the quantizer slot
    with pytest.raises(ValueError, match="is stage PRE_TRANSFORM"):
        Pipeline(quantizer=_Fake("p", Stage.PRE_TRANSFORM, log))
