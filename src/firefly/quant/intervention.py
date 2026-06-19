"""The intervention seam — the stable interface quantization techniques plug into.

Firefly does not own the quantization algorithms; it owns the *interface* they
implement and the diagnosis/verification around them. An intervention is one
opaque verb — ``apply(model, policy, calib) -> model'`` — so the interface stays
closed even as the list of techniques grows. The mechanism (per-channel scaling,
Hessian rounding, …) lives *inside* ``apply``; it is never an interface method,
which is exactly what stops the technique zoo from leaking into the contract.

Three things, deliberately, are *not* on an intervention:

* **Precision (mixed precision / granularity / bits)** is plain data —
  :class:`PrecisionPolicy` — that the agent edits, not an ``apply`` plugin.
* **Cost** is measured after the fact by :mod:`firefly.quant.cost`, not declared.
* **Verification** (run the eval, place it on the frontier) is the substrate's
  job. An intervention only *proposes* the change; the loop *proves* it.

Interventions compose in a short, fixed pipeline ordered by :class:`Stage`
(pre-transforms rewrite weights, then a single quantizer runs). Today the only
shipped intervention is :class:`RTNQuantizer` (round-to-nearest, the torchao
default); SmoothQuant / GPTQ / AWQ become drop-in :class:`Stage.PRE_TRANSFORM` /
:class:`Stage.QUANTIZER` adapters against this same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

from torch import nn

from firefly.quant.torchao import quantize_model

#: Failure-mode signatures an intervention can declare it ``treats`` — the
#: vocabulary :mod:`firefly.quant.diagnose` matches a diagnosis against. Strings,
#: not an enum, so a new adapter can introduce a new signature without editing
#: this module.
#:
#: Only signatures with a REAL detector on the activation-capture substrate live
#: here — we don't ship labels for detectors that don't exist:
#:   * ACTIVATION_OUTLIERS — quant-risk channel_concentration (→ SmoothQuant).
#:   * SINGLE_UNIT_DOMINANCE — a sensitivity sweep, one unit ≫ median (→ keep fp).
#: Deliberately absent: AWQ's salient-weight-channels (needs a new |W|·|X|
#: weight-side sensor) and GPTQ's diffuse-weight-loss (justified in weight-space
#: Hessian, not measurable from a forward pass). Add the signature only with the
#: detector — see firefly.quant.diagnose.
ACTIVATION_OUTLIERS = "activation_outliers"
SINGLE_UNIT_DOMINANCE = "single_unit_dominance"


class Stage(IntEnum):
    """Pipeline position. Ordered, so ``sorted`` by stage gives execution order:
    weight-rewriting pre-transforms first, the quantizer last."""

    PRE_TRANSFORM = 1   # output-equivalent weight rewrite (SmoothQuant, AWQ-scaling)
    QUANTIZER = 2       # the quantization step itself (RTN default, GPTQ)


@dataclass
class PrecisionPolicy:
    """Which Linears to quantize and at what scheme — the data the agent edits.

    This *is* the mixed-precision lever: FQNs not in ``quantize`` are kept in
    full precision. It carries the shared targeting/scheme context every stage
    needs, so :meth:`Intervention.apply` takes one uniform argument.
    """

    scheme: str = "w8a8"
    group_size: int = 32
    quantize: set[str] = field(default_factory=set)
    """FQNs to quantize; everything else stays fp (kept high-precision)."""

    def filter_fn(self) -> Callable[[nn.Module, str], bool]:
        """torchao ``filter_fn``: quantize exactly the FQNs in ``quantize``."""
        q = self.quantize
        return lambda _mod, fqn: fqn in q


@runtime_checkable
class Intervention(Protocol):
    """One quantization operation: ``model -> model'`` under a policy.

    The mechanism is entirely inside ``apply``. ``stage`` orders it in the
    pipeline; ``treats`` lists the failure-mode signatures it addresses (for the
    agent registry); ``name`` identifies it in reports.
    """

    name: str
    stage: Stage
    treats: frozenset[str]

    def apply(self, model: nn.Module, policy: PrecisionPolicy, calib: object | None = None) -> nn.Module:
        ...


class _TorchAOIntervention(ABC):
    """Shared scaffolding for torchao-backed interventions: identity fields and
    the scope→``filter_fn`` plumbing every adapter would otherwise re-derive.
    Concrete adapters (RTN today; SmoothQuant/GPTQ later) implement ``apply``."""

    name: str = "torchao"
    stage: Stage = Stage.QUANTIZER
    treats: frozenset[str] = frozenset()

    def config(self) -> dict:
        """Reconstruction kwargs for serialization (see firefly.quant.recipe_io).
        Default: no params. Override for interventions with state to round-trip."""
        return {}

    @abstractmethod
    def apply(self, model: nn.Module, policy: PrecisionPolicy, calib: object | None = None) -> nn.Module:
        ...


class RTNQuantizer(_TorchAOIntervention):
    """Round-to-nearest quantization — torchao's default ``quantize_``. The
    baseline quantizer; carries no special failure-mode treatment."""

    name = "rtn"
    stage = Stage.QUANTIZER
    treats = frozenset()

    def apply(self, model: nn.Module, policy: PrecisionPolicy, calib: object | None = None) -> nn.Module:
        if not policy.quantize:
            return model  # nothing to quantize → keep the model fp (mixed-precision extreme)
        return quantize_model(
            model, scheme=policy.scheme, group_size=policy.group_size,
            module_filter=policy.filter_fn(),
        )


@dataclass
class Pipeline:
    """An ordered set of interventions: pre-transforms (stage order), then one
    quantizer. Mutates the model in place through the stages (torchao ``quantize_``
    is in-place), so the *caller* is responsible for passing a fresh copy."""

    pre_transforms: list[Intervention] = field(default_factory=list)
    quantizer: Intervention = field(default_factory=RTNQuantizer)

    def __post_init__(self) -> None:
        for it in self.pre_transforms:
            if it.stage != Stage.PRE_TRANSFORM:
                raise ValueError(f"{it.name!r} is stage {it.stage.name}, not a PRE_TRANSFORM")
        if self.quantizer.stage != Stage.QUANTIZER:
            raise ValueError(f"quantizer {self.quantizer.name!r} is stage {self.quantizer.stage.name}")

    def run(self, model: nn.Module, policy: PrecisionPolicy, calib: object | None = None) -> nn.Module:
        m = model
        for it in sorted(self.pre_transforms, key=lambda i: i.stage):
            m = it.apply(m, policy, calib)
        return self.quantizer.apply(m, policy, calib)
