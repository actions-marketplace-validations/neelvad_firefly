"""Op-level drill-down: which ATen op inside a module first diverges.

The bottom rung of the granularity ladder (layer -> head -> ... -> op). Firefly's
module-level gate localizes a divergence to, say, ``layer.7.self_attn``; this
zooms *inside* that module to the individual ATen ops (matmul, softmax, add, ...)
and reports the first op where two executions' numbers part.

Mechanism (validated by scripts/spike_torch_dispatch.py): a ``TorchDispatchMode``
records every ATen op, gated by module forward hooks so only the target subtree
is captured, in execution order. It's an opt-in drill-down, not a default — op
interception is eager-only (bypassed under CUDA graphs), Python-per-op slow, and
produces thousands of ops model-wide, so it's scoped to one flagged module.

Op-by-op alignment assumes both runs execute the *same* op graph — true for
same-architecture precision/nondeterminism diffs (fp32 vs bf16). When the graphs
differ (e.g. a quantized kernel emits different ops) the first op-name mismatch
is reported as a *structural* divergence, which is itself the finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from firefly.capture import load_golden_inputs, load_model_and_tokenizer, parse_dtype
from firefly.determinism import set_deterministic


@dataclass
class OpRecord:
    index: int
    op: str
    output: torch.Tensor | None
    """The op's output as detached fp32 CPU (None for non-float / non-tensor)."""


@dataclass
class OpDivergence:
    index: int
    op: str
    rel: float | None
    """Relative L1 between the two runs' op outputs (None if not comparable)."""
    structural: bool
    """True if the op *names* differ at this index — the op graphs diverged."""
    exceeds: bool


@dataclass
class OpDiffResult:
    module: str
    n_ref_ops: int
    n_cand_ops: int
    tol: float
    divergences: list[OpDivergence]

    @property
    def first_divergent(self) -> OpDivergence | None:
        return next((d for d in self.divergences if d.structural or d.exceeds), None)

    @property
    def any_exceeded(self) -> bool:
        return self.first_divergent is not None


def _tensor_of(out: object) -> torch.Tensor | None:
    t = out if isinstance(out, torch.Tensor) else (
        out[0] if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor)
        else None
    )
    if t is None or not t.is_floating_point() or t.numel() == 0:
        return None
    return t.detach().float().cpu()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    denom = a.abs().mean().item() or 1.0
    return (a - b).abs().mean().item() / denom


class _ScopedOpRecorder(TorchDispatchMode):
    """Records ``OpRecord``s for ATen ops while ``active`` (toggled by module
    hooks). ``_in`` guards the re-entrant dispatch that storing the output
    triggers (``.detach().float().cpu()`` are themselves ops)."""

    def __init__(self) -> None:
        super().__init__()
        self.active = False
        self._in = False
        self.records: list[OpRecord] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self.active and not self._in:
            self._in = True
            try:
                self.records.append(
                    OpRecord(index=len(self.records), op=func._schema.name, output=_tensor_of(out))
                )
            finally:
                self._in = False
        return out


def capture_module_ops(model: nn.Module, batch: dict, module_fqn: str) -> list[OpRecord]:
    """Run ``model(**batch)``, capturing the ATen ops executed inside the
    ``module_fqn`` subtree, in execution order."""
    target = model.get_submodule(module_fqn)
    rec = _ScopedOpRecorder()
    handles = [
        target.register_forward_pre_hook(lambda *_: setattr(rec, "active", True)),
        target.register_forward_hook(lambda *_: setattr(rec, "active", False)),
    ]
    try:
        with rec, torch.no_grad():
            model(**batch)
    finally:
        for h in handles:
            h.remove()
    return rec.records


def diff_op_sequences(
    ref: list[OpRecord], cand: list[OpRecord], tol: float = 0.01
) -> OpDiffResult:
    """Align two op sequences by execution index and flag the first divergence.

    A divergence is *structural* (op names differ / a run is shorter) or
    *numerical* (aligned ops whose output relative L1 exceeds ``tol``)."""
    divergences: list[OpDivergence] = []
    for i in range(max(len(ref), len(cand))):
        r = ref[i] if i < len(ref) else None
        c = cand[i] if i < len(cand) else None
        if r is None or c is None:
            present = c or r
            divergences.append(OpDivergence(i, present.op, None, structural=True, exceeds=False))
            continue
        if r.op != c.op:
            divergences.append(
                OpDivergence(i, f"{r.op} vs {c.op}", None, structural=True, exceeds=False)
            )
            continue
        rel = None
        if r.output is not None and c.output is not None and r.output.shape == c.output.shape:
            rel = _rel(r.output, c.output)
        divergences.append(
            OpDivergence(i, c.op, rel, structural=False, exceeds=rel is not None and rel > tol)
        )
    return OpDiffResult(module="", n_ref_ops=len(ref), n_cand_ops=len(cand), tol=tol, divergences=divergences)


def op_diff_dtypes(
    model_id: str,
    inputs_path: Path,
    module_fqn: str,
    ref_dtype: str = "float32",
    cand_dtype: str = "bfloat16",
    device: str = "cpu",
    tol: float = 0.01,
) -> OpDiffResult:
    """Drill into ``module_fqn`` and diff its ATen ops between the same model
    loaded at two dtypes — the op-level view of a precision divergence."""
    set_deterministic()
    ref_model, tok = load_model_and_tokenizer(model_id, device=device, dtype=parse_dtype(ref_dtype))
    batch = load_golden_inputs(inputs_path, tok, device)
    ref_ops = capture_module_ops(ref_model, batch, module_fqn)

    set_deterministic()
    cand_model, _ = load_model_and_tokenizer(model_id, device=device, dtype=parse_dtype(cand_dtype))
    cand_ops = capture_module_ops(cand_model, batch, module_fqn)

    result = diff_op_sequences(ref_ops, cand_ops, tol=tol)
    result.module = module_fqn
    return result
