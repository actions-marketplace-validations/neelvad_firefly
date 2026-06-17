"""Spike: can a TorchDispatchMode capture op-level outputs scoped to one module?

The op-level drill-down (layer -> head -> ... -> ATen op) hinges on this. The
plan: a `TorchDispatchMode` intercepts every ATen op, but it has no idea which
*module* an op belongs to — so we gate recording with module forward hooks
(pre-hook turns recording on, post-hook off). Only ops executed inside the
target module's forward get recorded, in execution order.

Pass criteria:
  1. Recording scoped to a module captures its ops (e.g. an MLP block shows
     addmm / gelu / add), and far fewer than the whole model.
  2. The op-name + shape + value sequence is DETERMINISTIC across two runs of
     the same model (so a later op-by-op diff is meaningful).
  3. Two structurally-identical blocks record the SAME op sequence but DIFFERENT
     values — confirming scoping actually isolates the target module.

Run:  uv run python scripts/spike_torch_dispatch.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from firefly.determinism import set_deterministic


def _stats(out: object) -> dict:
    """Summary of an op's output tensor (None for non-float/non-tensor)."""
    t = out if isinstance(out, torch.Tensor) else (
        out[0] if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor)
        else None
    )
    if t is None or not t.is_floating_point() or t.numel() == 0:
        return {"shape": None, "abs_max": None}
    return {"shape": tuple(t.shape), "abs_max": round(float(t.detach().abs().max()), 6)}


class OpRecorder(TorchDispatchMode):
    """Records ``(op_name, stats)`` for every ATen op while ``active`` is set.

    ``active`` is toggled by module hooks so only a target subtree is recorded.
    ``_in_stats`` guards the re-entrant dispatch that computing stats triggers
    (mean/abs/max are themselves ops) so we don't record our own bookkeeping.
    """

    def __init__(self) -> None:
        super().__init__()
        self.active = False
        self._in_stats = False
        self.records: list[tuple[str, dict]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self.active and not self._in_stats:
            self._in_stats = True
            try:
                self.records.append((func._schema.name, _stats(out)))
            finally:
                self._in_stats = False
        return out


def capture_ops_in(model: nn.Module, target: nn.Module | None, x: torch.Tensor) -> list[tuple[str, dict]]:
    """Run ``model(x)``; record ATen ops executed inside ``target``'s forward
    (or the whole model if ``target`` is None)."""
    rec = OpRecorder()
    handles = []
    if target is None:
        rec.active = True
    else:
        handles.append(target.register_forward_pre_hook(lambda *_: setattr(rec, "active", True)))
        handles.append(target.register_forward_hook(lambda *_: setattr(rec, "active", False)))
    try:
        with rec, torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return rec.records


class _Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))  # residual MLP block


class _Net(nn.Module):
    def __init__(self, d: int = 16, n: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(d) for _ in range(n))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def main() -> None:
    set_deterministic()
    model = _Net().eval()
    x = torch.randn(2, 8, 16)

    block1 = model.blocks[1]
    block0 = model.blocks[0]

    rec_block1_a = capture_ops_in(model, block1, x)
    rec_block1_b = capture_ops_in(model, block1, x)  # determinism: same again
    rec_block0 = capture_ops_in(model, block0, x)
    rec_all = capture_ops_in(model, None, x)

    ops_block1 = [name for name, _ in rec_block1_a]
    print(f"ops recorded in blocks[1]: {ops_block1}")
    print(f"  count: block1={len(rec_block1_a)}  block0={len(rec_block0)}  whole_model={len(rec_all)}")

    # 1. captured the block's ops, far fewer than the whole model
    pass_scoped = (
        len(rec_block1_a) > 0
        and any("addmm" in n or "mm" in n or "linear" in n for n in ops_block1)
        and len(rec_block1_a) < len(rec_all)
    )

    # 2. deterministic across two runs (op name + shape + value)
    pass_deterministic = rec_block1_a == rec_block1_b

    # 3. same op structure as block0 but different values (scoping isolates it)
    names0 = [n for n, _ in rec_block0]
    names1 = [n for n, _ in rec_block1_a]
    vals0 = [s["abs_max"] for _, s in rec_block0]
    vals1 = [s["abs_max"] for _, s in rec_block1_a]
    pass_isolation = names0 == names1 and vals0 != vals1

    print()
    print(f"  block0 op names == block1 op names: {names0 == names1}")
    print(f"  block0 values   != block1 values  : {vals0 != vals1}")
    print()
    print(f"PASS scoped capture:   {pass_scoped}")
    print(f"PASS deterministic:    {pass_deterministic}")
    print(f"PASS module isolation: {pass_isolation}")
    overall = pass_scoped and pass_deterministic and pass_isolation
    print("=" * 50)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
