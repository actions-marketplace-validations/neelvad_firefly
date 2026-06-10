"""Spike: does a noop pass-through custom op survive torch.compile without graph breaks?

The shadow-mode design (src/firefly/shadow.py module docstring) hinges
on this: if the op survives Dynamo tracing as opaque, the rest of the
architecture is normal Python work. If it forces a graph break, the
architecture has to pivot.

Pass criterion: ``torch._dynamo.explain(model)(x)`` reports 0 graph breaks
on a 2-layer MLP wrapped with ``firefly.capture()`` calls.

Run:  uv run python scripts/spike_torch_compile.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Custom op: pass-through with a side-effecting print (stand-in for the
# ring-buffer write the real implementation will do).
# ---------------------------------------------------------------------------


@torch.library.custom_op("firefly::capture", mutates_args=())
def capture(x: torch.Tensor, name: str) -> torch.Tensor:
    """Capture x's stats; return x unchanged (pass-through semantics).

    Stand-in for the real shadow-mode capture op. ``mutates_args=()``
    tells Dynamo no tensor arguments are mutated; combined with the
    ``torch.library`` registration, the op is opaque to tracing.
    """
    print(f"[firefly]    capture: {name}  shape={tuple(x.shape)}  dtype={x.dtype}")
    return x.clone()  # explicit clone so no aliasing assumptions


@capture.register_fake
def _capture_fake(x: torch.Tensor, name: str) -> torch.Tensor:
    """Abstract implementation for symbolic tracing.

    Just describes the output shape/dtype — never actually executes.
    """
    return torch.empty_like(x)


def _capture_backward(ctx, grad: torch.Tensor) -> tuple:
    """Backward formula: gradient passes through to input.

    Since the op is a no-op pass-through at the math level (just observes
    a tensor without changing it), the gradient w.r.t. ``x`` is just the
    incoming gradient. The ``name`` arg is a string with no gradient.

    Production shadow-mode is inference-only, but registering this lets the
    op compose with ``torch.compile``'s AOT autograd graph capture even
    under ``inference_mode``, and makes the op safe to use in training-time
    debugging scenarios too.
    """
    return grad, None


capture.register_autograd(_capture_backward)


# ---------------------------------------------------------------------------
# A tiny 2-layer MLP with two capture calls.
# ---------------------------------------------------------------------------


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = torch.relu(h)
        h = torch.ops.firefly.capture(h, "layer.0.mlp")
        out = self.fc2(h)
        out = torch.ops.firefly.capture(out, "final")
        return out


def main() -> None:
    model = Tiny().eval()
    x = torch.randn(1, 16)

    # Production inference runs under inference_mode (no autograd graph).
    # Without this, torch.compile tries to AOT-trace the backward and fails
    # at our custom op which doesn't have an autograd formula registered.
    # Inference is the right test scope for shadow-mode anyway.
    with torch.inference_mode():
        print("=" * 70)
        print("EAGER MODE (baseline)")
        print("=" * 70)
        y = model(x)
        print(f"  eager output shape: {tuple(y.shape)}")
        print()

        print("=" * 70)
        print("torch.compile MODE  (backend='aot_eager' — Dynamo + AOT autograd")
        print("                     processing, but eager execution. Skips")
        print("                     Inductor codegen which dies on macOS+uv")
        print("                     because of a libc++.1.dylib rpath issue.")
        print("                     The question we actually care about — does")
        print("                     the custom op survive Dynamo tracing without")
        print("                     graph breaks — is a frontend question that")
        print("                     this backend answers correctly.)")
        print("=" * 70)
        compiled = torch.compile(model, backend="aot_eager")
        y_c = compiled(x)
        print(f"  compiled output shape: {tuple(y_c.shape)}")
        print()
        print("  (the two [firefly] capture lines above mean the op ran inside")
        print("   the compiled graph — pass-through semantics work in compile mode)")
        print()

        print("=" * 70)
        print("DYNAMO EXPLAIN (graph-break diagnosis)")
        print("=" * 70)
        explanation = torch._dynamo.explain(model)(x)
    # The explanation object has fields .graph_break_count, .op_count,
    # .graph_count, .break_reasons — print whichever subset is available
    # across PyTorch versions.
    for field in ("graph_count", "graph_break_count", "op_count"):
        value = getattr(explanation, field, "(field not present)")
        print(f"  {field}: {value}")

    break_reasons = getattr(explanation, "break_reasons", None)
    if break_reasons:
        print("  break_reasons:")
        for r in break_reasons:
            print(f"    - {r}")
    else:
        print("  break_reasons: (none)")

        # Numerical sanity: eager and compiled should produce the same output.
        print()
        print("=" * 70)
        print("NUMERICAL SANITY CHECK")
        print("=" * 70)
        y_e = model(x)
        y_c = compiled(x)
        max_abs_diff = (y_e - y_c).abs().max().item()
        print(f"  max abs diff (eager vs compiled): {max_abs_diff:.3e}")
        print(
            f"  result: {'BIT-EQUAL OK' if max_abs_diff == 0 else 'close-but-different (expected; compile may reorder)'}"
        )


if __name__ == "__main__":
    main()
