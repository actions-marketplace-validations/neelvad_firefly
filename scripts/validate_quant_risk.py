"""Validate quant-risk predictions against real torchao W8A8 quantization.

Closes Finding 7's caveat: `firefly quant-risk` *simulates* int8 quantization
and predicts which tensors break. This confronts that prediction with reality —
real torchao kernels (int8 dynamic per-token activation + int8 per-channel
weight, W8A8) — and measures where it actually diverges.

The honest, apples-to-apples test lives in `firefly.quant_validate` and is what
`firefly quant-risk --validate` runs: capture each Linear's fp input (the tensor
torchao quantizes), push it through both the fp and real-torchao int8 Linear,
and correlate quant-risk's per-input prediction against the local output error.

This script wraps that, and ALSO prints the *naive* comparison for contrast —
per-tensor simulation of *residual-stream* taps vs the model's *accumulated*
divergence. That one is an apples-to-oranges confound (local predictor vs global
accumulated error, on tensors torchao never quantizes that way) and scores
negative. Seeing both side by side is the finding: granularity is the bug, and
the sign flips from ~-0.48 to ~+0.71 once you measure the right tensor locally.

    uv run python scripts/validate_quant_risk.py            # SmolLM on CPU
    uv run python scripts/validate_quant_risk.py --device cuda --model ...
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from firefly.capture import load_golden_inputs, load_model_and_tokenizer, run_capture
from firefly.determinism import set_deterministic
from firefly.quant_risk import analyze_quant_risk
from firefly.quant_validate import (
    quantize_w8a8,
    rel_l1,
    spearman,
    validate_against_torchao,
)
from firefly.report import render_torchao_validation


def _naive_contrast(model_id: str, device: str, bits: int) -> None:
    """The confounded comparison, kept to show WHY granularity matters."""
    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(model_id, device=device, dtype=torch.float32)
    inputs_path = Path(tempfile.mkdtemp()) / "golden.json"
    inputs_path.write_text(
        json.dumps({"texts": ["the quick brown fox jumps over the lazy dog"], "max_length": 16})
    )
    batch = load_golden_inputs(inputs_path, tok, device)
    fp_caps = run_capture(fp_model, batch)
    taps = list(fp_caps.keys())
    risks = {r.tap_name: r for r in analyze_quant_risk(fp_caps, taps, bits=bits)}

    set_deterministic()
    q_model, _ = load_model_and_tokenizer(model_id, device=device, dtype=torch.float32)
    quantize_w8a8(q_model)
    q_caps = run_capture(q_model, batch)

    pred = [risks[t].per_tensor_rel_err for t in taps]
    act = [rel_l1(fp_caps[t], q_caps[t]) for t in taps]
    print("\n=== NAIVE CONTRAST: per-tensor sim of residual taps vs ACCUMULATED divergence ===")
    print(f"residual-stream taps: {len(taps)}   (confound: local predictor vs global accumulated error)")
    print(f"Spearman( predicted per-tensor int{bits} err , accumulated divergence ): {spearman(pred, act):+.3f}")
    print("  ^ apples-to-oranges: these taps aren't what torchao quantizes, and the error has accumulated.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--no-contrast", action="store_true", help="skip the naive confounded comparison")
    args = ap.parse_args()

    result = validate_against_torchao(args.model, device=args.device, bits=args.bits)
    print(render_torchao_validation(result))

    if not args.no_contrast:
        _naive_contrast(args.model, args.device, args.bits)


if __name__ == "__main__":
    main()
