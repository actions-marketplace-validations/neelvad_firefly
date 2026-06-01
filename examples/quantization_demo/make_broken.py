"""Produce a deliberately-broken candidate from SmolLM-135M.

Adds small Gaussian noise to the down_proj weight of one MLP block, then
saves the modified model as a HF-loadable directory. Used by run_demo.sh
to demonstrate Firefly's per-layer divergence attribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REFERENCE_MODEL = "HuggingFaceTB/SmolLM-135M"
TARGET_LAYER = 7
NOISE_SCALE = 1e-3
NOISE_SEED = 42


def main(out_dir: Path) -> None:
    print(f"Loading {REFERENCE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(REFERENCE_MODEL, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(REFERENCE_MODEL)

    target = model.model.layers[TARGET_LAYER].mlp.down_proj
    print(
        f"Perturbing model.layers.{TARGET_LAYER}.mlp.down_proj.weight "
        f"with N(0, {NOISE_SCALE})"
    )
    torch.manual_seed(NOISE_SEED)
    with torch.no_grad():
        target.weight.add_(NOISE_SCALE * torch.randn_like(target.weight))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving broken model to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Done. Expected first divergence at: layer.{TARGET_LAYER}.mlp")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./broken-smollm")
    main(out)
