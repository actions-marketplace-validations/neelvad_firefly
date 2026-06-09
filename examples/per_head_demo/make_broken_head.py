"""Produce a candidate from SmolLM-135M with ONE attention head perturbed.

Adds Gaussian noise to the query-projection (``q_proj``) rows belonging to a
single attention head in one decoder layer. Because head H's query rows feed
only head H's attention scores → only head H's context vector → only head H's
slice of the attention-output-projection input, the divergence is localized to
exactly that head in the ``layer.{i}.attn_heads`` tap.

This holds even under grouped-query attention (GQA, which SmolLM-135M uses):
perturbing a *query* head is per-head clean, whereas perturbing a key/value
head would smear across its whole query group. That's the demo's point —
Firefly's per-head attribution recovers the exact head, not just the layer.

Used by run_demo.sh to demonstrate per-head divergence attribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REFERENCE_MODEL = "HuggingFaceTB/SmolLM-135M"
TARGET_LAYER = 7
TARGET_HEAD = 4
NOISE_SCALE = 1e-2
NOISE_SEED = 42


def main(out_dir: Path) -> None:
    print(f"Loading {REFERENCE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(REFERENCE_MODEL, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(REFERENCE_MODEL)

    config = model.config
    n_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // n_heads)
    if not (0 <= TARGET_HEAD < n_heads):
        raise SystemExit(f"TARGET_HEAD={TARGET_HEAD} out of range for {n_heads} heads")

    # q_proj weight is (n_heads * head_dim, hidden). Head H owns output rows
    # [H*head_dim : (H+1)*head_dim].
    q_proj = model.model.layers[TARGET_LAYER].self_attn.q_proj
    lo, hi = TARGET_HEAD * head_dim, (TARGET_HEAD + 1) * head_dim
    print(
        f"Perturbing model.layers.{TARGET_LAYER}.self_attn.q_proj.weight "
        f"rows [{lo}:{hi}] (head {TARGET_HEAD} of {n_heads}, head_dim={head_dim}) "
        f"with N(0, {NOISE_SCALE})"
    )
    torch.manual_seed(NOISE_SEED)
    with torch.no_grad():
        rows = q_proj.weight[lo:hi]
        rows.add_(NOISE_SCALE * torch.randn_like(rows))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving broken model to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(
        f"Done. Expected first divergence around layer.{TARGET_LAYER}, "
        f"with per-head attribution pointing at head {TARGET_HEAD} of "
        f"layer.{TARGET_LAYER}.attn_heads."
    )


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./broken-head-smollm")
    main(out)
