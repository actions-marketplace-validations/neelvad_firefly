"""One-off: drill into the Qwen-2.5-7B layer-27 FLASHINFER spike per head.

Finding 5 established that Qwen + FLASHINFER spikes 20× in relative error at
layer 27 (its final transformer block), plausibly outlier-feature magnitudes
× FLASHINFER's kernel difference. This script answers the open question:
is the spike concentrated in specific attention heads, and is the worst-
diverging head also the largest-magnitude (outlier-feature) head?

Usage:
    uv run python experiments/analyze_qwen_layer27.py \
        scripts/results/qwen_fi_flash_perhead \
        scripts/results/qwen_fi_flashinfer_perhead
"""

from __future__ import annotations

import sys
from pathlib import Path

from firefly.head_attribution import split_heads
from firefly.reference import read_reference


def main(ref_a_dir: Path, ref_b_dir: Path, layers: list[int]) -> None:
    manifest, ta = read_reference(ref_a_dir)
    _, tb = read_reference(ref_b_dir)

    for layer in layers:
        name = f"layer.{layer}.attn_heads"
        if name not in manifest.head_counts:
            print(f"{name}: not in head_counts, skipping")
            continue
        n_heads = manifest.head_counts[name]
        a = split_heads(ta[name].float(), n_heads)  # (tokens, heads, head_dim)
        b = split_heads(tb[name].float(), n_heads)

        diff_max = (a - b).abs().amax(dim=(0, 2))        # per-head max |Δ|
        mag_max = a.abs().amax(dim=(0, 2))               # per-head max |activation| (reference)
        # Per-head relative divergence: max|Δ| normalized by that head's own magnitude.
        rel = diff_max / mag_max.clamp(min=1e-12)

        order_by_diff = diff_max.argsort(descending=True)
        print(f"\n=== {name} ({n_heads} heads) ===")
        print(f"{'head':>4} {'max|Δ|':>12} {'max|act|':>12} {'rel':>9}")
        for h in order_by_diff[:6].tolist():
            print(f"{h:>4} {diff_max[h]:>12.4e} {mag_max[h]:>12.4e} {rel[h]:>8.2%}")
        sorted_d = diff_max.sort().values
        median_d = float(sorted_d[n_heads // 2])
        worst_d = float(sorted_d[-1])
        conc = worst_d / median_d if median_d > 0 else float("inf")
        print(f"  concentration (worst/median |Δ|): {conc:.1f}x")
        print(f"  worst-|Δ| head: {int(diff_max.argmax())}   "
              f"worst-|act| head: {int(mag_max.argmax())}   "
              f"worst-rel head: {int(rel.argmax())}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    layers = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [25, 26, 27]
    main(Path(sys.argv[1]), Path(sys.argv[2]), layers)
