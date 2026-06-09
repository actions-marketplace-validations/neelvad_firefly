"""Compare two captured reference artifacts with per-head attention attribution.

Companion to ``run_vllm_suite.py``: where that diffs two vLLM captures at the
tap level, this additionally drills attention taps down to the worst head.
Use it on two ``capture_vllm.py --per-head`` artifacts (e.g. FLASH_ATTN vs
XFORMERS) to see which attention head carries a cross-backend divergence.

Both artifacts must have been captured with ``--per-head`` so they carry the
``layer.{i}.attn_heads`` taps and a populated ``head_counts`` manifest field.

Usage:
    uv run python scripts/compare_per_head.py <ref_a_dir> <ref_b_dir>
    uv run python scripts/compare_per_head.py <ref_a_dir> <ref_b_dir> --atol 1e-6
"""

from __future__ import annotations

import argparse
from pathlib import Path

from firefly.attribution import attribute_first_divergence
from firefly.compare import TapTolerance, diff_captures
from firefly.head_attribution import attribute_divergent_heads
from firefly.reference import read_reference
from firefly.report import render_human


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_a", type=Path, help="Reference artifact dir A (the reference)")
    parser.add_argument("ref_b", type=Path, help="Reference artifact dir B (the candidate)")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Flat absolute tolerance applied to every tap (default 1e-6).",
    )
    args = parser.parse_args()

    manifest_a, tensors_a = read_reference(args.ref_a)
    _manifest_b, tensors_b = read_reference(args.ref_b)

    tolerances = {name: TapTolerance(atol=args.atol) for name in manifest_a.tap_points}
    divergences = diff_captures(
        tensors_a, tensors_b, manifest_a.tap_points, tolerances=tolerances
    )
    result = attribute_first_divergence(divergences)

    # Forward-order the per-head taps via the manifest tap list (head_counts
    # dict is lexical after JSON sort_keys), matching compare.py.
    ordered_head_counts = {
        name: manifest_a.head_counts[name]
        for name in manifest_a.tap_points
        if name in manifest_a.head_counts
    }
    if not ordered_head_counts:
        print(
            "WARNING: no per-head taps in the reference. Re-capture both "
            "artifacts with `capture_vllm.py ... --per-head`."
        )
    per_head = attribute_divergent_heads(tensors_a, tensors_b, ordered_head_counts)

    print(render_human(result, per_head=per_head))

    if result.first_divergent_tap and per_head:
        # Spotlight the worst head at the most-concentrated attention tap.
        worst_tap = max(per_head, key=lambda ph: ph.concentration)
        print(
            f"\nMost head-localized divergence: head {worst_tap.worst_head} of "
            f"{worst_tap.tap_name} "
            f"(max|Δ|={worst_tap.worst_max_abs_diff:.3e}, "
            f"{worst_tap.concentration:.1f}× the median head)."
        )
    return 1 if result.any_exceeded else 0


if __name__ == "__main__":
    raise SystemExit(main())
