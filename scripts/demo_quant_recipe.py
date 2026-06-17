"""Demo: compare mixed-precision strategies (isolated vs marginal) on one model.

Runs the `firefly quant-recipe` engine once per strategy, prints the recovery
curves side by side, and saves an overlaid plot. The headline this surfaces:
the *intuitive* "marginal" strategy (measure each layer's recovery directly)
builds WORSE recipes than "isolated" (rank by intrinsic quant difficulty) —
measured, not assumed. That's the attribution-guided story autoquant can't tell.

Writes:
  scripts/results/quant_recipe_<model>_<scheme>.json
  scripts/plots/quant_recipe_strategies_<model>_<scheme>.png

    uv run python scripts/demo_quant_recipe.py
    uv run python scripts/demo_quant_recipe.py --scheme int4wo --device cuda
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from firefly.quant_sensitivity import compute_recipe

_STRATEGIES = ("isolated", "marginal", "greedy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--inputs", default=None, help="golden-inputs JSON (default: a built-in prompt)")
    ap.add_argument("--scheme", default="w8a8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--k-values", default="1,2,4,8")
    ap.add_argument("--out-dir", default="scripts", help="writes results/ and plots/ under here")
    args = ap.parse_args()

    if args.inputs:
        inputs = Path(args.inputs)
    else:
        inputs = Path(tempfile.mkdtemp()) / "golden.json"
        inputs.write_text(json.dumps(
            {"texts": ["the quick brown fox jumps over the lazy dog"], "max_length": 16}
        ))

    ks = [int(x) for x in args.k_values.split(",") if x.strip()]
    results = {}
    for strat in _STRATEGIES:
        print(f"running strategy={strat} ...")
        results[strat] = compute_recipe(
            args.model, inputs, device=args.device, dtype=args.dtype,
            scheme=args.scheme, strategy=strat, k_values=ks,
        )

    full_div = results["isolated"].sensitivity.full_quant_divergence
    print(f"\nall {len(results['isolated'].sensitivity.layers)} layers quantized "
          f"({args.scheme}) -> {full_div:.2%} output divergence\n")
    by_strat = {s: {p.k: p for p in results[s].curve} for s in _STRATEGIES}
    header = "  ".join(f"{s:>12} recov" for s in _STRATEGIES)
    print(f"{'k':>3}  {header}")
    for k in ks:
        cells = "  ".join(f"{by_strat[s][k].recovery:>17.1%}" for s in _STRATEGIES)
        print(f"{k:>3}  {cells}")

    out = Path(args.out_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1].lower()}_{args.scheme}"

    payload = {
        "model": args.model,
        "scheme": args.scheme,
        "device": args.device,
        "full_quant_divergence": full_div,
        "strategies": {
            s: [
                {"k": p.k, "kept_layers": p.kept_layers,
                 "output_divergence": p.output_divergence, "recovery": p.recovery}
                for p in sorted(results[s].curve, key=lambda p: p.k)
            ]
            for s in _STRATEGIES
        },
    }
    json_path = out / "results" / f"quant_recipe_{tag}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strat, marker in (("isolated", "o"), ("marginal", "s"), ("greedy", "^")):
        if strat not in results:
            continue
        curve = sorted(results[strat].curve, key=lambda p: p.k)
        ax.plot(
            [p.k for p in curve], [p.recovery * 100 for p in curve],
            marker=marker, linewidth=2, label=strat,
        )
    ax.set_xlabel("high-precision layers kept (top-k by strategy)")
    ax.set_ylabel("output fidelity recovered (%)")
    ax.set_title(
        f"{args.model.split('/')[-1]} · {args.scheme}: recipe recovery by strategy\n"
        f"(all-quantized output divergence = {full_div:.1%})"
    )
    ax.set_ylim(0, 100)
    ax.legend(title="sensitivity strategy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out / "plots" / f"quant_recipe_strategies_{tag}.png"
    fig.savefig(plot_path, dpi=130)

    print(f"\nwrote {json_path}\n      {plot_path}")


if __name__ == "__main__":
    main()
