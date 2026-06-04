"""Render the per-tap diff and activation-magnitude plots.

Two subcommands:

  diff REF_A REF_B [--out PATH]
      Per-tap relative error between two captures, in forward order.
      Used to show "the first-divergence layer" for cross-version,
      cross-backend, or cross-engine comparisons.

  magnitudes REF [--out PATH]
      Per-layer activation magnitude (mean absolute value of the
      residual stream, attention output, and MLP output). Useful to
      explain *why* divergence appears where it does — outlier features
      and the BF16-rounding threshold conspire to set the first-divergence
      layer.

Default output path is ``scripts/plots/<auto-name>.png``.

Examples:
    uv run python scripts/plot_validation.py diff \\
        scripts/results/vllm_0_8_5_flash \\
        scripts/results/vllm_0_8_5_xformers \\
        --out scripts/plots/flash_vs_xformers.png

    uv run python scripts/plot_validation.py magnitudes \\
        scripts/results/vllm_0_8_5_flash \\
        --out scripts/plots/magnitudes_smollm.png
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Firefly validation plots.")


def _tap_order_key(name: str) -> tuple:
    """Match capture_vllm.py's tap ordering, including @prefill / @token_N suffixes."""
    base, suffix = (name.rsplit("@", 1) + [""])[:2] if "@" in name else (name, "")
    if suffix == "" or suffix == "prefill":
        suffix_key = 0
    elif suffix.startswith("token_"):
        try:
            suffix_key = 1 + int(suffix[len("token_"):])
        except ValueError:
            suffix_key = 10**6
    else:
        suffix_key = 10**6

    if base == "final_norm":
        return (10**9, 0, suffix_key, name)
    m = re.match(r"layer\.(\d+)(?:\.(self_attn|mlp))?$", base)
    if m:
        i = int(m.group(1))
        sub = m.group(2)
        return (i, {"self_attn": 0, "mlp": 1, None: 2}[sub], suffix_key, name)
    return (10**9 - 1, 0, suffix_key, name)


@app.command()
def diff(
    ref_a: Path = typer.Argument(..., help="Reference directory A (baseline)."),
    ref_b: Path = typer.Argument(..., help="Reference directory B (candidate)."),
    out: Path = typer.Option(
        Path("scripts/plots/diff.png"),
        help="Output PNG path.",
    ),
    title: str = typer.Option("", help="Plot title override."),
) -> None:
    """Plot per-tap relative error A vs B in forward order."""
    import matplotlib.pyplot as plt
    import torch

    from firefly.reference import read_reference

    manifest_a, tensors_a = read_reference(ref_a)
    manifest_b, tensors_b = read_reference(ref_b)

    common_taps = sorted(set(tensors_a) & set(tensors_b), key=_tap_order_key)
    if not common_taps:
        raise SystemExit("No overlapping tap names between A and B.")

    rel_pcts: list[float] = []
    max_abs: list[float] = []
    for name in common_taps:
        ta = tensors_a[name].float()
        tb = tensors_b[name].float()
        diff_t = (ta - tb).abs()
        m_abs = diff_t.max().item()
        ref_mag = ta.abs().mean().item()
        mean_abs = diff_t.mean().item()
        rel_pct = (mean_abs / ref_mag * 100) if ref_mag > 0 else 0.0
        rel_pcts.append(rel_pct)
        max_abs.append(m_abs)

    if not title:
        a_label = _ref_label(manifest_a)
        b_label = _ref_label(manifest_b)
        title = f"Per-tap relative error: {a_label}  vs  {b_label}"

    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_rel, ax_abs) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    x = list(range(len(common_taps)))

    # Use semilog for relative error so the layer-7 onset is visible while
    # the layer-29 / final_norm growth doesn't dominate.
    plotted = [v if v > 0 else 1e-6 for v in rel_pcts]  # log-safe floor
    ax_rel.semilogy(x, plotted, marker=".", linewidth=1.0, color="#0c6fa6")
    ax_rel.axhline(1e-6, color="#aaaaaa", linewidth=0.5, linestyle=":")
    ax_rel.set_ylabel("mean abs diff / ref mag (%)")
    ax_rel.set_title(title)

    ax_abs.semilogy([i for i in x], [v if v > 0 else 1e-6 for v in max_abs],
                    marker=".", linewidth=1.0, color="#d77f00")
    ax_abs.set_ylabel("max abs diff (log)")
    ax_abs.set_xlabel("tap (forward order)")

    # Mark first-nonzero tap
    for i, v in enumerate(rel_pcts):
        if v > 0:
            ax_rel.axvline(i, color="#c14d4d", linewidth=0.8, linestyle="--",
                           label=f"first divergence: {common_taps[i]}")
            ax_abs.axvline(i, color="#c14d4d", linewidth=0.8, linestyle="--")
            ax_rel.legend(loc="upper left")
            break

    # Tick labels: show layer index transitions only (every 3rd tap)
    label_every = 3
    tick_idx = [i for i in x if i % label_every == 0]
    ax_abs.set_xticks(tick_idx)
    ax_abs.set_xticklabels(
        [common_taps[i].replace("self_attn", "sa").replace("final_norm", "fn") for i in tick_idx],
        rotation=70, ha="right", fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")
    print(f"  total taps: {len(common_taps)}")
    print(f"  taps with nonzero diff: {sum(1 for v in rel_pcts if v > 0)}")
    if any(v > 0 for v in rel_pcts):
        first = next(i for i, v in enumerate(rel_pcts) if v > 0)
        print(f"  first divergence: {common_taps[first]} ({rel_pcts[first]:.4f}%)")
        print(f"  final tap relative: {rel_pcts[-1]:.4f}%")

    # Touch torch import to keep the linter happy about its presence (we
    # actually use it implicitly via .float() on the tensors above).
    _ = torch.float32


@app.command()
def decode_diff(
    ref_a: Path = typer.Argument(..., help="Reference directory A (baseline, decode-mode)."),
    ref_b: Path = typer.Argument(..., help="Reference directory B (candidate, decode-mode)."),
    out: Path = typer.Option(
        Path("scripts/plots/decode_diff.png"),
        help="Output PNG path.",
    ),
    title: str = typer.Option("", help="Plot title override."),
) -> None:
    """Plot per-tap relative error per token position (prefill + decode steps).

    For decode-mode references where tap names have ``@prefill`` /
    ``@token_N`` suffixes. One semilog line per position; layer-tap
    order on the x-axis. Lines colored cool-to-warm so propagation
    through decode is visually obvious.
    """
    import matplotlib.pyplot as plt
    import torch

    from firefly.reference import read_reference

    manifest_a, tensors_a = read_reference(ref_a)
    manifest_b, tensors_b = read_reference(ref_b)

    common = set(tensors_a) & set(tensors_b)
    suffixed = [n for n in common if "@" in n]
    if not suffixed:
        raise SystemExit("Neither reference contains @prefill / @token_N suffixed taps.")

    # Group by suffix (position); within each, sort by base name (layer order).
    positions: dict[str, list[str]] = {}
    for name in suffixed:
        base, suffix = name.rsplit("@", 1)
        positions.setdefault(suffix, []).append(name)
    for suffix in positions:
        positions[suffix].sort(key=_tap_order_key)

    def _pos_order(s: str) -> int:
        if s == "prefill":
            return 0
        if s.startswith("token_"):
            return 1 + int(s[len("token_"):])
        return 10**6

    ordered_positions = sorted(positions.keys(), key=_pos_order)
    n_pos = len(ordered_positions)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))

    cmap = plt.get_cmap("viridis")
    summary = []
    first_div_layer = None
    for i, suffix in enumerate(ordered_positions):
        names_at_pos = positions[suffix]
        rel_pcts = []
        for name in names_at_pos:
            a = tensors_a[name].float()
            b = tensors_b[name].float()
            diff = (a - b).abs()
            mean_abs = diff.mean().item()
            ref_mag = a.abs().mean().item()
            rel_pct = (mean_abs / ref_mag * 100) if ref_mag > 0 else 0.0
            rel_pcts.append(rel_pct)
        x = list(range(len(names_at_pos)))
        plotted = [v if v > 0 else 1e-7 for v in rel_pcts]
        color = cmap(i / max(n_pos - 1, 1))
        ax.semilogy(x, plotted, marker=".", linewidth=1.2, color=color, label=suffix)

        # Track first-divergence layer for an annotation
        for idx, v in enumerate(rel_pcts):
            if v > 0 and first_div_layer is None:
                first_div_layer = idx
                ax.axvline(idx, color="#c14d4d", linewidth=0.7, linestyle="--", alpha=0.6)
                break
        final_rel = rel_pcts[-1] if rel_pcts else 0
        summary.append((suffix, sum(1 for v in rel_pcts if v > 0), final_rel))

    if not title:
        a_label = _ref_label(manifest_a)
        b_label = _ref_label(manifest_b)
        title = f"Per-position diff: {a_label}  vs  {b_label}"
    ax.set_title(title)
    ax.set_xlabel("tap (layer forward order)")
    ax.set_ylabel("mean abs diff / ref mag (%)")
    ax.legend(loc="lower right", title="position", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=140)

    print(f"Saved {out}")
    print(f"  positions: {ordered_positions}")
    print(f"{'position':<12} {'n_diff':<8} {'final_rel(%)':<14}")
    print("-" * 40)
    for suffix, n_diff, final_rel in summary:
        print(f"{suffix:<12} {n_diff:<8} {final_rel:<14.4f}")

    # Suppress unused-import warning when torch isn't otherwise touched.
    _ = torch.float32


@app.command()
def magnitudes(
    ref: Path = typer.Argument(..., help="Reference directory."),
    out: Path = typer.Option(
        Path("scripts/plots/magnitudes.png"),
        help="Output PNG path.",
    ),
    title: str = typer.Option("", help="Plot title override."),
) -> None:
    """Plot per-layer MLP and self_attn activation magnitudes (mean + max).

    Note: in vLLM, ``LlamaDecoderLayer.forward`` returns the post-mlp
    hidden state, so the layer-level tap captures the same tensor as
    the ``mlp`` tap. We drop the redundant series and show only the
    two meaningful components.
    """
    import matplotlib.pyplot as plt

    from firefly.reference import read_reference

    manifest, tensors = read_reference(ref)

    layer_max = max(
        (int(m.group(1)) for m in (re.match(r"layer\.(\d+)", n) for n in tensors) if m),
        default=-1,
    )
    if layer_max < 0:
        raise SystemExit("No layer.* taps found in reference.")

    layers = list(range(layer_max + 1))
    attn_mean = [_mean_abs(tensors.get(f"layer.{i}.self_attn")) for i in layers]
    mlp_mean = [_mean_abs(tensors.get(f"layer.{i}.mlp")) for i in layers]
    attn_max = [_max_abs(tensors.get(f"layer.{i}.self_attn")) for i in layers]
    mlp_max = [_max_abs(tensors.get(f"layer.{i}.mlp")) for i in layers]

    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_max, ax_mean) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_max.semilogy(layers, attn_max, marker="^", label="self_attn", color="#d77f00")
    ax_max.semilogy(layers, mlp_max, marker="s", label="mlp", color="#3a8c4f")
    ax_max.set_ylabel("max abs (log)")
    ax_max.grid(True, alpha=0.3, which="both")
    ax_max.legend(loc="best")
    if not title:
        title = f"Per-layer activation magnitudes — {_ref_label(manifest)}"
    ax_max.set_title(title)

    ax_mean.semilogy(layers, attn_mean, marker="^", label="self_attn", color="#d77f00")
    ax_mean.semilogy(layers, mlp_mean, marker="s", label="mlp", color="#3a8c4f")
    ax_mean.set_ylabel("mean abs (log)")
    ax_mean.set_xlabel("layer index")
    ax_mean.grid(True, alpha=0.3, which="both")
    ax_mean.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")
    print(f"  mlp max abs range: [{min(mlp_max):.2f}, {max(mlp_max):.2f}]")
    print(f"  self_attn max abs range: [{min(attn_max):.2f}, {max(attn_max):.2f}]")


def _mean_abs(t) -> float:
    if t is None:
        return float("nan")
    return t.float().abs().mean().item()


def _max_abs(t) -> float:
    if t is None:
        return float("nan")
    return t.float().abs().max().item()


def _ref_label(manifest) -> str:
    parts = []
    if (v := manifest.env.get("vllm_version")):
        parts.append(f"vllm-{v}")
    if (b := manifest.env.get("attention_backend")) and b != "auto":
        parts.append(b)
    if manifest.dtype:
        parts.append(manifest.dtype)
    return " / ".join(parts) if parts else manifest.model_id


if __name__ == "__main__":
    app()
