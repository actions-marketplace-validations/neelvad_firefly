"""Shared helpers across serving-engine runners (vLLM, SGLang)."""

from __future__ import annotations

import re


def tap_order_key(name: str) -> tuple:
    """Sort key putting taps in forward order.

    self_attn < attn_heads < mlp < layer-level, then final_norm last; within a
    tap, prefill before token_0..N. Engine hooks fire in execution order
    (and the attn-output-projection hook fires *before* its parent self_attn
    returns), so captures need this canonical re-ordering before they become
    the manifest's tap_points list.
    """
    base, suffix = (name.rsplit("@", 1) + [""])[:2] if "@" in name else (name, "")
    if suffix in ("", "prefill"):
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
    m = re.match(r"layer\.(\d+)(?:\.(self_attn|attn_heads|mlp))?$", base)
    if m:
        layer_idx = int(m.group(1))
        within = {"self_attn": 0, "attn_heads": 1, "mlp": 2, None: 3}[m.group(2)]
        return (layer_idx, within, suffix_key, name)
    return (10**9 - 1, 0, suffix_key, name)
