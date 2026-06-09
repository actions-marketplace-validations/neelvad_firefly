"""Per-attention-head divergence attribution.

:mod:`firefly.attribution` names the earliest *tap* that diverges. For an
attention tap this module drills one level deeper: given the input to the
attention output projection (``o_proj``) — the concatenated per-head
attention outputs of shape ``(..., n_heads * head_dim)`` — it splits by head
and ranks which head carries the divergence.

This is only meaningful on the o_proj *input*: post-projection the heads are
linearly mixed by a dense matrix and are no longer separable. Firefly
captures that input under the ``layer.{i}.attn_heads`` tap when
``capture --per-head`` is used, and records the head count in
``manifest.head_counts``.

Like the rest of Firefly's analysis layer, this module is pure: it takes
tensors and head counts, returns dataclasses, and does no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class HeadDivergence:
    """Divergence between reference and candidate for a single attention head."""

    head_idx: int
    max_abs_diff: float
    mean_abs_diff: float


@dataclass
class PerHeadAttribution:
    """Per-head breakdown of divergence at one ``attn_heads`` tap."""

    tap_name: str
    n_heads: int
    heads: list[HeadDivergence] = field(default_factory=list)
    worst_head: int = -1
    worst_max_abs_diff: float = 0.0
    median_max_abs_diff: float = 0.0

    @property
    def concentration(self) -> float:
        """``worst_max_abs_diff / median_max_abs_diff``.

        A high ratio means the divergence is localized to one (or a few)
        heads rather than spread evenly — the signature of a head-specific
        kernel difference (e.g. an outlier-feature head). Returns ``inf``
        when the median head is bit-identical (median diff is 0) but the
        worst head is not.
        """
        if self.median_max_abs_diff > 0:
            return self.worst_max_abs_diff / self.median_max_abs_diff
        return float("inf") if self.worst_max_abs_diff > 0 else 0.0


def split_heads(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Reshape ``(..., n_heads * head_dim)`` into ``(..., n_heads, head_dim)``.

    Raises ``ValueError`` if the last dimension is not divisible by
    ``n_heads`` — that means the tap isn't the concatenated per-head output
    we expect (or ``n_heads`` is wrong for this model).
    """
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    last = t.shape[-1]
    if last % n_heads != 0:
        raise ValueError(
            f"Last dim {last} is not divisible by n_heads={n_heads}; "
            f"this tensor doesn't look like concatenated per-head outputs."
        )
    head_dim = last // n_heads
    return t.reshape(*t.shape[:-1], n_heads, head_dim)


def per_head_divergence(
    tap_name: str,
    ref_t: torch.Tensor,
    cand_t: torch.Tensor,
    n_heads: int,
) -> PerHeadAttribution:
    """Split both tensors by head, compute per-head max/mean abs diff, rank.

    Raises ``ValueError`` on shape mismatch — a structural problem, not
    numerical divergence, consistent with :func:`firefly.compare.diff_captures`.
    """
    if ref_t.shape != cand_t.shape:
        raise ValueError(
            f"Shape mismatch at {tap_name!r}: reference {tuple(ref_t.shape)} "
            f"vs candidate {tuple(cand_t.shape)}"
        )

    r = split_heads(ref_t.float(), n_heads)
    c = split_heads(cand_t.float(), n_heads)
    diff = (c - r).abs()

    # Reduce over every dim except the head axis (second-to-last after split).
    head_axis = diff.ndim - 2
    reduce_dims = tuple(i for i in range(diff.ndim) if i != head_axis)
    max_per_head = diff.amax(dim=reduce_dims)
    mean_per_head = diff.mean(dim=reduce_dims)

    heads = [
        HeadDivergence(
            head_idx=i,
            max_abs_diff=float(max_per_head[i].item()),
            mean_abs_diff=float(mean_per_head[i].item()),
        )
        for i in range(n_heads)
    ]

    sorted_maxes = sorted(h.max_abs_diff for h in heads)
    median = sorted_maxes[len(sorted_maxes) // 2]
    worst = max(heads, key=lambda h: h.max_abs_diff)

    return PerHeadAttribution(
        tap_name=tap_name,
        n_heads=n_heads,
        heads=heads,
        worst_head=worst.head_idx,
        worst_max_abs_diff=worst.max_abs_diff,
        median_max_abs_diff=median,
    )


def attribute_divergent_heads(
    reference_tensors: dict[str, torch.Tensor],
    candidate_tensors: dict[str, torch.Tensor],
    head_counts: dict[str, int],
) -> list[PerHeadAttribution]:
    """Run :func:`per_head_divergence` for every tap in ``head_counts``.

    Taps absent from either capture are skipped (the main diff already
    surfaces a missing/extra tap as a hard error; here it's diagnostic
    only). Results are returned in ``head_counts`` iteration order, which
    preserves the forward order the manifest recorded them in.
    """
    out: list[PerHeadAttribution] = []
    for tap_name, n_heads in head_counts.items():
        if tap_name not in reference_tensors or tap_name not in candidate_tensors:
            continue
        out.append(
            per_head_divergence(
                tap_name,
                reference_tensors[tap_name],
                candidate_tensors[tap_name],
                n_heads,
            )
        )
    return out
