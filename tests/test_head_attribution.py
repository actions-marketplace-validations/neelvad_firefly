"""Tests for per-attention-head divergence attribution.

Covers the pure core (split_heads, per_head_divergence, attribute_divergent_heads),
the input-capturing tap path, the manifest head_counts plumbing, and report
rendering — all offline on hand-rolled fakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from firefly.head_attribution import (
    HeadDivergence,
    PerHeadAttribution,
    attribute_divergent_heads,
    per_head_divergence,
    split_heads,
)

# --- pure core: split_heads ------------------------------------------------


def test_split_heads_shape() -> None:
    t = torch.randn(2, 4, 12)
    out = split_heads(t, n_heads=3)
    assert out.shape == (2, 4, 3, 4)  # head_dim = 12 / 3


def test_split_heads_rejects_non_divisible() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        split_heads(torch.randn(2, 4, 10), n_heads=3)


def test_split_heads_rejects_nonpositive_heads() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        split_heads(torch.randn(2, 4, 12), n_heads=0)


# --- pure core: per_head_divergence ----------------------------------------


def test_identical_tensors_zero_divergence() -> None:
    t = torch.randn(2, 4, 12)
    attr = per_head_divergence("layer.0.attn_heads", t, t.clone(), n_heads=3)
    assert attr.n_heads == 3
    assert attr.worst_max_abs_diff == 0.0
    assert attr.median_max_abs_diff == 0.0
    assert attr.concentration == 0.0
    assert all(h.max_abs_diff == 0.0 for h in attr.heads)


def test_localizes_divergence_to_one_head() -> None:
    """A perturbation injected into a single head's slice is attributed to it."""
    torch.manual_seed(0)
    ref = torch.randn(2, 4, 12)  # 3 heads × head_dim 4
    cand = ref.clone()
    # Head 1 occupies columns [4:8]. Inject a large diff only there.
    cand[..., 4:8] += 5.0

    attr = per_head_divergence("layer.7.attn_heads", ref, cand, n_heads=3)

    assert attr.worst_head == 1
    assert attr.worst_max_abs_diff == pytest.approx(5.0, abs=1e-5)
    # The other two heads are bit-identical → median diff is 0 → inf concentration.
    assert attr.median_max_abs_diff == 0.0
    assert attr.concentration == float("inf")


def test_concentration_ratio_finite() -> None:
    """When every head diverges, concentration is worst/median (finite)."""
    ref = torch.zeros(1, 1, 6)  # 3 heads × head_dim 2
    cand = ref.clone()
    # Per-head abs diffs: head0=1, head1=2, head2=4. median=2, worst=4 → 2.0×.
    cand[..., 0:2] = 1.0
    cand[..., 2:4] = 2.0
    cand[..., 4:6] = 4.0

    attr = per_head_divergence("t", ref, cand, n_heads=3)
    assert attr.worst_head == 2
    assert attr.median_max_abs_diff == pytest.approx(2.0)
    assert attr.concentration == pytest.approx(2.0)


def test_per_head_divergence_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        per_head_divergence(
            "t", torch.randn(2, 4, 12), torch.randn(2, 5, 12), n_heads=3
        )


def test_heads_are_in_index_order() -> None:
    ref = torch.zeros(1, 6)
    cand = ref.clone()
    cand[..., 2:4] = 9.0  # head 1
    attr = per_head_divergence("t", ref, cand, n_heads=3)
    assert [h.head_idx for h in attr.heads] == [0, 1, 2]


# --- pure core: attribute_divergent_heads ----------------------------------


def test_attribute_divergent_heads_multiple_taps() -> None:
    ref = {
        "layer.0.attn_heads": torch.zeros(1, 8),
        "layer.1.attn_heads": torch.zeros(1, 8),
    }
    cand = {
        "layer.0.attn_heads": ref["layer.0.attn_heads"].clone(),
        "layer.1.attn_heads": ref["layer.1.attn_heads"].clone(),
    }
    cand["layer.1.attn_heads"][..., 0:2] += 3.0  # head 0 of layer 1

    out = attribute_divergent_heads(ref, cand, {"layer.0.attn_heads": 4, "layer.1.attn_heads": 4})
    assert [a.tap_name for a in out] == ["layer.0.attn_heads", "layer.1.attn_heads"]
    assert out[0].worst_max_abs_diff == 0.0
    assert out[1].worst_head == 0
    assert out[1].worst_max_abs_diff == pytest.approx(3.0)


def test_attribute_divergent_heads_skips_missing_taps() -> None:
    ref = {"layer.0.attn_heads": torch.zeros(1, 8)}
    cand: dict[str, torch.Tensor] = {}  # candidate missing the tap
    out = attribute_divergent_heads(ref, cand, {"layer.0.attn_heads": 4})
    assert out == []


# --- tap-point selection with per_head -------------------------------------


class _AttnWithProj(nn.Module):
    """Attention sub-module with a distinct o_proj input width to prove
    input-capture: maps dim -> inner -> dim where inner = n_heads * head_dim."""

    def __init__(self, dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        inner = n_heads * head_dim
        self.proj_in = nn.Linear(dim, inner)
        self.o_proj = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.proj_in(x))


class _Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _Layer(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.self_attn = _AttnWithProj(dim, n_heads, head_dim)
        self.mlp = _Mlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class _Inner(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(50, dim)
        self.layers = nn.ModuleList(
            [_Layer(dim, n_heads, head_dim) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)


class _FakeAttnLM(nn.Module):
    """Llama-shaped fake whose attention has a real o_proj."""

    def __init__(self, dim: int = 8, n_layers: int = 2, n_heads: int = 2, head_dim: int = 3) -> None:
        super().__init__()
        self.model = _Inner(dim, n_layers, n_heads, head_dim)
        self.lm_head = nn.Linear(dim, 50)
        # Mimic an HF config object for num_attention_heads discovery.
        self.config = SimpleNamespace(num_attention_heads=n_heads)
        self._inner = (dim, n_heads, head_dim)

    def forward(self, input_ids: torch.Tensor | None = None, **_kw) -> torch.Tensor:
        x = self.model.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


def test_select_per_head_adds_attn_heads_taps() -> None:
    from firefly.tap_points import select_llm_tap_points

    taps = select_llm_tap_points(_FakeAttnLM(n_layers=2), per_head=True)
    names = [t.name for t in taps]
    assert names == [
        "layer.0.self_attn",
        "layer.0.attn_heads",
        "layer.0.mlp",
        "layer.0",
        "layer.1.self_attn",
        "layer.1.attn_heads",
        "layer.1.mlp",
        "layer.1",
        "final_norm",
    ]
    head_tap = next(t for t in taps if t.name == "layer.0.attn_heads")
    assert head_tap.capture_input is True
    assert head_tap.module_path == "model.layers.0.self_attn.o_proj"


def test_select_default_has_no_attn_heads_taps() -> None:
    from firefly.tap_points import select_llm_tap_points

    taps = select_llm_tap_points(_FakeAttnLM(n_layers=1))
    assert all(not t.name.endswith(".attn_heads") for t in taps)
    assert all(t.capture_input is False for t in taps)


def test_find_attn_output_proj_probes_names() -> None:
    from firefly.tap_points import _find_attn_output_proj

    layer = _Layer(dim=8, n_heads=2, head_dim=3)
    assert _find_attn_output_proj(layer) == "self_attn.o_proj"

    class _NoAttn(nn.Module):
        pass

    assert _find_attn_output_proj(_NoAttn()) is None


# --- capture: input-hook captures the o_proj input -------------------------


def test_run_capture_per_head_captures_o_proj_input() -> None:
    from firefly.capture import run_capture

    torch.manual_seed(0)
    # dim=8, inner = n_heads*head_dim = 2*3 = 6, so o_proj input width is 6.
    model = _FakeAttnLM(dim=8, n_layers=2, n_heads=2, head_dim=3).eval()
    batch = {"input_ids": torch.randint(0, 50, (2, 4))}

    captured = run_capture(model, batch, per_head=True)

    assert "layer.0.attn_heads" in captured
    # The per-head tap is the o_proj INPUT (width 6), not its output (width 8).
    assert captured["layer.0.attn_heads"].shape == (2, 4, 6)
    # The self_attn output tap is post-projection (width 8).
    assert captured["layer.0.self_attn"].shape == (2, 4, 8)


def test_num_attention_heads_reads_config() -> None:
    from firefly.capture import num_attention_heads

    assert num_attention_heads(_FakeAttnLM(n_heads=2)) == 2

    class _NoConfig(nn.Module):
        pass

    assert num_attention_heads(_NoConfig()) is None


def test_num_attention_heads_probes_alt_names() -> None:
    from firefly.capture import num_attention_heads

    m = nn.Module()
    m.config = SimpleNamespace(n_head=12)  # GPT-2 style
    assert num_attention_heads(m) == 12


# --- manifest round-trips head_counts --------------------------------------


def test_manifest_round_trips_head_counts(tmp_path: Path) -> None:
    from firefly.reference import ReferenceManifest, read_manifest, write_reference

    manifest = ReferenceManifest(
        model_id="fake",
        model_fingerprint="abc",
        tap_points=["layer.0.attn_heads"],
        shapes={"layer.0.attn_heads": [2, 4, 6]},
        dtypes={"layer.0.attn_heads": "float32"},
        captured_at="2026-06-09T00:00:00+00:00",
        head_counts={"layer.0.attn_heads": 2},
    )
    write_reference(tmp_path, manifest, {"layer.0.attn_heads": torch.zeros(2, 4, 6)})

    loaded = read_manifest(tmp_path)
    assert loaded.head_counts == {"layer.0.attn_heads": 2}


def test_manifest_defaults_head_counts_empty(tmp_path: Path) -> None:
    """References written without per-head taps load with empty head_counts."""
    from firefly.reference import ReferenceManifest, read_manifest, write_reference

    manifest = ReferenceManifest(
        model_id="fake",
        model_fingerprint="abc",
        tap_points=["final_norm"],
        shapes={"final_norm": [2, 4, 8]},
        dtypes={"final_norm": "float32"},
        captured_at="2026-06-09T00:00:00+00:00",
    )
    write_reference(tmp_path, manifest, {"final_norm": torch.zeros(2, 4, 8)})
    assert read_manifest(tmp_path).head_counts == {}


# --- report rendering ------------------------------------------------------


def _sample_per_head() -> list[PerHeadAttribution]:
    return [
        PerHeadAttribution(
            tap_name="layer.7.attn_heads",
            n_heads=4,
            heads=[
                HeadDivergence(0, 0.1, 0.01),
                HeadDivergence(1, 0.1, 0.01),
                HeadDivergence(2, 3.2, 0.5),
                HeadDivergence(3, 0.1, 0.01),
            ],
            worst_head=2,
            worst_max_abs_diff=3.2,
            median_max_abs_diff=0.1,
        )
    ]


def test_render_human_includes_per_head() -> None:
    from firefly.attribution import AttributionResult
    from firefly.report import render_human

    result = AttributionResult(first_divergent_tap=None, any_exceeded=False, divergences=[])
    text = render_human(result, per_head=_sample_per_head())
    assert "Per-head attention attribution" in text
    assert "layer.7.attn_heads" in text
    assert "2 / 4" in text  # worst head / n_heads


def test_render_markdown_includes_per_head() -> None:
    from firefly.attribution import AttributionResult
    from firefly.report import render_markdown

    result = AttributionResult(first_divergent_tap=None, any_exceeded=False, divergences=[])
    md = render_markdown(result, per_head=_sample_per_head())
    assert "Per-head attention attribution" in md
    assert "`layer.7.attn_heads`" in md


def test_write_json_includes_per_head_concentration(tmp_path: Path) -> None:
    from firefly.attribution import AttributionResult
    from firefly.report import write_json

    result = AttributionResult(first_divergent_tap=None, any_exceeded=False, divergences=[])
    out = tmp_path / "report.json"
    write_json(result, out, per_head=_sample_per_head())

    payload = json.loads(out.read_text())
    assert "per_head" in payload
    assert payload["per_head"][0]["worst_head"] == 2
    # concentration is a property, not a dataclass field — must be serialized explicitly.
    assert payload["per_head"][0]["concentration"] == pytest.approx(32.0)


# --- slow: real-model end-to-end -------------------------------------------


@pytest.mark.slow
def test_capture_per_head_smollm_end_to_end(tmp_path: Path) -> None:
    """Real model: --per-head capture populates head_counts and attn_heads taps."""
    from firefly.capture import capture_reference
    from firefly.reference import read_reference

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    out_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, out_dir, per_head=True)

    manifest, tensors = read_reference(out_dir)
    head_taps = [name for name in manifest.tap_points if name.endswith(".attn_heads")]
    assert head_taps, "expected per-head taps in the manifest"
    assert manifest.head_counts, "expected head_counts to be populated"
    for tap in head_taps:
        n_heads = manifest.head_counts[tap]
        assert tensors[tap].shape[-1] % n_heads == 0


@pytest.mark.slow
def test_check_per_head_self_compare_is_clean(tmp_path: Path) -> None:
    """Capturing SmolLM with --per-head then checking it against itself yields
    zero divergence and a per-head attribution for every attn_heads tap."""
    from firefly.capture import capture_reference
    from firefly.compare import compare_to_reference_per_head

    inputs_path = tmp_path / "golden.json"
    inputs_path.write_text(json.dumps({"texts": ["hello world"], "max_length": 8}))
    out_dir = tmp_path / "reference"

    capture_reference("HuggingFaceTB/SmolLM-135M", inputs_path, out_dir, per_head=True)

    divergences, per_head = compare_to_reference_per_head(
        reference_dir=out_dir,
        candidate_model_id="HuggingFaceTB/SmolLM-135M",
        inputs_path=inputs_path,
    )
    # CPU+fp32 self-compare is bit-exact: nothing exceeds, every head diff is 0.
    assert not any(d.exceeds_tolerance for d in divergences)
    assert per_head, "expected per-head attributions"
    assert all(ph.worst_max_abs_diff == 0.0 for ph in per_head)
