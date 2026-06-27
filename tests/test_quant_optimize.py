"""Tests for the optimize orchestration — the ship/quality-bar/headroom
decisions and the report. The heavy steps (auto_quant select, export, vLLM
benchmark) are mocked; the end-to-end run is validated on Modal
(scripts/validate_optimize.py). classify_recipe runs for real on real Recipes.
"""

from __future__ import annotations

import pytest

from firefly.bench import BenchmarkConfig, BenchmarkResult
from firefly.quant import optimize as optmod
from firefly.quant.deploy import DeployArtifact
from firefly.quant.intervention import RTNQuantizer
from firefly.quant.optimize import choose_ship_recipe, optimize
from firefly.quant.recipe_io import Recipe, serialize_intervention


def _recipe(*, scheme="w8a8", pre=None) -> Recipe:
    return Recipe(
        model_id="m", scheme=scheme, group_size=128, granularity="layer",
        quantize_fqns=[], kept_fp_fqns=[], pre_transforms=pre or [],
        quantizer=serialize_intervention(RTNQuantizer()),
    )


def _auto(*, accepted, routed_pre=None, fp=10.0, plain=15.0, routed=11.0) -> dict:
    return {
        "recipe_obj": _recipe(pre=routed_pre),
        "accepted": accepted,
        "recovery": 0.8,
        "perplexity": {"fp": fp, "plain": plain, "routed": routed},
        "recipe": {"quantizer": "rtn", "pre_transforms": [p["name"] for p in (routed_pre or [])]},
        "diagnosis_summary": {"ACTIVATION_OUTLIERS": 3},
    }


class TestChooseShipRecipe:
    def test_ships_routed_when_accepted_and_deployable(self):
        # routed is plain uniform RTN (no pre-transform) → deployable.
        auto = _auto(accepted=True)
        recipe, kind, ppl = choose_ship_recipe(auto, "m", "w8a8", 128)
        assert kind == "routed" and ppl == 11.0

    def test_ships_plain_when_routed_not_deployable(self):
        # routed uses SmoothQuant → not directly deployable → fall back to uniform.
        auto = _auto(accepted=True, routed_pre=[{"name": "smoothquant", "params": {}}])
        recipe, kind, ppl = choose_ship_recipe(auto, "m", "w8a8", 128)
        assert kind == "plain" and ppl == 15.0

    def test_ships_plain_when_rejected(self):
        auto = _auto(accepted=False)
        _, kind, ppl = choose_ship_recipe(auto, "m", "w8a8", 128)
        assert kind == "plain" and ppl == 15.0


class TestOptimizePlan:
    def test_plan_only_no_export(self, monkeypatch):
        monkeypatch.setattr(optmod, "auto_quant", lambda *a, **k: _auto(accepted=True))
        r = optimize("m", "in.json", ["t"], scheme="w8a8", dtype="bfloat16")
        assert r["artifact"] is None and r["measured"] is None
        # bf16(16) → w8a8(8) ≈ 2× weight compression estimate.
        assert r["compression_estimate"] == pytest.approx(2.0)

    def test_quality_bar_met_and_missed(self, monkeypatch):
        # shipped = plain perplexity 15 vs fp 10 → +50%.
        monkeypatch.setattr(optmod, "auto_quant", lambda *a, **k: _auto(accepted=False))
        assert optimize("m", "i", ["t"], quality_bar=0.6)["meets_bar"] is True
        assert optimize("m", "i", ["t"], quality_bar=0.4)["meets_bar"] is False

    def test_headroom_when_better_recipe_not_servable(self, monkeypatch):
        auto = _auto(accepted=True, routed_pre=[{"name": "smoothquant", "params": {}}], routed=10.5)
        monkeypatch.setattr(optmod, "auto_quant", lambda *a, **k: auto)
        h = optimize("m", "i", ["t"])["headroom"]
        assert h is not None and "smoothquant" in h["pre_transforms"]
        assert h["perplexity"] == 10.5

    def test_no_headroom_when_shipping_the_winner(self, monkeypatch):
        monkeypatch.setattr(optmod, "auto_quant", lambda *a, **k: _auto(accepted=True))
        assert optimize("m", "i", ["t"])["headroom"] is None


class TestOptimizeExportAndBenchmark:
    def test_export_and_benchmark_fold_into_result(self, monkeypatch, tmp_path):
        monkeypatch.setattr(optmod, "auto_quant", lambda *a, **k: _auto(accepted=False))

        def fake_export(recipe, out_dir, **kw):
            return DeployArtifact(
                path=tmp_path, scheme=recipe.scheme, compressed_tensors_scheme="W8A8",
                serve_command=f"vllm serve {tmp_path}", manifest={"scheme": recipe.scheme},
            )

        monkeypatch.setattr(optmod, "export_deployable", fake_export)

        bench_result = BenchmarkResult(
            engine="vllm", dtype="bfloat16", quantization=None, config=BenchmarkConfig(),
            decode_throughput_tok_s=5000.0, prefill_throughput_tok_s=60000.0,
            ttft_ms=100.0, e2e_latency_ms=0.0, weight_memory_bytes=1.78e9,
        )

        class _B:
            def benchmark(self, *a, **k):
                return bench_result

        monkeypatch.setattr("firefly.bench.get_benchmarker", lambda name: _B())

        r = optimize(
            "m", "i", ["t"], dtype="bfloat16", out_dir=tmp_path, benchmark=True,
            bench_config=BenchmarkConfig(),
        )
        assert r["artifact"]["serve_command"].startswith("vllm serve")
        assert r["measured"]["decode_tok_s"] == 5000.0
        assert r["measured"]["weight_mb"] == pytest.approx(1780.0)
        # the measurement is folded into the written manifest
        import json

        written = json.loads((tmp_path / "firefly_serving.json").read_text())
        assert written["measured"]["decode_tok_s"] == 5000.0


class TestCrossBackendReeval:
    def test_served_quality_drives_the_bar(self, monkeypatch, tmp_path):
        # selection (torchao) ships plain ppl 15 (+50% vs fp 10) → would MISS a 60%
        # bar; but the served (compressed-tensors) model re-evals at 11 (+10%), so
        # the honest bar (against served) MEETS, and bar_basis flips to 'served'.
        monkeypatch.setattr(optmod, "auto_quant", lambda *a, **k: _auto(accepted=False))
        monkeypatch.setattr(
            optmod, "export_deployable",
            lambda recipe, out_dir, **kw: DeployArtifact(
                path=tmp_path, scheme=recipe.scheme, compressed_tensors_scheme="W8A8",
                serve_command="vllm serve x", manifest={},
            ),
        )
        monkeypatch.setattr(optmod, "evaluate_deployed", lambda *a, **k: 11.0)

        r = optimize("m", "i", ["t"], out_dir=tmp_path, reeval_quality=True, quality_bar=0.2)
        assert r["quality"]["served"] == 11.0
        assert r["quality"]["backend_delta"] == pytest.approx(11.0 - 15.0)  # served − torchao
        assert r["bar_basis"] == "served"
        assert r["meets_bar"] is True  # +10% served ≤ 20% bar (despite +50% torchao)

    def test_backend_delta_surfaces_disagreement(self, monkeypatch, tmp_path):
        monkeypatch.setattr(optmod, "auto_quant", lambda *a, **k: _auto(accepted=False, plain=12.0))
        monkeypatch.setattr(
            optmod, "export_deployable",
            lambda recipe, out_dir, **kw: DeployArtifact(
                path=tmp_path, scheme=recipe.scheme, compressed_tensors_scheme="W8A8",
                serve_command="vllm serve x", manifest={},
            ),
        )
        # compressed-tensors disagrees materially with torchao for the same scheme.
        monkeypatch.setattr(optmod, "evaluate_deployed", lambda *a, **k: 20.0)
        r = optimize("m", "i", ["t"], out_dir=tmp_path, reeval_quality=True)
        assert r["quality"]["backend_delta"] == pytest.approx(8.0)


def test_render_optimize_smoke(monkeypatch):
    from firefly.report import render_optimize

    monkeypatch.setattr(optmod, "auto_quant", lambda *a, **k: _auto(accepted=True, routed_pre=[{"name": "smoothquant", "params": {}}]))
    text = render_optimize(optimize("m", "i", ["t"], scheme="w8a8", quality_bar=0.05))
    assert "optimize" in text and "ship:" in text and "headroom" in text
