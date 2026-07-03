"""Pure-logic tests for the benchmark seam.

The engine stand-up (vLLM/SGLang) is GPU-only and validated on Modal
(``scripts/validate_benchmark.py``); here we test the parts that decide
*meaning* — the timing arithmetic, the cost-axis property, the measured Pareto
wiring, and config validation — none of which need a GPU.
"""

from __future__ import annotations

import pytest

from firefly.bench import (
    BenchmarkConfig,
    BenchmarkResult,
    available_benchmarkers,
    get_benchmarker,
    summarize_timings,
    throughput_frontier,
)


def _result(decode_tok_s: float, *, quant: str | None = None) -> BenchmarkResult:
    return BenchmarkResult(
        engine="vllm", dtype="bfloat16", quantization=quant, config=BenchmarkConfig(),
        decode_throughput_tok_s=decode_tok_s, prefill_throughput_tok_s=0.0,
        ttft_ms=0.0, e2e_latency_ms=0.0,
    )


class TestSummarizeTimings:
    def test_throughput_arithmetic(self):
        # 1 prefill iter of 1.0s, 1 full iter of 2.0s; batch 8, in 512, out 128.
        m = summarize_timings([1.0], [2.0], batch_size=8, input_len=512, output_len=128)
        assert m["prefill_throughput_tok_s"] == pytest.approx(8 * 512 / 1.0)
        # decode time = full - prefill = 1.0s over 8*128 decode tokens.
        assert m["decode_throughput_tok_s"] == pytest.approx(8 * 128 / 1.0)
        assert m["ttft_ms"] == pytest.approx(1000.0)
        assert m["e2e_latency_ms"] == pytest.approx(2000.0)

    def test_uses_median_across_iters(self):
        # Median of prefill ignores the stray slow 10.0; median full = 2.0.
        m = summarize_timings([1.0, 1.0, 10.0], [2.0, 2.0, 2.0],
                              batch_size=1, input_len=10, output_len=10)
        assert m["ttft_ms"] == pytest.approx(1000.0)  # median prefill = 1.0s

    def test_clamps_nonpositive_decode_time(self):
        # full <= prefill (noise): decode time clamps to a tiny positive, no div0.
        m = summarize_timings([2.0], [2.0], batch_size=1, input_len=10, output_len=10)
        assert m["decode_throughput_tok_s"] > 0
        assert m["decode_throughput_tok_s"] != float("inf")

    def test_requires_timings(self):
        with pytest.raises(ValueError):
            summarize_timings([], [1.0], batch_size=1, input_len=1, output_len=1)


class TestBenchmarkResult:
    def test_ms_per_output_token_is_inverse_throughput(self):
        assert _result(1000.0).ms_per_output_token == pytest.approx(1.0)
        assert _result(250.0).ms_per_output_token == pytest.approx(4.0)

    def test_zero_throughput_is_infinite_cost(self):
        assert _result(0.0).ms_per_output_token == float("inf")


class TestThroughputFrontier:
    def test_faster_and_better_dominates(self):
        # r0: slow + low quality (perplexity high=bad). r1: fast + better.
        r0, r1 = _result(100.0), _result(400.0)
        # quality = perplexity, lower better.
        frontier, knee = throughput_frontier([r0, r1], [30.0, 12.0],
                                              quality_higher_is_better=False)
        assert frontier == [1]  # r1 dominates r0 on both axes
        assert knee == 1

    def test_genuine_tradeoff_keeps_both(self):
        # r0 faster but worse quality; r1 slower but better → both on frontier.
        r0, r1 = _result(400.0), _result(100.0)
        frontier, _ = throughput_frontier([r0, r1], [25.0, 12.0],
                                          quality_higher_is_better=False)
        assert set(frontier) == {0, 1}


class TestConfig:
    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            BenchmarkConfig(batch_size=0)
        with pytest.raises(ValueError):
            BenchmarkConfig(output_len=0)
        with pytest.raises(ValueError):
            BenchmarkConfig(warmup_iters=-1)

    def test_defaults_are_decode_heavy(self):
        cfg = BenchmarkConfig()
        assert cfg.output_len >= 1 and cfg.input_len >= 1


class TestRegistry:
    def test_lists_both_engines(self):
        assert set(available_benchmarkers()) == {"vllm", "sglang"}

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmarker"):
            get_benchmarker("tensorrt")

    @pytest.mark.parametrize("name", ["vllm", "sglang"])
    def test_known_names_resolve_lazily(self, name):
        # Construction must not require the engine to be installed (lazy import
        # lives inside benchmark()); the class object is enough here.
        b = get_benchmarker(name)
        assert b.name == name


class TestBenchmarkerOptions:
    @pytest.mark.parametrize(
        "module", ["firefly.bench.vllm", "firefly.bench.sglang"]
    )
    def test_trust_remote_code_is_opt_in(self, module):
        # Security default: executing repo-shipped Python requires an explicit opt-in.
        import importlib

        parse = importlib.import_module(module)._parse_options
        assert parse({})["trust_remote_code"] is False
        assert parse({"trust_remote_code": "true"})["trust_remote_code"] is True
