"""The Benchmarker protocol, its config/result types, and the registry.

This is the *measured-cost* half of Firefly. The :mod:`firefly.quant.cost`
module computes a recipe's weight footprint analytically (pure param shapes +
bit-widths) — that's exact for memory and free, but it cannot tell you the one
thing people actually deploy for: **throughput**. Whether a quantized model
serves *faster* depends on kernels, the batch/sequence regime, and the
hardware (int4 weight-only helps memory-bound decode but its dequant can cost
compute-bound prefill; w8a8 helps only where int8 tensor cores do). None of
that is predictable from bit-widths — it has to be *measured on the serving
engine*. That is exactly the same claim the quality gate makes ("you can't
trust a recipe across models, measure it"), pointed at the cost axis.

A :class:`Benchmarker` stands a model up in a real serving engine and measures
prefill/decode throughput, time-to-first-token, and peak memory at a target
``(batch, input_len, output_len)``. Unlike the capture :class:`Runner` — which
forces ``enforce_eager=True`` so forward hooks fire — the benchmarker runs with
**CUDA graphs on**, because that is the throughput a user would actually serve.

Kept dependency-light: no engine import here, so the heavy backends load lazily
in :func:`get_benchmarker` only when requested.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class BenchmarkConfig:
    """The serving regime to measure. Defaults are a small, decode-heavy shape
    that fits a single mid-tier GPU and exercises the memory-bound regime where
    weight quantization is supposed to help."""

    batch_size: int = 8
    input_len: int = 512
    output_len: int = 128
    n_iters: int = 3
    warmup_iters: int = 1

    def __post_init__(self) -> None:
        for fname in ("batch_size", "input_len", "output_len", "n_iters"):
            if getattr(self, fname) < 1:
                raise ValueError(f"BenchmarkConfig.{fname} must be >= 1")
        if self.warmup_iters < 0:
            raise ValueError("BenchmarkConfig.warmup_iters must be >= 0")


@dataclass
class BenchmarkResult:
    """Measured serving cost of one model configuration.

    ``decode_throughput_tok_s`` is the headline (steady-state generation speed,
    the QPS proxy users feel); ``prefill_throughput_tok_s`` and ``ttft_ms``
    cover the prompt-processing side. Weight quantization typically helps decode
    (memory-bound) while *costing* prefill (dequant overhead on a compute-bound
    pass) — measured, e.g. fp8 on Qwen2.5-1.5B: +20% decode, -24% prefill. That
    split is the whole reason to measure rather than assume "quant = faster".

    ``weight_memory_bytes`` is the meaningful memory signal (model footprint).
    ``peak_memory_bytes`` is best-effort and, under vLLM, reflects the reserved
    ``gpu_memory_utilization`` KV pool — roughly constant across configs, *not*
    the model footprint. Both are ``None`` if the engine didn't expose them."""

    engine: str
    dtype: str
    quantization: str | None
    config: BenchmarkConfig
    decode_throughput_tok_s: float
    prefill_throughput_tok_s: float
    ttft_ms: float
    e2e_latency_ms: float
    peak_memory_bytes: float | None = None
    weight_memory_bytes: float | None = None
    raw: dict = field(default_factory=dict)

    @property
    def ms_per_output_token(self) -> float:
        """Per-token decode latency — the *lower-is-better* cost axis to pair
        with quality on a measured Pareto frontier (see
        :func:`throughput_frontier`)."""
        return 1000.0 / self.decode_throughput_tok_s if self.decode_throughput_tok_s > 0 else float("inf")


def summarize_timings(
    prefill_times_s: list[float],
    full_times_s: list[float],
    *,
    batch_size: int,
    input_len: int,
    output_len: int,
) -> dict[str, float]:
    """Turn raw per-iteration wall times into throughput metrics — pure, so the
    timing arithmetic is unit-testable without a GPU.

    ``prefill_times_s`` time a ``max_tokens=1`` generate (prompt processing
    only); ``full_times_s`` time the full ``output_len`` generate. Decode time
    is the difference, so decode throughput excludes prompt processing. Medians
    are used across iterations to shrug off a stray slow run.
    """
    if not prefill_times_s or not full_times_s:
        raise ValueError("need at least one prefill and one full timing")

    prefill = statistics.median(prefill_times_s)
    full = statistics.median(full_times_s)
    # Decode time is the full generate minus prompt processing. Clamp to a tiny
    # positive so a noisy near-zero difference can't divide to infinity.
    decode = max(full - prefill, 1e-6)

    prefill_tokens = batch_size * input_len
    decode_tokens = batch_size * output_len
    return {
        "prefill_throughput_tok_s": prefill_tokens / prefill if prefill > 0 else 0.0,
        "decode_throughput_tok_s": decode_tokens / decode,
        "ttft_ms": prefill * 1000.0,
        "e2e_latency_ms": full * 1000.0,
    }


def throughput_frontier(
    results: list[BenchmarkResult], qualities: list[float], *, quality_higher_is_better: bool
) -> tuple[list[int], int | None]:
    """Measured-cost Pareto frontier: trade decode latency (``ms_per_output_token``,
    lower better) against quality. Returns ``(frontier_positions, knee_position)``
    as indices into the inputs — the same shape as the analytic
    :func:`firefly.quant.cost.frontier_and_knee`, so callers can swap measured
    QPS in for estimated bytes without changing how they consume the frontier.
    """
    from firefly.quant.cost import frontier_and_knee

    costs = [r.ms_per_output_token for r in results]
    return frontier_and_knee(costs, qualities, quality_higher_is_better=quality_higher_is_better)


@runtime_checkable
class Benchmarker(Protocol):
    """Throughput/memory backend. ``benchmark`` is the whole contract."""

    name: str

    def benchmark(
        self,
        model_id: str,
        config: BenchmarkConfig | None = None,
        *,
        dtype: str = "bfloat16",
        quantization: str | None = None,
        device: str = "cuda",
        seed: int = 0,
        options: dict[str, str] | None = None,
    ) -> BenchmarkResult:
        """Stand ``model_id`` up in this engine and measure serving cost.

        ``quantization`` is the engine's own scheme name (e.g. vLLM's
        ``"torchao"`` / ``"fp8"``), or ``None`` for the unquantized baseline.
        ``options`` carries engine-specific knobs; backends ignore keys they
        don't recognize.
        """
        ...


def available_benchmarkers() -> list[str]:
    """Benchmarker names :func:`get_benchmarker` accepts."""
    return ["vllm", "sglang"]


def get_benchmarker(name: str) -> Benchmarker:
    """Instantiate a benchmarker by name. Engine modules are imported lazily."""
    if name == "vllm":
        from firefly.bench.vllm import VLLMBenchmarker

        return VLLMBenchmarker()
    if name == "sglang":
        from firefly.bench.sglang import SGLangBenchmarker

        return SGLangBenchmarker()
    raise ValueError(f"Unknown benchmarker {name!r}. Available: {available_benchmarkers()}")
