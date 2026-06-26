"""Measured serving cost ("benchmarkers") behind a common interface.

The analytic :mod:`firefly.quant.cost` model gives a recipe's weight footprint
exactly and for free — but it cannot tell you whether a quantized model serves
*faster*, because throughput depends on kernels, the batch/sequence regime, and
the hardware, none of which follow from bit-widths. A :class:`Benchmarker`
measures that on the real engine: prefill/decode throughput, time-to-first-token,
and peak memory at a target ``(batch, input_len, output_len)``.

Same shape as the capture :mod:`firefly.runners` seam — one verb behind a
registry, engine imports lazy — so adding an engine means adding a Benchmarker.
The default targets are :class:`~firefly.bench.vllm.VLLMBenchmarker` and
:class:`~firefly.bench.sglang.SGLangBenchmarker`. Both run with CUDA graphs on
(real serving perf), the opposite of the eager-forced capture runner.
"""

from __future__ import annotations

from firefly.bench.base import (
    BenchmarkConfig,
    Benchmarker,
    BenchmarkResult,
    available_benchmarkers,
    get_benchmarker,
    summarize_timings,
    throughput_frontier,
)

__all__ = [
    "Benchmarker",
    "BenchmarkConfig",
    "BenchmarkResult",
    "available_benchmarkers",
    "get_benchmarker",
    "summarize_timings",
    "throughput_frontier",
]
