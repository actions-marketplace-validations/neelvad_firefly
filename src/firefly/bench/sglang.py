"""SGLang benchmarker: the same throughput/memory contract on SGLang's engine.

SGLang has been gaining serving share quickly, so the measured-cost axis should
not be vLLM-only: a recipe that wins on one engine's kernels may not on the
other, which is the whole reason to *measure* rather than assume. This mirrors
:class:`firefly.bench.vllm.VLLMBenchmarker` against SGLang's offline ``Engine``
(CUDA graphs on by default; ``disable_cuda_graph`` opt to force eager).

Requires ``pip install 'firefly-ml[sglang]'`` and a CUDA GPU. Memory reads are not
wired yet (SGLang's offline engine doesn't expose a worker RPC as cleanly as
vLLM's ``collective_rpc``); ``peak_memory_bytes`` / ``weight_memory_bytes`` are
left ``None`` and the throughput numbers — the headline — are fully measured.

GPU-unvalidated as of writing: the throughput method is engine-agnostic (time a
1-token generate vs a full generate, same as the vLLM path), but the SGLang
``Engine`` construction/generate arg names should be confirmed on a GPU run.
"""

from __future__ import annotations

import time

from firefly.bench.base import BenchmarkConfig, BenchmarkResult, summarize_timings

_BOOL_TRUE = {"1", "true", "yes", "on"}


def _parse_options(options: dict[str, str] | None) -> dict:
    opts = dict(options or {})
    known = {"mem_fraction_static", "disable_cuda_graph", "disable_radix_cache", "context_length", "trust_remote_code"}
    unknown = set(opts) - known
    if unknown:
        raise ValueError(
            f"Unknown SGLang benchmarker option(s): {sorted(unknown)}. Supported: {sorted(known)}."
        )
    return {
        "mem_fraction_static": float(opts["mem_fraction_static"]) if "mem_fraction_static" in opts else None,
        "disable_cuda_graph": opts.get("disable_cuda_graph", "false").lower() in _BOOL_TRUE,
        # SGLang's radix cache is prefix caching; off by default here so the batch
        # doesn't collapse to a single shared prefill (matches the vLLM path).
        "disable_radix_cache": opts.get("disable_radix_cache", "true").lower() in _BOOL_TRUE,
        "context_length": int(opts["context_length"]) if "context_length" in opts else None,
        # Off by default: executes Python shipped with the model repo.
        "trust_remote_code": opts.get("trust_remote_code", "false").lower() in _BOOL_TRUE,
    }


class SGLangBenchmarker:
    """Measure serving throughput via SGLang's offline ``Engine``."""

    name = "sglang"

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
        cfg = config or BenchmarkConfig()
        opt = _parse_options(options)

        from firefly.capture import dtype_to_name, parse_dtype

        canonical_dtype = dtype_to_name(parse_dtype(dtype))

        try:
            import sglang as sgl
        except ImportError as e:
            raise ImportError(
                "The SGLang benchmarker needs SGLang installed and a CUDA GPU. "
                "Install with: pip install 'firefly-ml[sglang]'."
            ) from e

        engine_kwargs = dict(
            model_path=model_id,
            dtype=canonical_dtype,
            random_seed=seed,
            disable_cuda_graph=opt["disable_cuda_graph"],
            disable_radix_cache=opt["disable_radix_cache"],
            trust_remote_code=opt["trust_remote_code"],
        )
        if quantization:
            engine_kwargs["quantization"] = quantization
        if opt["mem_fraction_static"] is not None:
            engine_kwargs["mem_fraction_static"] = opt["mem_fraction_static"]
        if opt["context_length"] is not None:
            engine_kwargs["context_length"] = opt["context_length"]
        else:
            engine_kwargs["context_length"] = cfg.input_len + cfg.output_len + 16

        engine = sgl.Engine(**engine_kwargs)
        try:
            input_ids = self._build_prompts(cfg)
            prefill_sp = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
            full_sp = {"temperature": 0.0, "max_new_tokens": cfg.output_len, "ignore_eos": True}

            for _ in range(cfg.warmup_iters):
                engine.generate(input_ids=input_ids, sampling_params=full_sp)

            prefill_times: list[float] = []
            full_times: list[float] = []
            for _ in range(cfg.n_iters):
                t0 = time.perf_counter()
                engine.generate(input_ids=input_ids, sampling_params=prefill_sp)
                prefill_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                engine.generate(input_ids=input_ids, sampling_params=full_sp)
                full_times.append(time.perf_counter() - t0)
        finally:
            engine.shutdown()

        metrics = summarize_timings(
            prefill_times, full_times,
            batch_size=cfg.batch_size, input_len=cfg.input_len, output_len=cfg.output_len,
        )
        return BenchmarkResult(
            engine="sglang",
            dtype=canonical_dtype,
            quantization=quantization,
            config=cfg,
            decode_throughput_tok_s=metrics["decode_throughput_tok_s"],
            prefill_throughput_tok_s=metrics["prefill_throughput_tok_s"],
            ttft_ms=metrics["ttft_ms"],
            e2e_latency_ms=metrics["e2e_latency_ms"],
            peak_memory_bytes=None,  # SGLang worker memory not wired yet
            weight_memory_bytes=None,
            raw={"prefill_times_s": prefill_times, "full_times_s": full_times},
        )

    @staticmethod
    def _build_prompts(cfg: BenchmarkConfig) -> list[list[int]]:
        """``batch_size`` distinct token-id prompts of exactly ``input_len`` tokens
        (a per-prompt offset keeps the radix cache from sharing prefills)."""
        lo, span = 100, 29000
        return [
            [lo + ((b * cfg.input_len + j) % span) for j in range(cfg.input_len)]
            for b in range(cfg.batch_size)
        ]
