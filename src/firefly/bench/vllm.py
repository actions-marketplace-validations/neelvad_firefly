"""vLLM benchmarker: real serving throughput + memory for a model config.

Stands up an in-process vLLM ``LLM`` with **CUDA graphs on** (``enforce_eager``
defaults False — the opposite of the capture runner, which forces eager so
hooks fire) and times prompt processing and steady-state decode at a target
``(batch, input_len, output_len)``. This is the measurement the analytic cost
model can't make: whether a given quantization actually serves faster *here*.

Requires ``pip install 'firefly-ml[vllm]'`` and a CUDA GPU.

**One config per process.** An ``LLM`` claims ~``gpu_memory_utilization`` of the
device and vLLM caches engine state per process, so a second config in the same
process OOMs. To compare configs (fp baseline vs a quant), run each
``benchmark()`` in its own process/container — the same constraint the vLLM
capture runner documents.

Method (version-robust on purpose — no dependence on per-request metrics, which
have drifted across vLLM V0/V1):

* Build ``batch_size`` distinct token-id prompts of exactly ``input_len`` tokens
  (prefix caching disabled, so the batch doesn't collapse to one shared prefill).
* Time a ``max_tokens=1`` generate → prompt-processing (prefill) latency / TTFT.
* Time a ``max_tokens=output_len, ignore_eos=True`` generate → full latency;
  decode time is the difference, so decode throughput excludes prefill.
* Peak / weight memory are read from the worker via ``collective_rpc`` (V1) /
  ``apply_model`` (V0); best-effort — ``None`` if the engine doesn't expose them.
"""

from __future__ import annotations

import os
import time

from firefly.bench.base import BenchmarkConfig, BenchmarkResult, summarize_timings

# ---------------------------------------------------------------------------
# Worker-side functions. Top-level so they pickle to vLLM's worker process.
# ---------------------------------------------------------------------------


def _reset_peak_memory_impl(model) -> int:
    import torch

    torch.cuda.reset_peak_memory_stats()
    return 0


def _read_peak_memory_impl(model) -> int:
    import torch

    return int(torch.cuda.max_memory_allocated())


def _read_weight_bytes_impl(model) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def _reset_peak_memory(model) -> int:
    return _reset_peak_memory_impl(model)


def _read_peak_memory(model) -> int:
    return _read_peak_memory_impl(model)


def _read_weight_bytes(model) -> int:
    return _read_weight_bytes_impl(model)


def _v1_reset_peak_memory(worker) -> int:
    return _reset_peak_memory_impl(worker.model_runner.model)


def _v1_read_peak_memory(worker) -> int:
    return _read_peak_memory_impl(worker.model_runner.model)


def _v1_read_weight_bytes(worker) -> int:
    return _read_weight_bytes_impl(worker.model_runner.model)


_BOOL_TRUE = {"1", "true", "yes", "on"}


def _parse_options(options: dict[str, str] | None) -> dict:
    opts = dict(options or {})
    known = {"engine", "gpu_memory_utilization", "max_model_len", "enforce_eager", "enable_prefix_caching", "trust_remote_code"}
    unknown = set(opts) - known
    if unknown:
        raise ValueError(
            f"Unknown vLLM benchmarker option(s): {sorted(unknown)}. Supported: {sorted(known)}."
        )
    engine = opts.get("engine", "v1")
    if engine not in {"v0", "v1"}:
        raise ValueError(f"engine must be 'v0' or 'v1', got {engine!r}")
    return {
        "engine": engine,
        "gpu_memory_utilization": float(opts.get("gpu_memory_utilization", "0.9")),
        "max_model_len": int(opts["max_model_len"]) if "max_model_len" in opts else None,
        "enforce_eager": opts.get("enforce_eager", "false").lower() in _BOOL_TRUE,
        "enable_prefix_caching": opts.get("enable_prefix_caching", "false").lower() in _BOOL_TRUE,
        # Off by default: executes Python shipped with the model repo.
        "trust_remote_code": opts.get("trust_remote_code", "false").lower() in _BOOL_TRUE,
    }


def _unwrap(rpc_result):
    return rpc_result[0] if isinstance(rpc_result, list) and rpc_result else rpc_result


class VLLMBenchmarker:
    """Measure serving throughput/memory via an in-process vLLM ``LLM``."""

    name = "vllm"

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

        engine = opt["engine"]
        os.environ["VLLM_USE_V1"] = "0" if engine == "v0" else "1"
        # Needed so the worker memory reads above can be pickled through
        # collective_rpc on newer vLLM (in-process only; see runners/vllm.py
        # for the trust-boundary note).
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "The vLLM benchmarker needs vLLM installed and a CUDA GPU. "
                "Install with: pip install 'firefly-ml[vllm]'."
            ) from e

        max_model_len = opt["max_model_len"] or (cfg.input_len + cfg.output_len + 16)
        llm_kwargs = dict(
            model=model_id,
            dtype=canonical_dtype,
            enforce_eager=opt["enforce_eager"],
            max_model_len=max_model_len,
            gpu_memory_utilization=opt["gpu_memory_utilization"],
            enable_prefix_caching=opt["enable_prefix_caching"],
            seed=seed,
            trust_remote_code=opt["trust_remote_code"],
        )
        if quantization:
            llm_kwargs["quantization"] = quantization

        llm = LLM(**llm_kwargs)
        dispatch = llm.collective_rpc if engine == "v1" else llm.apply_model

        prompts = self._build_prompts(llm, cfg)
        prefill_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        full_params = SamplingParams(temperature=0.0, max_tokens=cfg.output_len, ignore_eos=True)

        for _ in range(cfg.warmup_iters):
            llm.generate(prompts, full_params, use_tqdm=False)

        weight_bytes = self._safe_rpc(dispatch, engine, "weight")
        self._safe_rpc(dispatch, engine, "reset")

        prefill_times: list[float] = []
        full_times: list[float] = []
        for _ in range(cfg.n_iters):
            t0 = time.perf_counter()
            llm.generate(prompts, prefill_params, use_tqdm=False)
            prefill_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            llm.generate(prompts, full_params, use_tqdm=False)
            full_times.append(time.perf_counter() - t0)

        peak_bytes = self._safe_rpc(dispatch, engine, "peak")

        metrics = summarize_timings(
            prefill_times, full_times,
            batch_size=cfg.batch_size, input_len=cfg.input_len, output_len=cfg.output_len,
        )
        return BenchmarkResult(
            engine="vllm",
            dtype=canonical_dtype,
            quantization=quantization,
            config=cfg,
            decode_throughput_tok_s=metrics["decode_throughput_tok_s"],
            prefill_throughput_tok_s=metrics["prefill_throughput_tok_s"],
            ttft_ms=metrics["ttft_ms"],
            e2e_latency_ms=metrics["e2e_latency_ms"],
            peak_memory_bytes=peak_bytes,
            weight_memory_bytes=weight_bytes,
            raw={"prefill_times_s": prefill_times, "full_times_s": full_times,
                 "max_model_len": max_model_len},
        )

    @staticmethod
    def _build_prompts(llm, cfg: BenchmarkConfig):
        """``batch_size`` distinct token-id prompts of exactly ``input_len`` tokens.

        Distinct prompts (a per-prompt offset on the token ids) keep prefix
        caching from collapsing the batch into one shared prefill. Token ids are
        kept inside the vocab and away from special-token lows.
        """
        try:
            vocab = llm.get_tokenizer().vocab_size
        except Exception:  # noqa: BLE001 — fall back to a safe vocab floor
            vocab = 30000
        lo, hi = 100, max(101, vocab - 1)
        span = hi - lo
        prompts = []
        for b in range(cfg.batch_size):
            ids = [lo + ((b * cfg.input_len + j) % span) for j in range(cfg.input_len)]
            prompts.append(ids)
        try:
            from vllm import TokensPrompt

            return [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
        except ImportError:
            return prompts  # older vLLM accepts a list[list[int]] of token ids

    @staticmethod
    def _safe_rpc(dispatch, engine: str, what: str) -> float | None:
        """Best-effort worker memory read; ``None`` if the engine won't expose it."""
        fns = {
            "v1": {"reset": _v1_reset_peak_memory, "peak": _v1_read_peak_memory, "weight": _v1_read_weight_bytes},
            "v0": {"reset": _reset_peak_memory, "peak": _read_peak_memory, "weight": _read_weight_bytes},
        }[engine]
        try:
            val = _unwrap(dispatch(fns[what]))
            return None if what == "reset" else float(val)
        except Exception:  # noqa: BLE001 — memory read is a nicety, not the measurement
            return None
