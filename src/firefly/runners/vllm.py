"""vLLM runner: in-process activation capture for a real serving engine.

This is the library form of the capture logic that ``scripts/capture_vllm.py``
pioneered on Modal. Here it runs **in-process** — `firefly check --runner vllm`
imports vLLM, stands up an ``LLM`` on the local GPU, captures, and tears it
down. Modal is only an experiment harness for the blog repros; it is never on
a product run's path.

Requires ``pip install 'firefly[vllm]'`` and a CUDA GPU. The runner raises a
clear error if vLLM is missing.

Capture mechanics (unchanged from the validated script):

* Load with ``enforce_eager=True`` so forward hooks fire (CUDA graphs would
  skip them). Fine for a CI diagnostic; the always-on shadow path is
  ``firefly.shadow`` instead.
* Hooks store ``.detach()`` only; a single bulk d2h at drain time.
* The register/drain/probe functions are module-level so they pickle cleanly
  to vLLM's worker (``collective_rpc`` on V1, ``apply_model`` on V0).
* The attention backend is selected via the ``attention_backend`` engine arg
  on modern vLLM (the ``VLLM_ATTENTION_BACKEND`` env var was removed) and
  **verified** by reading the live ``Attention.impl`` class — a silently
  ignored selector would mislabel a same-backend comparison.

A ``--runner vllm`` reference and candidate should both be captured with this
runner: vLLM flattens batch/seq into a single token axis, so its tensor shapes
differ from the HF runner's padded batches. Compare like with like.
"""

from __future__ import annotations

import os
from pathlib import Path

from firefly.runners.base import CaptureResult

# ---------------------------------------------------------------------------
# Worker-side functions. Top-level so they pickle to vLLM's worker process.
# (Mirrors scripts/capture_vllm.py; kept in sync deliberately.)
# ---------------------------------------------------------------------------

_VLLM_ATTN_OUTPUT_PROJ_NAMES = ("o_proj", "out_proj", "dense", "c_proj")
_VLLM_NUM_HEADS_ATTRS = ("num_attention_heads", "n_head", "num_heads", "n_heads")


def _find_model_layout(model):
    """Probe the vLLM model tree → (layers, attn_attr, mlp_attr, final_norm).

    Llama/Gemma/Qwen/Mistral/Yi/SmolLM/Phi-3 use ``model.model.layers[i]``
    with ``self_attn`` / ``mlp`` and ``model.model.norm``. Falcon/BLOOM use
    ``model.transformer.h[i]`` with ``self_attention`` / ``mlp`` and
    ``transformer.ln_f``. MPT uses ``transformer.blocks[i]`` with ``attn`` /
    ``ffn``. Names are normalized downstream regardless of source family.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "self_attn", "mlp", model.model.norm

    if hasattr(model, "transformer"):
        t = model.transformer
        if hasattr(t, "h"):
            layers = t.h
            first = layers[0]
            attn_attr = "self_attention" if hasattr(first, "self_attention") else "attn"
            mlp_attr = "mlp" if hasattr(first, "mlp") else "ffn"
            norm = getattr(t, "ln_f", None) or getattr(t, "norm_f", None)
            return layers, attn_attr, mlp_attr, norm
        if hasattr(t, "blocks"):
            layers = t.blocks
            first = layers[0]
            attn_attr = "attn" if hasattr(first, "attn") else "self_attn"
            mlp_attr = "ffn" if hasattr(first, "ffn") else "mlp"
            norm = getattr(t, "norm_f", None) or getattr(t, "ln_f", None)
            return layers, attn_attr, mlp_attr, norm

    raise RuntimeError(
        f"Unknown model layout for {type(model).__name__}; "
        f"module attrs: {[a for a in dir(model) if not a.startswith('_')][:30]}"
    )


def _find_attn_output_proj(attn_module):
    for proj_name in _VLLM_ATTN_OUTPUT_PROJ_NAMES:
        proj = getattr(attn_module, proj_name, None)
        if proj is not None:
            return proj
    return None


def _register_capture_hooks_impl(model, capture_decode: bool, per_head: bool = False) -> int:
    """Install forward hooks on Firefly's tap points; stash on the model.

    Tap names match the HF convention (``layer.{i}.self_attn`` / ``.mlp`` /
    ``layer.{i}`` / ``final_norm``). ``capture_decode`` suffixes ``@prefill`` /
    ``@token_N``. ``per_head`` adds an input-capturing hook on each attention
    output projection (``layer.{i}.attn_heads``) — the concatenated per-head
    context vectors, the only place heads are separable before o_proj mixes
    them.
    """
    import torch

    captures: dict[str, torch.Tensor] = {}
    step_counters: dict[str, int] = {}
    handles: list = []

    def make_hook(name: str, capture_input: bool = False):
        def hook(_module, input_, output):
            if capture_input:
                tensor = input_[0] if isinstance(input_, tuple) else input_
            else:
                tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor) or tensor.dim() < 1:
                return
            leading = tensor.shape[0]
            if leading > 1:
                key = f"{name}@prefill" if capture_decode else name
                if key in captures:
                    return
                captures[key] = tensor.detach()
            elif capture_decode and leading == 1:
                i = step_counters.get(name, 0)
                step_counters[name] = i + 1
                captures[f"{name}@token_{i}"] = tensor.detach()

        return hook

    layers, attn_attr, mlp_attr, final_norm = _find_model_layout(model)
    for i, layer in enumerate(layers):
        attn = getattr(layer, attn_attr)
        handles.append(attn.register_forward_hook(make_hook(f"layer.{i}.self_attn")))
        if per_head:
            o_proj = _find_attn_output_proj(attn)
            if o_proj is not None:
                handles.append(
                    o_proj.register_forward_hook(
                        make_hook(f"layer.{i}.attn_heads", capture_input=True)
                    )
                )
        handles.append(getattr(layer, mlp_attr).register_forward_hook(make_hook(f"layer.{i}.mlp")))
        handles.append(layer.register_forward_hook(make_hook(f"layer.{i}")))
    handles.append(final_norm.register_forward_hook(make_hook("final_norm")))

    model._firefly_captures = captures
    model._firefly_handles = handles
    return len(handles)


def _register_capture_hooks(model) -> int:
    return _register_capture_hooks_impl(model, capture_decode=False)


def _register_capture_hooks_with_decode(model) -> int:
    return _register_capture_hooks_impl(model, capture_decode=True)


def _register_capture_hooks_per_head(model) -> int:
    return _register_capture_hooks_impl(model, capture_decode=False, per_head=True)


def _read_n_heads_impl(model) -> int:
    config = getattr(model, "config", None)
    if config is None:
        return 0
    for attr in _VLLM_NUM_HEADS_ATTRS:
        val = getattr(config, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    return 0


def _read_n_heads(model) -> int:
    return _read_n_heads_impl(model)


def _v1_read_n_heads(worker) -> int:
    return _read_n_heads_impl(worker.model_runner.model)


def _read_attn_impl_impl(model) -> str:
    """Report the attention implementation class actually in use (e.g.
    ``FlashAttentionImpl``, ``FlashInferImpl``) by reading the live model,
    not by trusting env vars / kwargs."""
    for m in model.modules():
        impl = getattr(m, "impl", None)
        if impl is not None and type(impl).__name__.endswith("Impl"):
            return type(impl).__name__
    return "unknown"


def _read_attn_impl(model) -> str:
    return _read_attn_impl_impl(model)


def _v1_read_attn_impl(worker) -> str:
    return _read_attn_impl_impl(worker.model_runner.model)


def _drain_captures(model) -> dict:
    captures = getattr(model, "_firefly_captures", {})
    cpu_tensors = {name: t.cpu().contiguous() for name, t in captures.items()}
    for handle in getattr(model, "_firefly_handles", []):
        handle.remove()
    if hasattr(model, "_firefly_captures"):
        del model._firefly_captures
    if hasattr(model, "_firefly_handles"):
        del model._firefly_handles
    return cpu_tensors


def _v1_register_capture_hooks(worker) -> int:
    return _register_capture_hooks(worker.model_runner.model)


def _v1_register_capture_hooks_with_decode(worker) -> int:
    return _register_capture_hooks_with_decode(worker.model_runner.model)


def _v1_register_capture_hooks_per_head(worker) -> int:
    return _register_capture_hooks_per_head(worker.model_runner.model)


def _v1_drain_captures(worker) -> bytes:
    """V1 ``collective_rpc`` summarizes tensor returns to dtype strings; ship
    a ``torch.save``'d dict as bytes so the payload survives the RPC bus."""
    import io

    import torch

    raw = _drain_captures(worker.model_runner.model)
    buf = io.BytesIO()
    torch.save(raw, buf)
    return buf.getvalue()


def _tap_order_key(name: str) -> tuple:
    """Forward order: self_attn < attn_heads < mlp < layer, then final_norm;
    within a tap, prefill before token_0..N."""
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
    import re

    m = re.match(r"layer\.(\d+)(?:\.(self_attn|attn_heads|mlp))?$", base)
    if m:
        layer_idx = int(m.group(1))
        within = {"self_attn": 0, "attn_heads": 1, "mlp": 2, None: 3}[m.group(2)]
        return (layer_idx, within, suffix_key, name)
    return (10**9 - 1, 0, suffix_key, name)


# ---------------------------------------------------------------------------
# Options parsing: the CLI passes --runner-opt key=value as strings.
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"1", "true", "yes", "on"}


def _parse_options(options: dict[str, str] | None) -> dict:
    """Coerce string runner-opts into the typed vLLM knobs, with defaults."""
    opts = dict(options or {})
    unknown = set(opts) - {
        "engine",
        "attention_backend",
        "max_seq_len",
        "gpu_memory_utilization",
        "capture_decode",
        "speculative_tokens",
    }
    if unknown:
        raise ValueError(
            f"Unknown vLLM runner option(s): {sorted(unknown)}. "
            f"Supported: engine, attention_backend, max_seq_len, "
            f"gpu_memory_utilization, capture_decode, speculative_tokens."
        )
    engine = opts.get("engine", "v1")
    if engine not in {"v0", "v1"}:
        raise ValueError(f"engine must be 'v0' or 'v1', got {engine!r}")
    return {
        "engine": engine,
        "attention_backend": opts.get("attention_backend", ""),
        "max_seq_len": int(opts.get("max_seq_len", "1024")),
        "gpu_memory_utilization": float(opts.get("gpu_memory_utilization", "0.9")),
        "capture_decode": opts.get("capture_decode", "false").lower() in _BOOL_TRUE,
        "speculative_tokens": int(opts.get("speculative_tokens", "0")),
    }


class VLLMRunner:
    """Capture via an in-process vLLM ``LLM`` (eager, forward hooks)."""

    name = "vllm"

    def capture(
        self,
        model_id: str,
        inputs_path: Path,
        *,
        device: str = "cuda",
        seed: int = 0,
        domain: str = "llm",
        dtype: str = "float32",
        per_head: bool = False,
        options: dict[str, str] | None = None,
    ) -> CaptureResult:
        if domain != "llm":
            raise ValueError(
                f"The vLLM runner only supports the 'llm' domain, got {domain!r}."
            )
        opt = _parse_options(options)
        if per_head and opt["capture_decode"]:
            raise ValueError(
                "per_head and capture_decode are mutually exclusive "
                "(per-head attribution is prefill-only)."
            )

        # Normalize dtype to vLLM's accepted canonical names (it rejects 'fp16').
        from firefly.capture import dtype_to_name, parse_dtype

        canonical_dtype = dtype_to_name(parse_dtype(dtype))

        prompts = self._load_prompts(inputs_path)
        engine = opt["engine"]

        os.environ["VLLM_USE_V1"] = "0" if engine == "v0" else "1"
        # Newer vLLM refuses to msgspec-serialize arbitrary callables through
        # collective_rpc without this; harmless on older versions.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        if opt["attention_backend"]:
            os.environ["VLLM_ATTENTION_BACKEND"] = opt["attention_backend"]

        try:
            import vllm
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "The vLLM runner needs vLLM installed and a CUDA GPU. "
                "Install with: pip install 'firefly[vllm]'."
            ) from e

        llm_kwargs = dict(
            model=model_id,
            dtype=canonical_dtype,
            enforce_eager=True,
            max_model_len=opt["max_seq_len"],
            gpu_memory_utilization=opt["gpu_memory_utilization"],
            seed=seed,
            trust_remote_code=True,
        )
        if opt["speculative_tokens"] > 0:
            llm_kwargs["speculative_config"] = {
                "method": "ngram",
                "num_speculative_tokens": opt["speculative_tokens"],
                "prompt_lookup_max": 4,
            }

        # Modern vLLM selects the backend via an engine arg (the env var was
        # removed); pass it when supported. The env var above covers old vLLM.
        if opt["attention_backend"]:
            from vllm.engine.arg_utils import EngineArgs

            if "attention_backend" in getattr(EngineArgs, "__dataclass_fields__", {}):
                try:
                    from vllm.v1.attention.backends.registry import (
                        AttentionBackendEnum,
                    )

                    llm_kwargs["attention_backend"] = AttentionBackendEnum[
                        opt["attention_backend"]
                    ]
                except (ImportError, KeyError):
                    llm_kwargs["attention_backend"] = opt["attention_backend"]

        llm = LLM(**llm_kwargs)

        capture_decode = opt["capture_decode"]
        if engine == "v0":
            register_fn = (
                _register_capture_hooks_per_head if per_head
                else _register_capture_hooks_with_decode if capture_decode
                else _register_capture_hooks
            )
            read_heads_fn, read_impl_fn, drain_fn = (
                _read_n_heads,
                _read_attn_impl,
                _drain_captures,
            )
            dispatch = llm.apply_model
        else:
            register_fn = (
                _v1_register_capture_hooks_per_head if per_head
                else _v1_register_capture_hooks_with_decode if capture_decode
                else _v1_register_capture_hooks
            )
            read_heads_fn, read_impl_fn, drain_fn = (
                _v1_read_n_heads,
                _v1_read_attn_impl,
                _v1_drain_captures,
            )
            dispatch = llm.collective_rpc

        dispatch(register_fn)

        attn_impl = _unwrap(dispatch(read_impl_fn))
        if opt["attention_backend"]:
            want = opt["attention_backend"].replace("_", "").lower()
            got = str(attn_impl).lower()
            if want.startswith("flashinfer") != got.startswith("flashinfer"):
                raise RuntimeError(
                    f"Requested attention backend {opt['attention_backend']!r} but "
                    f"the model is running {attn_impl!r} — the selector was ignored. "
                    f"Aborting so a same-backend comparison isn't mislabeled."
                )

        n_heads = _unwrap(dispatch(read_heads_fn)) if per_head else 0

        max_tokens = 8 if capture_decode else 1
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        llm.generate(prompts, params)

        captures = self._drain(dispatch, drain_fn, engine)

        # Order taps in forward order so the manifest's tap_points list (built
        # from captured.keys()) drives the diff/attribution walk correctly.
        captures = {k: captures[k] for k in sorted(captures, key=_tap_order_key)}

        head_counts: dict[str, int] = {}
        if per_head and n_heads > 0:
            for name, t in captures.items():
                if name.endswith(".attn_heads") and t.shape[-1] % n_heads == 0:
                    head_counts[name] = n_heads

        env = {
            "engine": "vllm",
            "vllm_version": vllm.__version__,
            "vllm_engine": engine,
            "attention_backend": opt["attention_backend"] or "auto",
            "attn_impl": str(attn_impl),
            "capture_decode": str(capture_decode),
            "device": "cuda",
        }

        return CaptureResult(
            tensors=captures,
            # vLLM's parallel-wrapped params don't hash like HF's; a stable
            # placeholder is enough to detect "same model intent".
            fingerprint=f"vllm-{vllm.__version__}:{model_id}",
            head_counts=head_counts,
            env=env,
            dtype=canonical_dtype,
        )

    @staticmethod
    def _load_prompts(inputs_path: Path) -> list[str]:
        import json

        with inputs_path.open() as f:
            data = json.load(f)
        texts = data.get("texts")
        if not texts:
            raise ValueError(
                f"Golden inputs {inputs_path} must contain a non-empty 'texts' list."
            )
        return list(texts)

    @staticmethod
    def _drain(dispatch, drain_fn, engine: str) -> dict:
        import io

        import torch

        raw = dispatch(drain_fn)
        payload = raw[0] if isinstance(raw, list) and raw else raw
        if engine == "v1":
            if not isinstance(payload, (bytes, bytearray)):
                raise RuntimeError(
                    f"Expected bytes from V1 drain, got {type(payload).__name__}. "
                    "vLLM's RPC return-summarization rules may have changed."
                )
            captures = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        else:
            captures = payload
        if captures and isinstance(next(iter(captures.values())), list):
            captures = {k: v[0] for k, v in captures.items() if v}
        return captures


def _unwrap(rpc_result):
    """vLLM dispatch returns a per-worker list; for TP=1 take the head."""
    return rpc_result[0] if isinstance(rpc_result, list) and rpc_result else rpc_result
