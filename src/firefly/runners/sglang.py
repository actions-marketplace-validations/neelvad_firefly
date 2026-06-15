"""SGLang runner: in-process activation capture via SGLang's forward_hooks.

Unlike the vLLM runner (which leans on apply_model/collective_rpc), SGLang
ships a first-class extension point: ``ServerArgs.forward_hooks``. We hand
``sgl.Engine(forward_hooks=[...])`` a list of hook specs; the worker registers
them against the live model via ``register_forward_hooks``. Each spec is an
exact-match pattern for one tap, with the tap name in its config; our worker-
side factory (:mod:`firefly.runners._sglang_hooks`) captures and, on the
terminal tap, flushes everything to a file the caller reads back.

We get the layer/head counts from ``AutoConfig`` (config-only, no weights) to
generate the per-tap specs up front, since the model isn't loaded in this
process. Requires ``pip install 'firefly[sglang]'`` and a CUDA GPU.

Like vLLM, a reference and its candidates should use the same runner — SGLang
flattens batch/seq into one token axis, so shapes differ from the HF runner's
padded batches.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from firefly.runners._common import tap_order_key
from firefly.runners.base import CaptureResult

_HOOK_FACTORY = "firefly.runners._sglang_hooks.capture_hook_factory"
_BOOL_TRUE = {"1", "true", "yes", "on"}


def _parse_options(options: dict[str, str] | None) -> dict:
    """Coerce string runner-opts into typed SGLang knobs, with defaults."""
    opts = dict(options or {})
    known = {"attention_backend", "mem_fraction_static", "tp_size", "disable_cuda_graph"}
    unknown = set(opts) - known
    if unknown:
        raise ValueError(
            f"Unknown SGLang runner option(s): {sorted(unknown)}. "
            f"Supported: {sorted(known)}."
        )
    return {
        "attention_backend": opts.get("attention_backend", ""),
        "mem_fraction_static": float(opts.get("mem_fraction_static", "0.85")),
        "tp_size": int(opts.get("tp_size", "1")),
        # Capture is prefill (eager in SGLang); disabling CUDA graphs is belt-
        # and-suspenders so forward hooks always fire.
        "disable_cuda_graph": opts.get("disable_cuda_graph", "true").lower() in _BOOL_TRUE,
    }


def _model_dims(model_id: str) -> tuple[int, int]:
    """(num_layers, num_attention_heads) from the HF config (no weights)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if not n_layers:
        raise ValueError(f"Could not read num_hidden_layers from config of {model_id!r}")
    return int(n_layers), int(n_heads or 0)


def build_hook_specs(n_layers: int, per_head: bool, out_path: str) -> list[dict]:
    """Per-tap hook specs in our LLM tap convention (Llama-family layout).

    Exact-match patterns (one module each) so the shared-per-spec hook is
    effectively per-module and carries its name in config. ``final_norm``
    is the flush tap (fires last in the forward).
    """
    def spec(name: str, pattern: str, **cfg) -> dict:
        return {
            "name": name,
            "target_modules": [pattern],
            "hook_factory": _HOOK_FACTORY,
            "config": {"name": name, "out_path": out_path, **cfg},
        }

    specs: list[dict] = []
    for i in range(n_layers):
        specs.append(spec(f"layer.{i}.self_attn", f"model.layers.{i}.self_attn"))
        if per_head:
            specs.append(
                spec(
                    f"layer.{i}.attn_heads",
                    f"model.layers.{i}.self_attn.o_proj",
                    capture_input=True,
                )
            )
        specs.append(spec(f"layer.{i}.mlp", f"model.layers.{i}.mlp"))
        specs.append(spec(f"layer.{i}", f"model.layers.{i}"))
    specs.append(spec("final_norm", "model.norm", flush=True))
    return specs


def _load_prompts(inputs_path: Path) -> list[str]:
    with inputs_path.open() as f:
        data = json.load(f)
    texts = data.get("texts")
    if not texts:
        raise ValueError(
            f"Golden inputs {inputs_path} must contain a non-empty 'texts' list."
        )
    return list(texts)


class SGLangRunner:
    """Capture via an in-process SGLang ``Engine`` + ``forward_hooks``."""

    name = "sglang"

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
                f"The SGLang runner only supports the 'llm' domain, got {domain!r}."
            )
        opt = _parse_options(options)

        from firefly.capture import dtype_to_name, parse_dtype

        canonical_dtype = dtype_to_name(parse_dtype(dtype))
        prompts = _load_prompts(inputs_path)
        n_layers, n_heads = _model_dims(model_id)

        out_path = str(Path(tempfile.mkdtemp()) / "captures.pt")
        specs = build_hook_specs(n_layers, per_head, out_path)

        try:
            import sglang as sgl
        except ImportError as e:
            raise ImportError(
                "The SGLang runner needs SGLang installed and a CUDA GPU. "
                "Install with: pip install 'firefly[sglang]'."
            ) from e

        engine_kwargs = dict(
            model_path=model_id,
            dtype=canonical_dtype,
            forward_hooks=specs,
            mem_fraction_static=opt["mem_fraction_static"],
            tp_size=opt["tp_size"],
            random_seed=seed,
            disable_cuda_graph=opt["disable_cuda_graph"],
        )
        if opt["attention_backend"]:
            engine_kwargs["attention_backend"] = opt["attention_backend"]

        engine = sgl.Engine(**engine_kwargs)
        try:
            engine.generate(prompts, {"temperature": 0.0, "max_new_tokens": 1})
        finally:
            engine.shutdown()

        import torch

        if not Path(out_path).exists():
            raise RuntimeError(
                "SGLang capture produced no output file — the forward hooks "
                "likely didn't fire (check that the model uses the "
                "'model.layers.{i}' layout and that this SGLang build supports "
                "ServerArgs.forward_hooks)."
            )
        raw = torch.load(out_path, map_location="cpu", weights_only=True)
        captures = {k: raw[k] for k in sorted(raw, key=tap_order_key)}

        head_counts: dict[str, int] = {}
        if per_head and n_heads > 0:
            for name, t in captures.items():
                if name.endswith(".attn_heads") and t.shape[-1] % n_heads == 0:
                    head_counts[name] = n_heads

        env = {
            "engine": "sglang",
            "sglang_version": getattr(sgl, "__version__", "unknown"),
            "attention_backend": opt["attention_backend"] or "auto",
            "device": "cuda",
        }

        return CaptureResult(
            tensors=captures,
            fingerprint=f"sglang-{env['sglang_version']}:{model_id}",
            head_counts=head_counts,
            env=env,
            dtype=canonical_dtype,
        )
