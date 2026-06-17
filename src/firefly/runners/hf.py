"""HF-transformers runner: the default capture backend.

Wraps the eager forward-hook capture path that has always been Firefly's
core. This is a thin adapter over the helpers in :mod:`firefly.capture` —
the logic didn't move, it just lives behind the :class:`~firefly.runners.base.Runner`
interface now so other engines can slot in beside it.
"""

from __future__ import annotations

from pathlib import Path

from firefly.capture import (
    dtype_to_name,
    fingerprint_model,
    load_golden_inputs,
    load_model_and_tokenizer,
    num_attention_heads,
    parse_dtype,
    run_capture,
)
from firefly.determinism import set_deterministic
from firefly.reference import capture_env
from firefly.runners.base import CaptureResult


class HFRunner:
    """Capture via HF transformers + ``register_forward_hook`` (eager)."""

    name = "hf"

    def capture(
        self,
        model_id: str,
        inputs_path: Path,
        *,
        device: str = "cpu",
        seed: int = 0,
        domain: str = "llm",
        dtype: str = "float32",
        per_head: bool = False,
        options: dict[str, str] | None = None,
    ) -> CaptureResult:
        torch_dtype = parse_dtype(dtype)
        set_deterministic(seed=seed)
        model, tokenizer = load_model_and_tokenizer(
            model_id, device=device, dtype=torch_dtype
        )

        # Fingerprint the model BEFORE any quantization. Quantization is a
        # transform of the *same* model, not a different one — so a quant
        # candidate should still match its fp baseline's fingerprint (the diff
        # we go on to measure is the quantization itself, not a model swap). A
        # genuinely different candidate still mismatches and is caught upstream.
        fingerprint = fingerprint_model(model)

        # ``quantize=<scheme>`` (HF runner only) applies real torchao quant in
        # place after load — turning quantization into a candidate in the
        # standard capture/compare/attribute pipeline. ``group_size`` (int4
        # only) defaults to 32 to tolerate non-128-divisible weight dims.
        scheme = (options or {}).get("quantize")
        if scheme:
            from firefly.quant_torchao import quantize_model

            group_size = int((options or {}).get("group_size", 32))
            quantize_model(model, scheme=scheme, group_size=group_size)

        batch = load_golden_inputs(inputs_path, tokenizer, device)
        captured = run_capture(model, batch, domain=domain, per_head=per_head)

        head_counts: dict[str, int] = {}
        if per_head:
            n_heads = num_attention_heads(model)
            if n_heads is not None:
                for name, t in captured.items():
                    if name.endswith(".attn_heads") and t.shape[-1] % n_heads == 0:
                        head_counts[name] = n_heads

        return CaptureResult(
            tensors=captured,
            fingerprint=fingerprint,
            head_counts=head_counts,
            env=capture_env(),
            dtype=dtype_to_name(torch_dtype),
        )
