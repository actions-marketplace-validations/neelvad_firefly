"""Worker-side hook factory for the SGLang runner.

SGLang's ``ServerArgs.forward_hooks`` lets us hand the engine a list of hook
specs; the worker calls ``register_forward_hooks(model, specs)`` against the
live model. Each spec names an importable ``hook_factory`` (this module's
:func:`capture_hook_factory`) plus a ``config`` dict. The factory is called
once per spec with only its config, and the resulting hook is attached to the
spec's matched module(s) — so each spec targets exactly one module and carries
that module's tap name in ``config["name"]``.

Data leaves the worker by file: hooks accumulate captures into a process-global
dict; the ``final_norm`` spec (``config["flush"]=True``) ``torch.save``s the
whole dict to ``config["out_path"]`` after recording itself. SGLang's offline
engine scheduler runs in a child process that shares the filesystem with the
caller, so the runner reads that file once ``generate()`` returns.

Importable in the worker because ``firefly`` is installed in the SGLang image.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

# Worker-process-global accumulator. One capture pass per fresh engine process,
# so this never leaks across runs in production (each capture() spawns a new
# scheduler). Tests must clear it between cases.
_CAPTURES: dict[str, Any] = {}
_LOCK = threading.Lock()


def _reset() -> None:
    """Clear the accumulator (tests / defensive re-use)."""
    with _LOCK:
        _CAPTURES.clear()


def capture_hook_factory(config: dict[str, Any]) -> Callable:
    """Build a forward hook for one tap, per the SGLang hook-spec contract.

    ``config`` keys: ``name`` (tap name), ``out_path`` (where the flush writes),
    ``capture_input`` (capture the module's input[0] rather than output, for
    attn_heads o_proj-input taps), ``flush`` (write the whole accumulator after
    recording — set only on the terminal tap).
    """
    import torch

    name = config["name"]
    out_path = config["out_path"]
    capture_input = bool(config.get("capture_input", False))
    is_flush = bool(config.get("flush", False))

    def hook(_module, inputs, output):
        if capture_input:
            tensor = inputs[0] if isinstance(inputs, tuple) else inputs
        else:
            tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor) or tensor.dim() < 1:
            return
        # Prefill only: SGLang flattens to (num_tokens, hidden); the token axis
        # is > 1 for the prompt forward and 1 for each decode step. Capturing
        # prefill keeps the diff story aligned with the HF/vLLM runners.
        if tensor.shape[0] <= 1:
            return
        with _LOCK:
            _CAPTURES[name] = tensor.detach().cpu().contiguous()
            if is_flush:
                torch.save(dict(_CAPTURES), out_path)

    return hook
