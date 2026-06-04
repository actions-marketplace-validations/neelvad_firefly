"""Capture activations from a model running under vLLM.

Writes a standard Firefly reference artifact (``weights.safetensors`` +
``manifest.json``) so the existing ``firefly check`` / ``compare`` pipeline
can diff two vLLM captures the same way it diffs two HF captures.

Multiple vLLM versions are supported via per-version Modal functions —
each pinned with its own image. Choose a version with ``--vllm-tag`` at
the local entrypoint; e.g. ``--vllm-tag 0.7.3`` vs ``--vllm-tag 0.8.5``.
Add a new version by extending ``_VLLM_VERSIONS`` and registering a new
``@app.function`` wrapper below.

Design notes (see also AGENTS.md):

  * Loaded with ``enforce_eager=True`` to disable CUDA graphs, otherwise
    forward hooks force graph breaks. Acceptable for the CI diagnostic
    use case; not acceptable for shadow-mode capture against production
    traffic (would need a custom op).
  * Hooks store ``.detach()`` only — tensors stay on GPU during forward,
    one bulk d2h via a second ``apply_model`` call at the end. Collapses
    ~90 per-tap sync points into 1.
  * Prefill-only (v1). Decode capture planned for v1.5 with per-position
    indexing.
  * ``VLLM_USE_V1=0`` forces vLLM's V0 engine because V1 in 0.8.5 has a
    broken ``apply_model`` path. V0 also works in 0.7.x.

Usage:
    uv run modal run scripts/capture_vllm.py --vllm-tag 0.8.5
    uv run modal run scripts/capture_vllm.py --vllm-tag 0.7.3 --out vllm_run_0_7_3
"""

from __future__ import annotations

import os
import re

import modal

app = modal.App("firefly-capture-vllm")

_HF_TOKEN_SET = bool(os.environ.get("HF_TOKEN"))
_HF_SECRETS = (
    [modal.Secret.from_local_environ(["HF_TOKEN"])] if _HF_TOKEN_SET else []
)

# ---------------------------------------------------------------------------
# Per-version image definitions.
#
# vLLM's transformers requirement narrows as releases age; pin both together
# so pip never picks an incompatible mid-resolution transformers. If a new
# version's pip resolve breaks, the fastest fix is to set the transformers
# pin to a version known to work with that vLLM release (check vLLM's setup
# constraints in its github repo for the release tag).
# ---------------------------------------------------------------------------

_VLLM_VERSIONS: dict[str, dict[str, str]] = {
    "0.7.3": {"vllm": "vllm==0.7.3", "transformers": "transformers==4.48.3"},
    "0.8.5": {"vllm": "vllm==0.8.5", "transformers": "transformers==4.51.3"},
}


def _make_image(pins: dict[str, str]) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            pins["vllm"],
            pins["transformers"],
            "huggingface_hub>=0.24",
            "safetensors>=0.4",
        )
        .add_local_python_source("firefly")
    )


# ---------------------------------------------------------------------------
# Worker-side functions (must be top-level so they pickle to vLLM's worker).
# ---------------------------------------------------------------------------


def _register_capture_hooks(model) -> int:
    """Inside vLLM worker: install forward hooks on Firefly's tap points.

    Tap names follow the existing LLM-domain convention: per-layer
    ``self_attn``, ``mlp``, and full-layer (residual stream) plus a
    terminal ``final_norm``. Names match the HF reference convention so
    captures from this script are directly comparable to HF captures.
    """
    import torch

    captures: dict[str, torch.Tensor] = {}
    handles: list = []

    def make_hook(name: str):
        def hook(_module, _input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            # vLLM's V0 engine in eager mode runs forward with leading dim =
            # total prompt tokens during prefill, and = 1 per step during
            # decode. We capture prefill only; decode is v1.5 work.
            if tensor.dim() < 1 or tensor.shape[0] <= 1:
                return
            # First prefill call wins — dedup so a second prefill (e.g.
            # speculative re-run) doesn't overwrite.
            if name in captures:
                return
            captures[name] = tensor.detach()

        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(f"layer.{i}.self_attn")))
        handles.append(layer.mlp.register_forward_hook(make_hook(f"layer.{i}.mlp")))
        handles.append(layer.register_forward_hook(make_hook(f"layer.{i}")))
    handles.append(model.model.norm.register_forward_hook(make_hook("final_norm")))

    model._firefly_captures = captures
    model._firefly_handles = handles
    return len(handles)


def _drain_captures(model) -> dict:
    """Inside vLLM worker: bulk d2h, remove hooks, return tensor dict."""
    captures = getattr(model, "_firefly_captures", {})
    cpu_tensors = {name: t.cpu().contiguous() for name, t in captures.items()}

    for handle in getattr(model, "_firefly_handles", []):
        handle.remove()
    if hasattr(model, "_firefly_captures"):
        del model._firefly_captures
    if hasattr(model, "_firefly_handles"):
        del model._firefly_handles

    return cpu_tensors


# ---------------------------------------------------------------------------
# Shared capture body. Lives at module level so per-version Modal functions
# can call it; the actual vllm import happens *inside*, after VLLM_USE_V1 is
# set, so the env var takes effect.
# ---------------------------------------------------------------------------


def _do_capture(
    model_id: str,
    prompt: str,
    max_seq_len: int,
    dtype: str,
    attention_backend: str = "",
) -> dict:
    os.environ["VLLM_USE_V1"] = "0"
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend

    import vllm
    from vllm import LLM, SamplingParams

    print(
        f"vllm version: {vllm.__version__}  "
        f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}  "
        f"VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND', '(auto)')}"
    )
    print(f"Loading {model_id} dtype={dtype}")

    llm = LLM(
        model=model_id,
        dtype=dtype,
        enforce_eager=True,
        max_model_len=max_seq_len,
        gpu_memory_utilization=0.4,
    )

    n_hooks_raw = llm.apply_model(_register_capture_hooks)
    n_hooks = n_hooks_raw[0] if isinstance(n_hooks_raw, list) and n_hooks_raw else n_hooks_raw
    print(f"Registered {n_hooks} forward hooks")

    # max_tokens=1 keeps decode to a single step; the prefill filter in the
    # hook drops that single-token decode forward pass anyway.
    params = SamplingParams(temperature=0.0, max_tokens=1)
    _ = llm.generate([prompt], params)

    captures_raw = llm.apply_model(_drain_captures)
    captures = captures_raw[0] if isinstance(captures_raw, list) and captures_raw else captures_raw

    print(f"\nCaptured {len(captures)} taps:")
    for name in sorted(captures.keys(), key=_tap_order_key):
        t = captures[name]
        print(f"  {name:<32} shape={tuple(t.shape)} dtype={str(t.dtype).replace('torch.', '')}")

    return {
        "vllm_version": vllm.__version__,
        "model_id": model_id,
        "dtype": dtype,
        "prompt": prompt,
        "attention_backend": attention_backend or "auto",
        "captures": captures,
    }


def _tap_order_key(name: str) -> tuple:
    """Sort taps in forward order: self_attn < mlp < layer-level, then final_norm."""
    if name == "final_norm":
        return (10**9, 0, name)
    m = re.match(r"layer\.(\d+)(?:\.(self_attn|mlp))?$", name)
    if m:
        layer_idx = int(m.group(1))
        sub = m.group(2)
        within = {"self_attn": 0, "mlp": 1, None: 2}[sub]
        return (layer_idx, within, name)
    return (10**9 - 1, 0, name)


# ---------------------------------------------------------------------------
# Per-version Modal functions. Each pins its own image; the body is shared.
# Add new versions by appending to _VLLM_VERSIONS above and registering here.
# ---------------------------------------------------------------------------


@app.function(image=_make_image(_VLLM_VERSIONS["0.7.3"]), gpu="A10G", timeout=900, secrets=_HF_SECRETS)
def capture_at_v_0_7_3(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox jumps over the lazy dog",
    max_seq_len: int = 256,
    dtype: str = "bfloat16",
    attention_backend: str = "",
) -> dict:
    return _do_capture(model_id, prompt, max_seq_len, dtype, attention_backend)


@app.function(image=_make_image(_VLLM_VERSIONS["0.8.5"]), gpu="A10G", timeout=900, secrets=_HF_SECRETS)
def capture_at_v_0_8_5(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox jumps over the lazy dog",
    max_seq_len: int = 256,
    dtype: str = "bfloat16",
    attention_backend: str = "",
) -> dict:
    return _do_capture(model_id, prompt, max_seq_len, dtype, attention_backend)


_CAPTURE_BY_TAG = {
    "0.7.3": capture_at_v_0_7_3,
    "0.8.5": capture_at_v_0_8_5,
}


# ---------------------------------------------------------------------------
# Local entrypoint.
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    vllm_tag: str = "0.8.5",
    model: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox jumps over the lazy dog",
    gpu: str = "A10G",
    dtype: str = "bfloat16",
    attention_backend: str = "",
    out: str = "",
) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from firefly.reference import ReferenceManifest, write_reference

    if vllm_tag not in _CAPTURE_BY_TAG:
        raise SystemExit(
            f"Unknown --vllm-tag {vllm_tag!r}. Available: {sorted(_CAPTURE_BY_TAG)}"
        )

    if _HF_TOKEN_SET:
        print("HF_TOKEN found in local env — forwarding to GPU container.")
    backend_label = attention_backend or "(auto)"
    print(
        f"Launching {gpu} capture: vllm={vllm_tag}, model={model}, "
        f"dtype={dtype}, attention_backend={backend_label}"
    )

    fn = _CAPTURE_BY_TAG[vllm_tag]
    result = fn.with_options(gpu=gpu).remote(
        model_id=model, prompt=prompt, dtype=dtype, attention_backend=attention_backend,
    )

    vllm_version = result["vllm_version"]
    backend_used = result["attention_backend"]
    captures: dict = result["captures"]

    if not out:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backend_tag = f"_{backend_used.lower()}" if backend_used != "auto" else ""
        out = (
            f"vllm_{vllm_version.replace('.', '_')}{backend_tag}_"
            f"{gpu.lower().replace('-', '_')}_{timestamp}"
        )
    out_dir = Path(__file__).parent / "results" / out

    tap_points = sorted(captures.keys(), key=_tap_order_key)

    manifest = ReferenceManifest(
        model_id=model,
        # Placeholder fingerprint — vLLM's parallel-wrapped params don't hash
        # the same way HF's do. v1 only needs to detect same-model intent;
        # cross-version weight verification is future work.
        model_fingerprint=f"vllm-{vllm_version}:{model}",
        tap_points=tap_points,
        shapes={name: list(captures[name].shape) for name in tap_points},
        dtypes={name: str(captures[name].dtype).replace("torch.", "") for name in tap_points},
        captured_at=datetime.now(UTC).isoformat(),
        env={
            "engine": "vllm",
            "vllm_version": vllm_version,
            "attention_backend": backend_used,
            "gpu": gpu,
            "prompt": prompt,
        },
        domain="llm",
        dtype=dtype,
    )

    write_reference(out_dir, manifest, captures)

    print(f"\nWrote vLLM reference artifact to {out_dir}")
    print(
        f"  {len(captures)} taps, vllm={vllm_version}, dtype={dtype}, "
        f"attention_backend={backend_used}"
    )
