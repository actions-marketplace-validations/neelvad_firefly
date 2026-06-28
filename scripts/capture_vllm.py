"""Capture activations from a model running under vLLM.

Writes a standard Firefly reference artifact (``weights.safetensors`` +
``manifest.json``) so the existing ``firefly check`` / ``compare`` pipeline
can diff two vLLM captures the same way it diffs two HF captures.

Multiple vLLM versions are supported via per-version Modal functions —
each pinned with its own image. Choose a version with ``--vllm-tag`` at
the local entrypoint; e.g. ``--vllm-tag 0.7.3`` vs ``--vllm-tag 0.8.5``.
Add a new version by extending ``_VLLM_VERSIONS`` and registering a new
``@app.function`` wrapper below.

Design notes:

  * Loaded with ``enforce_eager=True`` to disable CUDA graphs, otherwise
    forward hooks force graph breaks. Acceptable for the CI diagnostic /
    capture use case (a candidate-vs-reference comparison), which is what
    this script does.
  * Hooks store ``.detach()`` only — tensors stay on GPU during forward,
    one bulk d2h via a second ``apply_model`` call at the end. Collapses
    ~90 per-tap sync points into 1.
  * Prefill by default; decode capture (per-position ``@token_N``
    indexing) via ``capture_decode=True``. Per-head capture via
    ``per_head=True``.
  * ``VLLM_USE_V1=0`` forces vLLM's V0 engine because V1 in 0.8.5 has a
    broken ``apply_model`` path. V0 also works in 0.7.x.

Usage:
    # SmolLM-135M on A10G (default)
    uv run modal run scripts/capture_vllm.py --vllm-tag 0.8.5

    # Llama-3-8B on A100-40GB. First run populates the shared HF-cache
    # Volume (~5 min for 16GB); subsequent runs reuse it.
    uv run modal run scripts/capture_vllm.py \
        --vllm-tag 0.8.5 \
        --model meta-llama/Llama-3.1-8B \
        --gpu A100-40GB \
        --gpu-memory-utilization 0.7
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-capture-vllm")

_HF_TOKEN_SET = bool(os.environ.get("HF_TOKEN"))
_HF_SECRETS = (
    [modal.Secret.from_local_environ(["HF_TOKEN"])] if _HF_TOKEN_SET else []
)

# Shared HF cache across all vllm-version × GPU runs. First run downloads;
# subsequent runs hit the cache directly. Cheap (cents/month for ~20GB)
# relative to per-run GPU-seconds. The Volume is auto-committed at function
# exit, so a Llama-3-8B run on Monday warms the cache for a SmolLM run on
# Tuesday and vice-versa.
_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
_HF_CACHE_MOUNT = "/root/.cache/huggingface"

# ---------------------------------------------------------------------------
# Per-version image definitions.
#
# vLLM's transformers requirement narrows as releases age; pin both together
# so pip never picks an incompatible mid-resolution transformers. If a new
# version's pip resolve breaks, the fastest fix is to set the transformers
# pin to a version known to work with that vLLM release (check vLLM's setup
# constraints in its github repo for the release tag).
# ---------------------------------------------------------------------------

_VLLM_VERSIONS: dict[str, dict] = {
    "0.7.3": {
        "vllm": "vllm==0.7.3",
        "transformers": "transformers==4.48.3",
        "extras": [],
    },
    "0.8.5": {
        "vllm": "vllm==0.8.5",
        "transformers": "transformers==4.51.3",
        "extras": [],
    },
    # FLASHINFER variant of 0.8.5. Install-path history:
    #
    #   1. debian_slim + `pip install flashinfer-python` → failed at
    #      vLLM init with "CUDA_HOME not set". Flashinfer's wheel
    #      needs the CUDA toolkit, not just torch's bundled runtime.
    #
    #   2. from_registry("vllm/vllm-openai:v0.8.5") → failed during
    #      image build. The image is Python 3.12; Modal's automatic
    #      runtime-dep install resolves an aiohttp old enough to lack
    #      a 3.12 wheel, and its C extension fails to compile against
    #      3.12's hidden PyLongObject layout. Pre-installing a newer
    #      aiohttp didn't stick — Modal does a downgrade pass over it.
    #
    #   3. Current: nvidia/cuda devel image + add_python=3.11 + we
    #      pip-install vllm + flashinfer ourselves. CUDA toolkit at
    #      /usr/local/cuda → flashinfer's runtime CUDA_HOME check
    #      passes. Python 3.11 → no Modal-deps vs 3.12 collisions.
    #      All package versions are under our control.
    "0.8.5-fi": {
        "base_image": "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        "vllm": "vllm==0.8.5",
        "transformers": "transformers==4.51.3",
        "flashinfer_index": "https://flashinfer.ai/whl/cu124/torch2.6/",
    },
    # Recent-vLLM variant for "does the bug still reproduce on current
    # FlashInfer?" checks. Modern vLLM declares flashinfer-python as a
    # regular dependency with real PyPI wheels, so no custom index is
    # needed — but FlashInfer JIT-compiles some kernels at runtime, so
    # the CUDA-devel base (nvcc + CUDA_HOME) is still required. V0
    # engine no longer exists at this version; run with --engine v1.
    "latest-fi": {
        "base_image": "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        "vllm": "vllm==0.22.1",
        "transformers": "transformers>=4.55",
    },
}


def _make_image(pins: dict) -> modal.Image:
    if pins.get("base_image"):
        # nvidia/cuda devel base: full CUDA toolkit at /usr/local/cuda,
        # no Python preinstalled. add_python=3.11 gives us a clean
        # Python under our control. CUDA_HOME export is what makes the
        # later flashinfer import succeed.
        base_pkgs = [
            pins["vllm"],
            pins["transformers"],
            "huggingface_hub>=0.24",
            "safetensors>=0.4",
        ]
        image = (
            modal.Image.from_registry(pins["base_image"], add_python="3.11")
            .env({"CUDA_HOME": "/usr/local/cuda"})
            .pip_install(*base_pkgs)
        )
        if pins.get("flashinfer_index"):
            # --no-deps because flashinfer's main runtime dep (torch)
            # is already installed via vllm. -i (not --extra-index-url)
            # to force pip past PyPI's stub package for this name.
            image = image.run_commands(
                f"pip install --no-deps flashinfer-python -i {pins['flashinfer_index']}"
            )
        return image.add_local_python_source("firefly")

    base_pkgs = [
        pins["vllm"],
        pins["transformers"],
        "huggingface_hub>=0.24",
        "safetensors>=0.4",
    ]
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*base_pkgs, *pins.get("extras", []))
        .add_local_python_source("firefly")
    )


# ---------------------------------------------------------------------------
# Capture body: delegates to the in-process firefly VLLMRunner. The worker-side
# hook/probe/drain logic and the capture orchestration now live in
# src/firefly/runners/vllm.py; this script is just the Modal harness around it
# (per-version images + entrypoint). Single source of truth for the capture
# logic is the library.
# ---------------------------------------------------------------------------


def _capture_via_runner(
    model_id: str,
    prompt: str,
    max_seq_len: int,
    dtype: str,
    attention_backend: str = "",
    engine: str = "v0",
    capture_decode: bool = False,
    max_tokens: int = 8,
    n_prompts: int = 1,
    speculative_tokens: int = 0,
    gpu_memory_utilization: float = 0.4,
    per_head: bool = False,
) -> dict:
    """Runs in the GPU container: capture via the library VLLMRunner, return a
    dict the local entrypoint turns into a reference artifact."""
    import json
    import tempfile
    from pathlib import Path as _Path

    from firefly.runners.vllm import VLLMRunner

    work = _Path(tempfile.mkdtemp())
    inputs = work / "golden.json"
    inputs.write_text(json.dumps({"texts": [prompt] * max(1, n_prompts)}))

    options = {
        "engine": engine,
        "attention_backend": attention_backend,
        "max_seq_len": str(max_seq_len),
        "gpu_memory_utilization": str(gpu_memory_utilization),
        "capture_decode": str(capture_decode),
        "max_tokens": str(max_tokens),
        "speculative_tokens": str(speculative_tokens),
    }
    result = VLLMRunner().capture(
        model_id, inputs, dtype=dtype, per_head=per_head, options=options
    )
    env = result.env
    print(
        f"vllm={env.get('vllm_version')} engine={env.get('vllm_engine')} "
        f"backend={env.get('attention_backend')} impl={env.get('attn_impl')} "
        f"taps={len(result.tensors)} head_taps={len(result.head_counts)}"
    )
    return {
        "vllm_version": env.get("vllm_version", "unknown"),
        "attention_backend": env.get("attention_backend", "auto"),
        "engine": env.get("vllm_engine", engine),
        "capture_decode": capture_decode,
        "max_tokens": max_tokens if capture_decode else 1,
        "n_prompts": max(1, n_prompts),
        "speculative_tokens": speculative_tokens,
        "per_head": per_head,
        "n_heads": next(iter(result.head_counts.values()), 0),
        "attn_impl": env.get("attn_impl", "unknown"),
        "fingerprint": result.fingerprint,
        "head_counts": result.head_counts,
        "captures": result.tensors,
    }


# ---------------------------------------------------------------------------
# Per-version Modal functions. Each pins its own image; the body is shared.
# Add new versions by appending to _VLLM_VERSIONS above and registering here.
# ---------------------------------------------------------------------------


@app.function(
    image=_make_image(_VLLM_VERSIONS["0.7.3"]),
    gpu="A10G",
    timeout=1800,
    secrets=_HF_SECRETS,
    volumes={_HF_CACHE_MOUNT: _HF_CACHE},
)
def capture_at_v_0_7_3(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox jumps over the lazy dog",
    max_seq_len: int = 1024,
    dtype: str = "bfloat16",
    attention_backend: str = "",
    engine: str = "v0",
    capture_decode: bool = False,
    max_tokens: int = 8,
    n_prompts: int = 1,
    speculative_tokens: int = 0,
    gpu_memory_utilization: float = 0.4,
    per_head: bool = False,
) -> dict:
    return _capture_via_runner(
        model_id, prompt, max_seq_len, dtype,
        attention_backend, engine, capture_decode, max_tokens, n_prompts,
        speculative_tokens, gpu_memory_utilization, per_head,
    )


@app.function(
    image=_make_image(_VLLM_VERSIONS["0.8.5"]),
    gpu="A10G",
    timeout=1800,
    secrets=_HF_SECRETS,
    volumes={_HF_CACHE_MOUNT: _HF_CACHE},
)
def capture_at_v_0_8_5(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox jumps over the lazy dog",
    max_seq_len: int = 1024,
    dtype: str = "bfloat16",
    attention_backend: str = "",
    engine: str = "v0",
    capture_decode: bool = False,
    max_tokens: int = 8,
    n_prompts: int = 1,
    speculative_tokens: int = 0,
    gpu_memory_utilization: float = 0.4,
    per_head: bool = False,
) -> dict:
    return _capture_via_runner(
        model_id, prompt, max_seq_len, dtype,
        attention_backend, engine, capture_decode, max_tokens, n_prompts,
        speculative_tokens, gpu_memory_utilization, per_head,
    )


@app.function(
    image=_make_image(_VLLM_VERSIONS["0.8.5-fi"]),
    gpu="A10G",
    timeout=1800,
    secrets=_HF_SECRETS,
    volumes={_HF_CACHE_MOUNT: _HF_CACHE},
)
def capture_at_v_0_8_5_fi(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox jumps over the lazy dog",
    max_seq_len: int = 1024,
    dtype: str = "bfloat16",
    attention_backend: str = "",
    engine: str = "v0",
    capture_decode: bool = False,
    max_tokens: int = 8,
    n_prompts: int = 1,
    speculative_tokens: int = 0,
    gpu_memory_utilization: float = 0.4,
    per_head: bool = False,
) -> dict:
    return _capture_via_runner(
        model_id, prompt, max_seq_len, dtype,
        attention_backend, engine, capture_decode, max_tokens, n_prompts,
        speculative_tokens, gpu_memory_utilization, per_head,
    )


@app.function(
    image=_make_image(_VLLM_VERSIONS["latest-fi"]),
    gpu="A10G",
    timeout=1800,
    secrets=_HF_SECRETS,
    volumes={_HF_CACHE_MOUNT: _HF_CACHE},
)
def capture_at_latest_fi(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox jumps over the lazy dog",
    max_seq_len: int = 1024,
    dtype: str = "bfloat16",
    attention_backend: str = "",
    engine: str = "v1",
    capture_decode: bool = False,
    max_tokens: int = 8,
    n_prompts: int = 1,
    speculative_tokens: int = 0,
    gpu_memory_utilization: float = 0.4,
    per_head: bool = False,
) -> dict:
    return _capture_via_runner(
        model_id, prompt, max_seq_len, dtype,
        attention_backend, engine, capture_decode, max_tokens, n_prompts,
        speculative_tokens, gpu_memory_utilization, per_head,
    )


_CAPTURE_BY_TAG = {
    "0.7.3": capture_at_v_0_7_3,
    "0.8.5": capture_at_v_0_8_5,
    "0.8.5-fi": capture_at_v_0_8_5_fi,
    "latest-fi": capture_at_latest_fi,
}


# ---------------------------------------------------------------------------
# Local entrypoint.
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    vllm_tag: str = "0.8.5",
    model: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox jumps over the lazy dog",
    prompt_file: str = "",
    gpu: str = "A10G",
    dtype: str = "bfloat16",
    attention_backend: str = "",
    engine: str = "v0",
    capture_decode: bool = False,
    max_tokens: int = 8,
    n_prompts: int = 1,
    speculative_tokens: int = 0,
    gpu_memory_utilization: float = 0.4,
    max_seq_len: int = 1024,
    per_head: bool = False,
    out: str = "",
) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from firefly.reference import ReferenceManifest, write_reference

    if vllm_tag not in _CAPTURE_BY_TAG:
        raise SystemExit(
            f"Unknown --vllm-tag {vllm_tag!r}. Available: {sorted(_CAPTURE_BY_TAG)}"
        )

    if engine not in {"v0", "v1"}:
        raise SystemExit(f"--engine must be 'v0' or 'v1', got {engine!r}")

    # --prompt-file overrides --prompt when both are set; lets long prompts
    # live in version-controlled files rather than CLI arg strings.
    if prompt_file:
        prompt = Path(prompt_file).read_text().strip()

    if _HF_TOKEN_SET:
        print("HF_TOKEN found in local env — forwarding to GPU container.")
    backend_label = attention_backend or "(auto)"
    print(
        f"Launching {gpu} capture: vllm={vllm_tag}, engine={engine}, "
        f"model={model}, dtype={dtype}, attention_backend={backend_label}, "
        f"capture_decode={capture_decode}, max_tokens={max_tokens}"
    )

    fn = _CAPTURE_BY_TAG[vllm_tag]
    result = fn.with_options(gpu=gpu).remote(
        model_id=model, prompt=prompt, dtype=dtype,
        attention_backend=attention_backend, engine=engine,
        capture_decode=capture_decode, max_tokens=max_tokens,
        n_prompts=n_prompts, speculative_tokens=speculative_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_seq_len=max_seq_len, per_head=per_head,
    )

    vllm_version = result["vllm_version"]
    backend_used = result["attention_backend"]
    engine_used = result["engine"]
    captures: dict = result["captures"]
    capture_decode_used = result.get("capture_decode", False)
    max_tokens_used = result.get("max_tokens", 1)
    per_head_used = result.get("per_head", False)
    n_heads_used = result.get("n_heads", 0)

    if not out:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backend_tag = f"_{backend_used.lower()}" if backend_used != "auto" else ""
        decode_tag = f"_decode{max_tokens_used}" if capture_decode_used else ""
        batch_tag = f"_x{result.get('n_prompts', 1)}" if result.get('n_prompts', 1) > 1 else ""
        spec_tag = f"_spec{result.get('speculative_tokens', 0)}" if result.get('speculative_tokens', 0) > 0 else ""
        out = (
            f"vllm_{vllm_version.replace('.', '_')}_{engine_used}{backend_tag}{decode_tag}{batch_tag}{spec_tag}_"
            f"{gpu.lower().replace('-', '_')}_{timestamp}"
        )
    out_dir = Path(__file__).parent / "results" / out

    # The VLLMRunner already returns tensors in forward order, so insertion
    # order is the tap order; head_counts comes straight from the runner.
    tap_points = list(captures.keys())
    head_counts: dict[str, int] = result.get("head_counts", {})

    manifest = ReferenceManifest(
        model_id=model,
        # Placeholder fingerprint — vLLM's parallel-wrapped params don't hash
        # the same way HF's do; the runner builds the same vllm-<ver>:<model>.
        model_fingerprint=result.get("fingerprint", f"vllm-{vllm_version}:{model}"),
        tap_points=tap_points,
        shapes={name: list(captures[name].shape) for name in tap_points},
        dtypes={name: str(captures[name].dtype).replace("torch.", "") for name in tap_points},
        captured_at=datetime.now(UTC).isoformat(),
        env={
            "engine": "vllm",
            "vllm_version": vllm_version,
            "vllm_engine": engine_used,
            "attention_backend": backend_used,
            "capture_decode": str(capture_decode_used),
            "max_tokens": str(max_tokens_used),
            "n_prompts": str(result.get("n_prompts", 1)),
            "speculative_tokens": str(result.get("speculative_tokens", 0)),
            "gpu": gpu,
            "prompt": prompt,
            "per_head": str(per_head_used),
            "n_heads": str(n_heads_used),
            "attn_impl": str(result.get("attn_impl", "unknown")),
        },
        domain="llm",
        dtype=dtype,
        head_counts=head_counts,
    )

    write_reference(out_dir, manifest, captures)

    print(f"\nWrote vLLM reference artifact to {out_dir}")
    print(
        f"  {len(captures)} taps, vllm={vllm_version} engine={engine_used}, "
        f"dtype={dtype}, attention_backend={backend_used}, "
        f"capture_decode={capture_decode_used}, max_tokens={max_tokens_used}, "
        f"per_head={per_head_used} ({len(head_counts)} head taps)"
    )
