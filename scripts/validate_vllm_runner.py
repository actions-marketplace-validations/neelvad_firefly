"""Step-4 validation: the library VLLMRunner reproduces the script's result.

Captures SmolLM-135M through the *new* in-process
`firefly.runners.vllm.VLLMRunner` (via the normal `capture_reference`
orchestrator), once per attention backend, then diffs with the library and
asserts the first divergence is `layer.7.self_attn` — the result
`scripts/capture_vllm.py` produced earlier this session. That proves the
extracted runner is faithful before the CLI's `--runner vllm` relies on it.

Each backend runs in its **own container**, spawned concurrently: vLLM caches
its attention backend at the first ``LLM()`` construction, and Modal reuses a
warm container across sequential ``.remote()`` calls — so two backends would
silently share the first one's. ``.spawn()`` both at once forces Modal to
scale out to two containers, each a fresh process that imports vLLM with its
own backend env var. The captured tensors come back to the laptop and the
diff runs locally on CPU. (The runner's _verify_backend guard hard-fails if a
backend didn't actually load, so a reuse slip surfaces loudly, never as a
wrong "no divergence".)

    uv run modal run scripts/validate_vllm_runner.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-validate-vllm-runner")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

# vLLM 0.8.5 keeps the XFORMERS backend (newer vLLM dropped it), so this is the
# version that can run the FLASH-vs-XFORMERS comparison. V0 engine because
# 0.8.5's V1 apply_model path is broken (documented in capture_vllm.py).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5",
        "transformers==4.51.3",
        "huggingface_hub>=0.24",
        "safetensors>=0.4",
    )
    .add_local_python_source("firefly")
)


@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/root/.cache/huggingface": _HF_CACHE})
def capture_backend(backend: str) -> dict:
    """Capture SmolLM under one attention backend; return CPU tensors + meta.

    One backend per call == one fresh process, so vLLM's global backend cache
    doesn't leak between backends. The runner's _verify_backend guard raises
    here if the requested backend didn't actually load.
    """
    import json
    from pathlib import Path

    import torch

    from firefly.capture import capture_reference
    from firefly.reference import read_reference
    from firefly.runners import get_runner

    work = Path("/tmp/ff_validate")
    work.mkdir(exist_ok=True)
    inputs = work / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox jumps over the lazy dog"]}))

    out_dir = work / backend.lower()
    capture_reference(
        "HuggingFaceTB/SmolLM-135M",
        inputs_path=inputs,
        out_dir=out_dir,
        dtype=torch.bfloat16,
        runner=get_runner("vllm"),
        options={"engine": "v0", "attention_backend": backend},
    )
    manifest, tensors = read_reference(out_dir)
    return {"tensors": tensors, "tap_points": manifest.tap_points, "env": manifest.env}


@app.local_entrypoint()
def main() -> None:
    from firefly.attribution import attribute_first_divergence
    from firefly.compare import TapTolerance, diff_captures

    # Spawn both concurrently so Modal scales out to two containers (one warm
    # container reused sequentially would leak vLLM's cached backend).
    flash_handle = capture_backend.spawn("FLASH_ATTN")
    xformers_handle = capture_backend.spawn("XFORMERS")
    flash = flash_handle.get()
    xformers = xformers_handle.get()

    tap_points = flash["tap_points"]
    tolerances = {name: TapTolerance(atol=1e-6) for name in tap_points}
    divs = diff_captures(flash["tensors"], xformers["tensors"], tap_points, tolerances=tolerances)
    result = attribute_first_divergence(divs)

    print("\n=== VLLMRunner validation (library path) ===")
    print(f"  taps: {len(tap_points)}")
    print(f"  FLASH env:    {flash['env'].get('attn_impl')}")
    print(f"  XFORMERS env: {xformers['env'].get('attn_impl')}")
    print(f"  first divergence (FLASH vs XFORMERS): {result.first_divergent_tap}")

    ok = result.first_divergent_tap == "layer.7.self_attn"
    print(
        f"\n{'PASS' if ok else 'FAIL'}: library VLLMRunner "
        f"{'reproduces' if ok else 'does NOT reproduce'} the known "
        f"FLASH-vs-XFORMERS result (layer.7.self_attn)."
    )
