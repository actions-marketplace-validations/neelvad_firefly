"""GPU validation for the SGLang runner.

Captures SmolLM-135M through the library SGLangRunner (via capture_reference)
and confirms the live engine path works end to end: SGLang accepts our
forward_hooks, the hooks fire on the prefill forward, our tap names come back,
and the captured tensors have sane shapes. This is the SGLang analog of
validate_vllm_runner.py — proving the extracted runner before the CLI relies
on it.

    uv run modal run scripts/validate_sglang_runner.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-validate-sglang-runner")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

# SGLang needs the CUDA toolkit (some kernels JIT-compile) and a recent build
# for ServerArgs.forward_hooks. sglang[all] pulls a consistent torch +
# transformers + flashinfer + kernels set — do NOT override transformers, that
# pulls a version whose hub_kernels integration mismatches sglang's pinned
# `kernels` and crashes at import.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    # sgl_kernel's prebuilt .so links libnuma; it's not in the cuda base image.
    .apt_install("libnuma1")
    .pip_install("sglang[all]==0.5.12")
    # The HF `kernels` accelerator package on PyPI advanced past transformers'
    # hub_kernels integration (its LayerRepository now requires a version arg
    # transformers doesn't pass), crashing at `import transformers.activations`.
    # `kernels` is optional — transformers degrades gracefully without it.
    .run_commands("pip uninstall -y kernels")
    .add_local_python_source("firefly")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=2400,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def validate() -> dict:
    import json
    from pathlib import Path

    import sglang as sgl
    import torch

    from firefly.capture import capture_reference
    from firefly.reference import read_reference
    from firefly.runners import get_runner

    print(f"sglang version: {getattr(sgl, '__version__', 'unknown')}")

    work = Path("/tmp/ffsg")
    work.mkdir(exist_ok=True)
    inputs = work / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox jumps over the lazy dog"]}))
    out_dir = work / "ref"

    capture_reference(
        "HuggingFaceTB/SmolLM-135M",
        inputs_path=inputs,
        out_dir=out_dir,
        dtype=torch.bfloat16,
        runner=get_runner("sglang"),
        options={"mem_fraction_static": "0.7"},
    )

    manifest, tensors = read_reference(out_dir)
    sample = next(iter(tensors.values()))
    return {
        "n_taps": len(manifest.tap_points),
        "first_taps": manifest.tap_points[:4],
        "last_tap": manifest.tap_points[-1],
        "has_final_norm": "final_norm" in manifest.tap_points,
        "sample_shape": list(sample.shape),
        "env": manifest.env,
        "fingerprint": manifest.model_fingerprint,
    }


@app.local_entrypoint()
def main() -> None:
    out = validate.remote()
    print("\n=== SGLangRunner validation ===")
    for k, v in out.items():
        print(f"  {k}: {v}")

    # SmolLM-135M is 30 layers → 30*(self_attn, mlp, layer) + final_norm = 91.
    ok = (
        out["has_final_norm"]
        and out["n_taps"] >= 90
        and out["first_taps"][0] == "layer.0.self_attn"
        and out["fingerprint"].startswith("sglang-")
    )
    print(
        f"\n{'PASS' if ok else 'FAIL'}: SGLangRunner "
        f"{'captures our tap convention end to end' if ok else 'did NOT produce the expected taps'}."
    )
