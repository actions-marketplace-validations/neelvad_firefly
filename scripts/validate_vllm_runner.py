"""Step-4 validation: the library VLLMRunner reproduces the script's result.

Runs the *new* in-process `firefly.runners.vllm.VLLMRunner` (via the normal
`capture_reference` orchestrator) on Modal, captures SmolLM-135M under
FLASH_ATTN and XFORMERS, diffs them with the library, and asserts the first
divergence is `layer.7.self_attn` — the result `scripts/capture_vllm.py`
produced earlier this session. This proves the extracted runner is faithful
before we route the CLI's `--runner vllm` at it for real.

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
def validate() -> dict:
    import json
    from pathlib import Path

    import torch

    from firefly.attribution import attribute_first_divergence
    from firefly.capture import capture_reference
    from firefly.compare import TapTolerance, diff_captures
    from firefly.reference import read_reference
    from firefly.runners import get_runner

    work = Path("/tmp/ff_validate")
    work.mkdir(exist_ok=True)
    inputs = work / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox jumps over the lazy dog"]}))

    runner = get_runner("vllm")
    common = dict(
        inputs_path=inputs,
        dtype=torch.bfloat16,
        runner=runner,
    )

    flash_dir = work / "flash"
    capture_reference(
        "HuggingFaceTB/SmolLM-135M", out_dir=flash_dir,
        options={"engine": "v0", "attention_backend": "FLASH_ATTN"}, **common,
    )
    xformers_dir = work / "xformers"
    capture_reference(
        "HuggingFaceTB/SmolLM-135M", out_dir=xformers_dir,
        options={"engine": "v0", "attention_backend": "XFORMERS"}, **common,
    )

    man_a, ta = read_reference(flash_dir)
    _, tb = read_reference(xformers_dir)
    tolerances = {name: TapTolerance(atol=1e-6) for name in man_a.tap_points}
    divs = diff_captures(ta, tb, man_a.tap_points, tolerances=tolerances)
    result = attribute_first_divergence(divs)

    # Self-consistency: FLASH vs FLASH (two captures, same backend) is bit-equal.
    flash2_dir = work / "flash2"
    capture_reference(
        "HuggingFaceTB/SmolLM-135M", out_dir=flash2_dir,
        options={"engine": "v0", "attention_backend": "FLASH_ATTN"}, **common,
    )
    _, ta2 = read_reference(flash2_dir)
    self_divs = diff_captures(ta, ta2, man_a.tap_points, tolerances=tolerances)
    self_clean = not any(d.exceeds_tolerance for d in self_divs)

    return {
        "n_taps": len(man_a.tap_points),
        "first_divergence_flash_vs_xformers": result.first_divergent_tap,
        "self_compare_flash_vs_flash_clean": self_clean,
        "manifest_env": man_a.env,
        "manifest_dtype": man_a.dtype,
    }


@app.local_entrypoint()
def main() -> None:
    out = validate.remote()
    print("\n=== VLLMRunner validation ===")
    for k, v in out.items():
        print(f"  {k}: {v}")
    ok = (
        out["first_divergence_flash_vs_xformers"] == "layer.7.self_attn"
        and out["self_compare_flash_vs_flash_clean"]
    )
    print(f"\n{'PASS' if ok else 'FAIL'}: library VLLMRunner "
          f"{'reproduces' if ok else 'does NOT reproduce'} the known result.")
