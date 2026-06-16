"""Confirm int4wo runs end-to-end through the real quant-diff path on GPU.

After fixing _quant_config to use the tile_packed_to_4d packing format (see
scripts/probe_int4_torchao.py), this runs the actual product path — HF runner
quantizes the model (int4wo) and captures, then we diff against the fp baseline
exactly like `firefly quant-diff` does — and confirms it no longer dies on the
`mslk` kernel dep. Qwen2.5-0.5B on A10G, bf16.

Run:  uv run modal run scripts/validate_int4_gpu.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-validate-int4-gpu")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "torchao>=0.7", "transformers>=4.44", "accelerate",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

MODEL = "Qwen/Qwen2.5-0.5B"


@app.function(
    image=image, gpu="A10G", timeout=900,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def validate() -> dict:
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.compare import diff_captures
    from firefly.runners.hf import HFRunner

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")

    inputs = Path(tempfile.mkdtemp()) / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox jumps"], "max_length": 16}))

    runner = HFRunner()
    fp = runner.capture(MODEL, inputs, device="cuda", dtype="bf16")
    # The fix under test: int4wo via the HF runner's quantize option.
    q = runner.capture(
        MODEL, inputs, device="cuda", dtype="bf16",
        options={"quantize": "int4wo", "group_size": "32"},
    )

    tap_order = list(fp.tensors)
    divs = diff_captures(fp.tensors, q.tensors, tap_order)
    ranked = sorted(divs, key=lambda d: d.rel_mean, reverse=True)
    final = next((d for d in divs if d.tap_name == "final_norm"), divs[-1])

    verdict = {
        "int4_quantized_without_mslk_error": True,  # reaching here means it worked
        "fingerprint_matches_baseline": fp.fingerprint == q.fingerprint,
        "same_tap_structure": set(fp.tensors) == set(q.tensors),
        "n_taps": len(divs),
        "final_norm_rel_divergence": round(final.rel_mean, 4),
        "top_divergent_taps": [(d.tap_name, round(d.rel_mean, 4)) for d in ranked[:8]],
    }
    verdict["overall_pass"] = bool(
        verdict["fingerprint_matches_baseline"]
        and verdict["same_tap_structure"]
        and final.rel_mean > 0
    )
    print(json.dumps(verdict, indent=2))
    print("OVERALL:", "PASS" if verdict["overall_pass"] else "FAIL")
    return verdict


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(validate.remote(), indent=2, default=str))
