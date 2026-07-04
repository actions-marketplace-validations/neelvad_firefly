"""Gemma 4 (dense 12B) through the parity engine: capture → calibrate → self-check.

First run of a day-one-2026 architecture through the HF-runner seam. Gemma 4
is a multimodal "unified" checkpoint — the text decoder nests at
``model.language_model.layers`` (the tap-path addition this validates) and the
attention stack interleaves sliding/global layers with p-RoPE. If the tap
conventions hold here, the cross-family table gains a 10th model and the
Gemma 4 example (quant-diff → optimize) is unblocked.

Needs transformers >= 5.5 (the v5 line is where ``gemma4_unified`` landed) —
installed in the Modal image, independent of the repo's locked version.

Pass criteria:
  1. tap selection finds the full per-layer set + final_norm (48 layers → 145 taps)
  2. capture → calibrate (4 runs) → self-check is CLEAN under calibrated tolerances
  3. manifest records a real fingerprint

Run:  uv run modal run scripts/validate_gemma4_parity.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-validate-gemma4-parity")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

MODEL_ID = "google/gemma-4-12B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==5.12.1",
        "safetensors>=0.4",
        "huggingface_hub>=0.24",
        "sentencepiece",
    )
    .add_local_python_source("firefly")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    memory=65536,  # the 24 GB bf16 checkpoint stages through CPU RAM before .to(cuda)
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def run_parity() -> dict:
    import json
    from pathlib import Path

    import torch

    from firefly.calibrate import calibrate
    from firefly.compare import compare_to_reference
    from firefly.reference import read_reference

    work = Path("/tmp/ff_gemma4")
    work.mkdir(exist_ok=True)
    inputs = work / "golden.json"
    inputs.write_text(
        json.dumps(
            {
                "texts": [
                    "The quick brown fox jumps over the lazy dog.",
                    "In 1969, humans first landed on the Moon.",
                    "Attention is all you need, the paper famously claimed.",
                    "def fibonacci(n): return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
                ],
                "max_length": 32,
            }
        )
    )

    from firefly.capture import capture_reference

    ref_dir = work / "reference"
    capture_reference(
        MODEL_ID,
        inputs_path=inputs,
        out_dir=ref_dir,
        device="cuda",
        dtype=torch.bfloat16,
    )
    manifest, tensors = read_reference(ref_dir)

    tolerances = calibrate(ref_dir, inputs, runs=4, device="cuda")

    divergences = compare_to_reference(ref_dir, MODEL_ID, inputs, device="cuda")
    diverged = [d.tap_name for d in divergences if d.exceeds_tolerance]

    return {
        "n_taps": len(manifest.tap_points),
        "first_taps": manifest.tap_points[:4],
        "last_tap": manifest.tap_points[-1],
        "fingerprint": manifest.model_fingerprint[:16],
        "n_calibrated": len(tolerances),
        "max_noise_floor": max(t.atol for t in tolerances.values()),
        "diverged_taps": diverged,
        "dtype": manifest.dtype,
    }


@app.local_entrypoint()
def main() -> None:
    r = run_parity.remote()

    print(f"\n=== Gemma 4 parity validation ({MODEL_ID}, bf16, A100) ===")
    print(f"  taps: {r['n_taps']}  ({r['first_taps'][0]} … {r['last_tap']})")
    print(f"  fingerprint: {r['fingerprint']}…")
    print(f"  calibrated tolerances: {r['n_calibrated']} taps, max atol {r['max_noise_floor']:.3e}")

    failures: list[str] = []
    # 48 decoder layers × (self_attn + mlp + residual) + final_norm = 145
    if r["n_taps"] != 145:
        failures.append(f"expected 145 taps (48 layers × 3 + final_norm), got {r['n_taps']}")
    if r["last_tap"] != "final_norm":
        failures.append(f"final tap is {r['last_tap']!r}, expected final_norm")
    if not r["fingerprint"]:
        failures.append("no model fingerprint recorded")
    if r["diverged_taps"]:
        failures.append(f"self-check NOT clean: {r['diverged_taps'][:5]}")

    for f in failures:
        print(f"  FAIL: {f}")
    if failures:
        raise SystemExit(1)
    print("  PASS: Gemma 4 dense captures, calibrates, and self-checks clean.")
