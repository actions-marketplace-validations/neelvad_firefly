"""Scheduled GPU smoke: does the vLLM runner still work on *current* vLLM?

The in-process capture path (forward hooks pickled over ``collective_rpc``,
live attention-backend verification, V1 engine wiring) touches vLLM internals
that drift across releases — and the fast CI suite runs on CPU, so it can't
exercise any of it. This is the cheapest end-to-end canary: install the
LATEST vLLM, capture SmolLM-135M through the library ``VLLMRunner`` via the
normal ``capture_reference`` orchestrator, and assert the capture contract:

  * every expected tap captured (per-layer self_attn / mlp / residual +
    final_norm), with finite tensors
  * the manifest records a real weight fingerprint and the runner metadata

Structural asserts only — no golden numerics — so a vLLM release can only
fail this by actually breaking the runner, never by benign kernel drift.
Exits non-zero on failure so the scheduled workflow turns red.

Run manually:   uv run modal run scripts/gpu_smoke.py
Scheduled by:   .github/workflows/gpu-smoke.yml (weekly + manual dispatch)
"""

from __future__ import annotations

import modal

app = modal.App("firefly-gpu-smoke")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

# Deliberately UNPINNED vllm: the smoke's job is to catch the next release
# breaking the runner, so it must track latest, not a known-good pin.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "huggingface_hub>=0.24",
        "safetensors>=0.4",
    )
    .add_local_python_source("firefly")
)


@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/root/.cache/huggingface": _HF_CACHE})
def smoke_capture() -> dict:
    """Capture SmolLM through the library VLLMRunner; return contract facts."""
    import json
    from pathlib import Path

    import torch
    import vllm

    from firefly.capture import capture_reference
    from firefly.reference import read_reference
    from firefly.runners import get_runner

    work = Path("/tmp/ff_smoke")
    work.mkdir(exist_ok=True)
    inputs = work / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox jumps over the lazy dog"]}))

    out_dir = work / "ref"
    capture_reference(
        "HuggingFaceTB/SmolLM-135M",
        inputs_path=inputs,
        out_dir=out_dir,
        dtype=torch.bfloat16,
        runner=get_runner("vllm"),
        options={},  # default engine + backend: what a fresh user gets
    )
    manifest, tensors = read_reference(out_dir)
    return {
        "vllm_version": vllm.__version__,
        "tap_points": manifest.tap_points,
        "fingerprint": manifest.model_fingerprint,
        "runner": manifest.runner,
        "finite": {name: bool(torch.isfinite(t).all()) for name, t in tensors.items()},
        "n_tensors": len(tensors),
    }


@app.local_entrypoint()
def main() -> None:
    r = smoke_capture.remote()

    failures: list[str] = []
    taps = r["tap_points"]
    # SmolLM-135M: 30 decoder layers → self_attn + mlp + residual each, + final_norm.
    if len(taps) != 91:
        failures.append(f"expected 91 taps (30 layers × 3 + final_norm), got {len(taps)}")
    if "final_norm" not in taps:
        failures.append("final_norm tap missing")
    if r["n_tensors"] != len(taps):
        failures.append(f"{len(taps)} taps but {r['n_tensors']} captured tensors")
    if not r["fingerprint"]:
        failures.append("manifest has no model fingerprint")
    if r["runner"] != "vllm":
        failures.append(f"manifest runner is {r['runner']!r}, expected 'vllm'")
    bad = [name for name, ok in r["finite"].items() if not ok]
    if bad:
        failures.append(f"non-finite values in taps: {bad[:5]}")

    print(f"\n=== vLLM runner smoke (vllm {r['vllm_version']}) ===")
    print(f"  taps captured: {r['n_tensors']}")
    for f in failures:
        print(f"  FAIL: {f}")
    if failures:
        raise SystemExit(1)
    print("  PASS: capture contract holds on current vLLM.")
