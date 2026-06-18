"""GPU validation for the vLLM weight-fingerprint (commit 14a1441).

The dev box is CPU-only, so the new vLLM fingerprint path could only be
unit-tested on `_fingerprint_impl` / `_combine_fingerprints` directly (CPU
nn.Module). What that *cannot* exercise is the part that only exists on a real
GPU worker:

  * does ``collective_rpc(_v1_read_fingerprint)`` (V1) / ``apply_model(
    _read_fingerprint)`` (V0) actually reach ``worker.model_runner.model``
    and walk its ``named_parameters()``?
  * does the 16-hex string survive the RPC bus? (collective_rpc summarizes
    *tensor* returns to dtype strings — that's why drain ships bytes — so a
    plain str must pass through, but that's worth confirming on hardware.)
  * is the fingerprint deterministic across two separate engine loads of the
    same weights, and does it actually change for *different* weights?

This runs the real production path — ``capture_reference`` through the
in-process ``VLLMRunner`` — and asserts, per engine:

  D (determinism)  same model loaded in two fresh containers → equal fp,
                   matching ``^vllm:[0-9a-f]{16}$`` (a real hash, not the old
                   ``vllm-{version}:{model_id}`` placeholder).
  S (sensitivity)  the base model vs its instruct fine-tune (same arch/shapes,
                   different weights) → different fp. This is exactly the
                   "republished fine-tune under a familiar name" case the old
                   name-only fingerprint let pass; it must now diverge.

Two engines, two vLLM versions (the repo's known-good split): V0 on 0.8.5
(``apply_model``), V1 on 0.22.1 (``collective_rpc``; V0 engine is gone there).
Each capture runs in its own fresh container (one model load per process), all
spawned concurrently.

    uv run modal run scripts/validate_vllm_fingerprint.py
"""

from __future__ import annotations

import re

import modal

app = modal.App("firefly-validate-vllm-fingerprint")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

_FP_RE = re.compile(r"^vllm:[0-9a-f]{16}$")

# Base model and its instruct fine-tune: same architecture and parameter
# shapes, different weights — the clean "is the hash weight-derived?" probe.
_BASE = "HuggingFaceTB/SmolLM-135M"
_FINETUNE = "HuggingFaceTB/SmolLM-135M-Instruct"

# V0 engine: vLLM 0.8.5 (apply_model dispatch; V1 is broken at this version).
image_v0 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5",
        "transformers==4.51.3",
        "huggingface_hub>=0.24",
        "safetensors>=0.4",
    )
    .add_local_python_source("firefly")
)

# V1 engine: vLLM 0.22.1 (collective_rpc dispatch; V0 engine no longer exists).
# Needs the CUDA devel image — V1's engine-core subprocess fails to start on
# debian_slim (matches the repo's known-good 0.22.1 setup in
# repro_flashinfer_zero_heads.py).
image_v1 = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "vllm==0.22.1",
        "transformers>=4.55",
        "huggingface_hub>=0.24",
        "safetensors>=0.4",
    )
    .add_local_python_source("firefly")
)


def _capture_fingerprint(model_id: str, engine: str) -> dict:
    """Run the real VLLMRunner capture path and return the manifest fingerprint.

    One model load per process, so two calls to the same model_id are two
    independent loads — the honest determinism test.
    """
    import json
    import os
    from pathlib import Path

    import torch

    if engine == "v0":
        os.environ["VLLM_USE_V1"] = "0"

    from firefly.capture import capture_reference
    from firefly.reference import read_reference
    from firefly.runners import get_runner

    work = Path("/tmp/ff_fp_validate")
    work.mkdir(exist_ok=True)
    inputs = work / "golden.json"
    inputs.write_text(json.dumps({"texts": ["the quick brown fox"]}))

    out_dir = work / f"{engine}_{model_id.split('/')[-1]}"
    capture_reference(
        model_id,
        inputs_path=inputs,
        out_dir=out_dir,
        dtype=torch.bfloat16,
        runner=get_runner("vllm"),
        options={"engine": engine},
    )
    manifest, _ = read_reference(out_dir)
    return {
        "model_id": model_id,
        "engine": engine,
        "fingerprint": manifest.model_fingerprint,
        "attn_impl": manifest.env.get("attn_impl"),
    }


@app.function(image=image_v0, gpu="A10G", timeout=1800, volumes={"/root/.cache/huggingface": _HF_CACHE})
def capture_v0(model_id: str) -> dict:
    return _capture_fingerprint(model_id, "v0")


@app.function(image=image_v1, gpu="A10G", timeout=1800, volumes={"/root/.cache/huggingface": _HF_CACHE})
def capture_v1(model_id: str) -> dict:
    return _capture_fingerprint(model_id, "v1")


def _check_engine(engine: str, base_a, base_b, finetune) -> bool:
    print(f"\n=== {engine.upper()} ===")
    for label, res in (("base A", base_a), ("base B", base_b), ("fine-tune", finetune)):
        if isinstance(res, Exception):
            print(f"  [FAIL] {label} capture raised: {type(res).__name__}: {res}")
            return False
    fp_a, fp_b, fp_ft = base_a["fingerprint"], base_b["fingerprint"], finetune["fingerprint"]
    print(f"  attn_impl:    {base_a['attn_impl']}")
    print(f"  base run A:   {fp_a}")
    print(f"  base run B:   {fp_b}")
    print(f"  fine-tune:    {fp_ft}")

    well_formed = bool(_FP_RE.match(fp_a))
    deterministic = fp_a == fp_b
    sensitive = fp_a != fp_ft
    print(f"  [{'PASS' if well_formed else 'FAIL'}] real hash, not placeholder (^vllm:[0-9a-f]{{16}}$)")
    print(f"  [{'PASS' if deterministic else 'FAIL'}] D: same weights, two loads → equal fingerprint")
    print(f"  [{'PASS' if sensitive else 'FAIL'}] S: base vs instruct fine-tune → different fingerprint")
    return well_formed and deterministic and sensitive


@app.local_entrypoint()
def main() -> None:
    # Spawn everything concurrently — fresh container per call so each is an
    # independent model load (and vLLM's per-process caches don't leak).
    handles = {
        "v0_base_a": capture_v0.spawn(_BASE),
        "v0_base_b": capture_v0.spawn(_BASE),
        "v0_ft": capture_v0.spawn(_FINETUNE),
        "v1_base_a": capture_v1.spawn(_BASE),
        "v1_base_b": capture_v1.spawn(_BASE),
        "v1_ft": capture_v1.spawn(_FINETUNE),
    }
    # Gather resiliently: one engine's env failure must not hide the other's
    # result (V1's CUDA-devel image issues shouldn't mask a V0 PASS).
    r: dict = {}
    for k, h in handles.items():
        try:
            r[k] = h.get()
        except Exception as e:  # noqa: BLE001 — report, don't abort
            r[k] = e

    v0_ok = _check_engine("v0", r["v0_base_a"], r["v0_base_b"], r["v0_ft"])
    v1_ok = _check_engine("v1", r["v1_base_a"], r["v1_base_b"], r["v1_ft"])

    print("\n" + "=" * 60)
    ok = v0_ok and v1_ok
    print(
        f"{'PASS' if ok else 'FAIL'}: vLLM fingerprint is a real, deterministic, "
        f"weight-derived hash on {'both engines' if ok else 'at least one engine FAILED'}."
    )
    if not ok:
        raise SystemExit(1)
