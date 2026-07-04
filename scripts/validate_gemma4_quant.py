"""Gemma 4 12B quantization signature: diagnose + measured per-layer quant-diff.

Step 2 of the Gemma 4 example (after `validate_gemma4_parity.py` PASSed): what
is this family's fragility signature? Two measurements in one container:

1. **Diagnosis from stored activations** (the cheap sensor): capture the bf16
   reference, then emit ACTIVATION_OUTLIERS findings at int8 and int4 — which
   layers carry Dettmers-style outlier channels, how concentrated, and how much
   per-channel quant rescues them. This is what `optimize` routes on.
2. **Measured quant-diff** (the ground truth): torchao-quantize the same model
   (w8a8, then int4wo g128) and diff it against the fp reference through the
   standard capture/compare pipeline — per-layer relative divergence ranking +
   first divergent tap in forward order.

Report-only (no gate): the point is the signature, compared against the known
families (Qwen: late-MLP down_projs, ×61 outliers @1.5B; SmolLM: layer-11 MLP).

Run:  uv run modal run scripts/validate_gemma4_quant.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-validate-gemma4-quant")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

MODEL_ID = "google/gemma-4-12B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==5.12.1",
        "torchao",
        "safetensors>=0.4",
        "huggingface_hub>=0.24",
        "sentencepiece",
    )
    .add_local_python_source("firefly")
)


def _free_cuda() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    memory=65536,
    volumes={"/root/.cache/huggingface": _HF_CACHE},
)
def run_quant_signature() -> dict:
    import json
    from pathlib import Path

    import torch

    from firefly.attribution import attribute_first_divergence
    from firefly.capture import capture_reference
    from firefly.compare import compare_to_reference
    from firefly.quant.diagnose import diagnose
    from firefly.reference import read_reference

    work = Path("/tmp/ff_gemma4_quant")
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

    ref_dir = work / "reference"
    capture_reference(MODEL_ID, inputs_path=inputs, out_dir=ref_dir, device="cuda", dtype=torch.bfloat16)
    manifest, tensors = read_reference(ref_dir)
    _free_cuda()

    # -- 1. cheap sensor: outlier diagnosis from the stored activations
    diagnosis = {}
    for bits in (8, 4):
        d = diagnose(tensors, manifest.tap_points, bits=bits)
        diagnosis[f"int{bits}"] = [
            {
                "location": f.location,
                "evidence": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.evidence.items()},
                "explanation": f.explanation,
            }
            for f in d.by_signature("activation_outliers")
        ]
    del tensors
    _free_cuda()

    # -- 2. ground truth: measured per-layer divergence of real torchao quant
    schemes = {}
    for scheme, group_size in (("w8a8", "128"), ("int4wo", "128")):
        divs = compare_to_reference(
            ref_dir,
            MODEL_ID,
            inputs,
            device="cuda",
            options={"quantize": scheme, "group_size": group_size},
        )
        ranked = sorted(divs, key=lambda d: d.rel_mean, reverse=True)
        schemes[scheme] = {
            "first_divergent_tap": attribute_first_divergence(divs).first_divergent_tap,
            "top10_rel_mean": [(d.tap_name, round(d.rel_mean, 5)) for d in ranked[:10]],
            "final_norm_rel_mean": round(
                next(d.rel_mean for d in divs if d.tap_name == "final_norm"), 5
            ),
        }
        _free_cuda()

    return {"diagnosis": diagnosis, "schemes": schemes}


@app.local_entrypoint()
def main() -> None:
    r = run_quant_signature.remote()

    print(f"\n=== Gemma 4 quant signature ({MODEL_ID}, bf16 baseline) ===")
    for bits, findings in r["diagnosis"].items():
        print(f"\n-- diagnosis @ {bits}: {len(findings)} activation-outlier finding(s)")
        for f in findings[:8]:
            print(f"   {f['location']}: {f['explanation']}")
        if len(findings) > 8:
            print(f"   … and {len(findings) - 8} more")

    for scheme, s in r["schemes"].items():
        print(f"\n-- measured quant-diff: {scheme}")
        print(f"   first divergent tap (forward order): {s['first_divergent_tap']}")
        print(f"   final_norm rel_mean: {s['final_norm_rel_mean']}")
        print("   worst 10 taps by rel_mean:")
        for tap, rel in s["top10_rel_mean"]:
            print(f"     {tap:28s} {rel}")
