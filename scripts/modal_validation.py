"""Modal-driven validation of the synthetic-noise calibration methodology.

Runs on an NVIDIA GPU (default A10G; configurable via ``--gpu``):

    1. Captures a reference from a small HF transformer.
    2. Calibrates three configurations against it:
         A. Full lockdown — set_deterministic(), no TF32.
         B. Hardware noise, no TF32.
         C. Hardware noise + TF32.
    3. Returns the per-config per-tap noise_floor + atol dictionaries.

The local entrypoint writes the returned JSON to scripts/results/ so a plot
script (or notebook) can render the depth-amplification curves and compare
across hardware generations.

Usage:
    uv run modal token new                          # one-time auth
    uv run modal run scripts/modal_validation.py
    uv run modal run scripts/modal_validation.py --gpu A100
    uv run modal run scripts/modal_validation.py --gpu H100 --model HuggingFaceTB/SmolLM-360M
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validation")

# Forward HF_TOKEN to the GPU container if the user has one set locally.
# Public models (SmolLM, Llama base, etc.) work without it but emit a
# rate-limit warning; gated models require it. Opt-in, not enforced.
_HF_TOKEN_SET = bool(os.environ.get("HF_TOKEN"))
_HF_SECRETS = (
    [modal.Secret.from_local_environ(["HF_TOKEN"])] if _HF_TOKEN_SET else []
)

# PyTorch 2.7+ with CUDA 12.8 is required for Blackwell (sm_100, B200).
# Older images (e.g., 2.6.0-cuda12.4) will UserWarning + fail on B200 because
# the wheels were compiled before sm_100 was added to TORCH_CUDA_ARCH_LIST.
# H100 / H200 / A100 / A10G all still work on the newer image — Hopper's sm_90
# and Ampere's sm_80 are supported in both 2.6 and 2.7.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "transformers>=4.40",
        "safetensors>=0.4",
        "typer>=0.12",
        "rich>=13",
        "numpy",
        "hf_xet",
    )
    .add_local_python_source("firefly")
)


@app.function(gpu="A10G", image=image, timeout=900, secrets=_HF_SECRETS)
def run_validation(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    texts: list[str] | None = None,
    max_length: int = 16,
    runs: int = 8,
) -> dict:
    """Run capture + the three calibration configs on GPU; return all per-tap data."""
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import torch

    from firefly.calibrate import calibrate
    from firefly.capture import capture_reference
    from firefly.noise import NoiseSpec

    if texts is None:
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "to be or not to be",
        ]

    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}  device: {device_name}")
    print(f"Torch: {torch.__version__}")

    results: dict = {
        "metadata": {
            "model_id": model_id,
            "texts": texts,
            "max_length": max_length,
            "runs": runs,
            "device": device_name,
            "torch_version": torch.__version__,
        },
        "configs": {},
    }

    configs: dict[str, NoiseSpec] = {
        "A_strict_no_tf32": NoiseSpec(mode="none"),
        "B_hardware_no_tf32": NoiseSpec(mode="hardware", allow_tf32=False),
        "C_hardware_tf32": NoiseSpec(mode="hardware", allow_tf32=True),
    }

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inputs_path = tmp_path / "golden.json"
        inputs_path.write_text(json.dumps({"texts": texts, "max_length": max_length}))
        ref_dir = tmp_path / "reference"

        print(f"\nCapturing reference: {model_id}")
        capture_reference(model_id, inputs_path, ref_dir, device="cuda")

        for name, spec in configs.items():
            print(f"\n=== Calibrating: {name} (mode={spec.mode}, allow_tf32={spec.allow_tf32}) ===")
            tolerances = calibrate(
                reference_dir=ref_dir,
                inputs_path=inputs_path,
                runs=runs,
                noise=spec,
                device="cuda",
            )
            results["configs"][name] = {
                tap_name: {
                    "noise_floor": tol.noise_floor,
                    "atol": tol.atol,
                }
                for tap_name, tol in tolerances.items()
            }
            n_above = sum(1 for tol in tolerances.values() if tol.noise_floor > 0)
            max_floor = max((tol.noise_floor for tol in tolerances.values()), default=0.0)
            print(f"  {n_above}/{len(tolerances)} taps with noise > 0; max noise_floor = {max_floor:.3e}")

    return results


# Approximate Modal on-demand pricing as of 2026-Q2, for the cost hint
# printed at launch. Rates change; treat as a sanity check, not a quote.
_GPU_COST_PER_HR = {
    "T4": 0.59,
    "L4": 0.80,
    "L40S": 1.95,
    "A10G": 1.10,
    "A100": 2.10,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 4.56,
    "H200": 4.56,
    "B200": 6.25,
}


def _gpu_filename_tag(gpu: str) -> str:
    return gpu.lower().replace("-", "_")


@app.local_entrypoint()
def main(
    model: str = "HuggingFaceTB/SmolLM-135M",
    runs: int = 8,
    gpu: str = "A10G",
) -> None:
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    if _HF_TOKEN_SET:
        print("HF_TOKEN found in local env — forwarding to GPU container.")
    else:
        print(
            "No HF_TOKEN in local env — using anonymous HF access "
            "(rate-limited; fine for public models)."
        )

    hourly = _GPU_COST_PER_HR.get(gpu)
    cost_hint = (
        f"(approx ${hourly:.2f}/hr; full run ≈ ${hourly * 5 / 60:.2f})"
        if hourly is not None
        else "(unknown cost — check modal.com/pricing)"
    )
    print(f"Launching {gpu} job for model={model}, runs={runs}  {cost_hint}")

    results = run_validation.with_options(gpu=gpu).remote(
        model_id=model, runs=runs,
    )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"modal_validation_{_gpu_filename_tag(gpu)}_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nResults written to {out_path}")
    print(f"Device reported by torch: {results['metadata']['device']}")
    print("\n--- Quick summary ---")
    for name, config in results["configs"].items():
        floors = [tap["noise_floor"] for tap in config.values()]
        n_above = sum(1 for f in floors if f > 0)
        max_floor = max(floors, default=0.0)
        print(f"  {name}: {n_above}/{len(floors)} taps noisy, max_floor={max_floor:.3e}")
