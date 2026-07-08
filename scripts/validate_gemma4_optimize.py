"""Gemma 4 12B-it headline: multi-scheme optimize vs Google's own QAT checkpoint.

Step 3 of the Gemma 4 example. The quant-signature run showed the most
outlier-saturated family yet measured (118/145 taps flagged @int8, pervasive
rather than concentrated) — the architecture Google considered hard enough to
ship official QAT checkpoints for. So the headline question is two-part:

1. **Can measurement-gated PTQ ship this family at all?**
   `optimize_over_schemes({int4wo, int8wo}, bar=10%)` — diagnose → route the
   recovery (int4 → GPTQ/AWQ) → export compressed-tensors → re-eval the
   *served* checkpoint → ship the most-compressed scheme that meets the bar.
2. **How close does one afternoon of PTQ get to Google's QAT?**
   Score `google/gemma-4-12B-it-qat-w4a16-ct` (their official W4A16
   compressed-tensors QAT export) with the *same* perplexity evaluator on the
   *same* eval set — the vendor baseline our int4 artifact is judged against.

Uses the -it model throughout so the fp baseline, our PTQ export, and the QAT
checkpoint share the same underlying weights family.

Run:  uv run modal run scripts/validate_gemma4_optimize.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-gemma4-optimize")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

MODEL = "google/gemma-4-12B-it"
QAT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
QUALITY_BAR = 0.10
EVAL_MAX_LENGTH = 128  # assistant turns run up to ~80 words
CHAT_EVAL = "/root/chat_eval_dolly.json"  # 200 dolly chat pairs — the SERVED distribution

# llmcompressor first (its resolver picks compatible compressed-tensors/datasets),
# then transformers pinned to the v5 line gemma4 requires — later layer wins.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch", "llmcompressor", "torchao", "accelerate", "sentencepiece")
    .pip_install("transformers==5.12.1")
    .add_local_python_source("firefly")
    .add_local_file(
        str(__import__("pathlib").Path(__file__).parent / "prompts" / "chat_eval_dolly.json"),
        CHAT_EVAL,
    )
)

_CALIB = [
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
    "In 1969, Apollo 11 landed the first humans on the Moon during the Space Race.",
    "Supply and demand determine prices in a competitive market economy.",
    "A binary search halves the search space at every comparison step.",
    "The water cycle moves moisture through evaporation, condensation, and precipitation.",
    "Shakespeare wrote tragedies, comedies, and histories in early modern English.",
    "Plate tectonics explains the slow drift of continents over geological time.",
    "A compiler translates source code into machine instructions before execution.",
    "The immune system distinguishes self from foreign antigens to fight infection.",
    "Inflation erodes purchasing power when the money supply grows faster than output.",
    "Genes encode proteins through transcription into RNA and translation by ribosomes.",
    "The speed of light in a vacuum is a fundamental constant of the universe.",
    "Object-oriented programming organizes code around encapsulated data and methods.",
    "Tectonic stress released along faults produces earthquakes and seismic waves.",
    "Markets allocate scarce resources through the price signals of buyers and sellers.",
    "Neurons communicate via electrochemical signals across synaptic junctions.",
    "The Renaissance revived classical art and learning across fifteenth-century Europe.",
    "A database index trades storage for faster lookups on a queried column.",
    "Thermodynamics governs how energy is conserved and dispersed in physical systems.",
    "Natural selection favors heritable traits that improve reproductive success.",
    "Encryption protects data by transforming it into a form only a key can reverse.",
    "Fiscal policy uses government spending and taxation to steer the economy.",
]

@app.function(
    image=image, gpu="A100-80GB", timeout=7200, memory=65536,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def run() -> dict:
    import gc
    import json
    import tempfile
    from pathlib import Path

    import llmcompressor
    import torch
    import torchao
    import transformers

    from firefly.quant.deploy import evaluate_deployed
    from firefly.quant.optimize import optimize_over_schemes
    from firefly.report import render_optimize_schemes

    print(
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"torchao {torchao.__version__} | llmcompressor {llmcompressor.__version__} | "
        f"{torch.cuda.get_device_name(0)}"
    )

    inputs = Path(tempfile.mkdtemp()) / "calib.json"
    inputs.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    # Gate on the SERVED distribution: 200 chat pairs (dolly), scored on the
    # assistant turn — the raw-text bar refused int4 for damage that only
    # exists off-distribution. (Calibration stays raw text; GPTQ needs
    # activations, not a metric.)
    from firefly.quant.evaluate import load_eval_texts

    eval_samples = load_eval_texts(Path(CHAT_EVAL))

    r = optimize_over_schemes(
        MODEL, inputs, eval_samples, schemes=("int4wo", "int8wo"), quality_bar=QUALITY_BAR,
        group_size=128, device="cuda", dtype="bfloat16", max_length=EVAL_MAX_LENGTH,
        out_dir="/tmp/gemma4_opt", benchmark=False,
    )
    print(render_optimize_schemes(r))

    gc.collect()
    torch.cuda.empty_cache()

    # Vendor baseline: Google's QAT W4A16 compressed-tensors export, scored with
    # the identical evaluator + eval set the optimize run used.
    qat_ppl = evaluate_deployed(
        QAT_MODEL, eval_samples, max_length=EVAL_MAX_LENGTH, device="cuda", dtype="bfloat16"
    )

    fp_ppl = r["winner"]["quality"]["fp"]
    result = {
        "chosen_scheme": r["chosen_scheme"],
        "met_bar": r["met_bar"],
        "per_scheme": r["per_scheme"],
        "fp_ppl": fp_ppl,
        "served_ppl": r["winner"]["quality"]["served"],
        "served_rel_to_fp": r["winner"]["quality"]["served_rel_to_fp"],
        "treatments": (r["winner"].get("artifact") or {}).get("treatments"),
        "method_fallback": (r["winner"].get("artifact") or {}).get("method_fallback"),
        "diagnosis_summary": r["winner"].get("diagnosis_summary"),
        "headroom": r["winner"].get("headroom"),
        "qat_ppl": qat_ppl,
        "qat_rel_to_fp": (qat_ppl - fp_ppl) / fp_ppl if fp_ppl else None,
    }
    # Durable results: the local client streaming this run has died to network
    # flake twice, cancelling the in-flight call. Print in-container (logs are
    # retrievable via `modal app logs`) AND persist to the volume, so a
    # finished run can always be read back with the `fetch` entrypoint.
    print("RESULT_JSON: " + json.dumps(result, default=str))
    result_path = Path("/root/.cache/huggingface/firefly_results")
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / "gemma4_optimize.json").write_text(json.dumps(result, indent=2, default=str))
    hf_cache.commit()
    return result


@app.function(image=image, timeout=120, volumes={"/root/.cache/huggingface": hf_cache})
def read_result() -> dict:
    import json
    from pathlib import Path

    p = Path("/root/.cache/huggingface/firefly_results/gemma4_optimize.json")
    if not p.exists():
        raise FileNotFoundError("No persisted result yet — the run hasn't finished.")
    return json.loads(p.read_text())


@app.local_entrypoint()
def fetch() -> None:
    """Read back a finished run's persisted result (modal run <script>::fetch)."""
    _render(read_result.remote())


@app.local_entrypoint()
def main() -> None:
    """Fire-and-forget: .spawn() + --detach makes the run immune to local
    client death (a .remote() await that dies cancels the in-flight call —
    it killed two runs mid-GPTQ). Results land in the volume; read them with
    `modal run <script>::fetch` once the app shows finished."""
    handle = run.spawn()
    print(f"spawned: {handle.object_id} — poll with `modal run {__file__}::fetch`")


def _render(r: dict) -> None:
    print(f"\n=== Gemma 4 12B-it: optimize vs Google QAT (bar {QUALITY_BAR:.0%}) ===")
    print(f"  fp perplexity:            {r['fp_ppl']:.3f}")
    for s in r["per_scheme"]:
        rel = s["served_rel_to_fp"]
        rel_s = f"{rel:+.1%}" if rel is not None else "n/a"
        print(f"  {s['scheme']:8s} served rel_to_fp {rel_s}  ({s['compression']:.1f}x est)  meets_bar={s['meets_bar']}")
    print(f"  chosen: {r['chosen_scheme']}  met_bar={r['met_bar']}  treatments={r['treatments']}")
    if r.get("method_fallback"):
        fb = r["method_fallback"]
        print(f"  method fallback: {fb['from']} → {fb['to']} ({fb['reason'][:80]}…)")
    print(f"  diagnosis: {r['diagnosis_summary']}")
    if r["headroom"]:
        print(f"  headroom: {r['headroom']}")
    print(f"\n  Google QAT W4A16 ({QAT_MODEL.rsplit('/', 1)[-1]}):")
    print(f"    perplexity {r['qat_ppl']:.3f}  rel_to_fp {r['qat_rel_to_fp']:+.1%}")
    if r["chosen_scheme"] == "int4wo" and r["served_rel_to_fp"] is not None:
        gap = r["served_rel_to_fp"] - r["qat_rel_to_fp"]
        print(f"    PTQ-vs-QAT gap (served rel_to_fp): {gap:+.1%}")
