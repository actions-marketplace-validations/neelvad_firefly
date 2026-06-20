"""(a)-path spike: does wrapping torchao's AWQ as a QUANTIZER intervention work,
and does it beat plain int4 — closing the distributed-damage gap the
mixed-precision recipe couldn't (7B int4 recipe recovered only ~9%)?

Compares, on one model, all-Linear int4:
  * plain int4wo (RTNQuantizer) — the baseline the recipe struggled to recover.
  * AWQ int4 (AWQQuantizer wrapping torchao.prototype.awq) — same 4-bit
    footprint, activation-aware scaling.
through the real Pipeline + measurement code (divergence vs fp + perplexity).

De-risk on 1.5B/A10G first (does the torchao AWQ flow run through our seam?),
then bump MODEL/GPU to the 7B/A100 where the int4 gap was measured.

Run:  uv run modal run scripts/validate_awq_gpu.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-validate-awq-gpu")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "torchao>=0.7", "transformers>=4.44", "accelerate",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # bump to Qwen/Qwen2.5-7B-Instruct + A100-80GB
GPU = "A10G"
GROUP_SIZE = 128

_CALIB = [
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
    "In 1969, Apollo 11 landed the first humans on the Moon during the Space Race.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Supply and demand determine prices in a competitive market economy.",
]
_EVAL = [
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Gradient descent iteratively updates parameters to minimize a loss function.",
    "Mount Everest, on the border of Nepal and Tibet, is Earth's highest peak.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "A hash table offers average constant-time lookups using a hash function.",
    "DNA encodes genetic instructions using four nucleotide bases.",
    "The transformer architecture relies on self-attention to model sequences.",
    "Jupiter is the largest planet in the solar system, a gas giant with many moons.",
    "Recursion solves a problem by reducing it to smaller instances of itself.",
    "Electrons occupy discrete energy levels around an atomic nucleus.",
    "A neural network learns features hierarchically across its layers.",
    "Compound interest grows savings exponentially over long horizons.",
    "Entropy measures the disorder of a thermodynamic system.",
    "Batch normalization stabilizes training by rescaling layer activations.",
]


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/root/.cache/huggingface": hf_cache})
def run() -> dict:
    import copy
    import json
    import tempfile
    from pathlib import Path

    import torch

    from firefly.capture import load_golden_inputs, load_model_and_tokenizer, run_capture
    from firefly.quant.awq import AWQQuantizer
    from firefly.quant.evaluate import perplexity_evaluator
    from firefly.quant.intervention import Pipeline, PrecisionPolicy, RTNQuantizer
    from firefly.quant.sensitivity import discover_units
    from firefly.quant.torchao import rel_l1

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  model={MODEL} gs={GROUP_SIZE}")

    work = Path(tempfile.mkdtemp())
    calib = work / "calib.json"
    calib.write_text(json.dumps({"texts": _CALIB, "max_length": 64}))

    model, tok = load_model_and_tokenizer(MODEL, device="cuda", dtype=torch.bfloat16)
    batch = load_golden_inputs(calib, tok, "cuda")
    ref_out = run_capture(model, batch)["final_norm"]
    all_fqns = {f for fqns in discover_units(model, "layer").values() for f in fqns}
    ev = perplexity_evaluator(_EVAL, max_length=64)
    ppl_fp = ev(model, tok)

    policy = PrecisionPolicy(scheme="int4wo", group_size=GROUP_SIZE, quantize=all_fqns)

    def measure(m) -> tuple[float, float]:
        return rel_l1(ref_out, run_capture(m, batch)["final_norm"]), ev(m, tok)

    # plain int4wo
    m_plain = Pipeline(quantizer=RTNQuantizer()).run(copy.deepcopy(model), policy)
    d_plain, ppl_plain = measure(m_plain)
    del m_plain
    torch.cuda.empty_cache()

    # AWQ int4 (same footprint, activation-aware)
    m_awq = Pipeline(quantizer=AWQQuantizer(group_size=GROUP_SIZE)).run(copy.deepcopy(model), policy, batch)
    d_awq, ppl_awq = measure(m_awq)

    # recovery of the int4 perplexity gap that AWQ closes vs plain int4
    gap_plain = ppl_plain - ppl_fp
    gap_recovered = (ppl_plain - ppl_awq) / gap_plain if gap_plain > 0 else 0.0
    verdict = {
        "model": MODEL,
        "group_size": GROUP_SIZE,
        "perplexity": {"fp": round(ppl_fp, 3), "int4_plain": round(ppl_plain, 3), "int4_awq": round(ppl_awq, 3)},
        "divergence": {"int4_plain": round(d_plain, 4), "int4_awq": round(d_awq, 4)},
        "awq_recovers_int4_gap": round(gap_recovered, 4),
        "awq_beats_plain_int4": ppl_awq < ppl_plain,
    }
    print("\n" + "=" * 64)
    print(f"perplexity  fp {ppl_fp:.2f} → int4 {ppl_plain:.2f} → AWQ int4 {ppl_awq:.2f}")
    print(f"divergence  int4 {d_plain:.1%} → AWQ int4 {d_awq:.1%}")
    print(f"AWQ recovers {gap_recovered:.0%} of the int4 perplexity gap; beats plain int4: {ppl_awq < ppl_plain}")
    print("=" * 64)
    return verdict


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
