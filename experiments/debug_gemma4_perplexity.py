"""Debug: why does the perplexity evaluator return ~1e8 on Gemma 4?

The optimize run scored gemma-4-12B-it at fp perplexity 2.9e8 (loss ~19.5 —
worse than uniform over the 262k vocab), for fp AND the QAT checkpoint alike,
while the parity run proved the forward itself is healthy. So the suspect is
the loss path: `_perplexity` does `model(ids, labels=ids)` and trusts
`out.loss`, and Gemma4UnifiedForConditionalGeneration (multimodal CG class)
may shift/mask labels differently than a plain CausalLM.

This isolates it on gemma-4-E2B-it (same gemma4_unified class, ~2.3B):
  A. generation sanity — does "The capital of France is" continue with Paris?
  B. out.loss from model(ids, labels=ids)      (the evaluator's current path)
  C. manual shifted cross-entropy from out.logits
  D. B vs C with keyword input_ids=
  E. the text-only Gemma4UnifiedForCausalLM class, if loadable, same checks

Run:  uv run modal run experiments/debug_gemma4_perplexity.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-debug-gemma4-ppl")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

MODEL_ID = "google/gemma-4-E2B-it"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "sentencepiece")
    .add_local_python_source("firefly")
)


@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/root/.cache/huggingface": _HF_CACHE})
def probe() -> dict:
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    report: dict = {"model_class": type(model).__name__}

    import inspect

    sig = list(inspect.signature(model.forward).parameters)
    report["forward_params_head"] = sig[:6]

    # A. generation sanity
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
    gen = model.generate(ids, max_new_tokens=8, do_sample=False)
    report["generation"] = tok.decode(gen[0], skip_special_tokens=True)

    text = "Photosynthesis converts sunlight into chemical energy stored in glucose."
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"].to("cuda")

    with torch.no_grad():
        # B. positional (the evaluator's current call)
        out_pos = model(ids, labels=ids)
        report["loss_positional"] = float(out_pos.loss)

        # D. keyword
        out_kw = model(input_ids=ids, labels=ids)
        report["loss_keyword"] = float(out_kw.loss)

        # C. manual shifted CE from the same logits
        logits = out_kw.logits
        report["logits_shape"] = list(logits.shape)
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        manual = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        report["loss_manual_shifted"] = float(manual)
        # and the unshifted variant, in case the class shifts internally already
        aligned = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
            ids[:, :-1].reshape(-1),
        )
        report["loss_manual_unshifted_diag"] = float(aligned)

    report["ppl_positional"] = math.exp(report["loss_positional"])
    report["ppl_manual_shifted"] = math.exp(report["loss_manual_shifted"])

    # E. the text-only class, if transformers exposes it for this checkpoint
    try:
        from transformers import Gemma4UnifiedForCausalLM

        lm = Gemma4UnifiedForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
        with torch.no_grad():
            report["text_only_loss"] = float(lm(input_ids=ids, labels=ids).loss)
        del lm
    except Exception as e:  # noqa: BLE001 — diagnostic probe, record and move on
        report["text_only_loss"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"

    return report


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(probe.remote(), indent=2, default=str))
