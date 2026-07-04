"""Debug round 2: which knob un-breaks Gemma 4 inference under transformers?

Round 1 (debug_gemma4_perplexity.py) exonerated the evaluator — the model
itself emits ~uniform logits and loops during generation on E2B-it. Suspects,
by Gemma priors: attention implementation (Gemma-2's softcap-vs-sdpa bug),
missing BOS (Gemma quality craters without it), bf16 numerics, or a
transformers-5.12 bug in the new hybrid-attention path.

Matrix: {eager, sdpa} × {as-tokenized, forced-BOS} on one text, plus an fp32
eager point. Reports loss for each cell + tokenizer BOS facts + generation
under the best cell.

Run:  uv run modal run experiments/debug_gemma4_perplexity2.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-debug-gemma4-ppl2")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

MODEL_ID = "google/gemma-4-E2B-it"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "sentencepiece")
    .add_local_python_source("firefly")
)


@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/root/.cache/huggingface": _HF_CACHE})
def probe() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text = "Photosynthesis converts sunlight into chemical energy stored in glucose."
    enc = tok(text, return_tensors="pt")
    ids_plain = enc["input_ids"]

    report: dict = {
        "bos_token_id": tok.bos_token_id,
        "first_ids": ids_plain[0][:4].tolist(),
        "starts_with_bos": bool(tok.bos_token_id is not None and ids_plain[0, 0].item() == tok.bos_token_id),
    }

    ids_bos = ids_plain
    if tok.bos_token_id is not None and ids_plain[0, 0].item() != tok.bos_token_id:
        ids_bos = torch.cat(
            [torch.tensor([[tok.bos_token_id]], dtype=ids_plain.dtype), ids_plain], dim=1
        )

    def loss_for(model, ids) -> float:
        with torch.no_grad():
            return float(model(input_ids=ids.to("cuda"), labels=ids.to("cuda")).loss)

    results = {}
    for attn in ("eager", "sdpa"):
        model = (
            AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn)
            .to("cuda")
            .eval()
        )
        results[f"{attn}_bf16_plain"] = loss_for(model, ids_plain)
        results[f"{attn}_bf16_bos"] = loss_for(model, ids_bos)
        if attn == "eager":
            gen_in = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
            gen = model.generate(gen_in, max_new_tokens=8, do_sample=False)
            report["generation_eager"] = tok.decode(gen[0], skip_special_tokens=True)
        del model
        torch.cuda.empty_cache()

    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager")
        .to("cuda")
        .eval()
    )
    results["eager_fp32_plain"] = loss_for(model, ids_plain)
    results["eager_fp32_bos"] = loss_for(model, ids_bos)

    report["losses"] = results
    return report


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(probe.remote(), indent=2, default=str))
