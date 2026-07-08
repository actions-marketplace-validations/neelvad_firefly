"""Debug: is Google's Gemma 4 QAT W4A16 checkpoint really broken under vLLM 0.23,
or is our logprob scoring the broken part?

The adjudication scored it at +935,022% vs fp — but the community serves these
exact checkpoints successfully (deployment guides exist), so the prior is that
the checkpoint is fine and something in our harness/version is wrong.

Discriminator, one container, vLLM 0.23 (same env as the adjudication):
  A. greedy GENERATION from the QAT model — chat-templated and raw prompts.
     Coherent text → execution is fine and our scoring path is the suspect.
     Garbage → real execution/checkpoint problem in this environment.
  B. our prompt_logprobs perplexity on 3 eval texts (the adjudication method).
  C. per-position top-1 logprobs for one text — if logits saturate at the
     softcap wall (like vLLM issue #39407's FP8 double-scale), the pattern
     shows it.

Run:  uv run modal run experiments/debug_gemma4_qat.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-debug-gemma4-qat")

_HF_CACHE = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)

QAT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
FP_MODEL = "google/gemma-4-12B-it"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.23.0", "sentencepiece", "huggingface_hub>=0.24")
    .add_local_python_source("firefly")
)

_TEXTS = [
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Gradient descent iteratively updates parameters to minimize a loss function.",
]


@app.function(image=image, gpu="A100-80GB", timeout=1800, memory=65536,
              volumes={"/root/.cache/huggingface": _HF_CACHE})
def probe() -> dict:
    import math
    import os as _os

    _os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tok = AutoTokenizer.from_pretrained(FP_MODEL)
    llm = LLM(model=QAT_MODEL, dtype="bfloat16", max_model_len=256, gpu_memory_utilization=0.85)
    report: dict = {}

    # A. generation — chat-templated (community usage) and raw completion.
    chat = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        tokenize=False, add_generation_prompt=True,
    )
    gen_params = SamplingParams(temperature=0.0, max_tokens=24)
    outs = llm.generate([chat, "The capital of France is"], gen_params, use_tqdm=False)
    report["generation_chat"] = outs[0].outputs[0].text
    report["generation_raw"] = outs[1].outputs[0].text

    # B. the adjudication scoring method.
    def ids_for(text: str) -> list[int]:
        ids = tok(text, truncation=True, max_length=64)["input_ids"]
        if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
            ids = [tok.bos_token_id] + ids
        return ids

    prompts = [TokensPrompt(prompt_token_ids=ids_for(t)) for t in _TEXTS]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate(prompts, params, use_tqdm=False)
    nll, n = 0.0, 0
    per_pos: list[float] = []
    for out, p in zip(outs, prompts, strict=True):
        ids = p["prompt_token_ids"]
        for pos in range(1, len(ids)):
            entry = out.prompt_logprobs[pos]
            if entry is None:
                continue
            lp = entry[ids[pos]].logprob
            nll += -lp
            n += 1
            if len(per_pos) < 16:
                per_pos.append(round(lp, 3))
    report["ppl_promptlogprobs"] = math.exp(nll / n)
    report["first16_token_logprobs"] = per_pos
    return report


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(probe.remote(), indent=2, default=str))
