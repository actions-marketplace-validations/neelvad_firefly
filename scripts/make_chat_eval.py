"""Build a chat-formatted eval set from databricks-dolly-15k (CC-BY-SA 3.0).

Instruction-tuned / QAT models must be gated on their *served* distribution
(chat), not raw text — measured on Gemma 4, the same checkpoints score
near-uniform on raw prose and best-in-class on chat. This produces the eval
artifact both the optimize gate and the vLLM adjudication read: Firefly's
``{"chat": [{"user": ..., "assistant": ...}, ...]}`` eval schema.

Deterministic (seeded) selection: context-free open_qa/general_qa samples with
mid-length responses, so the assistant turn carries enough scorable tokens
without dragging eval time.

Run:  uv run python scripts/make_chat_eval.py   # writes scripts/prompts/chat_eval_dolly.json
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)
OUT = Path(__file__).parent / "prompts" / "chat_eval_dolly.json"
N = 200
SEED = 20260708


def main() -> None:
    raw = urllib.request.urlopen(URL).read().decode()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    candidates = [
        {"user": r["instruction"].strip(), "assistant": r["response"].strip()}
        for r in rows
        if r.get("category") in ("open_qa", "general_qa")
        and not r.get("context")
        and 15 <= len(r["response"].split()) <= 80
        and len(r["instruction"].split()) <= 40
    ]
    rng = random.Random(SEED)
    picked = rng.sample(candidates, N)
    OUT.write_text(
        json.dumps(
            {
                "_source": "databricks/databricks-dolly-15k (CC-BY-SA 3.0), "
                f"seed={SEED}, filtered to context-free open_qa/general_qa "
                "with 15-80-word responses",
                "chat": picked,
            },
            indent=1,
        )
    )
    n_words = sum(len(p["assistant"].split()) for p in picked)
    print(f"wrote {OUT} — {N} samples, ~{n_words} assistant words")


if __name__ == "__main__":
    main()
