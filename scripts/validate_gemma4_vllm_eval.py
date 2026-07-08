"""Gemma 4: the vLLM-served adjudication — fp vs our PTQ artifacts vs Google QAT.

The optimize run left two numbers that only serving can adjudicate:

* our int4-GPTQ was refused at +261% and Google's QAT W4A16 scored garbage —
  both measured through transformers' compressed-tensors loader, so the int4
  refusal could in principle be a loader artifact rather than a real quality
  collapse. Score everything through vLLM (the engine these checkpoints are
  built for) on the SAME ruler and find out.
* fp perplexity looked too high in absolute terms under transformers —
  transformers-vs-vLLM on identical weights is exactly the parity question.

Plan (one vLLM engine per container; results persist to the volume):

  export   — re-export our int8wo (RTN) and int4 (GPTQ) compressed-tensors
             artifacts into the volume via the hardened export_deployable.
  score ×4 — fp / int8wo artifact / int4 artifact / Google QAT, perplexity
             from vLLM prompt_logprobs over BOS-ensured token ids (identical
             math to the HF-side shifted NLL).
  bench ×2 — measured decode/prefill/memory for fp and the shipped int8wo.

A cheap CPU driver orchestrates and persists; the local entrypoint only
spawns it (local network flake cancelled two earlier .remote runs).

Run:    uv run modal run --detach scripts/validate_gemma4_vllm_eval.py::main
Fetch:  uv run modal run scripts/validate_gemma4_vllm_eval.py::fetch
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-validate-gemma4-vllm-eval")

hf_cache = modal.Volume.from_name("firefly-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

MODEL = "google/gemma-4-12B-it"
QAT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
ARTIFACT_ROOT = "/root/.cache/huggingface/firefly_results/gemma4_artifacts"
RESULT_PATH = "/root/.cache/huggingface/firefly_results/gemma4_vllm_eval.json"
EVAL_MAX_LENGTH = 128

# Two images. Scoring/bench pin vllm==0.23.0: 0.24.0 (4 days old) crashes at
# engine warmup importing minimax_m3's triton kernels (triton JIT
# source-parse bug), and the official vllm-openai docker image has no
# `python` on PATH for Modal's builder. The export leg gets the llmcompressor
# stack; co-resolving the two in one pip install forced a vllm source build
# (CUDA-mismatch death).
CHAT_EVAL = "/root/chat_eval_dolly.json"  # 200 dolly chat pairs (see make_chat_eval.py)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.23.0", "sentencepiece", "huggingface_hub>=0.24")
    .add_local_python_source("firefly")
    .add_local_file(
        str(__import__("pathlib").Path(__file__).parent / "prompts" / "chat_eval_dolly.json"),
        CHAT_EVAL,
    )
)
export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "llmcompressor", "accelerate", "sentencepiece")
    .pip_install("transformers==5.12.1")
    .add_local_python_source("firefly")
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
_EVAL = [
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Gradient descent iteratively updates parameters to minimize a loss function.",
    "Mount Everest, on the border of Nepal and Tibet, is Earth's highest peak.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "Jupiter is the largest planet in the solar system, a gas giant with many moons.",
    "Compound interest grows savings exponentially over long horizons.",
    "Newton's three laws describe the motion of objects under forces.",
    "A neural network learns features hierarchically across its layers.",
    "Entropy measures the disorder of a thermodynamic system.",
]


def _eval_token_ids() -> list[list[int]]:
    """BOS-ensured token ids for each eval text — the shared ruler."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    out = []
    for text in _EVAL:
        ids = tok(text, truncation=True, max_length=EVAL_MAX_LENGTH)["input_ids"]
        if tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
            ids = [tok.bos_token_id] + ids
        out.append(ids)
    return out


def _eval_chat_token_ids() -> list[tuple[list[int], int]]:
    """(token ids, score_start) per chat sample — the distribution
    instruction-tuned/QAT models were actually trained (and vetted) on.
    Scoring raw text instead sends them off-distribution: the QAT checkpoint
    generates perfectly in chat format yet assigns ~-27 logprobs to raw prose
    (630M "perplexity") — an eval artifact, not a broken artifact. 200 dolly
    pairs via the shared product encoder (masked template prefix)."""
    from transformers import AutoTokenizer

    from firefly.quant.evaluate import _encode_sample, load_eval_texts

    tok = AutoTokenizer.from_pretrained(MODEL)
    return [
        _encode_sample(tok, sample, EVAL_MAX_LENGTH)
        for sample in load_eval_texts(__import__("pathlib").Path(CHAT_EVAL))
    ]


@app.function(
    image=export_image, gpu="A100-80GB", timeout=3600, memory=65536,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def export_artifacts() -> dict:
    """Re-export int8wo (RTN) and int4 (GPTQ) compressed-tensors artifacts into
    the volume, via the hardened export_deployable (tokenizer re-save, tower
    ignores, explicit processor)."""
    from pathlib import Path

    from firefly.quant.deploy import export_deployable
    from firefly.quant.intervention import RTNQuantizer
    from firefly.quant.recipe_io import Recipe, serialize_intervention

    made = {}
    # Recipes carry the MEASUREMENT quantizer (rtn/awq); export_method maps
    # int4wo+rtn to a GPTQ export on its own.
    for scheme, quantizer, calib in (
        ("int8wo", serialize_intervention(RTNQuantizer()), None),
        ("int4wo", serialize_intervention(RTNQuantizer()), _CALIB),
    ):
        out = Path(ARTIFACT_ROOT) / scheme
        if (out / "config.json").exists():
            made[scheme] = "cached"
        else:
            recipe = Recipe(
                model_id=MODEL, scheme=scheme, group_size=128, granularity="layer",
                quantize_fqns=[], kept_fp_fqns=[], pre_transforms=[], quantizer=quantizer,
            )
            export_deployable(recipe, out, calib_texts=calib, calib_max_length=EVAL_MAX_LENGTH)
            made[scheme] = "exported"
        # Artifacts exported before the processor/ignore fixes lack
        # preprocessor_config etc. and carry a transformers-naming-only
        # quantization ignore list vLLM can't match. Both patches are
        # idempotent, so apply to cached dirs too.
        from firefly.quant.deploy import _copy_processor_files, _extend_config_ignores

        _copy_processor_files(MODEL, out)
        _extend_config_ignores(out)
    hf_cache.commit()
    return made


@app.function(
    image=image, gpu="A100-80GB", timeout=1800, memory=65536,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def score(source: str) -> dict:
    """Token-weighted perplexity under vLLM via prompt_logprobs, on the shared
    BOS-ensured token ids. One engine per container."""
    import math
    import os as _os

    # flashinfer's sampler JIT-compiles at warmup and needs nvcc, which the
    # slim image lacks; torch-native sampling is fine (greedy decode anyway).
    _os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import transformers
    import vllm as vllm_pkg
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(model=source, dtype="bfloat16", max_model_len=EVAL_MAX_LENGTH + 96,
              gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def _ppl(items: list[tuple[list[int], int]]) -> tuple[float, int]:
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _ in items]
        outs = llm.generate(prompts, params, use_tqdm=False)
        total_nll, total_tokens = 0.0, 0
        for out, (ids, start) in zip(outs, items, strict=True):
            plp = out.prompt_logprobs  # [None, {tok: Logprob}, ...] aligned with ids
            for pos in range(max(start, 1), len(ids)):
                entry = plp[pos]
                lp = entry[ids[pos]].logprob if entry is not None else None
                if lp is None:
                    continue
                total_nll += -lp
                total_tokens += 1
        return math.exp(total_nll / total_tokens), total_tokens

    ppl_raw, n_raw = _ppl([(ids, 1) for ids in _eval_token_ids()])
    ppl_chat, n_chat = _ppl(_eval_chat_token_ids())
    return {
        "source": source,
        "perplexity": ppl_raw,  # raw-text ppl (off-distribution for -it/QAT models)
        "perplexity_chat": ppl_chat,  # the distribution-matched number
        "n_tokens": n_raw,
        "n_tokens_chat": n_chat,
        "vllm": vllm_pkg.__version__,
        "transformers": transformers.__version__,
    }


@app.function(
    image=image, gpu="A100-80GB", timeout=1800, memory=65536,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def bench(source: str) -> dict:
    """Measured decode/prefill/memory via the library benchmarker (CUDA graphs on)."""
    import os as _os

    _os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # no nvcc in slim image

    from firefly.bench import BenchmarkConfig, get_benchmarker

    b = get_benchmarker("vllm").benchmark(
        source, BenchmarkConfig(batch_size=8, input_len=512, output_len=128),
        dtype="bfloat16",
    )
    return {
        "source": source,
        "decode_tok_s": b.decode_throughput_tok_s,
        "prefill_tok_s": b.prefill_throughput_tok_s,
        "ttft_ms": b.ttft_ms,
        "weight_mb": (b.weight_memory_bytes / 1e6) if b.weight_memory_bytes else None,
        "peak_mb": (b.peak_memory_bytes / 1e6) if b.peak_memory_bytes else None,
    }


@app.function(
    image=image, timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache}, secrets=[hf_secret],
)
def drive() -> dict:
    """CPU orchestrator: export -> parallel score/bench -> persist. Runs fully
    server-side so no local client can cancel it."""
    import json
    from pathlib import Path

    # fp + QAT don't need the export; start them immediately.
    fp_h = score.spawn(MODEL)
    qat_h = score.spawn(QAT_MODEL)
    fp_bench_h = bench.spawn(MODEL)

    # A failed export must not lose the fp/QAT legs — record and carry on.
    try:
        exported = export_artifacts.remote()
        hf_cache.reload()  # pick up the artifacts the export just committed
        int8_h = score.spawn(f"{ARTIFACT_ROOT}/int8wo")
        int4_h = score.spawn(f"{ARTIFACT_ROOT}/int4wo")
        int8_bench_h = bench.spawn(f"{ARTIFACT_ROOT}/int8wo")
    except Exception as e:  # noqa: BLE001 — persist partial results either way
        exported = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        int8_h = int4_h = int8_bench_h = None

    def _get(handle, label: str):
        if handle is None:
            return {"source": label, "error": "skipped: export failed"}
        try:
            return handle.get()
        except Exception as e:  # noqa: BLE001 — record per-leg failure, keep the rest
            return {"source": label, "error": f"{type(e).__name__}: {str(e)[:300]}"}

    result = {
        "exported": exported,
        "perplexity": {
            "fp": _get(fp_h, MODEL),
            "int8wo": _get(int8_h, "int8wo"),
            "int4wo_gptq": _get(int4_h, "int4wo"),
            "qat_w4a16": _get(qat_h, QAT_MODEL),
        },
        "benchmark": {
            "fp": _get(fp_bench_h, MODEL),
            "int8wo": _get(int8_bench_h, "int8wo"),
        },
    }
    print("RESULT_JSON: " + json.dumps(result, default=str))
    Path(RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT_PATH).write_text(json.dumps(result, indent=2, default=str))
    hf_cache.commit()
    return result


@app.function(image=image, timeout=120, volumes={"/root/.cache/huggingface": hf_cache})
def read_result() -> dict:
    import json
    from pathlib import Path

    p = Path(RESULT_PATH)
    if not p.exists():
        raise FileNotFoundError("No persisted result yet — the run hasn't finished.")
    return json.loads(p.read_text())


def _render(r: dict) -> None:
    import json

    print("\n=== Gemma 4 12B-it under vLLM: fp vs PTQ artifacts vs Google QAT ===")
    ppl = r["perplexity"]
    fp = ppl["fp"].get("perplexity")
    fp_chat = ppl["fp"].get("perplexity_chat")
    if fp:
        print(f"  engine: vllm {ppl['fp'].get('vllm')} / transformers {ppl['fp'].get('transformers')}")
    for name in ("fp", "int8wo", "int4wo_gptq", "qat_w4a16"):
        e = ppl[name]
        if "error" in e:
            print(f"  {name:12s} ERROR: {e['error']}")
            continue
        rel = f" ({(e['perplexity'] - fp) / fp:+.1%})" if fp and name != "fp" else ""
        chat = e.get("perplexity_chat")
        rel_c = (
            f" ({(chat - fp_chat) / fp_chat:+.1%})" if chat and fp_chat and name != "fp" else ""
        )
        chat_s = f"{chat:.3f}{rel_c}" if chat else "n/a"
        print(f"  {name:12s} chat-ppl {chat_s}   raw-ppl {e['perplexity']:.3f}{rel}")
    print("\n  measured serving (batch 8, in 512, out 128):")
    for name in ("fp", "int8wo"):
        b = r["benchmark"][name]
        if "error" in b:
            print(f"  {name:12s} ERROR: {b['error']}")
        else:
            wm = f"{b['weight_mb']:.0f} MB weights" if b["weight_mb"] else "weights n/a"
            print(
                f"  {name:12s} decode {b['decode_tok_s']:.0f} tok/s  "
                f"prefill {b['prefill_tok_s']:.0f} tok/s  ttft {b['ttft_ms']:.0f} ms  {wm}"
            )
    print("\n  raw: " + json.dumps(r["exported"]))


@app.local_entrypoint()
def main() -> None:
    handle = drive.spawn()
    print(f"spawned driver: {handle.object_id} — poll with `modal run {__file__}::fetch`")


@app.local_entrypoint()
def fetch() -> None:
    _render(read_result.remote())
