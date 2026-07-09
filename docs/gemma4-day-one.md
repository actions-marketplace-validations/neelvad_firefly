---
layout: default
title: "Quantizing a day-one model: a case study in being wrong twice"
---

# Quantizing a day-one model: a case study in being wrong twice

*Gemma 4 12B went through our quantization gate the week it landed in our
stack. The gate refused our int4 export at +581% perplexity, and scored
Google's own QAT checkpoint at 34 million — six orders of magnitude broken.
Both verdicts were measured on the served artifact, replicated across two
serving stacks, and wrong. This is the story of the week: what a day-one
architecture breaks, what it costs to be honest about your own numbers, and
the result that survived — a 4× smaller model the gate initially refused to
ship.*

Everything below is measured; the scripts are in the repo. The punchline table,
for the impatient:

| gemma-4-12B-it, served via vLLM | chat eval (what it serves) | raw-text eval |
|---|---|---|
| fp (24.0 GB) | 25.2 | 3,665 |
| **int8wo, ours, 2×** (13.2 GB) | +0.7% | −0.6% |
| **int4-GPTQ, ours, 4×** | **+3.0%** | **+581%** |
| **Google QAT W4A16, 4×** | **−10.4%** (better than fp) | +936,000% |

Same four artifacts in both columns. The only thing that changes is what you
ask them to do.

## Act 1 — what a day-one architecture actually costs

The plan was ordinary: run the newest well-regarded open family through our
standard loop — activation-parity check, per-layer quantization signature,
then `optimize` (diagnose → route a recovery → export a servable checkpoint →
re-evaluate what was exported) with Google's official QAT release as the
comparison baseline.

The parity leg was almost anticlimactic. Gemma 4 is a multimodal "unified"
architecture — the text decoder nests under `model.language_model` next to
vision and audio towers — and supporting it took a two-line addition to the
tap-path conventions. After that: 48 layers, 145 tap points, capture →
calibrate → self-check clean on an A100. Tenth model family through the
engine.

The quantization leg was not anticlimactic. Six consecutive runs failed, each
on a different seam, and every failure was a real gap that is now a fix with a
unit test:

| # | what broke | what it taught the product |
|---|---|---|
| 1 | llm-compressor can't auto-create a processor for multimodal checkpoints | pass the tokenizer explicitly at export |
| 2 | GPTQ was about to quantize vision/audio tower Linears that receive **zero** activations under text-only calibration | tower ignore-patterns, mirroring what Google's own QAT export excludes |
| 3 | AWQ (which our diagnosis correctly routed to) crashes: llm-compressor's smooth-layer mappings don't resolve on an architecture its registry doesn't know | automatic AWQ → GPTQ fallback, recorded in the artifact manifest, never silent |
| 4 | the model scored *near-uniform* perplexity on everything | Gemma 4's tokenizer (transformers v5) doesn't add BOS, and the model collapses without it — the evaluator now ensures BOS |
| 5 | the exported checkpoint wasn't self-contained — no loadable tokenizer | export copies the source checkpoint's tokenizer/processor *files* (instantiating the processor class needs vision/audio deps a quant environment shouldn't carry) |
| 6 | vLLM refused our artifact: it constructs fp-kept modules as quantized and dies loading their fp weights | llm-compressor resolves ignore patterns against the *transformers* module tree, but vLLM *renames modules* (`model.embed_vision.patch_dense` → `vision_embedder.patch_dense`); the export now re-appends raw regex patterns, which match at load time under any engine's naming |

And the environment fought back independently of our code: the vLLM release
that was four days old crashed at engine warmup for *any* model (an unrelated
model's Triton kernels fail to JIT at import); the flashinfer sampler
JIT-compiles at startup and requires `nvcc` that slim images don't have;
pip-resolving vLLM and llm-compressor together silently forces a vLLM source
build that dies on a CUDA version mismatch.

None of this is a complaint. It's the honest bill for "supports new
architectures," itemized. A day-one model is a stress test that visits every
assumption your export path never knew it was making — and each line in that
table is a class of model that now works.

## Interlude — the signature

Before quantizing, the per-layer sensor pass said Gemma 4 was the most
outlier-saturated family we had measured: 118 of 145 tap points flagged for
activation outliers at int8 (previous families flagged a *concentrated*
handful — Qwen's late-MLP `down_proj`s, SmolLM's single layer-11 hotspot),
with residual-stream channel concentration over 100× by layer 1. Outlier
features are exactly what int4 crushes first. Hold that thought.

## Act 2 — two catastrophic verdicts, and the probe that killed them

The optimize run finished on the seventh attempt and delivered its verdicts.
Our int4-GPTQ export: **+581% perplexity vs fp, served** (the same GPTQ
recovery that lands within ~5% of fp on Qwen). The gate refused it and shipped
int8 at 2× instead. And Google's official QAT W4A16 checkpoint, scored by the
same evaluator on the same eval set: **34,309,611 perplexity**.

We want to be precise about how strong that second number looked. It was
measured on the served artifact — the discipline this whole tool exists to
enforce. It replicated across two independent serving stacks: transformers'
compressed-tensors loader said 31M, vLLM said 34M. The fp baseline scored
sanely under both (they agreed within 6%), so the harness wasn't broken. Our
own int8 artifact scored within noise of fp in the same environment, so the
loader wasn't broken either. Every cross-check we normally run had passed. The
headline wrote itself: *vendor ships QAT checkpoint that produces garbage as
served.*

The reason we didn't publish it is the same discipline, pointed inward:
people demonstrably serve these checkpoints in production, and deployment
guides for this exact artifact exist. Extraordinary verdicts about other
people's artifacts deserve one more probe. So, one discriminating experiment,
same environment, same vLLM version:

- Chat-formatted prompt: *"What is the capital of France?"* → **"The capital
  of France is Paris."** Flawless.
- Raw-text completion: *"The capital of France is"* → **"111111111111…"**
- Per-token logprobs on plain prose: **≈ −27**, when a *uniform* distribution
  over its 262k vocabulary would score −12.5.

Nothing was broken. The QAT checkpoint executes perfectly on its served
distribution and assigns essentially zero probability to raw prose — QAT
fine-tuning ran on chat-formatted data, and the raw-text language model
underneath is simply gone. Our eval was scoring chat models on a distribution
they never serve. The fp model's suspiciously high absolute perplexity
(~3,600 on short raw text, agreed by both stacks) was the same effect in
milder form — instruction tuning had already eaten most of its raw-text
ability, and nobody noticed because nobody serves it raw text.

Retraction, then, in full: **the checkpoint was fine; our eval was
off-distribution; there is no vendor bug.** And a worse realization
immediately behind it: our own int4 refusal had been produced by the same
eval.

## Act 3 — measured where it serves

The fix was structural, not a patch to a script: eval sets in the product now
accept chat pairs (`{"chat": [{"user": …, "assistant": …}]}`), scored on the
assistant turn with the template positions masked, so the entire select →
export → re-evaluate loop gates on the serving distribution. We built a
200-sample eval from dolly-15k (~11,000 scored tokens, 65× the pilot set) and
re-ran everything: the optimize gate end-to-end, and all four artifacts under
vLLM on both distributions.

The scoreboard at the top of this page is the result. What it says:

- **The gate now ships int4 at 4×.** Same code, same 10% bar: under the
  raw-text eval it had shipped 2× int8; on the served distribution, int4-GPTQ
  re-evaluates at **+2.1%** (transformers loader) / **+3.0%** (vLLM) and
  passes. The eval set decides what you can ship — which is exactly why it
  has to be yours, and has to match production.
- **Every earlier "catastrophe" resolves.** Google's QAT scores 22.5 — *better
  than the fp model* (−10.4%; QAT included further chat training) — and the
  two loaders agree on it to three significant figures (22.57 vs 22.54). The
  pilot eval's oddities (int4 scoring "better than fp" at −12.6% on 10
  sentences) wash out as small-sample noise at 200.
- **The honest PTQ-vs-QAT gap: +12.8%.** At identical 4× weight compression,
  Google's QAT beats our afternoon of GPTQ by about thirteen points of
  relative perplexity. Theirs cost a training run; ours cost one A100-hour
  and no data beyond 22 calibration sentences. Both are real options; now the
  trade is priced.
- **The serving win is stable and measured**: int8wo decodes +30–50% faster
  than fp across bench repetitions (499 vs 383 tok/s in the final run, batch 8)
  at 1.81× less weight memory — and pays 18–27%
  on compute-bound prefill (across bench repetitions), a regime split you only
  see by measuring.

## The finding that survived everything

Strip away the false alarms and one genuinely interesting result remains —
visible only because we ended up with both eval columns:

| | chat (head of distribution) | raw text (tails) |
|---|---|---|
| int8 PTQ | intact (+0.7%) | intact (−0.6%) |
| int4 PTQ | intact (+3.0%) | **destroyed (+581%)** |
| int4 QAT | **improved (−10.4%)** | **abandoned (+936,000%)** |

Quantization damage on this model is not uniform noise sprayed over the
distribution — it eats the distribution **tails-first**. int8 preserves
everything. int4 PTQ preserves the well-reinforced head behavior and destroys
the residual raw-text tail. QAT is the limiting case: it *improves* the head,
having traded the tails away entirely during training. The per-layer signature
suggests the mechanism — outlier features, which Gemma 4 has in unusual
pervasiveness, are widely implicated in carrying rare, long-tail capability,
and they are precisely what 4-bit weight grids crush first.

We fence this claim deliberately: one model family, perplexity-only, and a
lower-than-fp perplexity from a quantized model can partly reflect entropy
flattening rather than quality. It's a hypothesis with one clean data point,
not a law. But it reframes what a quantization gate should even ask: not "did
the model get worse?" but "*where* did it get worse, and do you serve there?"

## What we'd tell you to check in your own pipeline

1. **Gate on the distribution you serve.** An off-distribution eval doesn't
   add noise — it reverses verdicts in both directions (refuses shippable
   models, condemns healthy ones). If you serve chat, evaluate chat.
2. **Re-evaluate the exported artifact, not the model in your measurement
   framework.** Three of our six export failures produced checkpoints that
   would load *somewhere* and break *elsewhere* — the naming-scheme ignore bug
   only exists at the serving engine's loader.
3. **Two independent stacks, always.** Agreement between transformers and
   vLLM is what let us trust the retraction as much as the original alarm.
4. **Probe alarms with generation before publishing them.** Perplexity says
   "how broken"; thirty seconds of greedy decoding says "broken *how*" — and
   `"Paris"` vs `"111111"` told us more than any scalar.
5. **Pin your serving engine a release back.** The four-day-old vLLM crashed
   at warmup for every model on the planet. Day-one models are enough
   excitement without day-four engines.

## Honest scope

- One model family (Gemma 4 12B-it); the tails-first mechanism needs
  cross-family replication before it's a claim rather than a hypothesis.
- Perplexity-only; a downstream task metric is the next evidence step.
- Our int4 uses GPTQ at group-128 vs the QAT checkpoint's group-32 — the
  +12.8% gap conflates method and granularity.
- GPTQ calibration used raw-text sentences even in the final run; calibrating
  on served-distribution data is an obvious next experiment.
- The per-layer signature magnitudes were measured before the BOS fix and are
  treated as rankings only.

*The engine, the gate, and every script referenced here are in the
[repo](https://github.com/neelvad/firefly); the companion piece — four more
ways a quantization number lies, measured — is
[here](writeup-draft.html).*
