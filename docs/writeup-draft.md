---
layout: default
title: "Your quantization numbers were measured in the wrong place"
---

# Your quantization numbers were measured in the wrong place

*Draft. Every number below is measured; every limit is stated. The point is not
"use our tool" — it's four specific, checkable ways a quantization measurement
lies, and one discipline that catches them.*

Quantization tooling asks you to trust a number. You quantize a model, run an
eval, see "1.2% perplexity increase" or "98% recovery," and ship. The problem is
that the number was almost always produced somewhere other than where the model
runs in production — a different quantization backend, a proxy metric, a smaller
model, a single layer in isolation — and **the number does not transfer.**

We built a measurement engine (it started life as a numerical-parity CI gate for
model deployments) and pointed it at quantization, with one rule: *always measure
the model you actually ship, on the metric that matters.* That rule kept catching
the tooling lying. Here are four cases, each one a thing worth checking in your
own pipeline.

## Finding 1 — the backend you measure in isn't the backend you serve in

The most common quantization split in practice: you measure with one library
(it has the nice per-layer hooks and filters) and deploy with another (it's what
your serving engine loads). We measure with **torchao** and deploy with
**compressed-tensors**. Same scheme name, same bit-width — different
implementation.

On Qwen2.5-1.5B, int4 weight-only:

| | perplexity |
|---|---|
| fp | 9.38 |
| int4, measured in **torchao** | 21.45 |
| int4, served via **compressed-tensors** | **18.73** |

That's a **~29%-of-fp gap between the number you measured and the model you
ship** — and, notably, the deployed model is *better* than the measurement said.
For int8 the two backends agree to within ~2%; for int4 they diverge hard,
because int4 packing and calibration differ between implementations.

**The lesson:** a quantization number is only valid for the exact backend that
produced it. If you measure in framework A and serve from framework B, re-measure
in B. We made this a default step — re-evaluate the *served* checkpoint — and it
paid for itself immediately, twice more below.

## Finding 2 — "98% recovery" can be a serving no-op

SmoothQuant is a well-known technique for recovering int8 activation-quantization
quality. In torchao, on Qwen2.5-1.5B, it works exactly as advertised: plain w8a8
wrecks the model (18.1 perplexity vs 9.4 fp), and SmoothQuant brings it back to
9.7 — a near-full recovery.

We wired the *same* SmoothQuant into the served (compressed-tensors) export,
expecting the same recovery. Instead:

| w8a8 on Qwen2.5-1.5B (served) | perplexity |
|---|---|
| plain RTN | 20.82 |
| **+ SmoothQuant** | **20.82** |

Bit-identical. SmoothQuant did *nothing* when served — the calibration ran, the
smoothing was applied to every layer (we checked the logs), and the output was
unchanged. The reason is mechanical: SmoothQuant migrates outliers from
activations into weights, but compressed-tensors' W8A8 uses a per-token /
per-channel activation-quant granularity that is *invariant* to exactly that
rescaling. The recovery it shows in torchao is real — and specific to torchao's
activation-quant granularity. **It is a measurement artifact of the framework you
measured in, and it does not exist in the one you serve from.**

**The lesson:** a recovery technique's benefit is a property of the (technique ×
quant-granularity × backend) triple, not the technique alone. "SmoothQuant
recovers X%" is meaningless without naming where it was measured. Our re-eval gate
flagged this automatically — it re-scored the served model, saw 20.82 not 9.7, and
refused to claim a recovery.

## Finding 3 — per-layer mixed precision helps at 1.5B and *hurts* at 7B

"Keep the fragile layers in higher precision, quantize the rest" is the intuitive
recipe for mixed-precision. We tested whether a *cheap* per-layer fragility
ranking (measure each layer's int4 sensitivity once) predicts which layers, kept
at fp16, actually recover the served int4 model.

On Qwen2.5-1.5B it works and transfers cleanly. Keeping 4 layers fp16, choosing
*which* four:

| kept fp16 (1.5B) | served perplexity |
|---|---|
| all-int4 | 12.66 |
| top-4 (by our ranking) | **11.28** |
| random-4 | 11.73 |
| bottom-4 | 11.96 |

Monotonic: the ranking is real. So we scaled to 7B, expecting a *larger* effect
(outlier features sharpen with model size). Instead it **inverted**:

| kept fp16 (7B), K=4 | served perplexity |
|---|---|
| all-int4 | **9.93** |
| top-4 | 10.10 |
| random-4 | 10.35 |
| bottom-4 | 10.16 |

At 7B, keeping *any* layers fp16 made the served model *worse* than plain
all-int4 — and the "most fragile" layers were the worst to protect. Why:
**bigger models are more int4-robust, so int4+GPTQ is already near-lossless at
7B — there is no per-layer fragility left to exploit, because GPTQ's calibration
correction already absorbed it.** The cheap ranking measures fragility *without*
the recovery method; the served model *has* it. And pulling the early layers out
of the GPTQ set disrupts its sequential error-compensation, so protecting them
backfires.

**The lesson:** per-layer sensitivity measured on a bare quantizer does not
predict a served model that includes a recovery method — and the whole
optimization's value *shrinks* with scale exactly where you'd want to deploy it.
The 1.5B result alone would have justified building a feature that's worthless at
7B. Measuring at the scale you deploy is the only thing that caught it.

## Finding 4 — for recsys, AUC tells you the wrong component to protect

Everything above is LLMs. Recommendation models are the more interesting
quantization target precisely because they're *heterogeneous* — big and small
embedding tables, cross layers, a deep MLP — so a single precision can't fit all
of them, and you have to decide per component. We trained a DCN-v2 on
MovieLens-1M and int4-quantized each component in isolation.

By **AUC** (the offline ranking metric), int4 is nearly free and flat across
components — nothing to see. But AUC is rank-based; it is *blind to calibration*,
and a recommendation model's output is a probability that feeds a downstream
auction, where a calibration shift is real money. Measured by **calibration error
(ECE)** instead:

| component (int4) | ΔAUC | ΔECE |
|---|---|---|
| head | −0.0025 | **+0.0058** |
| side embeddings | −0.0010 | **+0.0045** |
| cross layers | −0.0009 | +0.0023 |
| big embeddings | −0.0018 | +0.0014 |
| deep MLP | −0.0001 | −0.0004 |

The calibration ranking **is not the AUC ranking**: by AUC you'd protect the big
embeddings (2nd); by calibration they're 4th, and the side embeddings jump to 2nd.
The deep MLP is free on both. So the offline proxy metric points you at the *wrong*
component to keep in higher precision.

**The lesson:** measure the metric your deployment actually cares about, per
component — not the aggregate offline proxy. (Honest caveat: on a small model the
magnitudes are small; this effect wants production-scale tables to become large.
But the *ranking flip* is the point, and it's already visible.)

## What actually works (and ships)

The findings are cautionary; the constructive half is real too. Measuring the
served model let us build a loop that ships what it verifies:

- **int4 recovery that serves.** Plain int4 RTN serves at +113% perplexity;
  GPTQ recovers it to +4% (~96%), AWQ ~94% — *measured on the served checkpoint*,
  on Qwen2.5-1.5B. That's a real, deployable 4× smaller model.
- **Pick the scheme by a quality bar.** Give a perplexity bar; the tool ships the
  most-compressed scheme that meets it — bar 10% → int8wo (2×, +2.7%), bar 30% →
  int4wo (4×). Same model, the bar decides.
- **Measured cost, not estimated.** Real serving throughput/memory from vLLM:
  fp8 is +20% decode *and* −24% prefill (weight quant helps memory-bound decode,
  costs compute-bound prefill — a regime split only measurement reveals).

These live behind one command — `firefly optimize <model> --quality-bar <b>` →
a servable compressed-tensors checkpoint + a `vllm serve` line + the measured
evidence.

## The engine underneath

None of this is quantization-specific. The core is a divergence-attribution
engine — **capture → compare → attribute** — that hooks every layer's activations
and names the first place two model executions diverge, down to the attention head
or ATen op. Pointed at serving stacks instead of quantization, the same engine is
a numerical-parity CI gate; it's how we found, in an earlier phase, that
`FLASH_ATTN` vs `XFORMERS` diverge at exactly one attention head across a 60×
model-scale range, and surfaced a live vLLM/FlashInfer bug that silently zeroed
two of Qwen-7B's attention heads. The attribution is the moat: it's why the tool
can say *which* layer a quant broke and *whether the model it ships* matches the
one it measured — which is the whole story above.

## Honest scope

- The re-eval gate, int4 recovery, and multi-scheme search are built and
  GPU-validated on Qwen2.5 (1.5B/7B). Cross-architecture breadth (3–5 families +
  a downstream task metric) is the open evidence work.
- Per-layer mixed precision is a real mechanism whose payoff is modest on every
  regime we can currently access (int4-robust LLMs, toy-scale recsys); it's
  parked pending fp4/mx4 tooling or production-recsys scale, deliberately.
- The tool measures and verifies; it does not prove. There is no worst-case
  accuracy bound for post-training quantization — the honest guarantee is "we
  measured the model you're about to ship, on your metric," which is a great deal
  more than most tooling offers, and less than a proof.
