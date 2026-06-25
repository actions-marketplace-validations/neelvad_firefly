---
layout: default
title: "Firefly: measured, attribution-guided model compression"
---

# Firefly — a measurement oracle for model compression

*Draft. Honest about scope; the numbers are real and the limits are stated.*

## The one idea

Firefly is **one capability** — *detect and attribute numerical divergence in
model execution* — pointed at the problems where "did this change break the
model, and where?" is the hard question. It started as a numerical-parity CI
gate (does this serving stack still produce the same activations as my
reference?) and grew into the **measurement oracle** for a quantization/compression
agent: a thing that, given a proposed model transform, *applies it, verifies it
against a real metric, attributes the residual per-layer, and reports the cost* —
so a search over transforms is grounded in measurement instead of guesswork.

The thesis, in one line: **you cannot trust a compression recipe across
architectures — you have to measure it — and the value is in the measurement +
attribution + verification, not the technique.**

## Why this matters (the problem)

Quantization (and pruning, low-rank, …) ships on faith in an average. Three
gaps the field mostly papers over:

- **No accuracy bound.** Post-training quantization gives a *sample estimate on
  one distribution*, not a guarantee. Error is input-dependent and worst exactly
  in the tails benchmarks under-sample.
- **Aggregate metrics hide localized failure.** A 0.1 perplexity bump can conceal
  that one capability collapsed. Averaging washes out concentrated damage.
- **Recipes don't transfer.** The config that recovers 94% on one model can
  silently ship garbage on another (we measured exactly this — see below).

The honest consequence: the useful artifact isn't a faster blind search, it's a
**truthful, per-model, localized verdict** on what a transform did.

## The engine

Forward-hook telemetry captures per-layer activations at architecturally stable
tap points. Two runs (reference vs candidate) are diffed per tap; a forward-order
walk names the **first divergent tap** — the layer where two executions part. The
same machinery drills to the worst attention head, and (via `TorchDispatchMode`)
to the first diverging ATen op inside a layer. Tolerances are *calibrated* from
controlled-noise re-runs, so the gate distinguishes real divergence from FP
jitter.

This is the verifier. It's empirical (measures specific inputs), not a sound
proof — but it's cheap relative to retraining and *non-gameable*: you can't
reward-hack a measured perplexity the way you can an RLHF reward model.

## The compression loop

On top of the engine: **diagnose → route → verify.**

- **Sensors** emit failure-mode *signatures* from measurements: activation
  outliers (`channel_concentration`), salient weight channels (`|W|·|X|`
  concentration), single-unit dominance (one layer's quant sensitivity ≫ the
  rest).
- **A closed intervention seam** is the action space: `apply(model, policy,
  calib) -> model'` — one opaque verb, mechanism inside. Shipped interventions:
  RTN, SmoothQuant (a pre-transform that migrates activation outliers into the
  weights), AWQ (activation-aware int4, wrapping torchao). Precision (which layers
  at what bits) is *data* the agent edits, not a plugin.
- **The router** (deterministic, non-LLM) maps each signature to the intervention
  that treats it: activation-outliers → SmoothQuant, salient-weights → AWQ,
  single-unit → keep-fp.
- **The measurement gate** is the point: the router *proposes* by signature, the
  *measurement decides*. A routed recipe ships only if it actually beat the
  plain-quant baseline.

A recipe is a serialized `(PrecisionPolicy, [Intervention])` artifact — the
action is **sandboxed by construction** (it can only compose validated
interventions, never run code) and **measured by construction**.

## The agent framing

Firefly is the *oracle*; an external coding agent is the *searcher*. This is the
generator–verifier decomposition that makes Lean-style proof agents work — a
smart-but-unsound generator + a dumb-but-sound-enough, cheap, non-gameable
verifier — applied to compression. The closest prior art (CEG4N) pairs an SMT
equivalence checker with a *genetic* searcher; AlphaEvolve pairs an LLM searcher
with a *scoring* evaluator. Firefly occupies the reachable cell between them:
**LLM searcher × measured-parity verifier × the structured intervention algebra**,
with first-divergence attribution as the counterexample-localization that turns
"reject" into "here's the layer to re-tune."

Two proposers plug into the same `diagnose → recipe` slot:
- **Deterministic router** — reproducible, no hallucination surface; the default.
- **LLM proposer** (Anthropic tool-use) — for composition/tradeoff cases where a
  fixed rule can't reach the answer. It emits a *compact, structured* recipe;
  every proposal is verified; a bad one wastes one measurement, not a deploy.

## The evidence (real numbers, honest limits)

**SmoothQuant recovers w8a8 degradation — across families.** Small models where
w8a8 actually degrades, measured per-model with the gate:

| model | family | fp → w8a8 → +SmoothQuant | recovered | gate |
|---|---|---|---|---|
| Qwen2.5-1.5B | Qwen | 12.4 → 22.3 → 12.6 | 98% | accept |
| Qwen2.5-0.5B | Qwen | 18.7 → 23.3 → 19.2 | 88% | accept |
| SmolLM2-1.7B | Llama-arch | 11.4 → 53.9 → 27.4 | 62% | accept |
| Gemma-2-2b | Gemma | 25.7 → 25.5 → 25.8 | — | **reject** (w8a8 already lossless) |

The reject is the point: where w8a8 is lossless, the gate *declines* SmoothQuant
rather than faking a recovery.

**AWQ recovers int4 — but the tooling is Qwen-overfit, and the gate caught it.**
On Qwen2.5-7B int4, the deterministic auto-quant autonomously routes to AWQ and
recovers **91%** of the degradation (where mixed-precision recovers ~9%). But a
4-family int4 sweep surfaced that the int4/AWQ path *doesn't transfer*: torchao's
int4 silently *breaks* Mistral-7B (perplexity 179k — isolated to a path bug, not
the model), and AWQ *regresses below plain RTN* on Gemma/Llama — both of which the
measurement gate **rejected**. The headline isn't "AWQ wins everywhere" (false);
it's "the same recipe that recovers 91% on one model ships garbage on another,
and measurement is the only thing that tells you which."

**An LLM proposer composes a win a fixed rule can't (N=1).** On Qwen2.5-1.5B int4
with a "minimum memory at a perplexity bar" goal, AWQ-alone misses the bar; the
LLM adds an attribution-guided keep-fp set and navigates the memory frontier to a
verified recipe at **2× compression**. This is one model, a grounded sandboxed
demo — not yet a cross-architecture result.

## Honest scope

- It is a **diagnosis-routed, measured recipe selector + oracle-grounded search**,
  for the regime where no known recipe exists (custom / under-explored
  architectures). On Llama-family where the recipe is known, a fast blind search
  (torchao autoquant) wins.
- It is **not** an autonomous technique-search agent, and not a sound verifier —
  for the property that matters (deployment faithfulness), checking ≈ running the
  model, so the cheap-verifier asymmetry that makes Lean magic does not transfer.
- **Known limits, stated:** the cheap proxy (divergence) ≠ the real metric
  (perplexity); per-layer signals don't generalize across families (we falsified
  our own `channel_concentration` predictor: ρ=0.71 on one family, 0.23–0.42 on
  another); and the eval is currently ~50-text perplexity — below the grade
  ("thousands of sequences + a task metric") that "better than a human" would
  require.

## Maturity

| surface | status |
|---|---|
| Parity CI gate | mature, broadly validated (3 runners: HF/vLLM/SGLang) |
| Quant diagnosis + deterministic auto-quant | built & validated, cross-family |
| LLM proposer / search harness | grounded sandboxed harness; one composition win (N=1) |
| General autonomous agent | aspirational |

## What would make it convincing (not more code — more measurement)

Run the LLM-vs-router comparison on 3–5 architectures; grow the eval to thousands
of sequences + a downstream task metric (the `Evaluator` callable seam already
takes one — the set is the gap); calibrate the diagnosis thresholds across a model
family. The infrastructure exists; it's an evidence-breadth problem now. That's a
good place to be.
