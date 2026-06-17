---
layout: default
title: "Attribution-guided mixed-precision quantization"
---

# Which layers should you keep in high precision?

[Firefly][repo] started as a CI gate that diffs a model's per-layer activations
against a reference. The same machinery answers a question quantization people
actually have: **when int8/int4 hurts my model, *which layers* are responsible,
and what do I keep in high precision to get the quality back?**

`torchao`'s `autoquant` searches for a fast config that clears a tolerance, but
when it doesn't, it can't tell you *why* or *what to try next*. Firefly measures
the cause directly.

## The mechanism

For each decoder layer, quantize **only that layer** (everything else stays fp)
with real torchao kernels, and measure the resulting divergence at the model's
output. That ranks layers by how much their quantization hurts. Then **build and
verify a recipe**: keep the top-k most-sensitive layers in high precision,
quantize the rest, and measure how much of the degradation you actually recover.

Nothing here is predicted from a proxy — every number is a measured forward pass.

## This is a feature-selection problem

Choosing which layers to keep in high precision is, structurally, **subset
selection**: a set of "features" (layers), and you pick which to keep fp to
minimize quality loss under a budget. (The general K-bit version is categorical
per layer — a knapsack/bit-allocation problem — but the keep-fp-vs-quantize case
is exactly binary subset selection.) That maps the methods people use onto the
classic feature-selection taxonomy:

- **Filter** (cheap per-item score, then threshold) → per-layer **sensitivity
  ranking**. Our `isolated`/`marginal`; HAWQ's Hessian-trace sensitivity; AWQ's
  activation salience.
- **Wrapper** (evaluate subsets by running the model) → **search**. Our
  `greedy` (sequential forward selection); HAQ's RL bit-allocation.
- **Embedded** (learned during training) → learnable bit-widths (DNAS, QAT).

And the reason wrapper methods beat filter methods in feature selection —
**interactions / non-additivity** — is exactly what shows up here.

## Three strategies, measured head to head

How you choose the keep-set is a pluggable strategy:

- **isolated** (filter) — quantize *only* layer `i`; rank by the divergence it
  causes alone. Surfaces the *intrinsically* hard-to-quantize layers.
- **marginal** (filter) — quantize *all but* layer `i`; rank by how much keeping
  it fp *recovers*. The seemingly more decision-relevant signal.
- **greedy** (wrapper) — sequential forward selection: add the layer that most
  reduces divergence *given what's already kept*, re-measure, repeat. More
  measurements; accounts for interactions.

![Recipe recovery by strategy on SmolLM-135M, W8A8](plots/quant_recipe_strategies.png)

On SmolLM-135M / W8A8, quantizing all 30 layers moves the output **66.7%**. All
three agree the single worst layer is `layer.28` (the model's massive-activation
layer) — keeping it alone recovers 37%. After that:

- **marginal** is the surprise loser — top-4 recovers only **46%**. It measures
  each layer's recovery in a context where *everything else is still quantized*,
  so it favors early layers whose recovery doesn't transfer to a multi-layer
  recipe.
- **isolated** recovers **71%** at top-4 — its intrinsic-difficulty ranking
  picks the late outlier layers that actually matter together.
- **greedy** is provably ≥ both and beats marginal clearly, but here it only
  *ties* isolated through k=4 and edges it at k=8 (**82.8% vs 81.0%**).

The interesting part is *why greedy barely beats isolated*: `layer.28` dominates
so heavily that the interactions are weak, so the cheap filter (`isolated`)
lands on essentially the greedy-optimal set. Greedy's wrapper advantage should
widen on models where sensitivity is more distributed — bigger models, int4. So
the practical rule is the feature-selection rule: **use the cheap filter when
one item dominates; spend the wrapper compute when interactions matter.**

## Reproduce it

```sh
# the comparison plot above (CPU, ~20s):
uv run python scripts/demo_quant_recipe.py

# rank layers by sensitivity:
firefly quant-sensitivity -m HuggingFaceTB/SmolLM-135M -i golden.json --scheme w8a8

# build + verify a recipe (try --strategy isolated | marginal | greedy):
firefly quant-recipe -m HuggingFaceTB/SmolLM-135M -i golden.json \
    --scheme w8a8 --strategy greedy --k-values 1,2,4,8
```

`--scheme int4wo` runs the same thing for int4 weight-only (needs a CUDA GPU).

[repo]: https://github.com/neelvad/firefly
