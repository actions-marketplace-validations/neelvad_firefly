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

## Two strategies, measured head to head

How you score "sensitivity" is a pluggable strategy:

- **isolated** — quantize *only* layer `i`; rank by the divergence it causes
  alone. This surfaces the *intrinsically* hard-to-quantize layers (the late,
  high-activation/outlier layers).
- **marginal** — quantize *all but* layer `i`; rank by how much keeping it fp
  *recovers*. The seemingly more decision-relevant signal.

I expected `marginal` to win. It doesn't:

![Recipe recovery by strategy on SmolLM-135M, W8A8](plots/quant_recipe_strategies.png)

On SmolLM-135M / W8A8, quantizing all 30 layers moves the output **66.7%**.
Both strategies agree the single worst layer is `layer.28` (the model's
massive-activation layer) — keeping it alone recovers 37%. But after that they
diverge: keeping the top-4 **isolated** layers recovers **71%**, while the top-4
**marginal** layers recover only **46%**.

Why? `marginal` measures each layer's recovery *in a context where every other
layer is still quantized* — dominated by catastrophic upstream error, that
signal favors early layers whose recovery doesn't transfer once you keep several
layers in fp. `isolated` ranks intrinsic difficulty, which picks the late
outlier layers that actually matter in a multi-layer recipe.

The honest takeaway: a single-pass ranking (either kind) is imperfect because a
layer's contribution is **non-additive**. The better recipe likely needs a
**greedy/iterative** strategy — add the layer that most improves the *current*
recipe, re-measure, repeat — which is exactly the "spend more compute, get a
better recipe" knob. The pluggable-strategy seam makes that a drop-in.

## Reproduce it

```sh
# the comparison plot above (CPU, ~20s):
uv run python scripts/demo_quant_recipe.py

# rank layers by sensitivity:
firefly quant-sensitivity -m HuggingFaceTB/SmolLM-135M -i golden.json --scheme w8a8

# build + verify a recipe:
firefly quant-recipe -m HuggingFaceTB/SmolLM-135M -i golden.json \
    --scheme w8a8 --k-values 1,2,4,8
```

`--scheme int4wo` runs the same thing for int4 weight-only (needs a CUDA GPU).

[repo]: https://github.com/neelvad/firefly
