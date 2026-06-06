---
layout: default
title: "Firefly: a numerical-parity CI gate for ML"
---

# What I found running per-layer activation diffs against vLLM

A few months ago I started building **[Firefly][repo]**, a small tool that
diffs a candidate ML model's per-layer activations against a calibrated
reference and tells you the first layer where they disagree. The intended
use case is a CI gate: you point it at your model on every PR and it
fails loud if the residual stream moves.

This post is about what fell out of running it against vLLM.

The most useful finding, before any setup:

> **vLLM 0.8.5 V0 → V1 is bit-equal at 9-token prompts and diverges at
> every single layer past the first PagedAttention block boundary**,
> with the *exact same attention kernel on both sides.* On
> Llama-3.1-8B, divergence saturates at ~2.8% final-layer relative
> error from 1k tokens through 4k tokens — it's a step function, not
> a slope. A unit test that uses short prompts would pass this
> comparison. Production would silently change.

![Llama-3.1-8B V0 vs V1 + FLASH_ATTN — per-tap rel error at 9, 1k, 2k, 4k tokens](plots/llama_v0_vs_v1_length_curve.png)

The rest of this post is how I got here.

## What's Firefly

It's a CLI plus a GitHub Action wrapper. The model is:

1. Register **forward hooks** on every decoder layer's `self_attn`, `mlp`,
   and residual-stream outputs (plus `final_norm`). For a 30-layer model
   that's 91 "tap points."
2. Run the model on a fixed batch of golden inputs and stash each tap's
   output tensor to disk in a `weights.safetensors` + `manifest.json`
   reference dir. That's the **reference**.
3. To check a candidate, run the candidate against the same inputs, diff
   per-tap, and walk the results in forward order to attribute the
   *first* tap where the diff exceeds a calibrated tolerance. That's the
   "first divergence" — actionable in a way `eval_score = 0.87 → 0.83`
   isn't.

The architecture is intentionally boring. The capture, diff, and
attribution modules are pure functions; the orchestrators wrap them with
the model-loading and file-I/O. There's no online inference layer, no
hosted dashboard, no novel ML. The interesting part is the picks:

- **Hook at the module boundary, not inside.** vLLM and HF transformers
  both expose `model.layers[i].self_attn` as a top-level submodule. Even
  though their internals differ (vLLM fuses QKV into one Linear, HF keeps
  them separate), the module-level *output* is a tensor of the same
  shape. We tap there.
- **Calibrate the noise floor empirically.** Every tap gets its own
  tolerance derived from re-running the reference 8 times. Flat
  thresholds are wrong in both directions — too tight on noisy taps,
  too loose on quiet ones.
- **One forward-ordered list drives everything.** The tap selector
  returns layer 0's `self_attn`, then `mlp`, then layer 0 residual, then
  layer 1's `self_attn`, ... etc. The capture loop, the diff loop, and
  the first-divergence walk all iterate this same list. "First
  divergence" semantics fall out for free.

## The setup I tested against

Everything below is **SmolLM-135M in BF16 on an NVIDIA A10G**, prompt =
`"the quick brown fox jumps over the lazy dog"` unless noted. Captures
ran on Modal. vLLM 0.8.5 with `enforce_eager=True` (to keep forward
hooks working; CUDA graphs would skip them).

The reason for `enforce_eager=True` deserves a flag: in eager mode, hooks
work because every forward op is dispatched through the Python
interpreter. With CUDA graphs enabled — which is what production vLLM
actually runs — hooks would force graph breaks. That means **Firefly as
currently written is a CI-time diagnostic, not a shadow-mode capture
against live traffic**. For shadow mode you'd need a custom op
(`torch.ops.firefly.capture(tensor, name)`) that Dynamo treats as
opaque-but-tensor-in/tensor-out. That's a separate engineering arc.

## The matrix

I ran 7 paired comparisons across the standard vLLM knobs — version,
engine, attention backend, prompt length, decode mode, batch size. The
short summary:

| Comparison | Result |
|---|---|
| 0.7.3 V0 vs 0.8.5 V0, auto backend | bit-equal |
| 0.8.5 V0 vs V1, auto backend | bit-equal |
| 0.8.5 V0 vs V1, FLASH_ATTN | bit-equal |
| 0.8.5 V0, FLASH_ATTN vs XFORMERS | **diverges, first at `layer.7.self_attn`** |
| Multi-request, V0 vs V1 (2 prompts) | shape mismatch (V0 packs, V1 doesn't) |
| **V0 vs V1, FLASH_ATTN, 300 tokens** | **diverges at every tap, first at `layer.0.self_attn`** |
| FLASH_ATTN vs XFORMERS, 8 decode tokens | **diverges everywhere, first at `layer.0.self_attn@token_0`** |

Three of these are real findings worth unpacking.

## Finding 1: Different attention kernels → first divergence at `layer.7.self_attn`

Same model, same vLLM, same hardware, same prompt. Only difference:
`VLLM_ATTENTION_BACKEND=FLASH_ATTN` versus `XFORMERS`. Both are correct
implementations of scaled dot-product attention; they use different
reduction orders for the matmul.

![FLASH_ATTN vs XFORMERS — prefill](plots/flash_vs_xformers.png)

Layers 0–6 are bit-equal. Then at layer 7, the divergence cuts in
sharply — `layer.7.self_attn` is the first non-zero tap, and from there
relative error climbs through the residual stream to 1.4% by the final
LayerNorm. The shape of the curve is the textbook
"compounding-rounding-error" pattern in a deep network.

The interesting part isn't *that* this happens — anyone who has worked
on numerical kernels expects different reduction orders to round
differently in low-precision. The interesting part is **why divergence
starts at layer 7 specifically, not layer 0.**

Both backends produce different intermediate values starting at layer 0.
But at low activation magnitudes (early layers), BF16's 7-bit mantissa
rounds those tiny differences to the same representable value. Around
layer 7 of SmolLM-135M, residual-stream magnitudes have grown enough
(this is the activation-magnitude phase transition Dettmers et al.
documented in [LLM.int8()][dettmers]) that the kernel-level differences
cross the BF16 representable threshold.

So "first divergence at layer 7" is really *"first layer where the
kernel-rounding difference exceeds the precision representable
threshold."* Firefly is correctly attributing to a *coupling of three
things* — kernel difference × activation magnitude × precision-format
floor.

I expected the boundary to move on a model with larger early-layer
activations. The next section is the experiment I ran to test that
prediction, and the result that made me rewrite this paragraph.

## Finding 1.5: I tried to break the layer-7 finding on Llama-3.1-8B. It didn't budge.

The cleanest stress test of the mechanism above is to rerun the
comparison on a much bigger model. A 7-8B model has ~7× wider residual
streams than SmolLM-135M, presumably hits the BF16-visible regime
earlier, and should move the first-divergence layer.

I ran FLASH_ATTN vs XFORMERS on `meta-llama/Llama-3.1-8B` on an
A100-40GB — same vLLM 0.8.5 V0, same BF16, same 10-token prompt. The
result:

![Llama-3.1-8B FLASH_ATTN vs XFORMERS, per-tap relative error in forward order](plots/llama_8b_flash_vs_xformers.png)

|  | SmolLM-135M | Llama-3.1-8B |
| --- | --- | --- |
| first divergent tap | `layer.7.self_attn` | `layer.7.self_attn` |
| taps diverging | 70/91 (77%) | 76/97 (78%) |
| final-norm relative error | ~1.4% | ~0.96% |
| hidden dim | 576 | 4096 |
| layers | 30 | 32 |

**First divergence at layer 7 — on a 60× larger model with a 7× wider
residual stream.** Same layer index. Similar fraction of taps diverging.
Similar overall divergence curve shape.

Plotting per-layer activation magnitudes explains why my prediction was
wrong:

![Per-layer activation magnitudes — Llama-3.1-8B](plots/magnitudes_llama_8b.png)

Early-layer magnitudes are comparable across both models (~2-4). The
residual stream is initialized from an embedding lookup that's roughly
unit-norm regardless of hidden dimension; the model-specific
activation-growth patterns only kick in after a handful of layers.
Llama's MLP output eventually grows to ~40 (vs SmolLM's ~14), but by
then we are well past the rounding-threshold crossover.

So the *mechanism* in Finding 1 was right but the prediction at the end
of it was wrong. The corrected story:

- The FLASH_ATTN vs XFORMERS kernel-reduction-order difference is
  scale-invariant per element.
- Early-layer activation magnitudes are similar across model scales
  because of how embeddings initialize the residual stream.
- BF16's representable threshold is a property of mantissa width, not of
  the model.
- The product (early-layer-magnitude × kernel-diff) crosses BF16's
  representable threshold at roughly the same depth in both networks.

This makes the finding *stronger* than I thought it was. "First
divergence at `layer.7.self_attn`" isn't a SmolLM-135M quirk; it's a
property of how these two attention kernels compose in BF16, robust
across 60× model scale. Same diagnostic, two architectures, same first
divergent tap — and the per-layer attribution made the universality
visible in one plot.

These next two findings still surprised me.

## Finding 2: Decode capture exposes that layer 0 itself diverges, via KV cache pollution

Same FLASH_ATTN vs XFORMERS comparison, but now I let the model generate
7 additional tokens after the prompt and captured activations at each
decode step. Tap names get suffixed: `layer.7.self_attn@prefill` for the
prompt forward, `layer.7.self_attn@token_0..token_6` for each decode
step.

![Per-position diff between FLASH_ATTN and XFORMERS through prefill and 7 decode steps](plots/flash_vs_xformers_decode.png)

Three things happen at once in this plot:

1. **Prefill (dark purple, bottom):** the familiar layer-7 onset story.
   Layers 0–6 bit-equal, sharp jump at 7, climbs to ~1.4% by
   `final_norm`. This is what we already had.

2. **token_0 (just above prefill):** *every layer* now diverges,
   including `layer.0.self_attn`. The prompt-time KV cache that
   layer 0's decode-step attention reads from already contains
   diverged tail-layer values from prefill. There is no
   "layer 0 starts clean" regime at decode time.

3. **token_1 through token_6 (stacked progressively higher):** each
   successive curve sits above the last. The KV cache lengthens with
   every generated token and accumulates more diverged outputs, so each
   new token's forward computation reads from a *progressively more
   polluted* cache. By token 6, `final_norm` is at 3.7% relative error,
   up from 1.4% at prefill.

In forward order with the unified tap naming (prefill first, then
token_0, token_1, ... per tap), the *first* divergent tap is
**`layer.0.self_attn@token_0`**, not `layer.7.self_attn` as it is in
prefill-only mode. The KV cache propagation makes layer 0 itself the
entry point of divergence at decode time.

This destroys the "output-level monitoring will eventually catch it"
argument that ML observability vendors lean on. The final-LayerNorm
rescales by ~50× by the end of the network — output drift is *small
percent-of-scale*. An eval that thresholds at 1% accuracy delta passes
this comparison at token 0 and might keep passing for 50 tokens before
the accumulated drift crosses the threshold.

## Finding 3: The same engine swap that's safe at 9 tokens is broken at 1k — and stays broken

This is the load-bearing finding. The plot at the top of the post.

Same vLLM 0.8.5. Same FLASH_ATTN. Only difference: the V0 engine vs the
V1 engine. I ran the comparison at four prompt lengths on
Llama-3.1-8B (A100-40GB, BF16):

| prompt length | taps diverging | first divergence | final-norm rel error |
| --- | --- | --- | --- |
| 9 tokens | **0 / 97** (bit-equal) | — | 0% |
| 1k tokens | 97 / 97 | `layer.0.self_attn` | **2.84%** |
| 2k tokens | 97 / 97 | `layer.0.self_attn` | **2.62%** |
| 4k tokens | 97 / 97 | `layer.0.self_attn` | **2.86%** |

I expected "monotonically growing with length." That's *not* what
happens. **The curve is a step function**: bit-equal at very short
prompts, immediately maxed-out divergence past the first PagedAttention
block boundary, and roughly flat from 1k tokens onward. Length isn't
the threshold; block-count is.

Why? V0 uses flat attention — compute the full $QK^T \to \text{softmax}
\to V$ product in one go. V1 uses PagedAttention, which is vLLM's
marquee feature: the KV cache is sharded into 16-token blocks, attention
is computed block-by-block, and an online-softmax merge stitches the
per-block scores back together. The math is *equivalent*. The reduction
order is *different*. BF16 makes the difference visible.

At 9 tokens, only one block is involved. Single-block PagedAttention is
arithmetically identical to flat attention — same reduction order, same
bit pattern. At 1k tokens, 60+ block boundaries are crossed and the
online-softmax merge has accumulated enough rounding error that *every*
tap is past tolerance. Crossing from 1k to 4k adds more block
boundaries but the per-element error from the merge is already
saturated.

The implication for the product story is:

- **Short-prompt unit tests pass.** Anyone testing their vLLM upgrade
  with the typical 8-to-30-token prompts you find in test fixtures
  would see "V0 → V1 is bit-equal" and call the upgrade safe.
- **Any realistic production prompt is in the divergent regime.** 1k
  tokens is below most production prompt lengths. Whatever length you
  test at past the block boundary, you see the same ~2.8% final-norm
  drift.
- **The kernel is literally identical.** This is not a kernel-swap bug;
  it's a *blocking strategy* bug. The argument "FLASH_ATTN on V0 and
  FLASH_ATTN on V1 are doing the same math" turns out to be true only
  at trivially-short context — below the block-boundary threshold.

This is the finding I'd want a CI gate to catch — quietly,
automatically, before deploy. A short-prompt unit test would not. A
benchmark eval might or might not, depending on how sensitive the eval
metric is to ~3% absolute internal drift that final_norm rescales down.

## Finding 4: FLASHINFER diverges at layer 0 — and its error grows with length

vLLM ships three attention backends — FLASH_ATTN, XFORMERS, and
FLASHINFER. FLASHINFER is the one production stacks at Together,
Fireworks, and DeepSeek actually use, because its split-K
parallelization beats FlashAttention 2 for single-query decode.
Getting it installed on Modal was annoying — `flashinfer-python` on
PyPI is a stub requiring a CUDA-specific wheel; attempts on
`debian_slim` failed with "CUDA_HOME not set", on the
`vllm/vllm-openai` Docker image with a Python 3.12 aiohttp ABI
collision, and finally worked on a clean `nvidia/cuda` devel base
with explicit pip-install of vLLM and flashinfer.

Once it was working, the result:

![Llama-3.1-8B V0 + FLASH_ATTN vs FLASHINFER — per-tap rel error at 9 and 1k tokens](plots/llama_flash_vs_flashinfer_length_curve.png)

| comparison | length | first divergent tap | early-layer rel | final-norm rel |
| --- | --- | --- | --- | --- |
| **SmolLM-135M FLASH vs FLASHINFER** | 9 tokens | `layer.0.self_attn` | 0.0519% | 2.49% |
| **Llama-3.1-8B FLASH vs FLASHINFER** | 9 tokens | `layer.0.self_attn` | 0.0516% | 1.31% |
| **Llama-3.1-8B FLASH vs FLASHINFER** | 1k tokens | `layer.0.self_attn` | 0.0749% | **3.29%** |

Two new things relative to the earlier findings:

**1. Layer 0, not layer 7 — and universal across model scale.** The
FLASHINFER vs FLASH_ATTN per-element kernel-difference is much larger
than XFORMERS vs FLASH_ATTN's: ~0.05% relative at layer 0 versus
~0.0001% at layer 0 for XFORMERS. The bigger per-element diff crosses
BF16's representable threshold *immediately* in early-layer
activations, instead of needing the magnitude growth through layers 0
through 6 that XFORMERS required. And the first-divergence layer is
the same on SmolLM-135M (0.0519%) and Llama-3.1-8B (0.0516%) — exactly
the cross-scale universality Finding 1.5 predicted, just at a different
layer index because the kernel changed.

**2. The length curve is monotonic, not a step function.** Final-norm
relative error grows from 1.31% at 9 tokens to 3.29% at 1k — a 2.5×
increase. That's a different shape than the V0 vs V1 length curve in
Finding 3, which plateaus from 1k through 4k. The explanation is that
FLASHINFER has two superimposed sources of difference:

- The per-attention-kernel reduction-order diff (visible at 9 tokens,
  baseline ~1.3% final).
- An additional length-dependent diff from FLASHINFER's own paging /
  block-merge strategy (compounds as more attention positions
  participate in each token's computation).

V0 vs V1 with the *same* FLASH_ATTN kernel only had the second source,
and that source saturated past one block. FLASH vs FLASHINFER has
both, and they sum.

**Synthesis.** Firefly's per-layer attribution distinguishes three
distinct failure modes that all look like "model output drifted" at
the eval level:

| failure mode | example | first divergent tap | length curve |
| --- | --- | --- | --- |
| kernel reduction-order (small) | FLASH vs XFORMERS | `layer.7.self_attn` | unknown |
| kernel reduction-order (large) | FLASH vs FLASHINFER | `layer.0.self_attn` | monotonic growth |
| blocking strategy | V0 vs V1, same kernel | bit-equal short, `layer.0.self_attn` long | step function |

Same attribution tool, three different signatures. Useful in
production: an SRE seeing "Firefly says first divergence is
layer.0.self_attn and the rel error grew between 1k and 4k" knows the
kernel itself changed; "first divergence is layer.7" knows it's a
subtler kernel swap; "bit-equal at short and divergent at long" knows
it's a blocking-strategy change.

## What I think this means for ML CI

A few unromantic takeaways:

**Numerical-parity testing is real, and the real failure mode is the
boring one.** I was hoping to find dramatic bugs: a quantization kernel
returning NaN, a kernel that off-by-ones at a boundary, a regression
shipped by accident. What I found instead was *correct code with
different reduction orders.* The vLLM team isn't doing anything wrong.
Their engine rewrite is mathematically equivalent. Their kernel swap is
mathematically equivalent. The numerical-parity failures come from
"mathematically equivalent" not being the same thing as "bit-identical
in low precision."

**Tolerance calibration is environment-sensitive.** Same-machine
calibration measures runs-on-this-machine variance; cross-machine FP
variation is a different (and often larger) noise distribution. I
discovered this the hard way when the first GitHub Actions run of
Firefly's own demo lit up like a Christmas tree because I'd calibrated
on Apple Silicon and the Action ran on x86 Ubuntu. The fix in the
product is a `--max-rel-error` ceiling that composes with the per-tap
calibration. The lesson: "calibrated tolerances" alone aren't portable;
they need an environment-stationarity escape hatch.

**Per-layer attribution earns its keep.** "Your eval dropped 2 points"
sends a developer on a multi-hour fishing expedition through git blame.
"`layer.7.self_attn` is the first divergent tap" points directly at the
attention kernel. The cost of producing this attribution is N forward
hooks and a sort — trivially cheap. The cost of *not* producing it
shows up every time someone debugs a serving-stack regression.

**Decode-step capture changes the story.** Prefill-only is the easy
case. Decode is where the production knobs (PagedAttention, scheduler,
spec decode) actually live, and it's where divergence compounds via KV
cache. If you're going to build numerical-parity tooling for LLM
inference, decode capture is the part that matters; prefill-only tools
will miss most real upgrade-time bugs.

## Limitations I should flag

- **Two model sizes, one model family.** SmolLM-135M and Llama-3.1-8B,
  both Meta-architecture. The layer-7 universality I show in Finding 1.5
  is N=2; a non-Meta architecture (Qwen, Mistral, Gemma) would tighten
  the claim from "robust on two Meta models" to "robust across model
  family." I haven't run that yet.
- **One precision format primarily.** Most of my runs are BF16; the
  earlier validation work showed FP32 is bit-deterministic on the same
  setup, and FP16 behaves like BF16 from a reproducibility standpoint
  (different mantissa, same general pattern). Quantized regimes
  (INT8/INT4) would surface different failure modes.
- **One inference engine.** Firefly currently has a vLLM-specific
  capture path. SGLang and TGI would each need their own. The engine-
  internal differences (apply_model vs collective_rpc, etc.) are the
  per-engine engineering cost.
- **Eager mode only.** As noted, hooks can't survive CUDA graphs, so
  Firefly today is a CI-time tool. Shadow-mode capture against live
  traffic would require a custom op. That's the v3 line item.

## Reproduce

The full repo is at **[github.com/neelvad/firefly][repo]**. To rerun the
matrix locally:

```sh
git clone https://github.com/neelvad/firefly && cd firefly
uv sync --all-extras
uv run python scripts/run_vllm_suite.py
```

The reference dirs for the V0/V1/FLASH_ATTN/XFORMERS combinations
referenced here aren't in the repo (they need a GPU to produce), but
`scripts/capture_vllm.py` is the script that produced each one and the
test suite YAML at `scripts/vllm_test_suite.yml` declares each
(reference_a, reference_b, expected) tuple so you can regenerate them.

The headline length-curve comparison is eight commands on Modal
A100-40GB, ~$2–5 total. A single 9-token + 1k pair is enough to see
the step-function behavior if you want to skip 2k / 4k:

```sh
# 9-token bit-equal baseline
uv run modal run scripts/capture_vllm.py \
  --vllm-tag 0.8.5 --engine v0 --attention-backend FLASH_ATTN \
  --model meta-llama/Llama-3.1-8B --gpu A100-40GB --gpu-memory-utilization 0.7 \
  --out llama_v0_flash_short

uv run modal run scripts/capture_vllm.py \
  --vllm-tag 0.8.5 --engine v1 --attention-backend FLASH_ATTN \
  --model meta-llama/Llama-3.1-8B --gpu A100-40GB --gpu-memory-utilization 0.7 \
  --out llama_v1_flash_short

# 1k-token divergent regime
uv run modal run scripts/capture_vllm.py \
  --vllm-tag 0.8.5 --engine v0 --attention-backend FLASH_ATTN \
  --model meta-llama/Llama-3.1-8B --gpu A100-40GB --gpu-memory-utilization 0.7 \
  --prompt-file scripts/prompts/long_1k.txt --max-seq-len 1100 \
  --out llama_v0_flash_long1k

uv run modal run scripts/capture_vllm.py \
  --vllm-tag 0.8.5 --engine v1 --attention-backend FLASH_ATTN \
  --model meta-llama/Llama-3.1-8B --gpu A100-40GB --gpu-memory-utilization 0.7 \
  --prompt-file scripts/prompts/long_1k.txt --max-seq-len 1100 \
  --out llama_v1_flash_long1k
```

`uv run python scripts/plot_validation.py diff scripts/results/llama_v0_flash_long1k scripts/results/llama_v1_flash_long1k` produces the per-tap curve at 1k tokens. The full overlay (9 / 1k / 2k / 4k on one chart) needs the 2k and 4k pairs as well; swap the prompt file and `--max-seq-len` accordingly.

## What's next

1. **Does the layer-7 (XFORMERS) / layer-0 (FLASHINFER) universality
   extend across model families?** Findings 1.5 and 4 confirm both
   across model scale (135M → 8B, both Meta). The natural next check
   is Qwen-7B or Mistral-7B — different tokenizer, different
   layer-norm placement, different MLP topology. If both layer
   indices still hold, the claim sharpens from "robust on Meta
   architectures" to "property of the attention kernels themselves."

2. **Does FLASHINFER vs FLASH_ATTN keep growing past 1k?** Finding 4
   shows 1.31% at 9 tokens, 3.29% at 1k. V0 vs V1 saturated past 1k
   so I'd expect FLASHINFER to also saturate eventually, but I haven't
   tested 2k and 4k yet. Two more captures on Llama-3.1-8B
   ($0.50-$1.00 of Modal time) would resolve it.

If you've hit numerical-regression bugs in serving stacks and want to
compare notes, or if you'd find Firefly useful for your own CI and want
to talk about what's missing, I'm at **neel.vadoothker@gmail.com**.

[repo]: https://github.com/neelvad/firefly
[dettmers]: https://arxiv.org/abs/2208.07339
