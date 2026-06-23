# Firefly

[![CI](https://github.com/neelvad/firefly/actions/workflows/ci.yml/badge.svg)](https://github.com/neelvad/firefly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Firefly instruments a model's internals and attributes where two executions
diverge** — down to the layer, attention head, or ATen op. It hooks every
decoder layer's attention/MLP outputs (and, on demand, per-head taps or
op-level traces via `TorchDispatchMode`), captures the activations, and walks
them in forward order to name the *first* place two runs disagree.

That one engine — **capture → compare → attribute** — is the shared core; each
use-case below is the same engine pointed at a different divergence:

| Surface | The question it answers | Maturity |
| --- | --- | --- |
| **Parity CI gate** — `firefly check` + GitHub Action | Did a kernel swap / dep bump / serving-stack drift silently change my model's activations? | **Shipped** — on the Marketplace, validated across 9 models / 8 families |
| **Quantization diagnosis** — `quant-diff` / `quant-sensitivity` / `quant-recipe` / `quant-diagnose` | *Which* layers does my quantized build break, what treats it, and which do I keep in higher precision? | **Built & verified** — diagnosis routes a detected failure mode to the intervention that treats it (activation-outliers → SmoothQuant; single-unit-dominance → mixed precision), verified against a real eval. A *general* technique-search agent is aspirational |
| **Shadow mode** — `firefly.shadow` | What are a *live production* model's internals doing? (survives `torch.compile` + CUDA graphs) | **Experimental mechanism** — GPU-validated but not deployed anywhere; overhead unmeasured |

## Parity CI gate

The core use-case is a **same-weights deployment-parity gate**: the reference
pins the exact model weights by fingerprint, and `check` re-runs *those weights*
in the candidate environment to ask "does this serving stack still produce the
same activations?" (A model whose *weights* changed — a fine-tune, or a
quantized build — is a different question: opt in with
`--allow-fingerprint-mismatch`, or use the quantization surface below.)

```yaml
# .github/workflows/firefly.yml
- uses: neelvad/firefly@v0.4.0
  with:
    reference: hf://my-org/my-firefly-ref  # captured from my-org/my-model
    candidate: my-org/my-model             # same weights, new serving stack
    inputs: tests/firefly-prompts.json
    jitter-floor: 0.001                    # ignore sub-0.1% cross-platform jitter
```

The action posts a markdown summary to `$GITHUB_STEP_SUMMARY` and exits
non-zero on divergence. On `pull_request` runs it also posts the summary as
a sticky PR comment (grant the workflow `permissions: pull-requests: write`;
a missing permission degrades to a warning, never a failed build). Outputs
(`first-divergent-tap`, `passed`, `report-path`) drive downstream steps.

## Quantization: diagnosis & mixed precision

### Gate on a quantized build (`mode: quant-diff`)

Set `mode: quant-diff` to gate on what *quantization* does: it diffs a
torchao-quantized candidate against the fp baseline, ranks the per-layer
divergence, and fails if any layer exceeds `rel-threshold`.

```yaml
# .github/workflows/firefly-quant.yml
permissions:
  pull-requests: write          # for the PR comment
jobs:
  quant-gate:
    runs-on: ubuntu-latest       # use a CUDA runner for scheme: int4wo
    steps:
      - uses: neelvad/firefly@v0.4.0
        with:
          mode: quant-diff
          reference: hf://my-org/my-firefly-ref  # fp baseline (same model)
          candidate: my-org/my-model
          inputs: tests/firefly-prompts.json
          scheme: w8a8                 # or int4wo (needs a CUDA runner)
          rel-threshold: 0.05          # fail if any layer diverges >5%
          firefly-extras: torchao      # quant-diff needs the torchao extra
```

Unlike `check`, `quant-diff` needs no calibration (it ranks by magnitude),
and the quantized candidate is *expected* to differ from the reference — the
fingerprint is taken pre-quantization, so it still matches the fp baseline and
no `allow-fingerprint-mismatch` is required.

### Attribution-guided mixed precision (`quant-sensitivity` / `quant-recipe`)

Beyond gating, Firefly does the thing `torchao autoquant` can't *explain*: it
measures, causally, how much each layer's (or Linear's) quantization hurts the
model output, then builds and **verifies** a mixed-precision recipe — keep the
most-sensitive units in high precision, quantize the rest, and report how much
output fidelity that recovers.

```sh
# rank units by how much their quantization hurts the output
firefly quant-sensitivity -m my-org/my-model -i prompts.json --scheme int4wo

# build + verify a recipe (a recovery curve, ranked by divergence)
firefly quant-recipe -m my-org/my-model -i prompts.json \
    --scheme int4wo --strategy greedy --k-values 1,2,4,8
```

It's a feature-selection problem: `--strategy {isolated,marginal,greedy}` trades
compute for quality (cheap per-unit filters → wrapper search) and
`--granularity {layer,linear}` sets the unit. Full writeup, the strategy
comparison, and the int4 result where `greedy` wins:
[docs/quant-recipe.md](docs/quant-recipe.md).

**Diagnosis-routed auto-quant (built & validated):** `firefly quant-auto`
diagnoses the failure mode, routes it to the intervention that treats it
(activation-outliers→SmoothQuant, salient-weights→AWQ, single-unit→mixed
precision), and **verifies** — shipping the recipe only if the measurement says
it actually helped. On Qwen2.5-7B int4 it autonomously reaches AWQ and recovers
~91% of the degradation (where mixed-precision recovers ~9%).

An **LLM proposer** (`firefly.quant.search`, Anthropic tool-use) plugs into the
same diagnosis→recipe slot for composition/tradeoff cases. Its own result is
*separate from the 91% above* (which is the deterministic router): on
**Qwen2.5-1.5B int4**, AWQ-alone misses a within-10%-of-fp perplexity bar, and
the LLM adds an attribution-guided keep-fp set to clear it at **2× compression** —
a composition the router's fixed rules can't reach, grounded and sandboxed (it
emits a `Recipe`, every proposal verified). Honest scope: this is **one model
(N=1)**, not a cross-architecture demonstration, and on the 7B (tiny AWQ gap) it
couldn't clear the bar within the eval's noise floor. The evidence breadth
(more architectures, a larger eval + task metric, calibrated thresholds) is the
open work; a fully general autonomous agent remains aspirational.

## Why this exists

Output-level ML monitoring (Arize, Galileo, Evidently) deliberately operates
*after* the final layer norm. By that point, layer-norm has rescaled
internal state — a residual stream that's diverged 600× internally can
collapse back to a 5% output difference after `final_norm`. Output
metrics that pass at token 0 can silently fail by token 50.

Firefly operates *inside* the residual stream. We hook every layer's
self-attention and MLP outputs, diff per-tap against a calibrated
reference, and attribute the first divergent layer in forward order.
That's actionable — "your PR moved `layer.7.self_attn`" tells you exactly
where to look — in a way "your eval dropped 2 points" doesn't.

## What we caught

The headline reproduction: same vLLM version, same model, same prompt,
same hardware. Only difference: `VLLM_ATTENTION_BACKEND=FLASH_ATTN` vs
`XFORMERS`. Both are correct kernel implementations; they just use
different reduction orders.

![Per-position diff between FLASH_ATTN and XFORMERS through prefill and 7 decode steps](scripts/plots/flash_vs_xformers_decode.png)

Three findings at once:

1. **First divergence: `layer.7.self_attn`.** The kernel difference is
   present at every layer, but BF16 rounding masks it until activation
   magnitudes grow large enough to expose it (~layer 7 on SmolLM-135M).
2. **Decode exposes layer 0 immediately.** At the first generated
   token, *all* layers diverge — including `layer.0.self_attn`, which
   was bit-equal in prefill. The decode attention path is a different
   kernel from prefill, so the rounding that stayed sub-threshold at
   layer 0 no longer does. There's no "layer 0 starts clean" regime
   once you're decoding.
3. **Divergence compounds with each token.** By token 6 the final-layer
   relative error is **3.7%** vs **1.4%** at prefill. Output monitoring
   gets a longer runway before it sees the drift.

We've validated this pattern across 9 NVIDIA GPUs × 3 storage dtypes
(FP32 / BF16 / FP16), 9 models across 8 architecture families, and the
standard vLLM engine × attention-backend matrix. Highlights: the
FLASH-vs-XFORMERS divergence starts at **exactly one attention head**
(on both SmolLM-135M and Llama-3.1-8B), and per-head attribution
surfaced a live FlashInfer bug that silently zeroes two of Qwen-2.5-7B's
attention heads. Full findings with plots in
**[the writeup](https://neelvad.github.io/firefly/)** (`docs/index.md`).

## Quickstart

```sh
# 1. Capture a reference from your current good model
firefly capture \
    --model my-org/my-model-current \
    --inputs golden.json \
    --out reference/

# 2. Calibrate per-tap tolerances (one-time, ~3 min on CPU)
firefly calibrate \
    --reference reference/ \
    --inputs golden.json \
    --runs 8

# 3. Commit reference/ to your repo, add the GitHub Action,
#    and Firefly checks every PR's candidate model against it.
```

`firefly check` is what runs in CI. It refuses to gate without a
calibrated `tolerances.json` — flat default tolerances either spam
false positives or silently miss real regressions, neither of which
delivers product value.

### Capturing from a serving engine (vLLM / SGLang)

By default capture runs the model through HF transformers. To capture
from a real serving engine instead — the configuration your production
stack actually serves — use `--runner vllm` or `--runner sglang` (each
needs its extra, `pip install 'firefly[vllm]'` / `'firefly[sglang]'`, and
a CUDA GPU). Engine knobs go through repeatable `--runner-opt`:

```sh
firefly capture --runner vllm \
    --model my-org/my-model --inputs golden.json --out reference/ \
    --runner-opt attention_backend=FLASH_ATTN --runner-opt engine=v1

firefly check --runner vllm \
    --reference reference/ --candidate my-org/my-model --inputs golden.json \
    --runner-opt attention_backend=XFORMERS

# SGLang — same flow, its own --runner-opts (attention_backend, tp_size, ...)
firefly capture --runner sglang \
    --model my-org/my-model --inputs golden.json --out reference/
```

A reference and its candidates should use the same runner — the serving
engines flatten batch/seq into one token axis, so their tensor shapes
differ from the HF runner's padded batches. Compare like with like.

## Publishing a reference

Small references (≤100 MB) commit cleanly to your repo. Past that —
or if multiple repos share a reference — host it on HuggingFace Hub
or S3 and point the action at the URI.

```sh
# After calibrating, push the reference dir to HF Hub.
# Creates the repo if it doesn't exist (needs HF_TOKEN with write).
firefly publish --reference reference/ --to hf://my-org/my-firefly-ref

# Or fuse capture + publish in one step:
firefly capture \
    --model my-org/my-model-current \
    --inputs golden.json \
    --out reference/ \
    --push hf://my-org/my-firefly-ref

# Re-publishing after recalibration works the same way — calibrate
# writes tolerances.json into the reference dir, then `--push` ships
# the updated artifact back. Useful when the reference itself lives
# on Hub: calibrate in-place, then push.
firefly calibrate \
    --reference hf://my-org/my-firefly-ref \
    --inputs golden.json \
    --push hf://my-org/my-firefly-ref

# S3 works the same way — boto3 uses your default AWS credential chain.
firefly publish --reference reference/ --to s3://my-bucket/firefly-refs/v1
```

The action then reads the same URI in CI:

```yaml
- uses: neelvad/firefly@v0.4.0
  with:
    reference: hf://my-org/my-firefly-ref
    candidate: my-org/my-model
    inputs: tests/firefly-prompts.json
```

For cloud-hosted references, tell the action which extra to install via
the `firefly-extras` input (`s3`, `gcs`, or `azure`) so the matching SDK
is present on the runner:

```yaml
- uses: neelvad/firefly@v0.4.0
  with:
    reference: s3://my-bucket/firefly-refs/v1
    candidate: my-org/my-model
    inputs: tests/firefly-prompts.json
    firefly-extras: s3
```

## Tolerance knobs

Three axes you can tune:

| Knob | Where | Default | Use case |
|---|---|---|---|
| `safety_factor` | `firefly calibrate` | 6x | Multiplies per-tap noise floor |
| `--jitter-floor` | `firefly check` | 0 (off) | Ignore relative drift below this fraction of `max\|ref\|` |
| Per-tap atol | hand-edit `tolerances.json` | — | Override calibration for known-noisy taps |

`--jitter-floor` is what makes a calibrated reference *portable* across
environments. Calibration measures variance *on the calibration machine*;
cross-machine FP variation is a different (and often larger) noise source.
A 0.1% floor absorbs typical cross-platform jitter while staying comfortably
below the >1% threshold where real bugs live. **Direction matters:** the
floor can only *loosen* the gate (effective atol becomes
`max(calibrated, jitter_floor × max|ref|)`) — it ignores drift *below* the
floor, it is not a ceiling that fails when drift exceeds it.

### Calibrating for a CPU-gated CI

Calibration only earns its keep where there's real nondeterminism to measure
(GPU / TF32 / relaxed BF16). **Pure CPU+fp32 inference is bit-deterministic**,
so `firefly calibrate` on a CPU runner derives a noise floor of ~0 and every
tap just falls back to the `1e-5` default — calibrated-on-CPU tolerances are
effectively the flat defaults, stamped `source="calibrated"`. They give a
false sense of bespoke protection.

The workflow that actually works for a CPU-gated CI is:

1. **Calibrate once on a GPU machine** that matches (or is noisier than) prod,
   and commit `tolerances.json` — those per-tap floors carry the real signal.
2. **Gate on CPU** with a `--jitter-floor` ceiling for cross-platform jitter.

If you can't calibrate on GPU, gate on `--jitter-floor` alone and skip the
calibration step — a hand-set floor is doing the real work, and a flat default
is more honest than `source="calibrated"` numbers that measured nothing.

## Architecture

| Module | Responsibility |
| --- | --- |
| `tap_points.py` | Pick stable per-layer hook sites (LLM; recsys planned for v2) |
| `capture.py` | Capture orchestration; dispatches to a `Runner` (HF default) |
| `runners/` | Pluggable capture backends behind one interface: `hf.py` (transformers, eager hooks), `vllm.py` (in-process vLLM), `sglang.py` (in-process SGLang via its native `forward_hooks`) |
| `calibrate.py` | Re-run reference under controlled noise; derive per-tap atol |
| `reference.py` | Read/write the inspectable reference artifact (safetensors + JSON) |
| `compare.py` | Per-tap diff with effective-atol composition |
| `attribution.py` | Forward-order walk → first divergent tap |
| `head_attribution.py` | Per-attention-head drill-down: which head diverged, how concentrated |
| `quant/` | Quantization surface on the engine. **Interventions** (the seam): `intervention.py` (PrecisionPolicy + Pipeline + RTN), `smoothquant.py`, `awq.py`. **Sensors/analysis**: `torchao.py` (real w8a8/int4wo + preflight), `risk.py`, `sensitivity.py` (per-unit), `salience.py` (AWQ signal), `cost.py` (memory/Pareto/budget), `evaluate.py` (perplexity + accuracy bar). **Recipe/agent**: `recipe.py`+`bar.py` (curves), `recipe_io.py` (serialize/apply a recipe), `diagnose.py`+`route.py` (diagnosis→recipe), `auto.py` (deterministic auto-quant), `step.py` (agent step primitive), `llm.py`+`search.py` (LLM proposer harness) |
| `op_drill.py` | Op-level drill-down (engine attribution rung): `TorchDispatchMode` scoped to a module → first diverging ATen op |
| `shadow/` | Shadow-mode capture package: custom ops + Triton kernel + Tappers + sinks that survive torch.compile and CUDA graphs |
| `storage.py` | Reference resolution/publish for `hf://`, `s3://`, `gs://`, `az://` |
| `report.py` | Rich-terminal table + markdown PR-comment formatter |
| `cli/` | Flat `firefly` command surface in command modules (parity / quant / drill): capture / calibrate / check / quant-risk / quant-diff / quant-sensitivity / quant-recipe / op-diff / publish |
| `action.yml` | GitHub Action wrapper for `firefly check` and `quant-diff` (`mode:` input) |
| `scripts/capture_vllm.py` | Modal harness around the vLLM runner (multi-version blog repros) |
| `scripts/plot_validation.py` | Diff and magnitude figures for the writeup |

## Reference storage backends

| Scheme | Use case | Install |
| --- | --- | --- |
| local path | Reference checked into your repo | (built-in) |
| `hf://<org>/<repo>[@<rev>][/<subpath>]` | Reference hosted on HF Hub | (built-in) |
| `s3://<bucket>/<prefix>` | Reference in private AWS bucket | `pip install 'firefly[s3]'` |
| `gs://<bucket>/<prefix>` | Reference in private GCS bucket | `pip install 'firefly[gcs]'` |
| `az://<account>/<container>/<prefix>` | Reference in private Azure container | `pip install 'firefly[azure]'` |

All three cloud backends use their library's default credential chain —
env vars, local credential files, or instance/runner metadata. Azure
prefers `AZURE_STORAGE_CONNECTION_STRING` if set, otherwise falls back
to `DefaultAzureCredential` (managed identity, az CLI, env vars).
Files are mirrored into `$FIREFLY_CACHE_DIR` (or
`~/.cache/firefly/<scheme>/<bucket>/...`) with ETag-based incremental
sync, so re-runs on a persistent CI cache only re-download what
changed upstream.

## Roadmap

**Shipped:**

- Core CI flow: capture / calibrate / check, GitHub Action, markdown
  PR summaries, calibrated per-tap tolerances + `--jitter-floor`
- Storage backends: local, `hf://`, `s3://`, `gs://`, `az://`
- **Pluggable capture runners** — `--runner {hf,vllm,sglang}` behind one
  `Runner` seam. vLLM (V0 + V1, prefill + decode, attention-backend
  selection with live verification) and SGLang (in-process via its native
  `forward_hooks`) both capture from a real serving engine; engine knobs
  via `--runner-opt`. Adding an engine is one class, not pipeline surgery.
- Reproducible vLLM parity suite + cross-family validation (9 models,
  8 architecture families)
- **Per-head attention attribution** (`capture --per-head`) — drills
  the first divergent layer down to the specific attention head
- **Quantization-risk heuristic** (`firefly quant-risk`) — flags
  outlier-feature layers sensitive to int8/int4 from stored activations alone
  (cheap, no model run; for *measured* attribution use the items below)
- **Quantization diff** (`firefly quant-diff`, action `mode: quant-diff`) —
  diff a real torchao-quantized model (w8a8 / int4wo) against the fp baseline,
  ranked by per-layer relative divergence, with a CI threshold gate
- **Attribution-guided mixed precision** (`quant-sensitivity` / `quant-recipe`)
  — measure which layers/Linears to keep in high precision and *verify* the
  recovered fidelity (isolated / marginal / greedy strategies; layer or linear
  granularity)
- **Op-level drill-down** (`firefly op-diff`) — a `TorchDispatchMode` scoped to
  a flagged module finds the first ATen op where two runs diverge
- **Shadow-mode capture mechanism** (`firefly.shadow`) — custom ops + a
  Triton kernel that survive `torch.compile` and CUDA-graph replay, with
  local and S3/GCS/Azure streaming sinks. This is a *mechanism*, not yet a
  product: it's unit-tested and passes a synthetic-model integration test,
  but overhead is unmeasured and it hasn't been run against a real serving
  stack. It targets teams that can instrument their **own** torch model
  (`instrument()` / `@tap`) — vLLM and SGLang own their model forward, so
  this does not capture from those engines.
- Recsys domain selector (TorchRec / DLRM / DCN-v2 tap conventions)

**Planned:**

- More runners behind the seam (TensorRT-LLM, TGI) as demand warrants
- Mixed precision at scale — `greedy` / per-`linear` recipes on larger
  models + int4 (where the wrapper search pulls ahead), and a per-op A/B
  intercept-override mode via `TorchDispatchMode`
- Recsys capture end-to-end (embedding-table monitoring, O2O
  divergence) — the v2 domain expansion
- Hosted/report surface beyond the terminal table (local static HTML
  first)

## Development

```sh
uv sync                       # install deps
uv run pytest                 # fast unit tests
uv run pytest -m slow         # downloads SmolLM-135M for end-to-end
uv run ruff check .           # lint
uv run firefly --help         # CLI surface
```
