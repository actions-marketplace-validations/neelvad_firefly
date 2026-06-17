# Firefly

A numerical-parity CI gate for ML model deployments. Firefly catches the
class of bugs that silently change a model's outputs — kernel swaps
(FlashAttention vs xFormers), dependency bumps, serving-stack drift,
hardware moves — and attributes the divergence to the specific layer
where it originated.

Firefly is fundamentally a **same-weights deployment-parity gate**: the
reference pins the exact model weights by fingerprint, and `check`
re-runs *those weights* in the candidate environment to ask "does this
serving stack still produce the same activations?" Comparing a model
whose *weights* changed — a new fine-tune, or a quantized build — is a
different question; it's supported, but you have to opt in with
`--allow-fingerprint-mismatch` (the action's `allow-fingerprint-mismatch:
true`), because by default a weight change is treated as "you pointed me
at the wrong model."

```yaml
# .github/workflows/firefly.yml
- uses: neelvad/firefly@v0.4.0
  with:
    reference: hf://my-org/my-firefly-ref  # captured from my-org/my-model
    candidate: my-org/my-model             # same weights, new serving stack
    inputs: tests/firefly-prompts.json
    max-rel-error: 0.001                   # cross-platform safety margin
```

The action posts a markdown summary to `$GITHUB_STEP_SUMMARY` and exits
non-zero on divergence. On `pull_request` runs it also posts the summary as
a sticky PR comment (grant the workflow `permissions: pull-requests: write`;
a missing permission degrades to a warning, never a failed build). Outputs
(`first-divergent-tap`, `passed`, `report-path`) drive downstream steps.

### Quantization gate (`mode: quant-diff`)

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
| `--max-rel-error` | `firefly check` | 0 (off) | Global ceiling for cross-platform variation |
| Per-tap atol | hand-edit `tolerances.json` | — | Override calibration for known-noisy taps |

The `max-rel-error` knob in particular is what makes a calibrated
reference *portable* across environments. Calibration measures
runs-on-the-calibration-machine variance; cross-machine FP variation
is a different (and often larger) noise source. A 0.1% ceiling
absorbs typical cross-platform jitter while staying comfortably below
the >1% threshold where real bugs live.

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
| `quant_risk.py` | Simulated int8/int4 quantization risk from stored activations (heuristic) |
| `quant_validate.py` | Real torchao quantization (w8a8 / int4wo) for quant-diff + sensitivity |
| `quant_sensitivity.py` | Attribution-guided mixed precision: per-unit sensitivity + verified recipe (isolated / marginal / greedy; layer or linear granularity) |
| `op_drill.py` | Op-level drill-down: `TorchDispatchMode` scoped to a module → first diverging ATen op |
| `shadow/` | Shadow-mode capture package: custom ops + Triton kernel + Tappers + sinks that survive torch.compile and CUDA graphs |
| `storage.py` | Reference resolution/publish for `hf://`, `s3://`, `gs://`, `az://` |
| `report.py` | Rich-terminal table + markdown PR-comment formatter |
| `cli.py` | `firefly capture / calibrate / check / quant-risk / quant-diff / quant-sensitivity / quant-recipe / op-diff / publish` |
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
  PR summaries, calibrated per-tap tolerances + `--max-rel-error`
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
