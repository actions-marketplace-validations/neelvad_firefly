# Firefly

A numerical-parity CI gate for ML model deployments. Firefly catches the
class of bugs that silently change a model's outputs — quantization
round-trips, kernel swaps (FlashAttention vs xFormers), dependency bumps,
serving-stack drift — and attributes the divergence to the specific layer
where it originated.

```yaml
# .github/workflows/firefly.yml
- uses: neelvad/firefly@v0.1.0
  with:
    reference: hf://my-org/my-firefly-ref  # or a local path
    candidate: my-org/my-finetune-ckpt
    inputs: tests/firefly-prompts.json
    max-rel-error: 0.001                   # cross-platform safety margin
```

The action posts a markdown summary to `$GITHUB_STEP_SUMMARY` and exits
non-zero on divergence. Outputs (`first-divergent-tap`, `passed`,
`report-path`) drive downstream steps like Slack notifications or PR
comments.

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
2. **Decode pollutes immediately.** At the first generated token, *all*
   layers diverge — the KV cache built during prefill carries diverged
   activations forward, so even `layer.0.self_attn` reads polluted state.
3. **Divergence compounds with each token.** By token 6 the final-layer
   relative error is **3.7%** vs **1.4%** at prefill. Output monitoring
   gets a longer runway before it sees the drift.

We've validated this pattern across 9 NVIDIA GPUs × 3 storage dtypes
(FP32 / BF16 / FP16) and 5 vLLM (engine × backend) combinations. Full
findings in `MEMORY.md` and the supporting plots in `scripts/plots/`.

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

## Tolerance knobs

Three axes you can tune:

| Knob | Where | Default | Use case |
|---|---|---|---|
| `safety_factor` | `firefly calibrate` | 1.5x | Multiplies per-tap noise floor |
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
| `capture.py` | Register forward hooks, capture activations from a golden batch |
| `calibrate.py` | Re-run reference under controlled noise; derive per-tap atol |
| `reference.py` | Read/write the inspectable reference artifact (safetensors + JSON) |
| `compare.py` | Per-tap diff with effective-atol composition |
| `attribution.py` | Forward-order walk → first divergent tap |
| `report.py` | Rich-terminal table + markdown PR-comment formatter |
| `cli.py` | `firefly capture / calibrate / check` |
| `action.yml` | GitHub Action wrapper for `firefly check` |
| `scripts/capture_vllm.py` | vLLM-specific capture (V0 + V1 engines, prefill + decode) |
| `scripts/plot_validation.py` | Diff and magnitude figures for the writeup |

## Roadmap

**v1 (now):** local-filesystem reference dirs, HuggingFace Hub references
via `hf://org/repo`, HF transformers + vLLM capture paths, LLM domain.
Production-ready as a one-repo, one-team quality gate.

**v2 (planned):**

- S3 storage backend (`s3://`) for teams that host references on AWS
  rather than HF Hub
- Recsys domain selector — different tap-point convention for
  embedding-table + cross-net architectures
- Comprehensive vLLM test suite — FLASHINFER, multi-request batching,
  long-context PagedAttention boundaries, speculative decoding

**v3:** GCS / Azure storage backends; shadow-mode capture against
production traffic (requires a custom op so torch.compile / CUDA-graphed
inference doesn't break).

## Development

```sh
uv sync                       # install deps
uv run pytest                 # fast unit tests
uv run pytest -m slow         # downloads SmolLM-135M for end-to-end
uv run ruff check .           # lint
uv run firefly --help         # CLI surface
```
