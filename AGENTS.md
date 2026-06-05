# Firefly — Agent Context

Persistent state for AI agents working on this repo. Update when major
decisions change; don't bloat with day-to-day status.

## What this is

Firefly is a **numerical-parity CI gate for ML model deployments**. It
detects bugs that silently change a model's outputs (quantization errors,
serving-stack drift, hardware mismatches, dependency bumps) and attributes
the divergence to the first layer where it originated. Built around a
per-tap activation-capture mechanism with per-layer tolerance calibration.

Target users: small/mid LLM fine-tune shops. Distribution model: OSS core
+ paid hosted layer (GitHub Marketplace eventually). Goals are dual:
**(1)** a plausibly sellable small CI tool and **(2)** an interview-signal
artifact demonstrating ML-infra / numerical-methods depth. **NOT** a
10M-ARR play; favor visible architectural depth over scale.

Domain dispatcher (`tap_points.py`) makes LLM the only currently-supported
domain; recsys is the planned v2 expansion driven by user's Meta-Instagram
background (O2O divergence, online training, MTIA/heterogeneous-hardware).

## Status

Phase 1 (demoable artifact) — **DONE**. Phase 2 (calibration methodology
+ moat) — **MOSTLY DONE**:

- ✅ Tap-point selector (LLM domain), pure-core + orchestrator capture
- ✅ Reference artifact format (`weights.safetensors` + `manifest.json` + optional `tolerances.json`)
- ✅ `firefly capture` / `check` / `calibrate` CLI
- ✅ `compare` with first-divergence attribution + fingerprint mismatch check
- ✅ `TapTolerance` dataclass (atol + source + noise_floor + n_calibration_runs)
- ✅ Three noise modes: `none` / `synthetic` (Gaussian injection at one tap) / `hardware` (relax determinism for real GPU noise)
- ✅ `--dtype` flag for fp32 / bf16 / fp16 captures + calibrations
- ✅ Modal validation script (`scripts/modal_validation.py`) — capture + 3 noise configs, runs on any NVIDIA GPU via `--gpu`
- ✅ Activation-magnitude diagnostic (`scripts/analyze.py`)
- ✅ **Real GPU validation at scale** — done for FP32, BF16, and FP16 (27 GPU-runs across 9 GPUs × 3 dtypes)
- ✅ **vLLM capture** — V0 via apply_model, V1 via collective_rpc + bytes-encoded drain, prefill + decode modes, per-version Modal images
- ✅ **CI integration** — `action.yml` GitHub Action wrapper, `--ci-format markdown` for PR comments, `--max-rel-error` for cross-platform jitter, `--allow-default-tolerances` escape hatch
- ✅ **HF Hub storage backend** — `hf://org/repo[@rev][/subpath]`; GCS/Azure stubbed with planned-for-vN errors (`src/firefly/storage.py`)
- ✅ **S3 storage backend** — `s3://bucket/prefix`, boto3 default credential chain, ETag-based incremental sync into `$FIREFLY_CACHE_DIR` (v2 first item)
- ✅ **Recsys domain selector** — TorchRec / DLRM / DCN-v2 conventions in `tap_points.py`
- ✅ **Reproducible parity suite** — `scripts/vllm_test_suite.yml` + `scripts/run_vllm_suite.py`; 7 tests passing
- ✅ **v0.1.0 tag** — annotated tag created (commit before push); pinnable via `uses: neelvad/firefly@v0.1.0`
- ✅ **Blog post drafted** — `docs/index.md` + `docs/_config.yml`, Cayman theme, ~2300 words; 3 inline plots; ready for GH Pages + HN submission

## Architecture (load-bearing modules)

| File | Role |
| --- | --- |
| `src/firefly/determinism.py` | `set_deterministic` (locks PyTorch) vs `set_hardware_noise_baseline` (relaxes for real GPU noise) |
| `src/firefly/tap_points.py` | Domain-aware tap selector via `select_tap_points(model, domain="llm")`. LLM variant picks per-layer self_attn/mlp/residual + final_norm |
| `src/firefly/noise.py` | `NoiseSpec` dataclass + `_NoiseInjector` hook for synthetic mode |
| `src/firefly/capture.py` | `run_capture_repeated(model, batch, runs, noise)` — pure core. `capture_reference(...)` — orchestrator. Also `parse_dtype` / `dtype_to_name` / `load_model_and_tokenizer` |
| `src/firefly/reference.py` | `ReferenceManifest` + safetensors I/O; doc-only reference to `tolerances.json` (read/write lives in compare.py) |
| `src/firefly/compare.py` | `TapTolerance` + `diff_captures` (pure) + `compare_to_reference` (orchestrator, does fingerprint check + auto-loads tolerances.json) + `read_tolerances` / `write_tolerances` |
| `src/firefly/calibrate.py` | `derive_tolerances` (pure) + `calibrate(reference_dir, inputs_path, runs, safety_factor, noise, ...)` — writes `tolerances.json` |
| `src/firefly/attribution.py` | `attribute_first_divergence` — walks forward order, names first tap exceeding tolerance |
| `src/firefly/report.py` | Rich terminal table + structured JSON output |
| `src/firefly/cli.py` | `firefly capture / check / calibrate` subcommands with all flags |
| `scripts/modal_validation.py` | Modal app that runs capture + 3 noise configs on GPU, returns per-tap data |
| `scripts/analyze.py` | Local diagnostic: prints per-layer activation magnitudes from a captured reference |
| `examples/quantization_demo/` | Killer demo: capture SmolLM-135M, perturb layer 7's MLP weights, show Firefly correctly attributes to layer.7.mlp |

## Validation findings (so far, FP32-only)

Ran 9 NVIDIA GPUs through `modal_validation.py` at FP32 storage with
optional TF32 matmul. All results in `scripts/results/`. Key findings:

1. **Config A (strict deterministic)** = 0 noise everywhere across all GPUs. Determinism setup works.
2. **Config B (relaxed determinism, no TF32)** = 0 noise *everywhere across all GPUs*. **Pure FP32 inference of SmolLM-135M is bit-deterministic on A10G/A100/H100/H200/B200, even with deterministic algorithms disabled.** Atomics-based nondeterminism warnings are training-time concerns; forward inference at FP32 is reliably bit-stable.
3. **Config C (relaxed + TF32)** = noise on 91/91 taps; max noise_floor in the 12–17 range across GPUs.
4. **T4 (Turing, sm_75)** has no TF32 silicon → Config C = 0 noise. Validates the experiment is measuring TF32 specifically.
5. **PyTorch 2.6 vs 2.7 → bit-identical numbers** on A100/H100/H200. No version-induced confounder; all earlier data still comparable.
6. **Cross-architecture max noise (Config C):** B200 cleanest (12.5), Hopper (15.3), Ada (15.5), Ampere consumer A10G (14.5), Ampere DC A100 (17.3). Spread < 2× across 5 years of architectures.
7. **The phase transition is at `layer.11.mlp` on EVERY TF32-capable GPU.** Ratio 187× (B200) to 291× (A10G). Identical layer position across architectures → it's a **model phenomenon, not a hardware quirk.**
8. **Explanation (confirmed by `scripts/analyze.py`):** SmolLM-135M develops **outlier features** at layer 11 (Dettmers et al. 2022 phenomenon). Activation magnitude at layer.11.mlp jumps from ~50 to ~30,303 in one layer — a 610× jump. The residual stream then "locks in" at ~32k for layers 11–27. **TF32 produces roughly constant relative error (~0.05–0.1%) everywhere**, so the absolute noise scales with activation magnitude. The "phase transition" in noise is just the activation magnitude jumping.
9. **Layer-norm masks the symptom:** final_norm rescales 32k → 56. Output-level monitoring (Arize, Galileo, etc.) can't see this internal phenomenon; Firefly can. Strongest pro-Firefly argument we've generated.

10. **BF16 and FP16 inference are both bit-deterministic on all 9 GPUs, including with TF32 enabled.** Original hypothesis (mantissa width → noise) was wrong. Reason: `torch.backends.cuda.matmul.allow_tf32` is an *FP32-input-specific* downcast; BF16/FP16 storage takes the dedicated tensor-core path with deterministic reduction order. Verified each format actually loaded by checking that Config C atol stayed at the 1e-5 floor (vs ~7.8e+1 at layer.11.mlp in FP32). **Counterintuitive headline for blog: BF16 and FP16 are more reproducible than FP32+TF32 — and FP16 has the same 10-bit mantissa as TF32, which rules out "narrow mantissa tensor cores" as the cause. The noise is in cuBLAS's FP32-storage-with-TF32 *kernel dispatch path*, not in tensor-core arithmetic itself.**

## Roadmap (current state)

v1 is **code-complete and tagged at v0.1.0**. Blog post drafted in `docs/index.md`. The remaining v1 launch steps are user-actions, not engineering:

1. **Push commits and tag** (`git push && git push --tags`) — unpushed at session compact.
2. **Enable GH Pages** in repo settings → Pages → source: branch main / folder `/docs`. URL becomes `neelvad.github.io/firefly`.
3. **Submit to HN** (narrative-framing title from the post recommended over Show HN; Sunday evening / Monday morning US time).

v2 is **in progress**:

- ✅ **S3 storage backend** — `s3://bucket/prefix`, boto3 default credential chain, ETag-based incremental sync. Optional install via `pip install 'firefly[s3]'`.

Remaining v2 items (do not start without explicit go-ahead):

- **FLASHINFER backend test** — three install paths documented in `project_firefly_flashinfer_deferred.md`; pick one deliberately
- **Larger-model validation** (Llama-3-8B) — confirms the predicted layer-boundary shift; the post called out as not-yet-tested
- **Long-prompt series at 1k/2k/4k** — tightens the load-bearing finding to "diverges at every length past PagedAttention block boundary"

What's deferred to v3:

- GCS / Azure storage backends
- Shadow-mode capture against production traffic (needs custom op for CUDA-graph / torch.compile compat)
- Hosted dashboard (the monetization seam; hold for actual usage signal)

## Non-obvious decisions to preserve

- **CPU+fp32 is the dev target.** Same model twice = exactly 0.0 diff. Makes any flagged divergence real signal. Don't develop on MPS/CUDA — noise floor confuses things.
- **Pure-core + orchestrator split everywhere** (`run_capture` pure / `capture_reference` orchestrates; `diff_captures` pure / `compare_to_reference` orchestrates; `derive_tolerances` pure / `calibrate` orchestrates). 90% of tests use fake `nn.Module` fixtures and run in milliseconds; only slow integration tests download weights.
- **Reference artifact = inspectable directory.** `weights.safetensors` + `manifest.json` + optional `tolerances.json`. Never use pickle. Human-debuggable matters more than throughput here.
- **One forward-ordered list drives everything.** `select_tap_points` → `manifest.tap_points` → `diff_captures` iteration → `attribute_first_divergence`. "First divergence" semantic falls out for free.
- **`pytest -m slow` for integration tests** (download models). Defaults exclude. Inner loop is sub-second.
- **`uv` for env management.** Not conda, not docker. `pyproject.toml` is source of truth.
- **GPU-noise mode in `NoiseSpec` is mode="hardware"** — does NOT register a hook; relies on the *hardware itself* producing variance via relaxed determinism. Synthetic mode is the one that registers a hook.
- **Tolerances live INSIDE the reference dir** (`<ref>/tolerances.json`), auto-loaded by `compare_to_reference` if present. User can hand-edit. Calibration writes; check reads.
- **Fingerprint check at compare time** fails loudly if candidate weights differ from manifest's recorded fingerprint. `--allow-fingerprint-mismatch` escape hatch exists.
- **Modal image is `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`** — needed for Blackwell (sm_100) support. Older 2.6+cu12.4 image breaks on B200 (no sm_100 in compiled arch list).
- **HF_TOKEN forwarding is opt-in.** If set locally, modal script forwards via `Secret.from_local_environ`; if not, anonymous HF access (rate-limited but works for public models).
- **The user is a former Meta-Instagram-Search engineer** with deep model-internal-state telemetry expertise (PyTorch/Triton hooks, custom ops, checkpoint analyses). Don't over-explain PyTorch internals; do explain ML-product/market concepts.
- **Don't squash merge.** Commit history is itself an artifact for the interview-signal goal. Each commit should be coherent and pass tests on its own (intermediate-commit hygiene relaxed where forward-looking test imports made this hard).
- **vLLM capture design (decided 2026-06-03):** load with `enforce_eager=True` so CUDA graphs are disabled — forward hooks then work without graph breaks. Hooks keep tensors on GPU (`.detach()` only, no `.cpu()`); a separate `apply_model(_drain)` call bulk-transfers to CPU at end. Collapses ~90 per-tap sync points into 1. **v1 captures prefill only** (filter on `out.shape[0] > 1`) — keeps the diff story simple and rules out KV-cache-state confounders.
- **Decode capture is the planned v1.5 extension** and is arguably *more* valuable than prefill for catching real regressions. Most vLLM optimization happens in the decode hot path (PagedAttention, speculative decoding, FlashAttention-decode kernels), so most version-bump bugs land there. KV-cache state at decode step N is deterministically derived from prompt + earlier steps, so decode IS reproducible at temp=0 — the "cache state is hard to control" framing I used initially was wrong. Implementation: drop prefill filter, add per-token-position indexing (`{tap}@token_{i}`), bound with `max_tokens=N`. Prefill and decode captures should be **separate tap groups** because their tensor shapes differ — prefill is `(prompt_tokens, hidden)`, decode is `(1, hidden)` per step.
- **Eager mode is acceptable for the CI use case but NOT for production capture.** In CI, Firefly spins up its own vLLM instance for a one-off diagnostic run — slowness from eager mode is fine, the user doesn't see it. For *shadow-mode* capture (Firefly listening to production traffic), nobody runs vLLM in eager mode in prod — they need CUDA graphs + torch.compile for throughput. That capture path needs **a custom op** (e.g., `torch.ops.firefly.capture(tensor, name)`) that Dynamo treats as opaque tensor-in/tensor-out so it doesn't force a graph break. This is significant work — weeks, not days — and gated on v2/v3 product direction. Note explicitly so we don't accidentally architect the v1 CI capture as if shadow-mode were already a goal.

## How to pick up a cold session

1. Read this file.
2. Read `MEMORY.md` for user/project memories (especially `project_recsys_v2_angle.md` if recsys comes up).
3. Check `git log --oneline -15` to see recent work.
4. Check `scripts/results/` to see latest validation data.
5. Run `uv run pytest -q` to confirm green state.
6. Ask user what specifically they want to work on next; don't assume.
