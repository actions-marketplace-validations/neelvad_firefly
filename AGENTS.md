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
- ✅ **HF Hub publish flow** — `firefly publish --reference <dir> --to <uri>`, plus `--push <uri>` on `capture` / `calibrate`. Supports hf:// and s3://.
- ✅ **Recsys domain selector** — TorchRec / DLRM / DCN-v2 conventions in `tap_points.py`
- ✅ **Reproducible parity suite** — `scripts/vllm_test_suite.yml` + `scripts/run_vllm_suite.py`; 7 tests passing
- ✅ **v0.1.0 tag** — annotated tag created (commit before push); pinnable via `uses: neelvad/firefly@v0.1.0`
- ✅ **Blog post — 5 findings** — `docs/index.md` covers Finding 1 (FLASH vs XFORMERS layer 7 on SmolLM), Finding 1.5 (cross-scale + cross-family check: within-Meta holds, breaks across families), Finding 2 (decode KV-cache propagation), Finding 3 (V0 vs V1 step function across 9/1k/2k/4k tokens on Llama), Finding 4 (FLASH vs FLASHINFER layer-0 universal across 4 models), Finding 5 (Qwen+FLASHINFER catastrophic layer-27 spike, 20× outlier).
- ✅ **Modal image variants in `scripts/capture_vllm.py`** — `0.7.3` and `0.8.5` on `debian_slim`; `0.8.5-fi` on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` with `add_python="3.11"` for the FLASHINFER backend. Shared HF cache via `firefly-hf-cache` Modal Volume mounted at `/root/.cache/huggingface`. `--max-seq-len` and `--gpu-memory-utilization` flags for production-scale runs.

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

v1 is **code-complete and tagged at v0.1.0**. Blog post in `docs/index.md` now has 5 findings (1, 1.5, 2, 3, 4, 5) including the cross-family rewrite of Finding 1.5 and the Qwen layer-27 catastrophic divergence (Finding 5). Remaining v1 launch steps are user-actions, not engineering:

1. **Push commits and tag** (`git push && git push --tags`) — many unpushed commits.
2. **Enable GH Pages** in repo settings → Pages → source: branch main / folder `/docs`. URL becomes `neelvad.github.io/firefly`.
3. **Submit to HN** (narrative-framing title from the post recommended over Show HN; Sunday evening / Monday morning US time).

v2 is **mostly done**:

- ✅ **S3 storage backend** — `s3://bucket/prefix`, boto3 default credential chain, ETag-based incremental sync. Optional install via `pip install 'firefly[s3]'`.
- ✅ **HF Hub publish flow** — `firefly publish --reference <dir> --to <uri>` plus `--push <uri>` on `capture` and `calibrate`. HF + S3 both supported.
- ✅ **Llama-3.1-8B validation** — drives Finding 1.5 (within-Meta universality at layer 7) and Finding 4 (cross-model FLASHINFER at layer 0).
- ✅ **Long-prompt series (1k/2k/4k)** — Finding 3 rewritten as a step function: bit-equal at 9 tokens, saturated at ~2.8% final-norm rel from 1k through 4k. Block-boundary > 1 is the threshold, not length.
- ✅ **FLASHINFER backend** — drives Finding 4. **Working install path: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` + `add_python="3.11"` + explicit pip-install of vllm + flashinfer**. Other paths (debian_slim, vllm-openai docker image) failed and are documented in `project_firefly_flashinfer_finding.md`.
- ✅ **Cross-family check (9 models, 8 families)** — drives Finding 1.5 rewrite + Finding 5. SmolLM, Llama (Meta), Qwen, Mistral, Phi-3, Yi, Gemma-2, Falcon-7B (MQA+RoPE), BLOOM-7B1 (MHA+ALiBi). XFORMERS layer-7 within-Meta only; non-Meta is layer 0 on 6 of 7 (Yi is the lone layer-2 outlier) regardless of MQA-vs-GQA or RoPE-vs-ALiBi. FLASHINFER layer-0 universality holds across all 8 supported models. **BLOOM+FLASHINFER is a SECOND catastrophic outlier** (22% from layer 0, flat through network — kernel-fundamental ALiBi mismatch, different mechanism from Qwen's layer-27 spike). Qwen+FLASHINFER 22.83% spike reproduces bit-identically (N=2). Phi-3 hard-incompatible with FLASHINFER (head_dim=96). MPT was attempted as second ALiBi point but mosaicml/mpt-7b returns 404 (Mosaic acquired by Databricks, model deprecated).
- ✅ **Llama 2k/4k FLASHINFER captures** — Finding 4 length curve now 4 points (9/1k/2k/4k). Plateaus past 1k just like V0 vs V1, but at a higher plateau (~3% vs ~2.8%).
- ✅ **Cross-family entries in `scripts/vllm_test_suite.yml`** — 12 tests total (7 original SmolLM + 5 cross-family Qwen/Mistral).
- ✅ **Model-layout probe** — `scripts/capture_vllm.py:_find_model_layout` handles Llama/Gemma/Qwen/Mistral/Yi (model.model.layers), Falcon/BLOOM (model.transformer.h), MPT (model.transformer.blocks) without special-casing.
- ✅ **`trust_remote_code=True` in vLLM init** — required for MPT-class repos; safe because the script runs in an isolated Modal container with explicit model IDs.

v3 is **in progress**:

- ✅ **GCS storage backend** — `gs://bucket/prefix` (or `gcs://`), Application Default Credentials, ETag-based incremental sync. Optional install via `pip install 'firefly[gcs]'`.
- ✅ **Azure Blob storage backend** — `az://account/container/prefix`, `AZURE_STORAGE_CONNECTION_STRING` or `DefaultAzureCredential` (managed identity / az CLI / env vars). Optional install via `pip install 'firefly[azure]'`.
- ✅ **Shadow-mode capture against production traffic** — shipped 2026-06-10 in `src/firefly/shadow.py`. Both spikes passed (`torch.compile` survival + CUDA-graph survival via Triton). End-to-end Modal integration test passes (200 records / 21 blobs / sidecar / round-trip). What landed: `Tapper` + `@tap` for eager + torch.compile path; `StaticTapper` + `@tap_static` + Triton kernel for CUDA-graph path; both paths support `first_n_steps` / `every_n_steps` / `on_alert` full-tensor policies; `LocalLogSink` plus streaming `S3Sink` / `GCSSink` / `AzureSink` (sharded `stats-NNNNN.jsonl`, 5s/500-event flush cadence, errors-don't-crash-inference contract); `aggregate()` handles both single-file and sharded layouts; `load_tap_index()` reads the `tap_index.json` sidecar. 52 shadow tests; full suite 186. The design memory `project_firefly_shadow_mode_design.md` has the full architecture and the corrected "weeks of work" estimate that turned out to be ~2 sessions of normal Python + a 30-line Triton kernel.

v3 deferred (multi-session):

- **Hosted dashboard** — the monetization seam; hold for actual usage signal.

v3.5+ (separate features, not part of shadow-mode):

- **Embedding-table / ID-drift monitoring** — recsys-leaning. Snapshot + diff at training time, not streaming during inference. Adjacent to shadow-mode but a different mechanism.
- **Per-head attention capture** — would let us diagnose the Qwen layer-27 spike mechanistically rather than speculating. Refinement of CI-time tooling.

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
