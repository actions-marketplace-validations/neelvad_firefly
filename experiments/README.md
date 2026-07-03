# experiments/

Research one-offs: hypothesis probes, debugging sessions, and spikes. Each
script answered a specific question during development; the answers shaped the
product but the scripts themselves are **not** on the product path and are not
maintained to the same bar as `src/` or `scripts/` (the GPU validation
harness + demos live there). Most run on Modal (`uv run modal run
experiments/<name>.py`); see each docstring for usage.

Kept on main because the findings are load-bearing for the
[writeup](../docs/index.md) and the repro trail matters.

## Kernel/backend divergence findings

| Script | Question | Outcome |
| --- | --- | --- |
| `analyze.py` | Is the layer-11 TF32 noise jump driven by activation-magnitude growth (SwiGLU saturation / outlier features)? | Yes — `max\|activation\|` ramps into layer 11, where the noise floor jumps ~250×. |
| `analyze_qwen_layer27.py` | Is Qwen-2.5-7B's 20× FLASHINFER error spike at layer 27 concentrated in specific heads? | Yes, decisively: FLASHINFER returns **exact zeros for heads 10 & 13** — a live kernel bug, not accumulated rounding. |
| `repro_flashinfer_zero_heads.py` | Standalone, Firefly-free repro of the zero-heads bug for the upstream issue. | Reproduces on vLLM 0.22.1 + flashinfer 0.6.11 (same heads); the core logic between the markers is the issue body. |
| `vllm_explore.py` | What does a HF model's `nn.Module` tree look like inside vLLM? | Reconnaissance that informed the vLLM runner's tap-name mapping. |

## Quantization / serving-backend findings

| Script | Question | Outcome |
| --- | --- | --- |
| `debug_smoothquant_export.py` | Does SmoothQuant actually move *served* (compressed-tensors) W8A8? | No — bit-identical no-op when served; its torchao "recovery" was a measurement artifact. Reframed w8a8 shipping around plain int8wo. |
| `debug_int4_recovery.py` | Does int4 recovery transfer to compressed-tensors serving? | Yes — GPTQ/AWQ recover most of served int4 degradation (RTN int4 serves at +113% perplexity; GPTQ ~+5%). This is why `optimize` exports int4 via GPTQ/AWQ. |
| `harden_backend_transfer.py` | Do the backend-gap and SmoothQuant-no-op findings replicate cross-family? | Replicated on Qwen2.5-1.5B, SmolLM2-1.7B, and gemma-2-2b (each in its own container). |
| `sanity_mistral_int4.py` | Why is plain int4wo on Mistral-7B catastrophically broken (~185k perplexity)? | Isolated to the int4wo path specifically (fp + w8a8 sane, Qwen control sane) — not the harness. |
| `probe_int4_torchao.py` | Which torchao int4 config actually runs on our GPU image (default dies on a missing `mslk` kernel dep)? | Found the working config matrix; the winner is wired into `firefly.quant.torchao._quant_config`. |

## Per-layer ranking & recsys probes

| Script | Question | Outcome |
| --- | --- | --- |
| `validate_ranking_transfer.py` | Does the cheap torchao per-layer fragility *ranking* transfer to served int4 quality? | Ranking transfers; margin over random-K is moderate (~1.5×) at 1.5B. |
| `probe_ranking_scale.py` | Does that margin grow with model scale (7B) and a tighter keep-fp budget (small K)? | K-sweep on Qwen2.5-7B; each export isolated in its own container. |
| `recsys_precision_fragility.py` | Is quant fragility component-specific in a heterogeneous recsys model (DCN-v2 on MovieLens-1M)? | Per-component int4 probe measuring calibration (logloss + ECE), not just AUC — the wedge experiment for the recsys v2 direction. |

## Spikes

| Script | Question | Outcome |
| --- | --- | --- |
| `spike_torch_dispatch.py` | Can a `TorchDispatchMode` capture op-level outputs scoped to one module (via forward-hook gating)? | Passed all criteria — became `firefly op-diff` (`src/firefly/op_drill.py`). |
