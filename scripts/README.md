# scripts/

One-off validation and exploration tooling. Not part of the product.

## modal_validation.py

Runs capture + three calibration configurations on a Modal-hosted NVIDIA
A10G, then writes the per-tap noise_floor / atol data to
`scripts/results/`. This is the v1 validation that the synthetic-noise
methodology generalizes to real hardware noise.

### First-time setup

```sh
uv run modal token new        # browser-based auth, creates ~/.modal/token
```

You will need a Modal account. The free tier ($30/mo of credits) is more
than enough — one full validation run on an A10G is well under $1.

### Running

```sh
uv run modal run scripts/modal_validation.py
# or with a different model:
uv run modal run scripts/modal_validation.py --model HuggingFaceTB/SmolLM-360M
```

### HuggingFace authentication (optional)

Public models work without auth but get a rate-limit warning. To silence it
and unlock faster downloads / gated models, set `HF_TOKEN` in your shell
before running — the script forwards it to the GPU container automatically
if present.

```sh
export HF_TOKEN=hf_...        # get from huggingface.co/settings/tokens
uv run modal run scripts/modal_validation.py
```

Nothing happens if it's not set — the script just notes which mode it's in.

Output lands in `scripts/results/modal_validation_<timestamp>.json`. The
directory is gitignored — these are experiment artifacts, not source.

### What gets compared

| Config | Determinism | TF32 | Expected behavior |
|---|---|---|---|
| `A_strict_no_tf32` | locked down (`set_deterministic`) | off | Closest to zero noise the hardware can produce |
| `B_hardware_no_tf32` | relaxed (`set_hardware_noise_baseline`) | off | Natural GPU noise from atomics / kernel selection |
| `C_hardware_tf32` | relaxed | on | Realistic production default; expect largest noise |

For each config, calibrate writes per-tap `noise_floor` (the observed max
deviation from the captured reference across N runs) and `atol`
(`max(safety_factor × noise_floor, DEFAULT_TOLERANCE)`).

A successful run shows:

* Config A's noise_floor near zero everywhere — confirms determinism works.
* Config B nonzero, growing with depth — confirms the methodology applies.
* Config C even larger than B at the same taps — confirms TF32's quality
  cost is measurable.

If B / C don't show depth-amplification, the methodology assumption is
wrong on real hardware and we need to revisit. That's the load-bearing
empirical claim.
