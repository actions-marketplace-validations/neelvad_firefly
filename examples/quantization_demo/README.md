# Quantization demo

The killer demo for Firefly. Shows the tool correctly attributing a planted
numerical bug to the layer that originated it.

## What it does

1. Captures a reference artifact from `HuggingFaceTB/SmolLM-135M`.
2. Runs `firefly check` against the unmodified model — expected to produce
   zero divergence and exit 0.
3. Runs `make_broken.py` to load SmolLM-135M, add `N(0, 1e-3)` Gaussian noise
   to `model.layers.7.mlp.down_proj.weight`, and save the modified model.
4. Runs `firefly check` against the broken model — expected to:
   - Report **First divergence: layer.7.mlp** (the tap immediately after the
     perturbed weight).
   - Show clean (max_abs_diff = 0.0) for all upstream taps including
     `layer.7.self_attn` (the perturbation is in MLP, not attention).
   - Show divergence growing through downstream layers.
   - Exit with code 1.

## Run it

```sh
bash run_demo.sh
```

Artifacts land in `_artifacts/` (gitignored).

Total runtime on a MacBook Air (M-series, CPU) is roughly a minute, most of
which is the initial SmolLM download.

## What this proves

- Activation capture via forward hooks on per-layer tap points works on a
  real HF decoder transformer.
- The same model captured twice produces bit-identical activations on CPU
  with determinism enabled — so any divergence flagged by the tool is signal,
  not noise from PyTorch's nondeterministic ops.
- First-divergence attribution correctly walks the forward order: it names
  the originating layer, not the loudest downstream amplification of it.
