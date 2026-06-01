# Firefly

A numerical-parity CI gate for ML model deployments.

Firefly detects bugs that silently change a model's outputs — quantization
errors, serving-stack drift, hardware mismatches, dependency bumps — and
attributes the divergence to the first layer where it originated.

## Concept

You register a **reference** (a known-good model + a fixed batch of
inputs). On every PR, Firefly runs the **candidate** model against the same
inputs at the same internal tap points (per-layer residual stream,
attention output, MLP output) and reports the first layer where behavior
diverges beyond tolerance.

```sh
firefly capture --model <ref-checkpoint> --inputs golden.json --out reference/
firefly check   --reference reference/   --candidate <candidate-checkpoint> --inputs golden.json
```

The check exits non-zero on divergence — drop it into any CI system as a
quality gate.

## Why

LLM observability tools (Arize, Galileo, etc.) deliberately operate at the
semantic layer and skip the numerical layer. Generic CI gates aren't
ML-aware. The numerical-parity surface — *"did your deployed model's
internal computation move, and where?"* — is currently solved by hand-rolled
`torch.allclose` in users' own CI. Firefly is the tool you'd otherwise write
yourself.

## Demo

The `examples/quantization_demo/` directory contains an end-to-end demo that:

1. Captures a reference from SmolLM-135M.
2. Perturbs one MLP weight tensor by `N(0, 1e-3)` Gaussian noise.
3. Shows Firefly correctly attributing the resulting divergence to
   `layer.7.mlp` — the tap immediately downstream of the perturbed weight —
   while reporting bit-identical (`max_abs_diff = 0.0`) on all upstream taps
   including `layer.7.self_attn`.

```sh
bash examples/quantization_demo/run_demo.sh
```

## Status

Phase 1 complete: a working CLI that captures references, compares
candidates, and attributes divergence at the layer level, demonstrated
end-to-end on a real HuggingFace transformer.

Next: per-layer tolerance calibration (the technical moat) and the
GitHub Action wrapper.

## Development

```sh
uv sync                     # install deps
uv run pytest               # run fast unit tests
uv run pytest -m slow       # run integration tests (downloads SmolLM-135M)
uv run ruff check .         # lint
uv run firefly --help       # CLI
```

Target for v1: HuggingFace decoder transformers, single device, CPU + fp32
for max determinism. Cross-hardware (GPU/MTIA) and continuous-prod
monitoring are deferred to v2.

## Architecture

| Module | Responsibility |
| --- | --- |
| `determinism.py` | Lock PyTorch into the most-deterministic backend available |
| `tap_points.py` | Select stable hook sites (currently `llm`; recsys/cv planned) |
| `capture.py` | Register forward hooks, run a golden batch, capture activations |
| `reference.py` | Read/write the on-disk reference artifact (safetensors + manifest.json) |
| `compare.py` | Per-tap diff between candidate and reference |
| `attribution.py` | Walk forward order, name the first tap that exceeds tolerance |
| `report.py` | Rich human report + structured JSON |
| `cli.py` | The three CLI subcommands wiring it all together |

The only domain-specific module is `tap_points.py`; everything else
generalizes across architectures.
