# Firefly

A numerical-parity CI gate for ML model deployments.

Firefly detects bugs that silently change a model's outputs — quantization errors, serving-stack drift, hardware mismatches, dependency bumps — and attributes the divergence to the first layer where it originated.

## Status

Early scaffolding. Not yet usable.

## Concept

You register a **reference** (a known-good model + a fixed batch of inputs). On every PR, Firefly runs the **candidate** model against the same inputs at the same internal tap points (per-layer residual stream, attention output, MLP output) and reports the first layer where behavior diverges beyond a calibrated tolerance.

```
firefly capture   --model <ref-checkpoint>  --inputs golden.json  --out reference/
firefly calibrate --reference reference/    --runs 16
firefly check     --reference reference/    --candidate <candidate-checkpoint>
```

The check exits non-zero on divergence — drop it into any CI system as a gate.

## Why

LLM observability tools (Arize, Galileo, etc.) deliberately operate at the semantic layer and skip the numerical layer. Generic CI gates aren't ML-aware. The numerical-parity surface — "did your deployed model's internal computation move, and where?" — is currently solved by hand-rolled `torch.allclose` in users' own CI. Firefly is the tool you'd otherwise write yourself.

## Development

```
uv sync
uv run pytest
uv run firefly --help
```

Target for v1: HuggingFace transformer fine-tunes, single device, CPU + fp32 for max determinism.
