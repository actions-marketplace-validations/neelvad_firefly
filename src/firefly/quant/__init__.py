"""Quantization surface: real torchao quant + diagnosis + mixed-precision recipes.

Sits on top of the engine (capture / compare / attribute). Submodules:
  torchao      — real torchao quantization helpers (w8a8 / int4wo), preflight
  sensitivity  — per-unit sensitivity + verified mixed-precision recipes
  risk         — cheap activation-only int8/int4 risk heuristic (no model run)
"""
