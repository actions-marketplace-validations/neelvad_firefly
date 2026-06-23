"""Quantization surface: real torchao quant + diagnosis + routed recipes.

Sits on top of the engine (capture / compare / attribute). Submodules:
  intervention — the seam: PrecisionPolicy + Pipeline + RTNQuantizer
  smoothquant  — SmoothQuant pre-transform (activation outliers)
  awq          — AWQ quantizer (salient weight channels), wraps torchao
  torchao      — real torchao quant helpers (w8a8 / int4wo), preflight
  risk         — cheap activation-only int8/int4 risk heuristic (no model run)
  sensitivity  — per-unit sensitivity sweep
  salience     — weight-salience (AWQ signal) sensor
  cost         — recipe memory cost, Pareto frontier, measurement budget
  evaluate     — real eval metric (perplexity / callable) + accuracy bar
  recipe, bar  — mixed-precision recipe curves / eval-bar optimization
  recipe_io    — serialize a recipe + re-apply it (Recipe artifact)
  diagnose     — measurements → failure-mode signatures
  route        — diagnosis → a concrete recipe (deterministic router)
  auto         — deterministic auto-quant harness (diagnose→route→verify)
  step         — the agent step primitive (apply + verify + attribute)
  llm, search  — LLM proposer + min-memory-at-bar search harness
"""
