"""Real task-metric evaluation + an accuracy bar for quant recipes.

The recipe's *inner* signal is output divergence vs the fp baseline — a cheap
surrogate computed from a forward pass on the calibration prompts. That ranks
units well but it isn't what anyone ships against. This module supplies the
*outer truth*: a real metric (perplexity, or a user-supplied callable) on a
held-out eval set, plus an **accuracy bar** stating how much degradation from
the fp baseline is acceptable.

The two are used at different tiers (see :func:`firefly.quant.sensitivity.
optimize_to_bar`): the cheap proxy ranks/shortlists units (the filter), and
these expensive evals are spent only to *gate acceptance* of candidate recipes
(the wrapper). That filter→wrapper split is what keeps the eval count ~log(N)
instead of O(N).

Not pure: the evaluators run models. The accuracy-bar math is pure and tested
on its own.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

#: Built-in metric names (anything with a ``:`` is treated as a callable spec).
BUILTIN_METRICS = ("perplexity",)


def load_eval_texts(path: Path) -> list[str]:
    """Load an eval set as a list of strings.

    Accepts a ``.jsonl`` file (one JSON value per line — a bare string or an
    object with a ``"text"`` key) or a ``.json`` file (either ``{"texts": [...]}``
    — the same shape as Firefly's golden inputs — or a bare list of strings).
    """
    path = Path(path)
    raw = path.read_text()
    if path.suffix == ".jsonl":
        texts: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj if isinstance(obj, str) else obj["text"])
    else:
        data = json.loads(raw)
        texts = data["texts"] if isinstance(data, dict) else data
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError(f"Eval set {path} must resolve to a non-empty list of strings.")
    return texts


@dataclass
class Evaluator:
    """A named scalar metric over a (model, tokenizer), with its direction.

    ``higher_is_better`` is load-bearing: it flips both how the accuracy bar is
    applied and which way "degradation" points (perplexity down-good, accuracy
    up-good).
    """

    name: str
    higher_is_better: bool
    fn: Callable[[nn.Module, object], float]

    def __call__(self, model: nn.Module, tokenizer: object) -> float:
        return float(self.fn(model, tokenizer))


def _perplexity(model: nn.Module, tokenizer: object, texts: list[str], max_length: int) -> float:
    """Token-weighted perplexity of a causal LM over ``texts``.

    exp(total NLL / total tokens). Runs on whatever device the model is on. The
    quantized model still computes ``loss`` from ``labels`` (torchao only swaps
    Linear weights; the forward is unchanged), which is the whole point — this
    is a real metric on the real (quantized) model.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    try:
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                ids = enc["input_ids"]
                # Ensure BOS: some tokenizers (Gemma 4 on transformers v5) don't
                # add it by default, and BOS-sensitive models score near-uniform
                # without it — a silently wrecked metric, not a wrecked model.
                # No-op when the tokenizer already emitted it or defines none.
                bos = getattr(tokenizer, "bos_token_id", None)
                if bos is not None and (ids.shape[-1] == 0 or ids[0, 0].item() != bos):
                    ids = torch.cat(
                        [torch.full((ids.shape[0], 1), bos, dtype=ids.dtype), ids], dim=1
                    )
                ids = ids.to(device)
                if ids.shape[-1] < 2:
                    continue  # need at least one (context, target) pair
                out = model(ids, labels=ids)
                n = ids.shape[-1] - 1  # HF averages loss over the shifted targets
                total_nll += float(out.loss) * n
                total_tokens += n
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Eval set produced no scorable tokens (all texts too short?).")
    return math.exp(total_nll / total_tokens)


def perplexity_evaluator(texts: list[str], max_length: int = 512) -> Evaluator:
    return Evaluator(
        name="perplexity",
        higher_is_better=False,
        fn=lambda model, tok: _perplexity(model, tok, texts, max_length),
    )


def callable_evaluator(spec: str) -> Evaluator:
    """Resolve a ``"module.path:function"`` metric.

    The function takes ``(model, tokenizer)`` and returns a float; it owns its
    own eval data (it knows its task). Treated as higher-is-better unless it
    carries a ``higher_is_better`` attribute set to ``False``; an optional
    ``metric_name`` attribute names it in reports.
    """
    mod_name, sep, fn_name = spec.partition(":")
    if not sep or not fn_name:
        raise ValueError(f"Callable metric must be 'module:function', got {spec!r}")
    try:
        fn = getattr(importlib.import_module(mod_name), fn_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not import metric {spec!r}: {e}") from e
    return Evaluator(
        name=str(getattr(fn, "metric_name", fn_name)),
        higher_is_better=bool(getattr(fn, "higher_is_better", True)),
        fn=fn,
    )


def resolve_evaluator(
    metric: str, eval_path: Path | None, *, max_length: int = 512
) -> Evaluator:
    """Map a ``--metric`` value to an :class:`Evaluator`.

    ``"perplexity"`` needs ``eval_path``; a ``"module:function"`` spec supplies
    its own data so ``eval_path`` is optional.
    """
    if metric == "perplexity":
        if eval_path is None:
            raise ValueError("Metric 'perplexity' requires an --eval set.")
        return perplexity_evaluator(load_eval_texts(eval_path), max_length=max_length)
    if ":" in metric:
        return callable_evaluator(metric)
    raise ValueError(
        f"Unknown metric {metric!r}; use {BUILTIN_METRICS} or a 'module:function' callable."
    )


@dataclass(frozen=True)
class AccuracyBar:
    """How much degradation from the fp baseline metric is acceptable.

    ``rel`` is a fraction of the baseline (``rel:0.01`` = within 1%); ``abs`` is
    an absolute delta in metric units. Direction-aware: for a higher-is-better
    metric the bar is a floor below the baseline, for lower-is-better a ceiling
    above it.
    """

    mode: str  # "rel" | "abs"
    value: float

    @classmethod
    def parse(cls, spec: str) -> AccuracyBar:
        mode, sep, value = spec.partition(":")
        if not sep or mode not in ("rel", "abs"):
            raise ValueError(f"--accuracy-bar must be 'rel:<frac>' or 'abs:<delta>', got {spec!r}")
        try:
            v = float(value)
        except ValueError as e:
            raise ValueError(f"--accuracy-bar value must be a number, got {value!r}") from e
        if v < 0:
            raise ValueError(f"--accuracy-bar must be non-negative, got {v}")
        return cls(mode, v)

    def threshold(self, baseline: float, higher_is_better: bool) -> float:
        """The worst metric value still inside the bar."""
        if self.mode == "rel":
            return baseline * (1 - self.value) if higher_is_better else baseline * (1 + self.value)
        return baseline - self.value if higher_is_better else baseline + self.value

    def passes(self, candidate: float, baseline: float, higher_is_better: bool) -> bool:
        thr = self.threshold(baseline, higher_is_better)
        return candidate >= thr if higher_is_better else candidate <= thr
