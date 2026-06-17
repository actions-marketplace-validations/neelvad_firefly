"""Confront quant-risk predictions with real torchao W8A8 kernels.

:mod:`firefly.quant_risk` *simulates* int8 quantization and predicts which
tensors break, from stored activations alone. This module checks that
prediction against reality, so the claim can be validated (and guarded against
regression) rather than asserted.

The test is **local and apples-to-apples**, which is the whole point. torchao's
``Int8DynamicActivationInt8WeightConfig`` (W8A8) quantizes the *inputs* to
Linear layers (per-token) and their *weights* (per-channel). So we:

  1. capture every Linear's fp input X — the tensor torchao actually quantizes,
  2. push that *same* X through both the fp Linear and the real torchao int8
     Linear and measure the local output error (no accumulation — the error of
     that one layer's quantization in isolation),
  3. ask whether quant-risk's prediction on X ranks the layers torchao hurts.

Mechanism under test: torchao's activation quant is *per-token* (per row), so
it does not rescale feature-channel outliers — exactly the failure mode
``channel_concentration`` measures, and exactly why SmoothQuant exists. On
SmolLM-135M this lands at Spearman ~0.7 (channel_concentration vs local error).

Unlike the rest of Firefly's analysis layer this module is **not pure**: it
loads and runs models, and depends on the optional ``torchao`` extra. Import
errors for ``torchao`` are surfaced with an actionable message.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from firefly.capture import load_model_and_tokenizer
from firefly.determinism import set_deterministic
from firefly.quant_risk import tap_quant_risk

_DEFAULT_PROMPT = "the quick brown fox jumps over the lazy dog"


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (Pearson on ranks), no scipy dependency."""

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """``mean|a - b| / mean|a|`` — relative L1, the local-divergence metric."""
    a, b = a.float(), b.float()
    denom = a.abs().mean().item() or 1.0
    return (a - b).abs().mean().item() / denom


@dataclass
class LinearRisk:
    """Predicted quant-risk vs measured torchao error for one Linear layer."""

    name: str
    channel_concentration: float
    per_tensor_rel_err: float
    mitigation_gain: float
    actual_local_err: float
    """``rel_l1(fp_linear(X), torchao_int8_linear(X))`` on the same fp input X."""


@dataclass
class TorchaoValidationResult:
    """Per-Linear measurement of real torchao quant divergence + how well the
    quant-risk proxies rank it. These are *measurements*, not a pass/fail
    verdict: the breadth sweep showed the proxies don't reliably rank divergence
    across model families, so this no longer carries a "validated" gate. The
    product surface for measuring real quant divergence is ``firefly quant-diff``.
    """

    model_id: str
    bits: int
    scheme: str = "w8a8"
    records: list[LinearRisk] = field(default_factory=list)

    @property
    def spearman_concentration(self) -> float:
        return spearman(
            [r.channel_concentration for r in self.records],
            [r.actual_local_err for r in self.records],
        )

    @property
    def spearman_per_tensor(self) -> float:
        return spearman(
            [r.per_tensor_rel_err for r in self.records],
            [r.actual_local_err for r in self.records],
        )

    @property
    def spearman_mitigation_gain(self) -> float:
        return spearman(
            [r.mitigation_gain for r in self.records],
            [r.actual_local_err for r in self.records],
        )


#: Quant schemes we validate against. ``w8a8`` quantizes *activations*
#: (per-token int8) + weights (per-channel int8) — the scheme quant-risk's
#: activation analysis is designed to predict. ``int4wo`` is int4 weight-only
#: (W4A16): activations stay in fp, only weights are quantized. quant-risk
#: looks at activations, so it is *expected* not to predict int4wo — running it
#: maps the tool's boundary, it isn't a target we expect to hit.
QUANT_SCHEMES = ("w8a8", "int4wo")


class QuantCompatibilityError(RuntimeError):
    """A model/scheme/device combination torchao can't quantize cleanly.

    Raised with actionable guidance instead of letting torchao's lower-level
    (often cryptic) errors surface raw — e.g. int4 on CPU, a missing int4
    kernel library, or a group_size that doesn't divide a weight dimension.
    """


def quant_preflight(scheme: str, device: str) -> None:
    """Cheap static checks before loading the model, so known-incompatible
    combinations fail fast with guidance rather than after a long download +
    load. Does not import torchao. Raises :class:`QuantCompatibilityError`."""
    if scheme not in QUANT_SCHEMES:
        raise QuantCompatibilityError(
            f"unknown quant scheme {scheme!r}; choose from {QUANT_SCHEMES}"
        )
    if scheme == "int4wo" and not str(device).startswith("cuda"):
        raise QuantCompatibilityError(
            "int4 weight-only (int4wo) needs a CUDA GPU — torchao's int4 kernels "
            "aren't available on CPU. Re-run with --device cuda, or use "
            "--scheme w8a8 (which runs on CPU)."
        )


def _translate_quant_error(scheme: str, exc: Exception) -> QuantCompatibilityError | None:
    """Map a known torchao quantization failure to actionable guidance.

    Returns ``None`` for unrecognized errors (the caller re-raises the
    original, so we never swallow a genuine bug behind a guess)."""
    low = str(exc).lower()
    if "mslk" in low:
        return QuantCompatibilityError(
            f"torchao's {scheme} path needs the 'mslk' int4 kernel library, which "
            f"isn't available in this environment ({str(exc).strip()}). Install it "
            f"on the GPU image, or use --scheme w8a8."
        )
    if "divisible" in low or "group_size" in low or "group size" in low:
        return QuantCompatibilityError(
            f"{scheme}: the group_size doesn't divide a weight dimension "
            f"({str(exc).strip()}). Try a smaller --group-size (e.g. 16)."
        )
    return None


def _quant_config(scheme: str, group_size: int = 32):
    """Build the torchao config for ``scheme``, handling API churn across
    torchao versions. ``group_size`` only applies to ``int4wo`` (smaller groups
    tolerate weight dims that aren't multiples of 128, e.g. SmolLM's 576)."""
    if scheme == "w8a8":
        try:
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig

            return Int8DynamicActivationInt8WeightConfig()
        except ImportError:  # older torchao API
            from torchao.quantization import int8_dynamic_activation_int8_weight

            return int8_dynamic_activation_int8_weight()
    if scheme == "int4wo":
        try:
            from torchao.quantization import Int4WeightOnlyConfig
        except ImportError:  # older torchao API
            from torchao.quantization import int4_weight_only

            return int4_weight_only(group_size=group_size)
        # torchao >= 0.x defaults int4 to the PLAIN packing format, which needs
        # an external 'mslk' kernel lib not shipped in our images. The
        # tile_packed_to_4d format uses the built-in tinygemm kernels and is the
        # only int4 config that runs cleanly on our GPUs (verified by
        # scripts/probe_int4_torchao.py on A10G, torchao 0.17). Pass the format
        # as a string so we don't depend on the enum's import path; fall back
        # for older torchao that lacks the kwarg.
        try:
            return Int4WeightOnlyConfig(
                group_size=group_size, int4_packing_format="tile_packed_to_4d"
            )
        except TypeError:
            return Int4WeightOnlyConfig(group_size=group_size)
    raise ValueError(f"unknown quant scheme {scheme!r}; choose from {QUANT_SCHEMES}")


def quantize_model(
    model: nn.Module,
    scheme: str = "w8a8",
    group_size: int = 32,
    module_filter: Callable[[nn.Module, str], bool] | None = None,
) -> nn.Module:
    """Apply real torchao quantization (``scheme``) in place.

    ``module_filter(module, fully_qualified_name) -> bool`` selects which
    modules to quantize (torchao's ``filter_fn``); ``None`` quantizes every
    Linear (torchao's default). The filter is what lets the sensitivity loop
    quantize one layer at a time. Raises a clear error if ``torchao`` is absent.
    """
    try:
        from torchao.quantization import quantize_
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "torchao is required for quant-risk validation. "
            "Install it with: uv pip install 'firefly[torchao]'"
        ) from e

    cfg = _quant_config(scheme, group_size=group_size)
    try:
        if module_filter is not None:
            quantize_(model, cfg, filter_fn=module_filter)
        else:
            quantize_(model, cfg)
    except Exception as e:  # translate known torchao failures; re-raise the rest
        translated = _translate_quant_error(scheme, e)
        if translated is not None:
            raise translated from e
        raise
    return model


def _make_batch(tokenizer, prompt: str, max_length: int, device: str) -> dict:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items()}


def capture_linear_inputs(model: nn.Module, batch: dict) -> dict[str, torch.Tensor]:
    """Run one forward, recording the fp input tensor to every ``nn.Linear``."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_mod, inputs, _out):
            if name not in captured and inputs:
                captured[name] = inputs[0].detach().clone()

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(**batch)
    finally:
        for h in handles:
            h.remove()
    return captured


def validate_against_torchao(
    model_id: str,
    device: str = "cpu",
    bits: int = 8,
    prompt: str = _DEFAULT_PROMPT,
    max_length: int = 16,
    dtype: torch.dtype = torch.float32,
    scheme: str = "w8a8",
    group_size: int = 32,
) -> TorchaoValidationResult:
    """Run the local confrontation and return per-Linear predictions vs reality.

    Loads two copies of ``model_id`` (one stays fp, one is torchao-quantized
    with ``scheme``), captures each Linear's fp input, and measures the local
    output error of the real quantized Linear on that input. Deterministic on
    CPU; ``int4wo`` needs CUDA.
    """
    set_deterministic()
    fp_model, tok = load_model_and_tokenizer(model_id, device=device, dtype=dtype)
    batch = _make_batch(tok, prompt, max_length, device)
    fp_inputs = capture_linear_inputs(fp_model, batch)
    fp_linears = {n: m for n, m in fp_model.named_modules() if isinstance(m, nn.Linear)}

    set_deterministic()
    q_model, _ = load_model_and_tokenizer(model_id, device=device, dtype=dtype)
    quantize_model(q_model, scheme=scheme, group_size=group_size)
    q_linears = {n: m for n, m in q_model.named_modules() if isinstance(m, nn.Linear)}

    records: list[LinearRisk] = []
    for name, x in fp_inputs.items():
        if name not in q_linears or name not in fp_linears:
            continue
        with torch.no_grad():
            y_fp = fp_linears[name](x)
            y_q = q_linears[name](x)
        risk = tap_quant_risk(name, x, bits=bits)
        records.append(
            LinearRisk(
                name=name,
                channel_concentration=risk.channel_concentration,
                per_tensor_rel_err=risk.per_tensor_rel_err,
                mitigation_gain=min(risk.mitigation_gain, 1e6),  # cap inf for ranking
                actual_local_err=rel_l1(y_fp, y_q),
            )
        )
    return TorchaoValidationResult(
        model_id=model_id, bits=bits, scheme=scheme, records=records
    )
