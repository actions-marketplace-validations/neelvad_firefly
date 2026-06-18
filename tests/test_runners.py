"""Tests for the runner abstraction (the --runner seam).

The HF capture path itself is covered by test_capture.py; these tests cover
the registry/dispatch and that the orchestrators route through a Runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from firefly.runners import CaptureResult, available_runners, get_runner


def test_get_runner_hf() -> None:
    runner = get_runner("hf")
    assert runner.name == "hf"


def test_available_runners_lists_hf() -> None:
    assert "hf" in available_runners()


def test_get_runner_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown runner"):
        get_runner("tensorrt-llm")


def test_get_runner_vllm_instantiates_without_vllm_installed() -> None:
    # The runner object constructs without importing vLLM (that happens lazily
    # in capture()), so it's usable on the dev box for wiring/registry checks.
    runner = get_runner("vllm")
    assert runner.name == "vllm"


def test_vllm_runner_rejects_unknown_options() -> None:
    from firefly.runners.vllm import _parse_options

    with pytest.raises(ValueError, match="Unknown vLLM runner option"):
        _parse_options({"bogus": "1"})


def test_vllm_runner_parses_options() -> None:
    from firefly.runners.vllm import _parse_options

    opt = _parse_options(
        {"engine": "v0", "attention_backend": "FLASH_ATTN", "max_seq_len": "2048",
         "capture_decode": "true", "speculative_tokens": "3"}
    )
    assert opt["engine"] == "v0"
    assert opt["attention_backend"] == "FLASH_ATTN"
    assert opt["max_seq_len"] == 2048
    assert opt["capture_decode"] is True
    assert opt["speculative_tokens"] == 3


def test_vllm_runner_rejects_non_llm_domain() -> None:
    from firefly.runners.vllm import VLLMRunner

    with pytest.raises(ValueError, match="only supports the 'llm' domain"):
        VLLMRunner().capture("m", Path("in.json"), domain="recsys")


def test_vllm_fingerprint_is_a_real_weight_hash() -> None:
    """The vLLM fingerprint must hash actual weights (so a republished
    fine-tune is caught), not just encode the model name. _fingerprint_impl
    only needs an nn.Module, so it's testable on CPU without vLLM."""
    import torch.nn as nn

    from firefly.runners.vllm import _fingerprint_impl

    model = nn.Linear(128, 128, bias=False)
    fp_before = _fingerprint_impl(model)
    assert _fingerprint_impl(model) == fp_before  # deterministic
    with torch.no_grad():
        model.weight[-1] += 1.0  # later-row change, like the HF strided test
    assert _fingerprint_impl(model) != fp_before


def test_vllm_combine_fingerprints_tp() -> None:
    from firefly.runners.vllm import _combine_fingerprints

    # TP=1: single worker's hash passes through unchanged.
    assert _combine_fingerprints(["abc123"]) == "abc123"
    assert _combine_fingerprints("abc123") == "abc123"
    # TP>1: order-independent and stable.
    combined = _combine_fingerprints(["aaa", "bbb"])
    assert combined == _combine_fingerprints(["bbb", "aaa"])
    assert combined != _combine_fingerprints(["aaa", "ccc"])


def test_verify_backend_accepts_matching_impl() -> None:
    from firefly.runners.vllm import _verify_backend

    # No raise when the live impl matches the requested backend.
    _verify_backend("FLASH_ATTN", "FlashAttentionImpl")
    _verify_backend("XFORMERS", "XFormersImpl")
    _verify_backend("FLASHINFER", "FlashInferImpl")
    _verify_backend("UNKNOWN_BACKEND", "WhateverImpl")  # unverifiable → trusted


def test_verify_backend_rejects_silent_fallback() -> None:
    from firefly.runners.vllm import _verify_backend

    # The exact bug the strengthened guard exists to catch: XFORMERS silently
    # running FlashAttention (cached backend / dropped backend).
    with pytest.raises(RuntimeError, match="backend selector was ignored"):
        _verify_backend("XFORMERS", "FlashAttentionImpl")
    with pytest.raises(RuntimeError, match="backend selector was ignored"):
        _verify_backend("FLASHINFER", "FlashAttentionImpl")


def test_capture_reference_dispatches_to_runner(tmp_path: Path) -> None:
    """capture_reference turns a runner's CaptureResult into the artifact."""
    import torch

    from firefly.capture import capture_reference
    from firefly.reference import read_reference

    class _FakeRunner:
        name = "fake"

        def capture(self, model_id, inputs_path, **kwargs):
            return CaptureResult(
                tensors={"layer.0": torch.zeros(2, 4), "final_norm": torch.ones(2, 4)},
                fingerprint="fake-fp",
                head_counts={},
                env={"engine": "fake"},
                dtype="float32",
            )

    inputs = tmp_path / "golden.json"
    inputs.write_text('{"texts": ["x"], "max_length": 4}')
    out = tmp_path / "reference"

    capture_reference("any/model", inputs, out, runner=_FakeRunner())

    manifest, tensors = read_reference(out)
    assert manifest.model_fingerprint == "fake-fp"
    assert manifest.env == {"engine": "fake"}
    assert set(tensors) == {"layer.0", "final_norm"}


def test_capture_reference_passes_dtype_name_to_runner(tmp_path: Path) -> None:
    """The torch.dtype the caller passes is handed to the runner as a name."""
    import torch

    from firefly.capture import capture_reference

    recorded: dict = {}

    class _FakeRunner:
        name = "fake"

        def capture(self, model_id, inputs_path, **kwargs):
            recorded.update(kwargs)
            return CaptureResult(tensors={"t": torch.zeros(1)}, fingerprint="fp")

    inputs = tmp_path / "golden.json"
    inputs.write_text('{"texts": ["x"], "max_length": 4}')

    capture_reference(
        "m", inputs, tmp_path / "ref", dtype=torch.bfloat16, runner=_FakeRunner()
    )
    assert recorded["dtype"] == "bfloat16"


# --- SGLang runner (pure pieces; engine path needs GPU) ---------------------


def test_get_runner_sglang_instantiates_without_sglang() -> None:
    runner = get_runner("sglang")
    assert runner.name == "sglang"


def test_sglang_available() -> None:
    assert "sglang" in available_runners()


def test_sglang_runner_rejects_unknown_options() -> None:
    from firefly.runners.sglang import _parse_options

    with pytest.raises(ValueError, match="Unknown SGLang runner option"):
        _parse_options({"bogus": "1"})


def test_sglang_runner_rejects_non_llm_domain() -> None:
    from firefly.runners.sglang import SGLangRunner

    with pytest.raises(ValueError, match="only supports the 'llm' domain"):
        SGLangRunner().capture("m", Path("in.json"), domain="recsys")


def test_sglang_build_hook_specs_shape() -> None:
    from firefly.runners.sglang import build_hook_specs

    specs = build_hook_specs(n_layers=2, per_head=False, out_path="/tmp/x.pt")
    names = [s["name"] for s in specs]
    assert names == [
        "layer.0.self_attn", "layer.0.mlp", "layer.0",
        "layer.1.self_attn", "layer.1.mlp", "layer.1",
        "final_norm",
    ]
    # Each spec targets exactly one module via an exact-match pattern.
    assert all(len(s["target_modules"]) == 1 for s in specs)
    assert specs[0]["target_modules"] == ["model.layers.0.self_attn"]
    assert specs[-1]["target_modules"] == ["model.norm"]
    # Only the terminal tap flushes.
    assert specs[-1]["config"]["flush"] is True
    assert all("flush" not in s["config"] for s in specs[:-1])
    # The factory path is importable as written.
    from firefly.runners._sglang_hooks import capture_hook_factory  # noqa: F401
    assert all(s["hook_factory"].endswith("capture_hook_factory") for s in specs)


def test_sglang_build_hook_specs_per_head_adds_o_proj_input_tap() -> None:
    from firefly.runners.sglang import build_hook_specs

    specs = build_hook_specs(n_layers=1, per_head=True, out_path="/tmp/x.pt")
    by_name = {s["name"]: s for s in specs}
    assert "layer.0.attn_heads" in by_name
    head_spec = by_name["layer.0.attn_heads"]
    assert head_spec["target_modules"] == ["model.layers.0.self_attn.o_proj"]
    assert head_spec["config"]["capture_input"] is True


def test_sglang_hook_factory_records_prefill_and_flushes(tmp_path: Path) -> None:
    import torch

    from firefly.runners import _sglang_hooks as H

    H._reset()
    out = tmp_path / "caps.pt"
    # Two taps writing to the same accumulator; the second flushes.
    h_attn = H.capture_hook_factory({"name": "layer.0.self_attn", "out_path": str(out)})
    h_final = H.capture_hook_factory({"name": "final_norm", "out_path": str(out), "flush": True})

    prefill = torch.randn(4, 8)   # token axis > 1
    decode = torch.randn(1, 8)    # decode step — must be skipped

    h_attn(None, (prefill,), prefill)
    h_attn(None, (decode,), decode)        # skipped (leading dim 1)
    h_final(None, (prefill,), prefill)     # records + flushes

    saved = torch.load(out, weights_only=True)
    assert set(saved) == {"layer.0.self_attn", "final_norm"}
    assert saved["layer.0.self_attn"].shape == (4, 8)
    H._reset()


def test_sglang_hook_factory_capture_input(tmp_path: Path) -> None:
    import torch

    from firefly.runners import _sglang_hooks as H

    H._reset()
    out = tmp_path / "caps.pt"
    h = H.capture_hook_factory(
        {"name": "layer.0.attn_heads", "out_path": str(out), "capture_input": True, "flush": True}
    )
    x_in = torch.randn(4, 6)
    y_out = torch.randn(4, 8)
    h(None, (x_in,), y_out)
    saved = torch.load(out, weights_only=True)
    # capture_input => stored the INPUT (width 6), not the output (width 8).
    assert saved["layer.0.attn_heads"].shape == (4, 6)
    H._reset()


def test_tap_order_key_forward_order() -> None:
    from firefly.runners._common import tap_order_key

    names = ["final_norm", "layer.10.mlp", "layer.2.self_attn", "layer.2.attn_heads",
             "layer.2.mlp", "layer.2", "layer.1.self_attn"]
    ordered = sorted(names, key=tap_order_key)
    assert ordered == [
        "layer.1.self_attn", "layer.2.self_attn", "layer.2.attn_heads",
        "layer.2.mlp", "layer.2", "layer.10.mlp", "final_norm",
    ]
