"""Tests for the runner abstraction (the --runner seam).

The HF capture path itself is covered by test_capture.py; these tests cover
the registry/dispatch and that the orchestrators route through a Runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from firefly.runners import CaptureResult, available_runners, get_runner


def test_get_runner_hf() -> None:
    runner = get_runner("hf")
    assert runner.name == "hf"


def test_available_runners_lists_hf() -> None:
    assert "hf" in available_runners()


def test_get_runner_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown runner"):
        get_runner("sglang")


def test_get_runner_vllm_not_yet_wired() -> None:
    # The vLLM runner is planned; until extracted it raises a clear error
    # rather than silently falling back to a different engine.
    with pytest.raises(NotImplementedError, match="vLLM runner"):
        get_runner("vllm")


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
