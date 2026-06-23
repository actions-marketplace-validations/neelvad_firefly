"""LLM proposer pure pieces (no API, no GPU): the compact-action → Recipe
expander, the tool-call parser, the prompt. The real LLM+GPU loop is the Modal
demo (scripts/demo_llm_agent.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from firefly.quant.llm import POLICY_TOOL, build_prompt, compact_to_recipe, parse_tool_action

_UNITS = {"layer.0": ["l0.q", "l0.mlp"], "layer.1": ["l1.q", "l1.mlp"], "layer.2": ["l2.q", "l2.mlp"]}
_ALL = {f for v in _UNITS.values() for f in v}


def _inp(tmp_path: Path) -> Path:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"texts": ["hi"]}))
    return p


def _expand(action, inp):
    return compact_to_recipe(
        action, model_id="m", scheme="int4wo", all_fqns=_ALL, unit_fqns=_UNITS,
        inputs_path=inp, default_group_size=128,
    )


def test_compact_action_awq_plus_keep_fp(tmp_path: Path) -> None:
    r = _expand({"quantizer": "awq", "keep_fp_units": ["layer.1"], "rationale": "x"}, _inp(tmp_path))
    assert r.quantizer["name"] == "awq"
    assert set(r.kept_fp_fqns) == {"l1.q", "l1.mlp"}
    assert set(r.quantize_fqns) == _ALL - {"l1.q", "l1.mlp"}


def test_compact_action_unknown_unit_ignored(tmp_path: Path) -> None:
    # sandbox: a hallucinated unit name can't break out — it's just dropped.
    r = _expand({"quantizer": "rtn", "keep_fp_units": ["layer.99"], "rationale": "x"}, _inp(tmp_path))
    assert r.kept_fp_fqns == [] and r.quantizer["name"] == "rtn"


def test_compact_action_smoothquant_pretransform(tmp_path: Path) -> None:
    r = _expand(
        {"quantizer": "rtn", "pre_transforms": ["smoothquant"], "keep_fp_units": [], "rationale": "x"},
        _inp(tmp_path),
    )
    assert [p["name"] for p in r.pre_transforms] == ["smoothquant"]


def test_parse_tool_action() -> None:
    class _Block:
        type = "tool_use"
        name = "propose_recipe"
        input = {"quantizer": "awq", "keep_fp_units": ["layer.0"], "rationale": "r"}

    class _Msg:
        content = [_Block()]

    action = parse_tool_action(_Msg())
    assert action["quantizer"] == "awq" and action["keep_fp_units"] == ["layer.0"]


def test_parse_tool_action_raises_without_tool_call() -> None:
    class _Text:
        type = "text"

    class _Msg:
        content = [_Text()]

    with pytest.raises(ValueError, match="propose_recipe"):
        parse_tool_action(_Msg())


def test_policy_tool_schema_shape() -> None:
    assert POLICY_TOOL["name"] == "propose_recipe"
    props = POLICY_TOOL["input_schema"]["properties"]
    assert set(props) >= {"quantizer", "keep_fp_units", "pre_transforms", "group_size", "rationale"}
    assert props["quantizer"]["enum"] == ["rtn", "awq"]


def test_build_prompt_includes_measurements_and_history() -> None:
    out = build_prompt({"goal": "min memory", "perplexity_fp": 11.7}, [{"step": 0, "perplexity": 12.2}])
    assert "propose_recipe" in out and "min memory" in out and "12.2" in out
