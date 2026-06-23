"""Deterministic router: diagnosis → recipe (fast, no model)."""

from __future__ import annotations

import json
from pathlib import Path

from firefly.quant.diagnose import Diagnosis, Finding
from firefly.quant.intervention import (
    ACTIVATION_OUTLIERS,
    SALIENT_WEIGHT_CHANNELS,
    SINGLE_UNIT_DOMINANCE,
)
from firefly.quant.route import route_recipe

_UNITS = {"layer.0": ["l0.q", "l0.mlp"], "layer.1": ["l1.q", "l1.mlp"]}
_ALL = {f for v in _UNITS.values() for f in v}


def _route(diagnosis, scheme, inp):
    return route_recipe(
        diagnosis, model_id="m", scheme=scheme, group_size=128,
        all_fqns=_ALL, unit_fqns=_UNITS, inputs_path=inp,
    )


def _inp(tmp_path: Path) -> Path:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"texts": ["hi"]}))
    return p


def test_int4_salient_routes_to_awq(tmp_path: Path) -> None:
    r = _route(Diagnosis([Finding(SALIENT_WEIGHT_CHANNELS, "l0.mlp", "awq", {}, "x")]), "int4wo", _inp(tmp_path))
    assert r.quantizer["name"] == "awq"
    assert "AWQ" in r.result["routing"][0]


def test_w8a8_activation_outliers_routes_to_smoothquant(tmp_path: Path) -> None:
    r = _route(Diagnosis([Finding(ACTIVATION_OUTLIERS, "layer.3.mlp", "smoothquant", {}, "x")]), "w8a8", _inp(tmp_path))
    assert r.quantizer["name"] == "rtn"
    assert [p["name"] for p in r.pre_transforms] == ["smoothquant"]


def test_single_unit_dominance_keeps_unit_fp(tmp_path: Path) -> None:
    r = _route(Diagnosis([Finding(SINGLE_UNIT_DOMINANCE, "layer.0", "mixed-precision", {}, "x")]), "w8a8", _inp(tmp_path))
    assert set(r.kept_fp_fqns) == {"l0.q", "l0.mlp"}
    assert set(r.quantize_fqns) == _ALL - {"l0.q", "l0.mlp"}


def test_int4_does_not_add_smoothquant(tmp_path: Path) -> None:
    # SmoothQuant treats activation quant — irrelevant for weight-only int4.
    r = _route(Diagnosis([Finding(ACTIVATION_OUTLIERS, "layer.3.mlp", "smoothquant", {}, "x")]), "int4wo", _inp(tmp_path))
    assert r.pre_transforms == [] and r.quantizer["name"] == "rtn"


def test_empty_diagnosis_falls_back_to_plain(tmp_path: Path) -> None:
    r = _route(Diagnosis([]), "int4wo", _inp(tmp_path))
    assert r.quantizer["name"] == "rtn" and r.pre_transforms == []
    assert "plain RTN" in r.result["routing"][0]
