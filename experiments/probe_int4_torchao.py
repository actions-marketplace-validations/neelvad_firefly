"""Probe: which torchao int4 weight-only config actually runs on the GPU?

The breadth sweep's int4wo runs all died with `ImportError: Requires mslk >=
1.0.0` — torchao 0.17's *default* Int4WeightOnlyConfig routes through a kernel
lib that isn't in our image. Rather than guess the right incantation, introspect
the installed torchao on real hardware: print the config signature + available
layouts/packing-formats, then try a matrix of int4 configs, reporting which one
both quantizes AND runs a forward cleanly.

The winner gets wired into firefly.quant.torchao._quant_config.

Run:  uv run modal run experiments/probe_int4_torchao.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-probe-int4")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.6",
    "torchao>=0.7",
    extra_index_url="https://download.pytorch.org/whl/cu124",
)


@app.function(image=image, gpu="A10G", timeout=600)
def probe() -> dict:
    import inspect

    import torch
    import torch.nn as nn
    import torchao
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    print(f"torchao {torchao.__version__}  torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    sig = inspect.signature(Int4WeightOnlyConfig.__init__)
    print(f"Int4WeightOnlyConfig signature: {sig}")
    params = set(sig.parameters)

    import torchao.quantization as q
    print("int4 symbols:", [s for s in dir(q) if "int4" in s.lower()])
    try:
        import torchao.dtypes as d
        print("layouts:", [s for s in dir(d) if "Layout" in s])
    except Exception as e:
        print("layout introspection failed:", e)

    def make_model():
        m = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
        return m.to("cuda").to(torch.bfloat16).eval()

    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)

    candidates: list[tuple[str, object]] = [
        ("Int4WeightOnlyConfig()", lambda: Int4WeightOnlyConfig()),
        ("Int4WeightOnlyConfig(group_size=32)", lambda: Int4WeightOnlyConfig(group_size=32)),
        ("Int4WeightOnlyConfig(group_size=128)", lambda: Int4WeightOnlyConfig(group_size=128)),
    ]
    if "version" in params:
        for v in (1, 2):
            candidates.append(
                (f"Int4WeightOnlyConfig(group_size=32, version={v})",
                 (lambda v=v: Int4WeightOnlyConfig(group_size=32, version=v)))
            )
    if "int4_packing_format" in params:
        for fmt in ("tile_packed_to_4d", "plain", "plain_int32", "marlin"):
            candidates.append(
                (f"Int4WeightOnlyConfig(group_size=32, int4_packing_format={fmt!r})",
                 (lambda f=fmt: Int4WeightOnlyConfig(group_size=32, int4_packing_format=f)))
            )
    if "layout" in params:
        try:
            from torchao.dtypes import TensorCoreTiledLayout
            candidates.append(
                ("Int4WeightOnlyConfig(group_size=32, layout=TensorCoreTiledLayout())",
                 lambda: Int4WeightOnlyConfig(group_size=32, layout=TensorCoreTiledLayout()))
            )
        except Exception as e:
            print("no TensorCoreTiledLayout:", e)
    try:
        from torchao.quantization import int4_weight_only
        candidates.append(
            ("int4_weight_only(group_size=32)", lambda: int4_weight_only(group_size=32))
        )
    except Exception:
        pass

    results: dict[str, str] = {}
    for label, build in candidates:
        try:
            cfg = build()
        except Exception as e:
            results[label] = f"CONFIG-BUILD-FAIL: {type(e).__name__}: {e}"
            print(label, "->", results[label])
            continue
        m = make_model()
        try:
            quantize_(m, cfg)
        except Exception as e:
            results[label] = f"QUANTIZE-FAIL: {type(e).__name__}: {e}"
            print(label, "->", results[label])
            continue
        try:
            with torch.no_grad():
                y = m(x)
            results[label] = f"OK (finite={bool(torch.isfinite(y).all())}, shape={tuple(y.shape)})"
        except Exception as e:
            results[label] = f"FORWARD-FAIL: {type(e).__name__}: {e}"
        print(label, "->", results[label])

    working = [k for k, v in results.items() if v.startswith("OK")]
    print("\nWORKING CONFIGS:", working)
    return {
        "torchao": torchao.__version__,
        "signature": str(sig),
        "results": results,
        "working": working,
    }


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(probe.remote(), indent=2, default=str))
