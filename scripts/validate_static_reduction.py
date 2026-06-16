"""GPU validation for the two-phase grid-stride reduction (commits dc94d90 +
a4f21a0). The dev box is CPU-only, so the rewritten Triton kernel could only be
bound-tested (``reduction_grid``) and drain-tested on CPU. This runs the real
``firefly.shadow`` code on a GPU and confronts it with the cases the old
single-block kernel couldn't handle.

Checks:
  A1 — eager correctness on a 4.2M-element tap ([1,1024,4096]). This is the
       case that blew up the old ``BLOCK = next_power_of_2(numel)`` kernel;
       here grid is capped at MAX_BLOCKS and each program grid-strides. Stats
       must match a torch reference.
  A2 — the same multi-block reduction captured into a CUDA graph and replayed,
       through the full StaticTapper + @tap_static + drain stack. Verifies the
       3-kernel sequence is graph-safe at scale and the drain produces the
       right record count.
  A3 — full-tensor blob copy of a >REDUCE_BLOCK tensor round-trips exactly
       (the blob copy is multi-block now too).
  D  — a bf16 tap reports dtype "bfloat16" + correct shape in the drained
       record and the tap_meta.json sidecar (not the old hardcoded fp32/[]).

Run:  uv run modal run scripts/validate_static_reduction.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-validate-static-reduction")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("firefly")
)


@app.function(image=image, gpu="A10G", timeout=600)
def validate() -> dict:
    import json
    import time
    from pathlib import Path

    import torch
    import torch.nn as nn

    from firefly import shadow
    from firefly.shadow.triton_stats import MAX_BLOCKS, REDUCE_BLOCK, reduction_grid

    print(f"torch: {torch.__version__}")
    try:
        import triton

        print(f"triton: {triton.__version__}")
    except ImportError:
        print("triton: <not importable>")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"REDUCE_BLOCK={REDUCE_BLOCK}  MAX_BLOCKS={MAX_BLOCKS}")
    print()

    dev = "cuda"
    verdict: dict = {}

    def fresh_buffers(n_slots=1, max_blob_numel=1):
        return {
            "stats_buf": torch.zeros((64, 5), device=dev, dtype=torch.float32),
            "counter": torch.zeros((1,), device=dev, dtype=torch.int32),
            "blob_buf": torch.zeros((n_slots, max_blob_numel), device=dev, dtype=torch.float32),
            "blob_meta": torch.zeros((n_slots, 2), device=dev, dtype=torch.int32),
            "blob_counter": torch.zeros((1,), device=dev, dtype=torch.int32),
            "alert": torch.zeros((1,), device=dev, dtype=torch.int32),
            "partials": torch.zeros((MAX_BLOCKS, 4), device=dev, dtype=torch.float32),
            "decision": torch.zeros((2,), device=dev, dtype=torch.int32),
        }

    def call_op(x, b, first_n=0, every_n=0, on_alert=0):
        torch.ops.firefly.capture_static(
            x, b["stats_buf"], b["counter"], 0,
            b["blob_buf"], b["blob_meta"], b["blob_counter"], b["alert"],
            b["partials"], b["decision"],
            first_n, every_n, on_alert,
        )
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # A1 — eager multi-block correctness on the 4.2M-element case.
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    x = torch.randn(1, 1024, 4096, device=dev)  # 4,194,304 elements
    grid = reduction_grid(x.numel())
    b = fresh_buffers()
    call_op(x, b)
    row = b["stats_buf"][0].cpu()
    xf = x.float()
    pop_std = ((xf - xf.mean()) ** 2).mean().sqrt()
    a1 = {
        "numel": x.numel(),
        "grid": grid,
        "grid_is_capped": grid == MAX_BLOCKS,
        "mean_ok": abs(row[0].item() - xf.mean().item()) < 1e-3,
        "abs_mean_ok": abs(row[1].item() - xf.abs().mean().item()) < 1e-3,
        "abs_max_ok": abs(row[2].item() - xf.abs().max().item()) / xf.abs().max().item() < 1e-4,
        "std_ok": abs(row[3].item() - pop_std.item()) / pop_std.item() < 1e-3,
        "counter_is_1": int(b["counter"].item()) == 1,
        "row": [round(v, 5) for v in row.tolist()],
        "ref": [round(xf.mean().item(), 5), round(xf.abs().mean().item(), 5),
                round(xf.abs().max().item(), 5), round(pop_std.item(), 5)],
    }
    a1["pass"] = all(a1[k] for k in ("grid_is_capped", "mean_ok", "abs_mean_ok",
                                     "abs_max_ok", "std_ok", "counter_is_1"))
    verdict["A1_eager_multiblock"] = a1
    print(f"A1 multi-block stats: {a1}")

    # ------------------------------------------------------------------
    # A3 — blob copy of a >REDUCE_BLOCK tensor round-trips exactly.
    # ------------------------------------------------------------------
    x3 = torch.randn(1, 64, 512, device=dev)  # 32,768 elements (> REDUCE_BLOCK)
    b3 = fresh_buffers(n_slots=4, max_blob_numel=200_000)
    call_op(x3, b3, first_n=1)
    blob = b3["blob_buf"][0, : x3.numel()].cpu()
    a3 = {
        "numel": x3.numel(),
        "blob_count_is_1": int(b3["blob_counter"].item()) == 1,
        "blob_matches": torch.allclose(blob, x3.float().flatten().cpu(), atol=1e-5),
        "meta_numel_ok": int(b3["blob_meta"][0, 0].item()) == x3.numel(),
    }
    a3["pass"] = all(a3[k] for k in ("blob_count_is_1", "blob_matches", "meta_numel_ok"))
    verdict["A3_blob_copy_multiblock"] = a3
    print(f"A3 blob copy: {a3}")

    # ------------------------------------------------------------------
    # A2 + D — full stack: bf16 multi-block tap, CUDA-graph capture/replay.
    # ------------------------------------------------------------------
    HIDDEN = 8192  # activation [1, 8192] -> grid = ceil(8192/2048) = 4 (multi-block)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, HIDDEN)
            self.fc2 = nn.Linear(HIDDEN, 16)

        def forward(self, x):
            h = self._tap(self.fc1(x))
            return self.fc2(h)

        @shadow.tap_static(idx=0, name="layer.0.fc1")
        def _tap(self, x):
            return x

    model = Net().eval().to(dev).to(torch.bfloat16)
    x2 = torch.randn(1, 16, device=dev, dtype=torch.bfloat16)
    log_dir = Path("/tmp/firefly-static-reduction")
    if log_dir.exists():
        import shutil

        shutil.rmtree(log_dir)

    N_REPLAYS = 50
    policy = shadow.StaticFullTensorPolicy(first_n_steps=3, max_blob_numel=HIDDEN + 16)
    with (
        shadow.StaticTapper(
            log_dir, {0: "layer.0.fc1"}, drain_interval_s=0.05,
            full_tensor_policy=policy, device=dev,
        ) as tapper,
        torch.inference_mode(),
    ):
        for _ in range(3):
            model(x2)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(x2)
        torch.cuda.current_stream().wait_stream(s)
        tapper.counter.zero_()
        tapper.blob_counter.zero_()
        tapper.stats_buf.zero_()
        tapper.blob_buf.zero_()
        tapper.blob_meta.zero_()
        with torch.cuda.graph(g):
            _ = model(x2)
        for _ in range(N_REPLAYS):
            g.replay()
        torch.cuda.synchronize()
        time.sleep(0.5)

    records = [
        json.loads(line)
        for line in (log_dir / "stats.jsonl").read_text().splitlines()
    ]
    meta = json.loads((log_dir / "tap_meta.json").read_text())
    a2d = {
        "graph_grid": reduction_grid(HIDDEN),
        "n_records": len(records),
        "records_ok": len(records) == N_REPLAYS,
        "dtype_ok": bool(records) and records[0]["stats"]["dtype"] == "bfloat16",
        "shape_ok": bool(records) and records[0]["stats"]["shape"] == [1, HIDDEN],
        "sidecar_dtype_ok": meta.get("0", {}).get("dtype") == "bfloat16",
        "n_blobs": sum(1 for r in records if r.get("blob_path")),
        "first_record_stats": records[0]["stats"] if records else None,
    }
    a2d["pass"] = all(a2d[k] for k in ("records_ok", "dtype_ok", "shape_ok", "sidecar_dtype_ok"))
    verdict["A2D_graph_bf16_fullstack"] = a2d
    print(f"A2+D full stack: {a2d}")

    verdict["overall_pass"] = bool(a1["pass"] and a3["pass"] and a2d["pass"])
    print()
    print("=" * 60)
    print(f"OVERALL: {'PASS' if verdict['overall_pass'] else 'FAIL'}")
    print("=" * 60)
    return verdict


@app.local_entrypoint()
def main() -> None:
    import json

    verdict = validate.remote()
    print()
    print("verdict:")
    print(json.dumps(verdict, indent=2, default=str))
