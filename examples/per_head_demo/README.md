# Per-head attention attribution demo

Firefly's `check` names the first decoder layer where a candidate diverges
from the reference. With `capture --per-head`, it goes one level deeper and
names the **attention head** that carries the divergence.

## Run it

```bash
bash examples/per_head_demo/run_demo.sh
```

Runs on CPU in fp32 (~30s after the SmolLM-135M download is cached). No GPU.

## What it does

1. **Capture** a reference from `HuggingFaceTB/SmolLM-135M` with `--per-head`.
   This adds a `layer.{i}.attn_heads` tap per layer that captures the *input*
   to the attention output projection (`o_proj`) — the concatenated per-head
   context vectors, before `o_proj` linearly mixes them. (Post-`o_proj` the
   heads are unrecoverable, which is why the tap is on the input.)
2. **Clean check** against the unmodified model → no divergence, exit 0.
3. **Break one head**: `make_broken_head.py` perturbs the `q_proj` rows for a
   single query head (layer 7, head 4) with `N(0, 1e-2)`. Perturbing a *query*
   head localizes cleanly to that one head even under grouped-query attention
   (perturbing a key/value head would smear across its whole query group).
4. **Broken check** → exit 1, with a per-head attribution table.

## Expected result

The first-divergence layer is `layer.7.self_attn`, and the per-head table
pinpoints the head:

```
                        Per-head attention attribution
┃ Tap                 ┃ Worst head ┃   max |Δ| ┃ median head ┃ concentration ┃
│ layer.7.attn_heads  │      4 / 9 │ 1.301e-02 │   0.000e+00 │          inf× │
│ layer.8.attn_heads  │      1 / 9 │ 4.607e-03 │   1.445e-03 │          3.2× │
│ layer.9.attn_heads  │      3 / 9 │ 2.990e-03 │   1.041e-03 │          2.9× │
```

Read the signature:

- **Layer 7, head 4** is the worst head, and `concentration = inf` because
  every *other* head at layer 7 is bit-identical (median diff is 0) — the
  perturbation is perfectly localized to the head we broke.
- **Downstream layers** (8, 9, …) show the divergence smearing out: every
  head diverges a little as the perturbation propagates through the residual
  stream, so concentration drops to single-digit multiples. High concentration
  marks the *source*; low concentration marks *propagation*.

The structured `--report-json` output includes a `per_head` array with the
full per-head breakdown and the `concentration` ratio for each tap.
