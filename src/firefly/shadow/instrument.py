"""Auto-instrumentation via torch.fx or named_modules."""

from __future__ import annotations

import functools
import re

import torch

from firefly.shadow.static import _active_static_tapper


@torch.fx.wrap
def _firefly_eager_tap(x: torch.Tensor, name: str) -> torch.Tensor:
    """FX-opaque wrapper for eager-mode capture. Inserted at instrumented
    sites by :func:`_instrument_via_fx`."""
    return torch.ops.firefly.capture(x, name)

@torch.fx.wrap
def _firefly_static_tap(x: torch.Tensor, tap_idx: int) -> torch.Tensor:
    """FX-opaque wrapper for CUDA-graph-mode capture.

    Looks up the active StaticTapper at runtime; the lookup happens during
    graph capture and the resulting op call (with its concrete buffer
    pointers baked in) is what survives into the captured graph for replay.
    """
    t = _active_static_tapper()
    if t is None:
        return x
    return torch.ops.firefly.capture_static(
        x, t.stats_buf, t.counter, tap_idx,
        t.blob_buf, t.blob_meta, t.blob_counter, t.alert_flag,
        t.full_tensor_policy.first_n_steps,
        t.full_tensor_policy.every_n_steps,
        1 if t.full_tensor_policy.on_alert else 0,
    )

def instrument(
    model: torch.nn.Module,
    pattern: str,
    *,
    mode: str = "eager",
    method: str = "auto",
    tap_index_start: int = 0,
) -> tuple[torch.nn.Module, dict[int, str]]:
    """Insert shadow capture ops at module sites matching ``pattern``.

    Args:
        model: The model to instrument. Modified in place for the
            ``named_modules`` path; replaced with a GraphModule for the
            ``fx`` path. Either way, the returned model is the one to
            use for inference.
        pattern: Regex matched against module names (e.g.
            ``r"layer\\.(7|15)\\.self_attn$"``).
        mode: ``"eager"`` (calls :func:`capture` with the tap name) or
            ``"static"`` (calls :func:`capture_static` with a tap index).
        method: ``"fx"`` forces torch.fx; ``"named_modules"`` wraps
            forwards; ``"auto"`` (default) tries fx first then falls
            back to named_modules if symbolic tracing fails.
        tap_index_start: For ``mode="static"``: starting tap_idx (useful
            when instrumenting multiple sub-models with disjoint indices).

    Returns:
        ``(instrumented_model, index_to_name)`` — the second element is a
        ``{tap_idx: module_name}`` map populated for ``mode="static"``,
        empty for ``mode="eager"``. Pass it to
        :class:`StaticTapper(index_to_name=...)` so the drain can re-attach
        names.
    """
    if mode not in ("eager", "static"):
        raise ValueError(f"mode must be 'eager' or 'static', got {mode!r}")
    if method not in ("fx", "named_modules", "auto"):
        raise ValueError(
            f"method must be 'fx', 'named_modules', or 'auto', got {method!r}"
        )

    if method == "fx":
        return _instrument_via_fx(model, pattern, mode, tap_index_start)
    if method == "named_modules":
        return _instrument_via_named_modules(model, pattern, mode, tap_index_start)
    # auto: try fx, fall back to named_modules on tracing failure.
    try:
        return _instrument_via_fx(model, pattern, mode, tap_index_start)
    except Exception as e:  # noqa: BLE001 — FX raises many types
        import sys
        print(
            f"[firefly] instrument: torch.fx tracing failed ({type(e).__name__}: "
            f"{e}); falling back to named_modules method.",
            file=sys.stderr,
        )
        return _instrument_via_named_modules(model, pattern, mode, tap_index_start)

def _instrument_via_fx(
    model: torch.nn.Module,
    pattern: str,
    mode: str,
    tap_index_start: int,
) -> tuple[torch.nn.Module, dict[int, str]]:
    """torch.fx symbolic-trace then insert capture call_function nodes."""
    pat = re.compile(pattern)
    gm = torch.fx.symbolic_trace(model)
    index_to_name: dict[int, str] = {}
    next_idx = tap_index_start

    # Iterate over a list copy — we mutate the graph during iteration.
    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue
        target_str = str(node.target)
        if not pat.search(target_str):
            continue

        if mode == "eager":
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    _firefly_eager_tap, args=(node, target_str)
                )
        else:  # static
            tap_idx = next_idx
            index_to_name[tap_idx] = target_str
            next_idx += 1
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    _firefly_static_tap, args=(node, tap_idx)
                )
        # Redirect every downstream use of `node` to read from `new_node`
        # instead. The replacement runs over all current uses; we then
        # restore `new_node`'s reference to the original `node` since the
        # capture wrapper needs the captured tensor as input.
        node.replace_all_uses_with(new_node)
        new_node.args = (node, *new_node.args[1:])

    gm.recompile()
    return gm, index_to_name

def _instrument_via_named_modules(
    model: torch.nn.Module,
    pattern: str,
    mode: str,
    tap_index_start: int,
) -> tuple[torch.nn.Module, dict[int, str]]:
    """Monkey-patch forward() on every module whose name matches the pattern."""
    pat = re.compile(pattern)
    index_to_name: dict[int, str] = {}
    next_idx = tap_index_start

    for name, mod in model.named_modules():
        if not pat.search(name):
            continue
        if mode == "eager":
            _wrap_forward_eager(mod, name)
        else:
            index_to_name[next_idx] = name
            _wrap_forward_static(mod, next_idx)
            next_idx += 1
    return model, index_to_name

def _wrap_forward_eager(mod: torch.nn.Module, name: str) -> None:
    """Replace ``mod.forward`` so its return value is routed through the
    eager capture op. Tuple-returning forwards get their first tensor
    element captured."""
    original = mod.forward

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return torch.ops.firefly.capture(out, name)
        if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            return (torch.ops.firefly.capture(out[0], name), *out[1:])
        return out

    mod.forward = wrapped

def _wrap_forward_static(mod: torch.nn.Module, tap_idx: int) -> None:
    """Like :func:`_wrap_forward_eager` but for CUDA-graph mode."""
    original = mod.forward

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        t = _active_static_tapper()
        if t is None:
            return out
        first_n = t.full_tensor_policy.first_n_steps
        every_n = t.full_tensor_policy.every_n_steps
        on_alert_enabled = 1 if t.full_tensor_policy.on_alert else 0
        if isinstance(out, torch.Tensor):
            return torch.ops.firefly.capture_static(
                out, t.stats_buf, t.counter, tap_idx,
                t.blob_buf, t.blob_meta, t.blob_counter, t.alert_flag,
                first_n, every_n, on_alert_enabled,
            )
        if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            head = torch.ops.firefly.capture_static(
                out[0], t.stats_buf, t.counter, tap_idx,
                t.blob_buf, t.blob_meta, t.blob_counter, t.alert_flag,
                first_n, every_n, on_alert_enabled,
            )
            return (head, *out[1:])
        return out

    mod.forward = wrapped
