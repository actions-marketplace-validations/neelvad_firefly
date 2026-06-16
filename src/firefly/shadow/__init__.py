"""Shadow-mode capture: passive activation observation for production inference.

Shadow-mode is the production-side complement to Firefly's CI-time capture.
The CI flow uses forward hooks, which are incompatible with torch.compile
(Dynamo can't trace Python hooks) and CUDA graphs (no callback hook point
after capture). Shadow-mode routes observation through registered custom ops
that Dynamo treats as opaque, plus a Triton-kernel op for the CUDA-graph path.

This package was split out of a single shadow.py; the public surface is
unchanged — everything is re-exported here so ``firefly.shadow.X`` keeps
working. Layout:

* ``buffer``       — _Event, _Buffer, DrainThread
* ``triton_stats`` — the CUDA-graph summary-stats Triton kernel
* ``sinks``        — LocalLogSink + S3/GCS/Azure streaming sinks + make_sink
* ``aggregate``    — aggregate() / load_tap_index()
* ``eager``        — Tapper, @tap, FullTensorPolicy (eager + torch.compile)
* ``static``       — StaticTapper, @tap_static (CUDA-graph replay)
* ``ops``          — the firefly::capture / firefly::capture_static custom ops
* ``instrument``   — instrument(): torch.fx / named_modules auto-wiring

Status: this is a capture *mechanism*, validated at the unit level and against
a synthetic compiled model. It is **not** a production-ready monitor — overhead
is unbenchmarked, there's no live-traffic integration, and no
``firefly check-shadow`` product loop. It applies to torch models you
instrument directly; it cannot capture from vLLM/SGLang, which own their model
forward.
"""

from __future__ import annotations

from firefly.shadow.aggregate import (
    TapAggregate,
    _iter_jsonl_records,
    aggregate,
    load_tap_index,
)
from firefly.shadow.buffer import DrainThread, _Buffer, _Event
from firefly.shadow.eager import (
    _TLS,
    FullTensorPolicy,
    Tapper,
    _active_tapper,
    _summary_stats,
    _TapperContextBase,
    _TapState,
    tap,
)
from firefly.shadow.instrument import (
    _firefly_eager_tap,
    _firefly_static_tap,
    _instrument_via_fx,
    _instrument_via_named_modules,
    _wrap_forward_eager,
    _wrap_forward_static,
    instrument,
)
from firefly.shadow.ops import (
    _capture_backward,
    _capture_fake,
    _capture_static_fake,
    capture,
    capture_static,
)
from firefly.shadow.sinks import (
    _SHARD_FLUSH_EVERY_N,
    _SHARD_FLUSH_EVERY_S,
    AzureSink,
    GCSSink,
    LocalLogSink,
    S3Sink,
    SinkHandle,
    _CloudSinkBase,
    make_sink,
    start_sink,
    stop_sink,
)
from firefly.shadow.static import (
    StaticFullTensorPolicy,
    StaticTapper,
    _active_static_tapper,
    _StaticDrainThread,
    tap_static,
)
from firefly.shadow.triton_stats import _STATS_KERNEL, _get_stats_kernel, _import_triton

__all__ = [
    "AzureSink",
    "DrainThread",
    "FullTensorPolicy",
    "GCSSink",
    "LocalLogSink",
    "S3Sink",
    "SinkHandle",
    "StaticFullTensorPolicy",
    "StaticTapper",
    "TapAggregate",
    "Tapper",
    "_Buffer",
    "_CloudSinkBase",
    "_Event",
    "_SHARD_FLUSH_EVERY_N",
    "_SHARD_FLUSH_EVERY_S",
    "_STATS_KERNEL",
    "_StaticDrainThread",
    "_TLS",
    "_TapState",
    "_TapperContextBase",
    "_active_static_tapper",
    "_active_tapper",
    "_capture_backward",
    "_capture_fake",
    "_capture_static_fake",
    "_firefly_eager_tap",
    "_firefly_static_tap",
    "_get_stats_kernel",
    "_import_triton",
    "_instrument_via_fx",
    "_instrument_via_named_modules",
    "_iter_jsonl_records",
    "_summary_stats",
    "_wrap_forward_eager",
    "_wrap_forward_static",
    "aggregate",
    "capture",
    "capture_static",
    "instrument",
    "load_tap_index",
    "make_sink",
    "start_sink",
    "stop_sink",
    "tap",
    "tap_static",
]
