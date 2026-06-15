"""Collapse shadow logs into per-tap distributions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TapAggregate:
    """Aggregated statistics for one tap name across all captures."""

    tap_name: str
    n_events: int
    abs_mean_p50: float
    abs_mean_p95: float
    abs_mean_max: float
    abs_max_p50: float
    abs_max_p95: float
    abs_max_max: float
    full_tensor_blobs: list[str] = field(default_factory=list)

def aggregate(shadow_log_dir: str | Path) -> dict[str, TapAggregate]:
    """Read a shadow log directory, group by tap, compute per-tap distributions.

    Handles two on-disk layouts:

    * Single-file (eager-mode / local sink default): ``stats.jsonl``
    * Sharded (cloud-streaming sink default): ``stats-00000.jsonl``,
      ``stats-00001.jsonl``, ...

    Shards are read in lexical order so events stay in capture order
    within each shard; cross-shard ordering preserves the wall-clock
    order the drain wrote them in.

    For each tap name observed, returns a :class:`TapAggregate` with
    p50 / p95 / max of the abs-mean and abs-max series, plus the list
    of full-tensor blobs available for that tap. Downstream code can
    compare these distributions against a reference's deterministic
    per-tap values, with tolerance derived from the spread.

    This is the thin-end of the shadow-vs-reference comparison flow.
    A future ``firefly check-shadow`` would consume this output, load
    the reference's per-tap means, and report taps whose live p95 has
    drifted past tolerance.
    """
    path = Path(shadow_log_dir)
    by_tap: dict[str, dict[str, list]] = {}
    for rec in _iter_jsonl_records(path):
        t = rec["tap_name"]
        slot = by_tap.setdefault(t, {"abs_mean": [], "abs_max": [], "blobs": []})
        slot["abs_mean"].append(rec["stats"]["abs_mean"])
        slot["abs_max"].append(rec["stats"]["abs_max"])
        if "blob_path" in rec:
            slot["blobs"].append(rec["blob_path"])

    out: dict[str, TapAggregate] = {}
    for tap_name, slot in by_tap.items():
        abs_means = sorted(slot["abs_mean"])
        abs_maxes = sorted(slot["abs_max"])
        n = len(abs_means)
        if n == 0:
            continue
        out[tap_name] = TapAggregate(
            tap_name=tap_name,
            n_events=n,
            abs_mean_p50=abs_means[n // 2],
            abs_mean_p95=abs_means[min(n - 1, int(n * 0.95))],
            abs_mean_max=abs_means[-1],
            abs_max_p50=abs_maxes[n // 2],
            abs_max_p95=abs_maxes[min(n - 1, int(n * 0.95))],
            abs_max_max=abs_maxes[-1],
            full_tensor_blobs=sorted(slot["blobs"]),
        )
    return out

def _iter_jsonl_records(path: Path):
    """Yield JSONL records from a shadow-log directory.

    Layout dispatch:
    * ``stats.jsonl`` present → read it as a single stream
    * ``stats-NNNNN.jsonl`` shards → read them in lexical order

    If both are present (e.g. legacy + new-style cohabiting), the
    sharded stream is preferred and the singleton is logged as
    skipped.
    """
    shards = sorted(path.glob("stats-*.jsonl"))
    if shards:
        if (path / "stats.jsonl").exists():
            import sys
            print(
                f"[firefly] aggregate: both stats.jsonl and shards present in "
                f"{path}; reading shards, ignoring singleton.",
                file=sys.stderr,
            )
        for shard in shards:
            with shard.open() as f:
                for line in f:
                    yield json.loads(line)
        return
    # No shards. Fall back to single file.
    stats_path = path / "stats.jsonl"
    if stats_path.exists():
        with stats_path.open() as f:
            for line in f:
                yield json.loads(line)

def load_tap_index(shadow_log_dir: str | Path) -> dict[int, str]:
    """Read the ``tap_index.json`` sidecar a :class:`StaticTapper` writes.

    Returns an empty dict if no sidecar is present (eager-mode logs
    don't write one — the tap names live directly in each event).
    Useful for downstream tools that want to enumerate all known taps
    without scanning the whole stats stream.
    """
    path = Path(shadow_log_dir) / "tap_index.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    # JSON keys are strings; restore to int.
    return {int(k): v for k, v in raw.items()}
