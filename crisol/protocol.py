"""Shared JSON-lines protocol helpers for Crisol workload scripts.

This module is the SINGLE canonical source for _emit and all typed event
helpers. Every workload script (matmul, attention, byte_ssm_setup, …) SHOULD
import from here instead of redefining its own copies. The old file-local
definitions are kept for backward compatibility but deprecated.

Supported event types (schema v2, backward-compatible addition):

  lifecycle  — start/end/benchmark_start/complete/alive events
  progress   — {type, phase, current, total}
  metrics    — {type, phase, rate, memory_gb, elapsed_s, …}
  histogram  — {type, name, values, bins, range} weight/grad distribution
  sparkline  — {type, name, values, window} loss/acc streaming sequence
  gpu_stats  — {type, vram_used_mb, vram_total_mb, temp_c, util_pct, …}
  result     — {type, status, …}
  error      — {type, error, …}
  artifact   — {type, descriptor, …}

All helpers are pure functions with zero dependencies beyond stdlib. They
can be imported and called from any workload script, unit test, or debugger
session without torch, numpy, or CUDA installed.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any


def _now() -> str:
    """Return an ISO-8601 timestamp with milliseconds, UTC, e.g.

    ``2026-07-21T14:30:00.123Z``.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"


def _emit(rec: dict) -> None:
    """Serialize ``rec`` as a single JSON line to stdout and flush.

    This is the ONE sink for the protocol. Every typed helper below calls
    this. Override with ``_emit_override`` for testing or alternative sinks.
    """
    sys.stdout.write(json.dumps(rec, default=str) + "\n")
    sys.stdout.flush()


# ── Lifecycle ────────────────────────────────────────────────────────────────


def _lifecycle(event: str, **kw) -> None:
    """Emit a lifecycle event (start, complete, alive, benchmark_start, …)."""
    _emit({"type": "lifecycle", "event": event, "ts": _now(), **kw})


# ── Progress ──────────────────────────────────────────────────────────────────


def _progress(phase: str, current: int | float, total: int | float, **kw) -> None:
    """Emit a progress update. ``current``/``total`` may be step counts or
    byte-based for streaming progress."""
    _emit({"type": "progress", "phase": phase, "current": current, "total": total, "ts": _now(), **kw})


# ── Metrics ───────────────────────────────────────────────────────────────────


def _metrics(phase: str, rate: float, memory_gb: float, elapsed_s: float, **kw) -> None:
    """Emit a metric measurement (TFLOPS, tok/s, loss, etc.)."""
    _emit({
        "type": "metrics", "phase": phase,
        "rate": rate,
        "memory_gb": round(memory_gb, 4),
        "elapsed_s": round(elapsed_s, 4),
        "ts": _now(), **kw,
    })


# ── Histogram (weight/gradient distribution) ──────────────────────────────────


def _histogram(
    name: str,
    values: list[float],
    *,
    bins: int = 64,
    range_min: float | None = None,
    range_max: float | None = None,
    **kw,
) -> None:
    """Emit a histogram distribution for weights, gradients, or activations.

    ``values`` is a downsampled / binned representation of the full tensor —
    never the raw tensor itself (which can be millions of elements). The
    receiver should plot these as a bar/step chart.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. ``"layer_0.weight"``.
    values : list[float]
        Binned frequency counts or percentile edges. Prefer ≤256 values.
    bins : int
        Number of bins used to produce ``values`` (for display purposes).
    range_min, range_max : float or None
        Observed min/max of the original data. If None, receivers should
        infer from values length and bins.
    """
    payload: dict[str, Any] = {
        "type": "histogram",
        "name": name,
        "values": values,
        "bins": bins,
        "ts": _now(),
    }
    if range_min is not None:
        payload["range_min"] = range_min
    if range_max is not None:
        payload["range_max"] = range_max
    payload.update(kw)
    _emit(payload)


# ── Sparkline (loss/accuracy streaming sequence) ──────────────────────────────


def _sparkline(
    name: str,
    values: list[float],
    *,
    window: int | None = None,
    **kw,
) -> None:
    """Emit a streaming time-series sequence for loss, accuracy, or any
    scalar metric that a UI should render as a sparkline / mini-chart.

    ``values`` is a bounded recent window (e.g. last 100 steps), NOT the
    full training history. The receiver appends/replaces its display window.

    Parameters
    ----------
    name : str
        Label, e.g. ``"train/loss"``, ``"val/acc"``.
    values : list[float]
        Ordered sequence of scalar values (most recent last).
    window : int or None
        Maximum window size. If None receivers should use values length.
    """
    payload: dict[str, Any] = {
        "type": "sparkline",
        "name": name,
        "values": values,
        "ts": _now(),
    }
    if window is not None:
        payload["window"] = window
    payload.update(kw)
    _emit(payload)


# ── GPU stats (VRAM, temperature, utilization) ────────────────────────────────


def _gpu_stats(
    vram_used_mb: float | None = None,
    vram_total_mb: float | None = None,
    temp_c: float | None = None,
    util_pct: float | None = None,
    **kw,
) -> None:
    """Emit a GPU hardware statistics snapshot.

    All fields are optional — emit only what is observable. Values are
    instantaneous snapshots, not running averages.

    Parameters
    ----------
    vram_used_mb : float or None
        GPU memory currently allocated/used, in MiB.
    vram_total_mb : float or None
        Total GPU memory available, in MiB.
    temp_c : float or None
        GPU temperature in Celsius.
    util_pct : float or None
        GPU compute utilization percent (0-100).
    """
    payload: dict[str, Any] = {"type": "gpu_stats", "ts": _now()}
    if vram_used_mb is not None:
        payload["vram_used_mb"] = vram_used_mb
    if vram_total_mb is not None:
        payload["vram_total_mb"] = vram_total_mb
    if temp_c is not None:
        payload["temp_c"] = temp_c
    if util_pct is not None:
        payload["util_pct"] = util_pct
    payload.update(kw)
    _emit(payload)


# ── Result / Error / Artifact ─────────────────────────────────────────────────


def _result(**kw) -> None:
    """Emit a terminal result record."""
    _emit({"type": "result", "ts": _now(), **kw})


def _error(message: str, **kw) -> None:
    _emit({"type": "error", "error": message, "ts": _now(), **kw})


def _artifact(descriptor: dict, **kw) -> None:
    """Emit an artifact descriptor (e.g. a compiled wheel ready on disk).

    The descriptor MUST contain ``filename``, ``bytes``, ``sha256``, and
    ``kind`` at minimum. See ``byte_ssm_setup._emit_wheel_ready`` for the
    canonical wheel artifact shape.
    """
    _emit({"type": "artifact", "descriptor": descriptor, "ts": _now(), **kw})
