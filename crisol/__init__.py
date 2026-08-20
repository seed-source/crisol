"""Crisol — zero-bloat heterogeneous compute dispatcher and benchmark runner.

Crisol is the open-source launch pad for Seedsource Novum: a small, installable
CLI that dispatches registered jobs, probes local nodes, and keeps an immutable
JSONL audit of every run. No Kubernetes, no Slurm, no Ray — for the 2-5 box lab
(and your one box today).

Commands: `crisol doctor`, `crisol status`, `crisol jobs`, `crisol run <job>`.
"""
from __future__ import annotations

__version__ = "0.1.1"

from crisol import core, protocol, receiver  # noqa: E402
from crisol.cli import main  # noqa: E402

__all__ = ["__version__", "main", "core", "protocol", "receiver"]
