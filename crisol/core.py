"""Slim Crisol core: Job registry, local node probe, JSONL run ledger, dispatch.

This module is the public launch-platform core. It deliberately does NOT
contain RunPod MCP/GraphQL, wheel persistence, M5/M2 SSH mirrors, or GCS push.
It dispatches jobs locally as subprocesses and records every run to an
immutable JSONL ledger (one line per run, keyed by date).

Public API
----------
- Job, NodeStatus              — core data types
- JOBS                         — registry (cpu-smoke always; torch-smoke if torch installed)
- probe_nodes()                — local node discovery (this machine only)
- append_run()                 — write a JSONL run record to results/
- dispatch()                   — run a job synchronously and record the run
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The canonical deterministic seed used by Crisol contracts.
DEFAULT_SEED = 1729

# Package root: crisol/core.py -> crisol/
PKG_ROOT = Path(__file__).resolve().parent

# Results ledger. Override with CRISOL_RESULTS_DIR for tests/platforms.
RESULTS_DIR = Path(
    os.environ.get("CRISOL_RESULTS_DIR", str(PKG_ROOT.parent / "results"))
)


@dataclass(frozen=True)
class Job:
    """A registered benchmark/run contract.

    ``module`` is a python module run via ``python -m <module>``. The module's
    workload script owns emitting protocol JSONL on stdout (lifecycle, metrics,
    result). ``default_seed`` is passed as ``--seed`` unless the operator
    overrides it on the CLI; set to None for jobs that take no seed.
    """

    name: str
    description: str
    module: str
    backends: tuple[str, ...] = ("cpu",)
    default_seed: int | None = DEFAULT_SEED
    min_memory_gb: float = 0.0
    timeout_s: int = 300
    requires_optional: bool = False
    """True if the job needs an optional dependency (e.g. torch). Such jobs are
    only listed/run when the dependency is importable on the probe node."""


@dataclass
class NodeStatus:
    name: str
    backend: str
    reachable: bool
    detail: str
    free_memory_gb: float | None = None


def _torch_available() -> bool:
    """Return True only if torch imports cleanly (no error, no warning)."""
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _has_optional(job: Job) -> bool:
    """A job only exists on a node when its optional deps are present."""
    if not job.requires_optional:
        return True
    return _torch_available()


def probe_nodes() -> list[NodeStatus]:
    """Return reachable nodes on THIS machine (local-only probe).

    - ``local-cpu`` — always present (stdlib execution surface).
    - ``local-torch`` — present only when torch is importable; reports CUDA
      when available. This is the optional-accelerated surface.
    """
    nodes: list[NodeStatus] = [
        NodeStatus(
            "local-cpu",
            "cpu",
            True,
            f"native {sys.platform}; Python {platform.python_version()}",
        )
    ]
    if _torch_available():
        import torch

        cuda = torch.cuda.is_available()
        backend = "cuda" if cuda else "cpu"
        detail = f"torch {torch.__version__}"
        if cuda:
            try:
                detail += f"; CUDA {torch.version.cuda}; {torch.cuda.get_device_name(0)}"
            except Exception:  # noqa: BLE001 — device name is best-effort
                detail += "; CUDA present"
        nodes.append(NodeStatus("local-torch", backend, True, detail))
    return nodes


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with milliseconds, matching protocol.ts format."""
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        + f".{int(time.time() * 1000) % 1000:03d}Z"
    )


def append_run(record: dict[str, Any]) -> Path:
    """Append a single JSON run record to today's ledger and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl"
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def dispatch(
    job: Job, node: NodeStatus, seed: int | None = None
) -> dict[str, Any]:
    """Run a job synchronously and record an immutable JSONL run record.

    Builds ``[sys.executable, -m, job.module, --seed, ...]``, captures the
    workload's protocol JSONL from stdout, and writes a summarized record to
    the ledger. ``seed`` defaults to the job's ``default_seed`` when the job
    accepts one.
    """
    effective_seed = seed if seed is not None else job.default_seed
    run_id = str(uuid.uuid4())
    argv = [sys.executable, "-m", job.module]
    if job.default_seed is not None:
        argv += ["--seed", str(effective_seed)]

    started = time.time()
    status = "completed"
    exit_code = 0
    stdout_text = ""
    stderr_text = ""
    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=job.timeout_s,
            check=False,
        )
        exit_code = proc.returncode
        stdout_text = proc.stdout
        stderr_text = proc.stderr
        if exit_code != 0:
            status = "failed"
    except subprocess.TimeoutExpired as error:
        timed_out = True
        status = "timed_out"
        exit_code = -1
        stdout_text = error.stdout or ""
        stderr_text = (error.stderr or "") + f"\n[timed out after {job.timeout_s}s]"
    elapsed = time.time() - started

    record: dict[str, Any] = {
        "run_id": run_id,
        "job": job.name,
        "node": node.name,
        "backend": node.backend,
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 4),
        "seed": effective_seed,
        "timed_out": timed_out,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "ts": _now_iso(),
    }
    ledger = append_run(record)
    record["ledger"] = str(ledger)
    return record


# ── Job registry ────────────────────────────────────────────────────────────────

JOBS: dict[str, Job] = {
    "cpu-smoke": Job(
        name="cpu-smoke",
        description="Stdlib-only CPU matmul smoke benchmark with honest metrics (no torch required).",
        module="crisol.jobs.cpu_smoke",
        backends=("cpu",),
        default_seed=DEFAULT_SEED,
        timeout_s=60,
    ),
    "torch-smoke": Job(
        name="torch-smoke",
        description="Optional torch micro-benchmark (CUDA matmul when present, else CPU). Listed only when torch is installed.",
        module="crisol.jobs.torch_smoke",
        backends=("cpu", "cuda"),
        default_seed=DEFAULT_SEED,
        timeout_s=120,
        requires_optional=True,
    ),
}


def listed_jobs() -> dict[str, Job]:
    """Return the jobs available on this machine (optional-dep jobs filtered)."""
    return {name: job for name, job in JOBS.items() if _has_optional(job)}


def resolve_job(name: str) -> Job:
    """Return a registered Job by name, or raise KeyError."""
    return JOBS[name]


def eligible_nodes(job: Job) -> list[NodeStatus]:
    """Return reachable, unlocked nodes whose backend the job supports."""
    eligible: list[NodeStatus] = []
    for node in probe_nodes():
        if node.reachable and node.backend in job.backends:
            eligible.append(node)
    return eligible


def choose_node(job: Job) -> NodeStatus:
    """Pick the first eligible node, preferring GPU-backed accelerators."""
    nodes = eligible_nodes(job)
    if not nodes:
        raise RuntimeError(
            f"no eligible node for job '{job.name}' (backends={job.backends}) — "
            f"probe: {[n.name for n in probe_nodes()]}"
        )
    # Prefer CUDA > cpu > other, preserving probe order as tiebreak.
    def rank(node: NodeStatus) -> int:
        return 0 if node.backend == "cuda" else 1

    return sorted(nodes, key=rank)[0]


def write_doctor_report() -> dict[str, Any]:
    """Return a doctor report: python, torch, results dir, node probes."""
    import shutil

    report: dict[str, Any] = {
        "python": f"{platform.python_version()} ({sys.executable})",
        "platform": sys.platform,
        "torch": None,
        "cuda": False,
        "results_dir": str(RESULTS_DIR),
        "results_writable": None,
        "nodes": [],
        "crisol_version": None,
    }
    if _torch_available():
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        probe = RESULTS_DIR / ".doctor-probe"
        shutil.rmtree(probe, ignore_errors=True)
        probe.mkdir()
        report["results_writable"] = True
        probe.rmdir()
    except OSError as exc:
        report["results_writable"] = f"WRITE FAILED: {exc}"
    for node in probe_nodes():
        report["nodes"].append(
            {
                "name": node.name,
                "backend": node.backend,
                "reachable": node.reachable,
                "detail": node.detail,
            }
        )
    return report
