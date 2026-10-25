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
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    pkg_parent = str(PKG_ROOT.parent)
    child_env["PYTHONPATH"] = (
        pkg_parent + os.pathsep + child_env["PYTHONPATH"]
        if "PYTHONPATH" in child_env
        else pkg_parent
    )
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=job.timeout_s,
            check=False,
            env=child_env,
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
    "mini-lyra-train": Job(
        name="mini-lyra-train",
        description="Mini-Lyra OLM massive training workload with 80/20 Cross-Entropy + River Phase Resonance hybrid loss on GPU.",
        module="crisol.jobs.mini_lyra_train",
        backends=("cuda",),
        default_seed=DEFAULT_SEED,
        timeout_s=3600,
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


# Config directory for named node aliases
CONFIG_DIR = Path(os.environ.get("CRISOL_CONFIG_DIR", str(Path.home() / ".crisol")))
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict[str, Any]:
    """Load ~/.crisol/config.json or return default empty schema."""
    if not CONFIG_FILE.exists():
        return {"nodes": {}}
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"nodes": {}}
            if "nodes" not in data or not isinstance(data["nodes"], dict):
                data["nodes"] = {}
            return data
    except Exception:
        return {"nodes": {}}


def save_config(cfg: dict[str, Any]) -> None:
    """Save config to ~/.crisol/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def resolve_node_target(target: str | None, token: str | None = None) -> tuple[str | None, str | None]:
    """Resolve a node target (URL or named alias) to (url, token)."""
    if not target:
        return None, token

    # Lookup in config aliases first
    cfg = load_config()
    nodes = cfg.get("nodes", {})
    if target in nodes:
        node_info = nodes[target]
        if isinstance(node_info, dict):
            url = node_info.get("url")
            resolved_token = token or node_info.get("token")
            return url, resolved_token
        elif isinstance(node_info, str):
            return node_info, token

    # Fallback to direct URL/host
    return target, token


def probe_remote_node(url: str, token: str | None = None) -> dict[str, Any]:
    """Probe a remote Crisol receiver over HTTP."""
    import urllib.request

    base_url = url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    req = urllib.request.Request(f"{base_url}/health")
    tok = token or os.environ.get("CRISOL_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("X-Crisol-Token", tok)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def dispatch_remote(
    url: str,
    job_name: str,
    seed: int | None = None,
    detach: bool = False,
    token: str | None = None,
    event_callback: Any = None,
) -> dict[str, Any]:
    """Dispatch a job to a remote Crisol receiver and stream events in real time or detach."""
    import urllib.request

    base_url = url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"

    tok = token or os.environ.get("CRISOL_TOKEN")
    payload = json.dumps({"job": job_name, "seed": seed, "detach": detach}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/run", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("X-Crisol-Token", tok)

    if detach:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            data["node_url"] = base_url
            return data

    run_id = str(uuid.uuid4())
    started = time.time()
    status = "completed"
    exit_code = 0
    backend = "remote"
    node_name = base_url
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            stdout_lines.append(line)
            try:
                event = json.loads(line)
            except Exception:
                event = {"type": "stdout", "line": line}

            if isinstance(event, dict):
                t = event.get("type")
                if t == "lifecycle":
                    ev = event.get("event")
                    if ev == "remote_dispatch_started":
                        backend = event.get("backend", backend)
                        node_name = event.get("node", node_name)
                    elif ev == "dispatch_complete":
                        status = event.get("status", status)
                        exit_code = event.get("exit_code", exit_code)
                elif t == "error":
                    stderr_lines.append(event.get("error", ""))
                    status = "failed"
                    exit_code = 1
                if event_callback:
                    event_callback(event)

    elapsed = time.time() - started
    record: dict[str, Any] = {
        "run_id": run_id,
        "job": job_name,
        "node": f"remote@{node_name}",
        "backend": backend,
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 4),
        "seed": seed if seed is not None else DEFAULT_SEED,
        "timed_out": False,
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
        "ts": _now_iso(),
    }
    ledger = append_run(record)
    record["ledger"] = str(ledger)
    return record


def listen_remote(
    url: str,
    run_id: str,
    token: str | None = None,
    event_callback: Any = None,
) -> dict[str, Any]:
    """Connect to a running or completed remote job's event stream."""
    import urllib.request

    base_url = url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"

    tok = token or os.environ.get("CRISOL_TOKEN")
    req = urllib.request.Request(f"{base_url}/runs/{run_id}/listen")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("X-Crisol-Token", tok)

    started = time.time()
    status = "completed"
    exit_code = 0
    backend = "remote"
    node_name = base_url
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            stdout_lines.append(line)
            try:
                event = json.loads(line)
            except Exception:
                event = {"type": "stdout", "line": line}

            if isinstance(event, dict):
                t = event.get("type")
                if t == "lifecycle":
                    ev = event.get("event")
                    if ev == "remote_dispatch_started":
                        backend = event.get("backend", backend)
                        node_name = event.get("node", node_name)
                    elif ev == "dispatch_complete":
                        status = event.get("status", status)
                        exit_code = event.get("exit_code", exit_code)
                elif t == "error":
                    stderr_lines.append(event.get("error", ""))
                    status = "failed"
                    exit_code = 1
                if event_callback:
                    event_callback(event)

    elapsed = time.time() - started
    record: dict[str, Any] = {
        "run_id": run_id,
        "job": "listen",
        "node": f"remote@{node_name}",
        "backend": backend,
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 4),
        "seed": DEFAULT_SEED,
        "timed_out": False,
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
        "ts": _now_iso(),
    }
    ledger = append_run(record)
    record["ledger"] = str(ledger)
    return record


def list_remote_runs(url: str, token: str | None = None) -> list[dict[str, Any]]:
    """List recent runs from a remote Crisol receiver."""
    import urllib.request

    base_url = url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    req = urllib.request.Request(f"{base_url}/runs")
    tok = token or os.environ.get("CRISOL_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("X-Crisol-Token", tok)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("runs", [])

