"""Crisol Receiver Daemon (Zero-bloat stdlib HTTP server for remote dispatch).

Provides a clean HTTP/NDJSON streaming interface so remote clients can probe,
discover jobs, dispatch workloads interactively or detached, and reconnect to
event streams without manual SSH, buffer stalls, or complex credentials.

Endpoints
---------
  GET  /health              Return node info, platform, and available jobs.
  GET  /jobs                Return detailed registered jobs list.
  POST /run                 Execute job (streaming or detached).
  GET  /runs                List background/recent runs.
  GET  /runs/<id>           Get status of a specific run.
  GET  /runs/<id>/listen    Stream events for a specific run (replay + live).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any

from crisol import __version__
from crisol import core


@dataclass
class ManagedRun:
    run_id: str
    job_name: str
    node_name: str
    backend: str
    seed: int | None
    started_at: float
    status: str = "running"
    exit_code: int | None = None
    elapsed_s: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    event_cond: threading.Condition = field(default_factory=threading.Condition)
    proc: subprocess.Popen | None = None
    completed: bool = False


# In-memory registry of recent runs on this receiver
RECENT_RUNS: dict[str, ManagedRun] = {}
REGISTRY_LOCK = threading.Lock()


def _register_run(run: ManagedRun) -> None:
    with REGISTRY_LOCK:
        if len(RECENT_RUNS) > 100:
            oldest = next(iter(RECENT_RUNS))
            del RECENT_RUNS[oldest]
        RECENT_RUNS[run.run_id] = run


def _get_run(run_id: str) -> ManagedRun | None:
    with REGISTRY_LOCK:
        return RECENT_RUNS.get(run_id)


def _execute_run_worker(run: ManagedRun, job: core.Job, timeout_s: int) -> None:
    argv = [sys.executable, "-u", "-m", job.module]
    if run.seed is not None:
        argv += ["--seed", str(run.seed)]

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    pkg_parent = str(core.PKG_ROOT.parent)
    child_env["PYTHONPATH"] = (
        pkg_parent + os.pathsep + child_env["PYTHONPATH"]
        if "PYTHONPATH" in child_env
        else pkg_parent
    )

    def _append_event(ev: dict[str, Any]) -> None:
        with run.event_cond:
            run.events.append(ev)
            run.event_cond.notify_all()

    _append_event({
        "type": "lifecycle",
        "event": "remote_dispatch_started",
        "run_id": run.run_id,
        "job": run.job_name,
        "node": run.node_name,
        "backend": run.backend,
        "seed": run.seed,
        "ts": core._now_iso(),
    })

    stdout_captured: list[str] = []
    stderr_captured: list[str] = []
    status = "completed"
    exit_code = 0
    timed_out = False

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
            bufsize=1,
        )
        run.proc = proc

        if proc.stdout:
            for line in proc.stdout:
                stdout_captured.append(line)
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("{"):
                    try:
                        parsed_event = json.loads(stripped)
                        _append_event(parsed_event)
                        continue
                    except json.JSONDecodeError:
                        pass
                _append_event({
                    "type": "stdout",
                    "line": stripped,
                    "ts": core._now_iso(),
                })

        proc.wait(timeout=timeout_s)
        exit_code = proc.returncode
        if proc.stderr:
            err_text = proc.stderr.read()
            if err_text:
                stderr_captured.append(err_text)
        if exit_code != 0:
            status = "failed"
            _append_event({
                "type": "error",
                "error": f"Job exited with code {exit_code}: {''.join(stderr_captured).strip()}",
                "ts": core._now_iso(),
            })

    except subprocess.TimeoutExpired:
        timed_out = True
        status = "timed_out"
        exit_code = -1
        if run.proc:
            run.proc.kill()
            if run.proc.stderr:
                stderr_captured.append(run.proc.stderr.read())
        _append_event({
            "type": "error",
            "error": f"Job timed out after {timeout_s}s",
            "ts": core._now_iso(),
        })
    except Exception as exc:
        status = "failed"
        exit_code = 1
        _append_event({
            "type": "error",
            "error": f"Subprocess dispatch error: {exc}",
            "ts": core._now_iso(),
        })

    elapsed = time.time() - run.started_at
    stdout_full = "".join(stdout_captured)
    stderr_full = "".join(stderr_captured)

    run_record: dict[str, Any] = {
        "run_id": run.run_id,
        "job": run.job_name,
        "node": run.node_name,
        "backend": run.backend,
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 4),
        "seed": run.seed,
        "timed_out": timed_out,
        "stdout": stdout_full,
        "stderr": stderr_full,
        "ts": core._now_iso(),
    }
    ledger_path = core.append_run(run_record)

    with run.event_cond:
        run.status = status
        run.exit_code = exit_code
        run.elapsed_s = round(elapsed, 4)
        run.completed = True
        run.events.append({
            "type": "lifecycle",
            "event": "dispatch_complete",
            "status": status,
            "exit_code": exit_code,
            "elapsed_s": round(elapsed, 4),
            "host_ledger": str(ledger_path),
            "ts": core._now_iso(),
        })
        run.event_cond.notify_all()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class CrisolReceiverHandler(BaseHTTPRequestHandler):
    server_version = f"CrisolReceiver/{__version__}"

    def address_string(self) -> str:
        # Avoid blocking reverse DNS lookups on local networks
        return self.client_address[0]

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[{core._now_iso()}] [receiver] {format % args}\n")
        sys.stderr.flush()

    def _check_auth(self) -> bool:
        token = getattr(self.server, "auth_token", None)
        if not token:
            return True
        auth_hdr = self.headers.get("Authorization", "")
        crisol_hdr = self.headers.get("X-Crisol-Token", "")
        if auth_hdr.startswith("Bearer "):
            bearer = auth_hdr[7:].strip()
            if bearer == token:
                return True
        if crisol_hdr.strip() == token:
            return True
        return False

    def _send_json(self, status_code: int, data: Any) -> None:
        payload = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Crisol-Token")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._check_auth():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized", "detail": "Valid token required"})
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("", "/health", "/status"):
            nodes = core.probe_nodes()
            jobs = core.listed_jobs()
            report = {
                "status": "ok",
                "crisol_version": __version__,
                "platform": sys.platform,
                "python": platform.python_version(),
                "torch_available": core._torch_available(),
                "nodes": [
                    {
                        "name": n.name,
                        "backend": n.backend,
                        "reachable": n.reachable,
                        "detail": n.detail,
                    }
                    for n in nodes
                ],
                "jobs": list(jobs.keys()),
            }
            self._send_json(HTTPStatus.OK, report)
            return

        if path == "/jobs":
            jobs = core.listed_jobs()
            out = {
                name: {
                    "name": j.name,
                    "description": j.description,
                    "backends": list(j.backends),
                    "default_seed": j.default_seed,
                    "module": j.module,
                }
                for name, j in jobs.items()
            }
            self._send_json(HTTPStatus.OK, out)
            return

        if path == "/runs":
            with REGISTRY_LOCK:
                summary = [
                    {
                        "run_id": r.run_id,
                        "job": r.job_name,
                        "node": r.node_name,
                        "backend": r.backend,
                        "status": r.status,
                        "exit_code": r.exit_code,
                        "elapsed_s": r.elapsed_s,
                        "completed": r.completed,
                        "event_count": len(r.events),
                    }
                    for r in RECENT_RUNS.values()
                ]
            self._send_json(HTTPStatus.OK, {"runs": summary})
            return

        if path.startswith("/runs/"):
            parts = path.split("/")[2:]
            run_id = parts[0]
            run = _get_run(run_id)
            if not run:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "run_not_found", "run_id": run_id})
                return

            if len(parts) == 1:
                self._send_json(HTTPStatus.OK, {
                    "run_id": run.run_id,
                    "job": run.job_name,
                    "node": run.node_name,
                    "backend": run.backend,
                    "status": run.status,
                    "exit_code": run.exit_code,
                    "elapsed_s": run.elapsed_s,
                    "completed": run.completed,
                    "event_count": len(run.events),
                })
                return

            if len(parts) == 2 and parts[1] == "listen":
                self._stream_run_events(run)
                return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": self.path})

    def _stream_run_events(self, run: ManagedRun) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def _send_chunk(rec: dict[str, Any]) -> bool:
            line = (json.dumps(rec, default=str) + "\n").encode("utf-8")
            chunk = f"{len(line):X}\r\n".encode("ascii") + line + b"\r\n"
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        idx = 0
        while True:
            with run.event_cond:
                while idx >= len(run.events) and not run.completed:
                    run.event_cond.wait(timeout=1.0)
                
                batch = run.events[idx:]
                idx = len(run.events)
                is_done = run.completed and idx >= len(run.events)

            for ev in batch:
                if not _send_chunk(ev):
                    return

            if is_done:
                break

        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        if not self._check_auth():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized", "detail": "Valid token required"})
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path != "/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": self.path})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            req = json.loads(body)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json", "detail": str(exc)})
            return

        job_name = req.get("job")
        if not job_name:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing_job", "detail": "Field 'job' is required"})
            return

        try:
            job = core.resolve_job(job_name)
        except KeyError:
            self._send_json(HTTPStatus.NOT_FOUND, {
                "error": "unknown_job",
                "detail": f"Job '{job_name}' is not registered on this node",
                "available": list(core.listed_jobs().keys()),
            })
            return

        if job.requires_optional and not core._torch_available():
            self._send_json(HTTPStatus.BAD_REQUEST, {
                "error": "missing_dependencies",
                "detail": f"Job '{job.name}' requires torch, which is not available on this node",
            })
            return

        seed = req.get("seed")
        effective_seed = seed if seed is not None else job.default_seed
        detach = bool(req.get("detach", False))

        try:
            node = core.choose_node(job)
        except RuntimeError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "node_selection_failed", "detail": str(exc)})
            return

        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run = ManagedRun(
            run_id=run_id,
            job_name=job.name,
            node_name=node.name,
            backend=node.backend,
            seed=effective_seed,
            started_at=time.time(),
        )
        _register_run(run)

        worker = threading.Thread(
            target=_execute_run_worker,
            args=(run, job, job.timeout_s),
            daemon=True,
        )
        worker.start()

        if detach:
            self._send_json(HTTPStatus.ACCEPTED, {
                "status": "dispatched",
                "run_id": run.run_id,
                "job": job.name,
                "node": node.name,
                "backend": node.backend,
                "seed": effective_seed,
                "listen_path": f"/runs/{run.run_id}/listen",
            })
            return

        self._stream_run_events(run)


def run_receiver(
    host: str = "0.0.0.0",
    port: int = 8430,
    token: str | None = None,
) -> int:
    """Start the Crisol receiver HTTP server on (host, port)."""
    server = ThreadedHTTPServer((host, port), CrisolReceiverHandler)
    server.auth_token = token  # type: ignore[attr-defined]

    print(f"Crisol Receiver {__version__} listening on http://{host}:{port}")
    print(f"  Auth: {'token-protected' if token else 'open (no token)'}")
    print(f"  Available jobs: {list(core.listed_jobs().keys())}")
    print(f"  Endpoints: GET /health, GET /jobs, POST /run, GET /runs, GET /runs/<id>/listen")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping receiver...")
    finally:
        server.server_close()
    return 0
