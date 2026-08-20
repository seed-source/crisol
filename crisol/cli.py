"""Crisol command line interface: status, jobs, run, doctor, receiver.

The zero-bloat launch platform. Dispatch a registered job, probe the local or
remote node, run the receiver daemon, and keep an immutable JSONL audit of every run.

Commands
--------
  doctor   Validate Python, optional torch/CUDA, and the results ledger.
  status   Probe reachable nodes (locally or on a remote receiver).
  jobs     List registered jobs available on this machine.
  run      Dispatch a registered job locally or to a remote receiver.
  receiver Start the Crisol receiver HTTP daemon on this host.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from crisol import __version__
from crisol import core


def cmd_doctor(args: argparse.Namespace) -> int:
    report = core.write_doctor_report()
    report["crisol_version"] = __version__
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        print(f"Crisol {__version__}")
        print(f"  python : {report['python']}")
        print(f"  platform: {report['platform']}")
        if report["torch"]:
            cuda = "yes" if report["cuda"] else "no"
            print(f"  torch  : {report['torch']} (cuda={cuda})")
        else:
            print("  torch  : not installed (cpu-smoke still works)")
        print(f"  results dir : {report['results_dir']}")
        print(f"  results dir writable: {report['results_writable']}")
        for node in report["nodes"]:
            print(f"  node : {node['name']} backend={node['backend']} "
                  f"reachable={node['reachable']} — {node['detail']}")
    # Non-zero only if the ledger is unusable (storage is the P0).
    return 0 if report["results_writable"] is True else 1


def cmd_status(args: argparse.Namespace) -> int:
    target_node = getattr(args, "node", None)
    if target_node:
        url, token = core.resolve_node_target(target_node, getattr(args, "token", None))
        if url:
            try:
                report = core.probe_remote_node(url, token=token)
                if args.json:
                    print(json.dumps(report, sort_keys=True, indent=2))
                else:
                    alias_info = f" (alias: {target_node})" if target_node != url else ""
                    print(f"Remote Node: {url}{alias_info} (Crisol {report.get('crisol_version', '?')})")
                    print(f"  platform: {report.get('platform')}; Python {report.get('python')}")
                    for node in report.get("nodes", []):
                        state = "reachable" if node.get("reachable") else "unreachable"
                        print(f"  {node.get('name')}: backend={node.get('backend')} {state} — {node.get('detail')}")
                    print(f"  available jobs: {', '.join(report.get('jobs', []))}")
                return 0
            except Exception as exc:
                print(f"error probing remote node {target_node} ({url}): {exc}", file=sys.stderr)
                return 3

    nodes = core.probe_nodes()
    for node in nodes:
        state = "reachable" if node.reachable else "unreachable"
        print(f"{node.name}: backend={node.backend} {state} — {node.detail}")
    if args.json:
        print(json.dumps(
            [{"name": n.name, "backend": n.backend, "reachable": n.reachable,
              "detail": n.detail} for n in nodes],
            sort_keys=True, indent=2,
        ))
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    jobs = core.listed_jobs()
    if args.json:
        print(json.dumps(
            [{"name": j.name, "description": j.description, "backends": list(j.backends),
              "default_seed": j.default_seed, "module": j.module} for j in jobs.values()],
            sort_keys=True, indent=2,
        ))
        return 0
    for job in jobs.values():
        print(f"{job.name}: backends={','.join(job.backends)} "
              f"default_seed={job.default_seed}")
        print(f"  {job.description}")
    if "torch-smoke" not in jobs:
        print("\n(torch-smoke not listed: torch not installed on this machine)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # ── Remote Dispatch path ───────────────────────────────────────────────────
    target_node = getattr(args, "node", None)
    if target_node:
        url, token = core.resolve_node_target(target_node, getattr(args, "token", None))
        if not url:
            print(f"could not resolve node target '{target_node}'", file=sys.stderr)
            return 2
        
        if args.dry_run:
            print(f"[dry-run] would dispatch job '{args.job}' to remote node '{target_node}' ({url}) "
                  f"(seed={args.seed}, detach={args.detach})")
            return 0

        if args.detach:
            try:
                res = core.dispatch_remote(
                    url=url,
                    job_name=args.job,
                    seed=args.seed,
                    detach=True,
                    token=token,
                )
                print(f"Job '{args.job}' successfully dispatched in background!")
                print(f"  Run ID:      {res.get('run_id')}")
                print(f"  Status:      {res.get('status')}")
                print(f"  Node:        {res.get('node')} ({res.get('backend')})")
                print(f"  Listen cmd:  crisol listen {res.get('run_id')} --node {target_node}")
                return 0
            except Exception as exc:
                print(f"remote dispatch error: {exc}", file=sys.stderr)
                return 3

        print(f"Dispatching '{args.job}' to remote receiver {target_node} ({url})...")
        
        def _on_event(event: dict) -> None:
            t = event.get("type")
            if t in ("metrics", "result"):
                print(f"  [{t}] {json.dumps({k: v for k, v in event.items() if k not in ('ts',)}, sort_keys=True)}")
            elif t == "progress":
                print(f"  [progress] {event.get('phase')}: {event.get('current')}/{event.get('total')}")
            elif t == "lifecycle":
                print(f"  [lifecycle] {event.get('event')}")
            elif t == "error":
                print(f"  [error] {event.get('error')}", file=sys.stderr)

        try:
            record = core.dispatch_remote(
                url=url,
                job_name=args.job,
                seed=args.seed,
                detach=False,
                token=token,
                event_callback=_on_event,
            )
        except Exception as exc:
            print(f"remote dispatch error: {exc}", file=sys.stderr)
            return 3

        print(f"\n{'=' * 50}")
        print(f"  Job:       {record['job']}")
        print(f"  Node:      {record['node']} ({record['backend']})")
        print(f"  Seed:      {record['seed']}")
        print(f"  Status:    {record['status']} (exit {record['exit_code']})")
        print(f"  Elapsed:   {record['elapsed_s']}s")
        print(f"  Ledger:    {record['ledger']}")
        print(f"{'=' * 50}")
        return 0 if record["status"] in ("completed",) else 1

    # ── Local Dispatch path ────────────────────────────────────────────────────
    try:
        job = core.resolve_job(args.job)
    except KeyError:
        known = ", ".join(sorted(core.JOBS))
        print(f"unknown job '{args.job}'. Known: {known}", file=sys.stderr)
        return 2
    if job.requires_optional and not core._torch_available():
        print(f"job '{job.name}' requires optional deps (torch) not installed here.",
              file=sys.stderr)
        return 2
    try:
        node = core.choose_node(job)
    except RuntimeError as exc:
        print(f"dispatch error: {exc}", file=sys.stderr)
        return 3
    seed = args.seed  # may be None -> job.default_seed applied in dispatch
    if args.dry_run:
        print(f"[dry-run] would run job '{job.name}' on node '{node.name}' "
              f"(backend={node.backend}, seed={seed if seed is not None else job.default_seed})")
        return 0
    record = core.dispatch(job, node, seed=seed)
    print(f"\n{'=' * 50}")
    print(f"  Job:       {job.name}")
    print(f"  Node:      {node.name} ({node.backend})")
    print(f"  Seed:      {record['seed']}")
    print(f"  Status:    {record['status']} (exit {record['exit_code']})")
    print(f"  Elapsed:   {record['elapsed_s']}s")
    print(f"  Ledger:    {record['ledger']}")
    print(f"{'=' * 50}")
    # Surface the workload's own result/metrics lines (compact).
    stdout = record.get("stdout") or ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = payload.get("type")
        if t in ("metrics", "result"):
            print(f"  [{t}] {json.dumps({k: v for k, v in payload.items() if k not in ('ts',)}, sort_keys=True)}")
    return 0 if record["status"] in ("completed",) else 1


def cmd_listen(args: argparse.Namespace) -> int:
    target_node = args.node
    url, token = core.resolve_node_target(target_node, getattr(args, "token", None))
    if not url:
        print(f"could not resolve node target '{target_node}'", file=sys.stderr)
        return 2

    print(f"Listening to run '{args.run_id}' on {target_node} ({url})...")

    def _on_event(event: dict) -> None:
        t = event.get("type")
        if t in ("metrics", "result"):
            print(f"  [{t}] {json.dumps({k: v for k, v in event.items() if k not in ('ts',)}, sort_keys=True)}")
        elif t == "progress":
            print(f"  [progress] {event.get('phase')}: {event.get('current')}/{event.get('total')}")
        elif t == "lifecycle":
            print(f"  [lifecycle] {event.get('event')}")
        elif t == "error":
            print(f"  [error] {event.get('error')}", file=sys.stderr)

    try:
        record = core.listen_remote(
            url=url,
            run_id=args.run_id,
            token=token,
            event_callback=_on_event,
        )
        print(f"\n{'=' * 50}")
        print(f"  Run ID:    {record['run_id']}")
        print(f"  Node:      {record['node']}")
        print(f"  Status:    {record['status']} (exit {record['exit_code']})")
        print(f"  Elapsed:   {record['elapsed_s']}s")
        print(f"  Ledger:    {record['ledger']}")
        print(f"{'=' * 50}")
        return 0 if record["status"] in ("completed",) else 1
    except Exception as exc:
        print(f"listen error: {exc}", file=sys.stderr)
        return 3


def cmd_runs(args: argparse.Namespace) -> int:
    target_node = args.node
    url, token = core.resolve_node_target(target_node, getattr(args, "token", None))
    if not url:
        print(f"could not resolve node target '{target_node}'", file=sys.stderr)
        return 2

    try:
        runs = core.list_remote_runs(url, token=token)
        if args.json:
            print(json.dumps(runs, sort_keys=True, indent=2))
        else:
            print(f"Recent runs on {target_node} ({url}):")
            if not runs:
                print("  (no runs recorded)")
            for r in runs:
                status_str = f"{r.get('status')} (exit {r.get('exit_code')})" if r.get('completed') else "running..."
                print(f"  {r.get('run_id')}: job={r.get('job')} node={r.get('node')} "
                      f"status={status_str} elapsed={r.get('elapsed_s')}s")
        return 0
    except Exception as exc:
        print(f"error listing runs from {target_node}: {exc}", file=sys.stderr)
        return 3


def cmd_config(args: argparse.Namespace) -> int:
    cfg = core.load_config()
    nodes = cfg.setdefault("nodes", {})

    if args.config_action == "set-node":
        nodes[args.name] = {
            "url": args.url,
            "token": args.token,
        }
        core.save_config(cfg)
        print(f"Node '{args.name}' configured -> {args.url} (token: {'yes' if args.token else 'no'})")
        return 0

    if args.config_action == "rm-node":
        if args.name in nodes:
            del nodes[args.name]
            core.save_config(cfg)
            print(f"Node '{args.name}' removed.")
        else:
            print(f"Node '{args.name}' not found in config.", file=sys.stderr)
        return 0

    if args.config_action == "list" or not args.config_action:
        if args.json:
            print(json.dumps(cfg, sort_keys=True, indent=2))
        else:
            print(f"Crisol Config ({core.CONFIG_FILE}):")
            if not nodes:
                print("  (no named nodes configured — use `crisol config set-node <name> <url>`)")
            for name, info in nodes.items():
                if isinstance(info, dict):
                    tok_str = " (token configured)" if info.get("token") else ""
                    print(f"  [{name}] -> {info.get('url')}{tok_str}")
                else:
                    print(f"  [{name}] -> {info}")
        return 0

    return 0


def cmd_receiver(args: argparse.Namespace) -> int:
    from crisol import receiver

    return receiver.run_receiver(
        host=args.host,
        port=args.port,
        token=args.token,
    )


MAIN_EPILOG = """
Quick Link & Remote Dispatch Workflow
-------------------------------------
1. Start Receiver on the remote compute host:
     crisol receiver --port 8430 --token my-secret

2. Configure named node alias on your local client:
     crisol config set-node m2 http://192.168.1.50:8430 --token my-secret

3. Check remote node health:
     crisol status --node m2

4. Run interactive streaming workload:
     crisol run cpu-smoke --node m2
     crisol run torch-smoke --node m2

5. Run detached in background & reconnect:
     crisol run torch-smoke --node m2 --detach
     crisol runs --node m2
     crisol listen <run_id> --node m2
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crisol",
        description=__doc__,
        epilog=MAIN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"crisol {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Validate python, torch/CUDA, results ledger.")
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON report.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_status = sub.add_parser("status", help="Probe reachable nodes (locally or remote).")
    p_status.add_argument("--node", default=None, help="Remote receiver URL or node alias (e.g. m2, http://host:8430).")
    p_status.add_argument("--token", default=None, help="Auth token for remote receiver.")
    p_status.add_argument("--json", action="store_true", help="Emit JSON.")
    p_status.set_defaults(func=cmd_status)

    p_jobs = sub.add_parser("jobs", help="List registered jobs available here.")
    p_jobs.add_argument("--json", action="store_true", help="Emit JSON.")
    p_jobs.set_defaults(func=cmd_jobs)

    p_run = sub.add_parser(
        "run",
        help="Dispatch a registered job (JSONL audit).",
        description="Dispatch a job locally or to a remote receiver over HTTP.",
        epilog="Examples:\n  crisol run cpu-smoke\n  crisol run torch-smoke --node m2\n  crisol run torch-smoke --node m2 --detach\n  crisol run cpu-smoke --node http://34.56.65.241:8430 --token secret",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_run.add_argument("job", help="Registered job name (see `crisol jobs`).")
    p_run.add_argument("--seed", type=int, default=None,
                       help="Override the job's default seed (default 1729 for smoke jobs).")
    p_run.add_argument("--node", default=None,
                       help="Dispatch to a remote receiver URL or node alias (e.g. m2, http://host:8430).")
    p_run.add_argument("--token", default=None, help="Auth token for remote receiver.")
    p_run.add_argument("--detach", action="store_true", help="Dispatch in background and return run ID immediately.")
    p_run.add_argument("--dry-run", action="store_true", help="Resolve without executing.")
    p_run.set_defaults(func=cmd_run)

    p_listen = sub.add_parser(
        "listen",
        help="Stream events from a running or completed remote job.",
        description="Connect to an active or past background job's event stream on a remote receiver.",
        epilog="Example:\n  crisol listen run_a9e9f510 --node m2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_listen.add_argument("run_id", help="The run ID to listen to (e.g. run_9f2a81bc).")
    p_listen.add_argument("--node", required=True, help="Remote receiver URL or node alias.")
    p_listen.add_argument("--token", default=None, help="Auth token for remote receiver.")
    p_listen.set_defaults(func=cmd_listen)

    p_runs = sub.add_parser(
        "runs",
        help="List recent runs from a remote receiver.",
        description="Query recent background runs and their current status from a remote receiver.",
        epilog="Example:\n  crisol runs --node m2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_runs.add_argument("--node", required=True, help="Remote receiver URL or node alias.")
    p_runs.add_argument("--token", default=None, help="Auth token for remote receiver.")
    p_runs.add_argument("--json", action="store_true", help="Emit JSON.")
    p_runs.set_defaults(func=cmd_runs)

    p_cfg = sub.add_parser(
        "config",
        help="Manage named node aliases and client settings.",
        description="Configure friendly node aliases stored in ~/.crisol/config.json.",
        epilog="Examples:\n  crisol config set-node m2 http://192.168.1.160:8430 --token secret\n  crisol config list\n  crisol config rm-node m2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cfg_sub = p_cfg.add_subparsers(dest="config_action")
    p_cfg_set = p_cfg_sub.add_parser("set-node", help="Set a named node alias.")
    p_cfg_set.add_argument("name", help="Alias name (e.g. m2, m5, gcloud).")
    p_cfg_set.add_argument("url", help="Receiver URL (e.g. http://192.168.100.160:8430).")
    p_cfg_set.add_argument("--token", default=None, help="Optional secret token.")

    p_cfg_rm = p_cfg_sub.add_parser("rm-node", help="Remove a named node alias.")
    p_cfg_rm.add_argument("name", help="Alias name to remove.")

    p_cfg_list = p_cfg_sub.add_parser("list", help="List configured node aliases.")
    p_cfg_list.add_argument("--json", action="store_true", help="Emit JSON.")
    p_cfg.set_defaults(func=cmd_config)

    p_rec = sub.add_parser(
        "receiver",
        help="Start the Crisol receiver HTTP daemon.",
        description="Start a lightweight, zero-dependency HTTP server to accept remote job dispatches and stream stdout NDJSON.",
        epilog="Examples:\n  crisol receiver --port 8430\n  crisol receiver --host 0.0.0.0 --port 8430 --token my-secret",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_rec.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0).")
    p_rec.add_argument("--port", type=int, default=8430, help="Port to listen on (default: 8430).")
    p_rec.add_argument("--token", default=None, help="Optional secret token required for requests.")
    p_rec.set_defaults(func=cmd_receiver)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
