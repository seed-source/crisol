"""Crisol command line interface: status, jobs, run, doctor.

The zero-bloat launch platform. Dispatch a registered job, probe the local
node, and keep an immutable JSONL audit of every run.

Commands
--------
  doctor   Validate Python, optional torch/CUDA, and the results ledger.
  status   Probe reachable nodes on this machine.
  jobs     List registered jobs available on this machine.
  run      Dispatch a registered job (writes a JSONL run record).
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crisol", description=__doc__)
    parser.add_argument("--version", action="version",
                        version=f"crisol {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Validate python, torch/CUDA, results ledger.")
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON report.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_status = sub.add_parser("status", help="Probe reachable nodes on this machine.")
    p_status.add_argument("--json", action="store_true", help="Emit JSON.")
    p_status.set_defaults(func=cmd_status)

    p_jobs = sub.add_parser("jobs", help="List registered jobs available here.")
    p_jobs.add_argument("--json", action="store_true", help="Emit JSON.")
    p_jobs.set_defaults(func=cmd_jobs)

    p_run = sub.add_parser("run", help="Dispatch a registered job (JSONL audit).")
    p_run.add_argument("job", help="Registered job name (see `crisol jobs`).")
    p_run.add_argument("--seed", type=int, default=None,
                       help="Override the job's default seed (default 1729 for smoke jobs).")
    p_run.add_argument("--node", default=None,
                       help="Unused in the slim public core (local-only dispatch); kept for CLI parity.")
    p_run.add_argument("--dry-run", action="store_true", help="Resolve without executing.")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
