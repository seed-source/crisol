# Crisol

**Zero-bloat heterogeneous compute dispatcher and benchmark runner.**

Kubernetes, Slurm, and Ray exist to herd a datacenter. Crisol exists for the
lab that actually exists: a laptop, a tower with an RTX, a spare headless box
— **2 to 5 machines**, not 500. One command, one seed, one honest JSONL
ledger per run.

```
$ crisol status
local-cpu: backend=cpu reachable — native win32; Python 3.12.8

$ crisol jobs
cpu-smoke: backends=cpu default_seed=1729
  Stdlib-only CPU matmul smoke benchmark with honest metrics (no torch required).

$ crisol run cpu-smoke
==================================================
  Job:       cpu-smoke
  Node:      local-cpu (cpu)
  Seed:      1729
  Status:    completed (exit 0)
  Elapsed:   0.8s
  Ledger:    ...\results\2026-08-18.jsonl
==================================================
```

## Why Crisol

- **The pain.** Setting up Ray/Slurm/K8s for a handful of local boxes is a
  nightmare of daemons, config, and wasted RAM. Hand-rolled SSH scripts are
  fragile, produce no traceability, and hide real numbers.
- **The reproducibility crisis.** People say "my model flies" while comparing
  different libraries, dimensions, and no warmup. Crisol pins the **seed**
  (default `1729`), the dimensions, and the warmup so numbers are comparable.
- **The audit.** Every run appends an immutable **JSONL** record: run id, job,
  node, backend, status, exit code, elapsed, seed, and full stdout/stderr.

## Names

| Surface | Name |
|---|---|
| GitHub | `seed-source/crisol` (public, MIT) |
| PyPI | **`crisol`** |
| CLI | **`crisol`** or **`sscri`** (same entrypoint) |
| Module | `py -m crisol` — use this if PATH still has another `crisol` |

## Install

```bash
pip install crisol
```

Then, from any directory:

```bash
# Prefer this if PATH still points at the house lab CLI:
py -m crisol doctor
crisol doctor     # validate python, torch/CUDA, results ledger
crisol status     # probe reachable nodes on this machine
crisol jobs       # list registered jobs
crisol run cpu-smoke    # dispatch a smoke benchmark
crisol run cpu-smoke --seed 42   # override the deterministic seed
```

`cpu-smoke` needs **no torch** — stdlib only. If `torch` is installed,
`crisol jobs` also lists `torch-smoke`, which runs a real FP16 matmul on CUDA
when available and reports honest TFLOPS.

## Remote Dispatch & Multi-Node Labs (Receiver & Client)

Crisol includes a zero-dependency HTTP receiver daemon for dispatching workloads to remote compute hosts (GPU rigs, Mac Silicon nodes, cloud VMs) with real-time NDJSON event streaming.

### 1. Start Receiver on Compute Host
```bash
crisol receiver --port 8430 --token my-secret
```

### 2. Configure Named Node on Local Machine
```bash
crisol config set-node m2 http://192.168.1.50:8430 --token my-secret
crisol config list
```

### 3. Check Remote Health & Dispatch
```bash
# Probe remote node
crisol status --node m2

# Interactive real-time streaming
crisol run torch-smoke --node m2

# Detached background execution (run & reconnect)
crisol run torch-smoke --node m2 --detach
# -> Job 'torch-smoke' dispatched in background! Run ID: run_a9e9f510

# List runs on remote node
crisol runs --node m2

# Stream/listen to the running or completed job
crisol listen run_a9e9f510 --node m2
```

## What you get

- **`crisol status [--node <name|url>]`** — probes reachable nodes locally or on a remote receiver.
- **`crisol jobs`** — lists the registered dispatch contract, including each job's default seed.
- **`crisol run <job> [--node <name|url>] [--seed N] [--detach]`** — dispatches the job locally or remotely with live event streaming and immutable JSONL audit.
- **`crisol listen <run_id> --node <name|url>`** — reconnects to a remote running or completed job's event stream.
- **`crisol runs --node <name|url>`** — lists recent runs and execution status from a remote node.
- **`crisol config [set-node|rm-node|list]`** — manages friendly node aliases (`~/.crisol/config.json`).
- **`crisol receiver [--host H] [--port P] [--token T]`** — starts the zero-bloat HTTP receiver daemon.
- **`crisol doctor`** — validates your Python, torch/CUDA availability, and that the results ledger is writable.

## The JSONL audit ledger

Each run appends a single line to `results/<YYYY-MM-DD>.jsonl`:

```json
{"backend": "cpu", "elapsed_s": 0.8, "exit_code": 0, "job": "cpu-smoke",
 "node": "local-cpu", "run_id": "...", "seed": 1729, "status": "completed",
 "stdout": "...", "stderr": "", "timed_out": false, "ts": "..."}
```

Workloads emit Crisol **protocol JSONL** on stdout (`lifecycle`, `progress`,
`metrics`, `result`, `error`) — see `docs/PROTOCOL.md`.

## Roadmap

- More registered hardware benchmarks (FP16 matmul, attention, tok/s).
- SSH transport for the 2-5 box lab — remote dispatch with the same JSONL
  contract and zero daemons.
- Optional web dashboard for comparing runs.

## License

MIT. See [LICENSE](LICENSE).

---

Crisol is the public launch pad of **Seedsource Novum**. It is deliberately
independent of any proprietary physics engine: a clean protocol + metrics +
JSONL dispatcher that any lab can adopt.
