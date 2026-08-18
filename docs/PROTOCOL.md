# Crisol Protocol

Crisol workloads emit **JSON-lines (JSONL)** on stdout. Each line is a single
JSON object, flushed immediately. The canonical event-type set lives in
`crisol/protocol.py` — that module is the single source of truth and has zero
dependencies beyond the Python standard library.

## Event types

| `type`       | Purpose                                                        | Key fields                                    |
|--------------|----------------------------------------------------------------|-----------------------------------------------|
| `lifecycle`  | start / complete / benchmark_start / alive                     | `event`, `ts`                                 |
| `progress`   | phase progress                                                 | `phase`, `current`, `total`, `ts`             |
| `metrics`    | a measured rate (TFLOPS, tok/s, ...)                           | `phase`, `rate`, `memory_gb`, `elapsed_s`, `ts` |
| `histogram`  | weight/gradient distribution (binned)                          | `name`, `values`, `bins`, `ts`                |
| `sparkline`  | streaming time-series (loss/acc)                               | `name`, `values`, `window`, `ts`              |
| `gpu_stats`  | GPU hardware snapshot (VRAM, temp, util)                       | `vram_used_mb`, `vram_total_mb`, `temp_c`, `util_pct` |
| `result`     | terminal outcome                                               | `status`, `ts`                                |
| `error`      | fatal error                                                    | `error`, `ts`                                 |
| `artifact`   | produced artifact descriptor                                   | `descriptor` (`filename`, `bytes`, `sha256`, `kind`) |

## Helpers

Use `crisol.protocol` helpers instead of writing raw JSON:

```python
from crisol import protocol as p

p._lifecycle("start", job="cpu-smoke", seed=1729)
p._progress("matmul", 1, 5)
p._metrics(phase="matmul", rate=1.234, memory_gb=0.0, elapsed_s=0.8, unit="GFLOP/s")
p._result(status="ok", job="cpu-smoke", seed=1729)
p._lifecycle("complete", job="cpu-smoke", status="ok")
```

All helpers are pure and importable without torch, numpy, or CUDA. The
receiver (e.g. the Crisol runner ledger, a TUI, or a dashboard) parses these
lines to build live progress and a final immutable record.
