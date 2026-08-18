"""torch-smoke: optional torch micro-benchmark (CUDA matmul when present).

A tiny optional probe that runs a real FP16 dense matmul on whatever torch
backend is available — CUDA if present, else CPU — and emits Crisol protocol
JSONL (lifecycle, metrics, result). This job is only listed when torch is
importable on the node.

Measured rate is TFLOPS = (2 * N^3 * iters) / elapsed_s / 1e12 on the
selected device. This is a launch/smoke check; use a registered hardware
benchmark for authoritative numbers.

Run as:  python -m crisol.jobs.torch_smoke --seed 1729
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

from crisol import protocol as p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=1729, help="Deterministic seed (default 1729)."
    )
    parser.add_argument("--n", type=int, default=2048, help="Matrix dimension N (default 2048).")
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations (default 5).")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device override (default: cuda if available else cpu).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed = args.seed
    n = args.n
    iters = max(1, args.iters)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    p._lifecycle("start", job="torch-smoke", seed=seed, device=device, torch=torch.__version__)
    p._progress("init", 0, 1)

    torch.manual_seed(seed)
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    p._progress("init", 1, 1)

    # Warmup: one untimed pass so CUDA kernels/caching are warm.
    _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    p._lifecycle("benchmark_start", job="torch-smoke", n=n, iters=iters, device=device)

    started = time.perf_counter()
    for i in range(iters):
        c = a @ b
        if device == "cuda":
            torch.cuda.synchronize()
        p._progress("matmul", i + 1, iters)
    elapsed = time.perf_counter() - started

    flops = 2.0 * (n**3) * iters
    tflops = flops / max(elapsed, 1e-9) / 1e12
    memory_gb = 0.0
    if device == "cuda":
        memory_gb = float(torch.cuda.memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    p._metrics(phase="matmul", rate=tflops, memory_gb=memory_gb, elapsed_s=elapsed,
               unit="TFLOPS", n=n, iters=iters, device=device, seed=seed)
    p._result(status="ok", job="torch-smoke", tflops=round(tflops, 4),
              elapsed_s=round(elapsed, 4), device=device, seed=seed,
              platform=platform.platform())
    p._lifecycle("complete", job="torch-smoke", status="ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
