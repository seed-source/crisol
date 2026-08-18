"""cpu-smoke: stdlib-only CPU matmul smoke benchmark with honest metrics.

A tiny, dependency-free probe that times a deterministic dense matmul using a
pure-Python row-dot-product kernel and emits Crisol protocol JSONL
(lifecycle, metrics, result) on stdout. No torch, no numpy, no CUDA.

The measured rate is GFLOPS = (2 * N^3 * iters) / elapsed_s / 1e9 and is
intentionally honest — a pure-Python loop is slow, and the number reflects
that. This is a launch/smoke check, not a hardware benchmark.

Run as:  python -m crisol.jobs.cpu_smoke --seed 1729
"""
from __future__ import annotations

import argparse
import platform
import random
import sys
import time

from crisol import protocol as p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=1729, help="Deterministic PRNG seed (default 1729)."
    )
    parser.add_argument("--n", type=int, default=64, help="Matrix dimension N (default 64).")
    parser.add_argument("--iters", type=int, default=2, help="Timed iterations (default 2).")
    return parser.parse_args(argv)


def _matmul_row_dot(a: list[list[float]], b: list[list[float]], n: int) -> list[list[float]]:
    """Pure-Python row-major matmul via dot products. Honest and slow."""
    bt = [list(col) for col in zip(*b)]
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed = args.seed
    n = args.n
    iters = max(1, args.iters)

    p._lifecycle("start", job="cpu-smoke", seed=seed, kernel="row_dot")
    p._progress("init", 0, 1)

    rng = random.Random(seed)
    a = [[rng.random() for _ in range(n)] for _ in range(n)]
    b = [[rng.random() for _ in range(n)] for _ in range(n)]
    p._progress("init", 1, 1)

    # Warmup: one untimed pass so first-run caches/paging are not metered.
    _matmul_row_dot(a, b, n)
    p._lifecycle("benchmark_start", job="cpu-smoke", n=n, iters=iters)

    started = time.perf_counter()
    for i in range(iters):
        c = _matmul_row_dot(a, b, n)
        p._progress("matmul", i + 1, iters)
    elapsed = time.perf_counter() - started

    flops = 2.0 * (n**3) * iters
    gflops = flops / max(elapsed, 1e-9) / 1e9
    p._metrics(phase="matmul", rate=gflops, memory_gb=0.0, elapsed_s=elapsed,
               unit="GFLOP/s", n=n, iters=iters, seed=seed)
    p._result(status="ok", job="cpu-smoke", gflops=round(gflops, 4),
              elapsed_s=round(elapsed, 4), seed=seed, platform=platform.platform())
    p._lifecycle("complete", job="cpu-smoke", status="ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
