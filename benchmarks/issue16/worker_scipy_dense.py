"""Worker: compute scipy.linalg.expm (dense) reference, emit JSON to stdout.

Usage:
    python worker_scipy_dense.py <operator> <N> <t>
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np


def build_dense(n: int, scale: float, op_type: str) -> "np.ndarray":
    """Build dense NxN matrix for the 1-D Laplacian."""
    import numpy as np

    A = np.zeros((n, n))
    for i in range(n):
        is_boundary = i == 0 or i == n - 1
        if op_type == "neumann":
            A[i, i] = scale if is_boundary else 2.0 * scale
        else:
            A[i, i] = 2.0 * scale
        if i > 0:
            A[i, i - 1] = -scale
        if i < n - 1:
            A[i, i + 1] = -scale
    return A


def run(op_type: str, n: int, scale: float, t: float) -> dict:
    """Compute dense expm reference; only feasible for small n."""
    import numpy as np
    import scipy.linalg

    A = build_dense(n, scale, op_type)
    v = np.linspace(1.0 / n, 1.0, n)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_start = time.perf_counter()
    out = scipy.linalg.expm(-t * A) @ v
    elapsed = time.perf_counter() - t_start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "engine": "scipy/expm_dense",
        "op_type": op_type,
        "n": n,
        "scale": scale,
        "t": t,
        "wallclock_s": elapsed,
        "peak_mem_mb": max(rss_before, rss_after) / 1024.0,
        "timeout": False,
        "sup_error": None,
        "out": out.tolist(),
    }


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: worker_scipy_dense.py <op_type> <N> <t>", file=sys.stderr)
        sys.exit(1)
    op_type = sys.argv[1]
    n = int(sys.argv[2])
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    t = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    result = run(op_type, n, scale, t)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
