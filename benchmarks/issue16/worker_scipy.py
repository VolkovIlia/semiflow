"""Worker: run scipy.sparse.linalg.expm_multiply, emit JSON to stdout.

Usage:
    python worker_scipy.py <operator> <N> <scale> <t>
"""

from __future__ import annotations

import json
import resource
import sys
import time


def build_sparse(n: int, scale: float, op_type: str):
    """Build scipy CSR matrix for the 1-D Laplacian."""
    import numpy as np
    import scipy.sparse as sp

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(n):
        is_boundary = i == 0 or i == n - 1
        if op_type == "neumann":
            diag = scale if is_boundary else 2.0 * scale
        else:
            diag = 2.0 * scale
        if i > 0:
            rows.append(i); cols.append(i - 1); data.append(-scale)
        rows.append(i); cols.append(i); data.append(diag)
        if i < n - 1:
            rows.append(i); cols.append(i + 1); data.append(-scale)
    return sp.csr_matrix(
        (np.array(data), (np.array(rows), np.array(cols))),
        shape=(n, n),
    )


def run(op_type: str, n: int, scale: float, t: float) -> dict:
    """Build sparse matrix, run expm_multiply, return result dict."""
    import numpy as np
    import scipy.sparse.linalg

    A = build_sparse(n, scale, op_type)
    v = np.linspace(1.0 / n, 1.0, n)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_start = time.perf_counter()
    out = scipy.sparse.linalg.expm_multiply(-t * A, v)
    elapsed = time.perf_counter() - t_start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    sup_error: float | None = None
    if op_type == "neumann" and scale >= 1e6:
        mean_v = float(v.mean())
        sup_error = float(np.max(np.abs(out - mean_v)))

    return {
        "engine": "scipy/expm_multiply",
        "op_type": op_type,
        "n": n,
        "scale": scale,
        "t": t,
        "wallclock_s": elapsed,
        "peak_mem_mb": max(rss_before, rss_after) / 1024.0,
        "timeout": False,
        "sup_error": sup_error,
    }


def main() -> None:
    if len(sys.argv) < 5:
        print("Usage: worker_scipy.py <op_type> <N> <scale> <t>", file=sys.stderr)
        sys.exit(1)
    op_type = sys.argv[1]
    n = int(sys.argv[2])
    scale = float(sys.argv[3])
    t = float(sys.argv[4])
    result = run(op_type, n, scale, t)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
