"""Worker: run one semiflow engine call, emit JSON to stdout.

Usage:
    python worker_semiflow.py <operator> <N> <scale> <t> <path> [n_steps]

operator: "dirichlet" | "neumann"
path:     "implicit" | "chebyshev" | "lanczos"
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np


def build_neumann_csr(n: int, scale: float):
    """Singular Neumann Laplacian scaled by scale (row sums = 0)."""
    import numpy as np

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for i in range(n):
        is_boundary = i == 0 or i == n - 1
        diag = scale if is_boundary else 2.0 * scale
        if i > 0:
            indices.append(i - 1)
            data.append(-scale)
        indices.append(i)
        data.append(diag)
        if i < n - 1:
            indices.append(i + 1)
            data.append(-scale)
        indptr.append(len(indices))
    return (
        np.array(indptr, dtype=np.int64),
        np.array(indices, dtype=np.int32),
        np.array(data, dtype=np.float64),
    )


def build_dirichlet_csr(n: int, scale: float):
    """Positive-definite 1-D Laplacian (Dirichlet BCs, full 2*scale diagonal)."""
    import numpy as np

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for i in range(n):
        if i > 0:
            indices.append(i - 1)
            data.append(-scale)
        indices.append(i)
        data.append(2.0 * scale)
        if i < n - 1:
            indices.append(i + 1)
            data.append(-scale)
        indptr.append(len(indices))
    return (
        np.array(indptr, dtype=np.int64),
        np.array(indices, dtype=np.int32),
        np.array(data, dtype=np.float64),
    )


def neumann_oracle(v, n: int) -> "np.ndarray":
    """Analytic e^{-tA}v for stiff Neumann Laplacian: surviving mode = mean(v)."""
    import numpy as np

    return np.full(n, float(v.mean()))


def run(op_type: str, n: int, scale: float, t: float, path: str, n_steps: int) -> dict:
    """Build operator, run evolve_batched, return result dict."""
    import numpy as np
    import semiflow

    if op_type == "neumann":
        indptr, indices, data = build_neumann_csr(n, scale)
        sym_tol = 1e-6
    else:
        indptr, indices, data = build_dirichlet_csr(n, scale)
        sym_tol = 1e-10

    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n, sym_tol)

    # Non-constant initial condition: linspace (non-trivial for Neumann)
    v = np.linspace(1.0 / n, 1.0, n).reshape(n, 1)

    kwargs: dict = {"path": path}
    if path == "implicit":
        kwargs["n_steps"] = n_steps

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_start = time.perf_counter()
    out = op.evolve_batched(t=t, v_nc=v, **kwargs)
    elapsed = time.perf_counter() - t_start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB (Linux)

    sup_error: float | None = None
    if op_type == "neumann" and scale >= 1e6:
        oracle = neumann_oracle(v, n)
        sup_error = float(np.max(np.abs(out.ravel() - oracle)))

    return {
        "engine": f"semiflow/{path}",
        "op_type": op_type,
        "n": n,
        "scale": scale,
        "t": t,
        "wallclock_s": elapsed,
        "peak_mem_mb": max(rss_before, rss_after) / 1024.0,  # KB -> MB
        "timeout": False,
        "sup_error": sup_error,
    }


def main() -> None:
    if len(sys.argv) < 6:
        print("Usage: worker_semiflow.py <op_type> <N> <scale> <t> <path> [n_steps]",
              file=sys.stderr)
        sys.exit(1)
    op_type = sys.argv[1]
    n = int(sys.argv[2])
    scale = float(sys.argv[3])
    t = float(sys.argv[4])
    path = sys.argv[5]
    n_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    result = run(op_type, n, scale, t, path, n_steps)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
