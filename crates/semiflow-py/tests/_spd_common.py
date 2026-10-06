"""Shared builders for the ADR-0202 Python tests (not a test module)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

import semiflow

try:
    import scipy.linalg
    import scipy.sparse
    import scipy.sparse.linalg

    HAS_SCIPY = True
except ImportError:
    # Bound name so Pyright stays clean; tests using scipy carry @needs_scipy.
    import types as _t

    scipy = _t.SimpleNamespace(  # type: ignore[assignment]
        linalg=_t.SimpleNamespace(),
        sparse=_t.SimpleNamespace(linalg=_t.SimpleNamespace()),
    )
    HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")


# ---------------------------------------------------------------------------
# Builders (pure numpy, no libm beyond what the cases state)
# ---------------------------------------------------------------------------


def _csr_from_rows(rows: list[list[tuple[int, float]]]):
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    idx: list[int] = []
    dat: list[float] = []
    for i, row in enumerate(rows):
        for j, v in sorted(row):
            idx.append(j)
            dat.append(v)
        indptr[i + 1] = len(idx)
    return indptr, np.array(idx, dtype=np.int32), np.array(dat, dtype=np.float64)


def _op(indptr, indices, data) -> "semiflow.SymmetricOperator":
    return semiflow.SymmetricOperator.from_csr(indptr, indices, data, len(indptr) - 1)


def _path_rows(n: int, cond) -> list[list[tuple[int, float]]]:
    rows: list[list[tuple[int, float]]] = []
    for i in range(n):
        left = cond(i - 1) if i > 0 else 0.0
        right = cond(i) if i + 1 < n else 0.0
        row = [(i, left + right)]
        if i > 0:
            row.append((i - 1, -left))
        if i + 1 < n:
            row.append((i + 1, -right))
        rows.append(row)
    return rows


def _grid_op(side: int) -> "semiflow.SymmetricOperator":
    """5-point Neumann Laplacian on a side x side grid (row-major)."""
    rows: list[list[tuple[int, float]]] = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            nbrs = []
            if r > 0:
                nbrs.append(i - side)
            if c > 0:
                nbrs.append(i - 1)
            if c + 1 < side:
                nbrs.append(i + 1)
            if r + 1 < side:
                nbrs.append(i + side)
            rows.append([(j, -1.0) for j in nbrs] + [(i, float(len(nbrs)))])
    return _op(*_csr_from_rows(rows))


def _op_1d(n: int, reaction: float = 1.0) -> "semiflow.SymmetricOperator":
    """Conservative Neumann carrier, k(x) = 1 + x^2, plus a reaction c > 0."""
    x = np.linspace(0.0, 1.0, n)
    op = semiflow.assemble_conservative_csr_1d(n, 0.0, 1.0, 1.0 + x * x)
    return op.with_diagonal(np.full(n, reaction))


def _to_scipy(op):
    indptr, indices, data = op.to_csr()
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(op.n(), op.n()))


def _dense(op) -> np.ndarray:
    indptr, indices, data = op.to_csr()
    a = np.zeros((op.n(), op.n()))
    for i in range(op.n()):
        for k in range(indptr[i], indptr[i + 1]):
            a[i, indices[k]] += data[k]
    return a


def _rel(got: npt.ArrayLike, want: npt.ArrayLike) -> float:
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def _rhs(n: int, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, n)


def _fnv(x: np.ndarray) -> int:
    h = 0xCBF29CE484222325
    for byte in np.ascontiguousarray(x, dtype="<f8").tobytes():
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
