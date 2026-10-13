"""Rational (shift-and-invert) Krylov path (ADR-0206)."""

from __future__ import annotations

import numpy as np
import pytest

import semiflow as sf  # pyright: ignore[reportMissingImports]


def neumann(n: int, scale: float = 1.0) -> sf.SymmetricOperator:
    """Neumann path Laplacian ``n×n`` times ``scale``."""
    rows, cols, vals = [], [], []
    for i in range(n):
        deg = (i > 0) + (i + 1 < n)
        if i > 0:
            rows.append(i); cols.append(i - 1); vals.append(-scale)
        rows.append(i); cols.append(i); vals.append(deg * scale)
        if i + 1 < n:
            rows.append(i); cols.append(i + 1); vals.append(-scale)
    indptr = np.searchsorted(np.array(rows), np.arange(n + 1)).astype(np.int64)
    return sf.SymmetricOperator.from_csr(
        indptr, np.array(cols, dtype=np.int32), np.array(vals), n
    )


def dense_expm(op: sf.SymmetricOperator, t: float, v: np.ndarray, mass: np.ndarray | None = None) -> np.ndarray:
    indptr, indices, data = op.to_csr()
    n = op.n()
    a = np.zeros((n, n))
    for i in range(n):
        a[i, indices[indptr[i]:indptr[i + 1]]] = data[indptr[i]:indptr[i + 1]]
    m = np.ones(n) if mass is None else mass
    s = 1.0 / np.sqrt(m)
    lam, q = np.linalg.eigh(s[:, None] * a * s[None, :])
    w = q.T @ (np.sqrt(m) * v)
    return s * (q @ (np.exp(-t * np.maximum(lam, 0.0)) * w))


def test_stiff_neumann_matches_eigen_reference() -> None:
    n = 400
    op = neumann(n)
    v = np.sin(np.linspace(0.0, 7.0, n)) + np.where(np.arange(n) % 3 == 0, 0.5, -0.25)
    for lam_t in (1e2, 1e6, 1e8):
        t = lam_t / op.lambda_max_bound()
        rk = op.rational_krylov(t, tol=1e-10)
        y, report = rk.apply(v)
        assert np.max(np.abs(y - dense_expm(op, t, v))) <= 1e-9
        assert 1 <= report["solves"] <= 30
        assert report["inner_iterations"] == 0  # tridiagonal LDLᵀ
        assert rk.gamma == pytest.approx(t / 10.0)


def test_batched_entry_points_agree() -> None:
    n = 60
    op = neumann(n, 3.0)
    v_nc = np.stack([np.cos(np.arange(n) * k * 0.1) for k in range(3)], axis=1)
    rk = op.rational_krylov(0.7, tol=1e-12)
    via_object = rk.apply_batched(v_nc)
    via_path = op.evolve_batched(0.7, v_nc, path="rational", tol=1e-12)
    assert np.array_equal(via_object, via_path)
    want = np.stack([dense_expm(op, 0.7, v_nc[:, c]) for c in range(3)], axis=1)
    assert np.max(np.abs(via_path - want)) <= 1e-10


def test_mass_lumped_rational_matches_reference() -> None:
    n = 50
    op = neumann(n, 2.0)
    mass = 0.5 + 0.02 * np.arange(n)
    v_nc = np.linspace(-1.0, 2.0, n)[:, None]
    got = sf.mass_lumped_evolve(op, mass, 3.0, v_nc, path="rational", tol=1e-12)
    want = dense_expm(op, 3.0, v_nc[:, 0], mass)
    assert np.max(np.abs(got[:, 0] - want)) <= 1e-10


def test_identity_and_errors() -> None:
    op = neumann(10)
    v = np.arange(10.0)
    y, report = op.rational_krylov(0.0).apply(v)
    assert np.array_equal(y, v) and report["solves"] == 0
    with pytest.raises(sf.SemiflowError):
        op.rational_krylov(1.0, tol=2.0)
    with pytest.raises(sf.SemiflowError):
        op.rational_krylov(1.0, solver="cholmod")
    with pytest.raises(sf.SemiflowError):
        op.rational_krylov(1.0).apply(np.ones(9))
    with pytest.raises(sf.SemiflowError, match="GridMismatch"):
        op.rational_krylov(1.0, mass=np.ones(9))
    with pytest.raises(sf.SemiflowError, match="'rational'"):
        op.evolve_batched(1.0, np.ones((10, 1)), path="nope")
    # A shift ratio the a-priori bound cannot resolve fails loudly, not silently.
    # (n = 100 > m_max = 64, so the iteration cannot end on the whole space.)
    with pytest.raises(sf.SemiflowError):
        neumann(100).rational_krylov(1.0, gamma=1e-7).apply(np.arange(100.0))


@pytest.mark.parametrize(
    ("solver", "precond"), [("tridiagonal", "ic0"), ("pcg", "ic0"), ("pcg", "jacobi")]
)
def test_solver_variants_match_reference(solver: str, precond: str) -> None:
    op = neumann(60, 1e3)
    mass = 1.0 + 0.5 * np.sin(np.arange(60.0))
    v = np.cos(np.linspace(0.0, 3.0, 60))
    rk = op.rational_krylov(0.7, mass=mass, solver=solver, precond=precond)
    y, report = rk.apply(v)
    ref = dense_expm(op, 0.7, v, mass)
    assert np.max(np.abs(y - ref)) <= 1e-9 * np.max(np.abs(v))
    assert report["solves"] <= 30
