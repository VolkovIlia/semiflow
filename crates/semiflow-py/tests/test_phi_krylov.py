"""Adaptive Krylov (Arnoldi) φ path: ``method="krylov"`` / ``phi="krylov"`` (ADR-0211)."""

from __future__ import annotations

import numpy as np
import pytest

import semiflow as sf  # pyright: ignore[reportMissingImports]


def csr(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense -> ``(indptr, indices, data)``."""
    indptr, indices, data = [0], [], []
    for row in a:
        nz = np.nonzero(row)[0]
        indices.extend(nz.tolist())
        data.extend(row[nz].tolist())
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int64), np.array(indices, dtype=np.int32), np.array(data)


def fokker_planck(n: int, d: float = 0.5) -> np.ndarray:
    """Dense ``A`` (``G = -A``) of the drifted Fokker–Planck generator on ``[-3, 3]``."""
    dx = 6.0 / (n - 1)
    g = np.zeros((n, n))
    for i in range(n - 1):
        b = 1.5 - (-3.0 + (i + 0.5) * dx)
        from_i = d / dx**2 + max(b, 0.0) / dx
        from_ip = d / dx**2 + max(-b, 0.0) / dx
        g[i + 1, i] += from_i
        g[i, i] -= from_i
        g[i, i + 1] += from_ip
        g[i + 1, i + 1] -= from_ip
    return -g


def general(a: np.ndarray) -> sf.GeneralOperator:
    indptr, indices, data = csr(a)
    return sf.GeneralOperator.from_csr(a.shape[0], indptr.tolist(), indices.tolist(), data.tolist())


def test_phi_combination_krylov_matches_taylor() -> None:
    n = 150
    a = fokker_planck(n)
    op = general(a)
    tau = 300.0 / op.norm_inf_bound()
    rng = np.random.default_rng(7)
    w = rng.standard_normal((4, n))
    want = sf.phi_combination(op, tau, w)
    got = sf.phi_combination(op, tau, w, method="krylov", tol=1e-12)
    assert np.max(np.abs(got - want)) <= 1e-8 * max(1.0, np.max(np.abs(want)))


def test_general_operator_expmv_krylov_matches_taylor() -> None:
    n = 200
    op = general(fokker_planck(n))
    t = 1000.0 / op.norm_inf_bound()
    v_nc = np.stack([np.exp(-np.linspace(-3, 3, n) ** 2), np.linspace(0, 1, n)], axis=1)
    want = op.evolve_batched(t, v_nc)
    got = op.evolve_batched(t, v_nc, method="krylov", tol=1e-12)
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) <= 1e-9 * np.max(np.abs(want))


def test_etdrk4_krylov_matches_taylor() -> None:
    n = 64
    a = np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    a[0, 0] = a[-1, -1] = 1.0
    indptr, indices, data = csr(a * n * n * 1e-2)
    op = sf.SymmetricOperator.from_csr(indptr, indices, data, n)
    u0 = 0.5 * np.sin(np.linspace(0.0, 2.0 * np.pi, n))
    taylor = sf.Etdrk4.from_symmetric_op(op, "allen_cahn", 0.01).integrate(u0, 20)
    krylov = sf.Etdrk4.from_symmetric_op(op, "allen_cahn", 0.01, phi="krylov", tol=1e-13)
    assert np.max(np.abs(krylov.integrate(u0, 20) - taylor)) <= 1e-10


def test_unknown_method_and_bad_tol_raise() -> None:
    op = general(fokker_planck(20))
    w = np.ones((2, 20))
    with pytest.raises(sf.SemiflowError):
        sf.phi_combination(op, 0.1, w, method="pade")
    with pytest.raises(sf.SemiflowError):
        sf.phi_combination(op, 0.1, w, method="krylov", tol=2.0)
    with pytest.raises(sf.SemiflowError):
        op.evolve_batched(0.1, np.ones((20, 1)), method="lanczos")
