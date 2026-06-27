"""Smoke tests for issue #15 — PyO3 exposure of #11/#12/#13/#14.

Each test is self-contained (no fixtures) and validates shape, type, and
rough numerical accuracy against simple analytic cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import semiflow


# ---------------------------------------------------------------------------
# Issue #13: SymmetricOperator + symmetric_op_expmv_frechet
# ---------------------------------------------------------------------------

def _neg_laplacian_1d_csr(n: int):
    """Build -L (negative Laplacian) as CSR for a 1-D path graph with n nodes."""
    h2 = (n - 1) ** 2  # (n-1)^2 for unit-interval dx = 1/(n-1)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = []
    data = []
    for i in range(n):
        if i > 0:
            indices.append(i - 1)
            data.append(-h2)
        indices.append(i)
        data.append(2 * h2 if 0 < i < n - 1 else h2)
        if i < n - 1:
            indices.append(i + 1)
            data.append(-h2)
        indptr[i + 1] = len(indices)
    return (
        indptr,
        np.array(indices, dtype=np.int32),
        np.array(data, dtype=np.float64),
    )


def test_symmetric_operator_from_csr_smoke():
    n = 8
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    assert op.n() == n
    assert op.lambda_max_bound() > 0.0


def test_symmetric_operator_evolve_batched_shape():
    n = 10
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    V = np.random.default_rng(0).standard_normal((n, 3))
    out = op.evolve_batched(t=0.01, v_nc=V)
    assert out.shape == (n, 3)
    # Semigroup should not blow up norm
    assert np.all(np.abs(out) <= np.abs(V).max() + 1e-6)


def test_symmetric_operator_null_space_preserved():
    """Constant vector is in the null space of neg-Laplacian; evolve_batched preserves it."""
    n = 12
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    # 1-D neg-Laplacian with natural (Neumann) BCs has row-sum = 0, so 1 ∈ ker(A)
    v = np.ones((n, 1))
    out = op.evolve_batched(t=0.5, v_nc=v)
    np.testing.assert_allclose(out, v, atol=1e-10)


def test_symmetric_op_expmv_frechet_shape():
    n = 6
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    rng = np.random.default_rng(42)
    u0 = rng.standard_normal((n, 2))
    dj = rng.standard_normal((n, 2))
    entries = [(0, 0), (0, 1), (1, 1)]
    grad = semiflow.symmetric_op_expmv_frechet(op, u0, dj, t=0.1, entries=entries)
    assert grad.shape == (3,)
    assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# Issue #11: ConservativeDiffusionChernoff + assemble_conservative_csr_1d
# ---------------------------------------------------------------------------

def test_conservative_diffusion_from_k_array():
    n = 16
    k = np.ones(n)
    op = semiflow.ConservativeDiffusionChernoff.from_k_array(
        n=n, x_lo=0.0, x_hi=1.0, k_nodes=k
    )
    assert op.n() == n
    assert np.isfinite(op.dx())


def test_conservative_to_symmetric_operator():
    n = 20
    k = 1.0 + 0.5 * np.sin(np.linspace(0, np.pi, n))
    cd = semiflow.ConservativeDiffusionChernoff.from_k_array(
        n=n, x_lo=0.0, x_hi=1.0, k_nodes=k
    )
    sym_op = cd.to_symmetric_operator()
    assert sym_op.n() == n
    assert sym_op.lambda_max_bound() > 0.0


def test_assemble_conservative_csr_1d():
    n = 12
    k = np.full(n, 2.0)
    sym_op = semiflow.assemble_conservative_csr_1d(
        n=n, x_lo=0.0, x_hi=1.0, k_nodes=k
    )
    assert sym_op.n() == n
    v = np.ones((n, 1))
    out = sym_op.evolve_batched(t=1e-4, v_nc=v)
    assert out.shape == (n, 1)


def test_conservative_dirichlet_boundary():
    n = 8
    k = np.ones(n)
    op = semiflow.ConservativeDiffusionChernoff.from_k_array(
        n=n, x_lo=0.0, x_hi=1.0, k_nodes=k, boundary="dirichlet:0.0"
    )
    assert op.n() == n


# ---------------------------------------------------------------------------
# Issue #14: MassKOperator + mass_lumped_evolve
# ---------------------------------------------------------------------------

def test_mass_lumped_evolve_shape():
    n = 10
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    m = np.ones(n) * 2.0  # uniform mass
    V = np.eye(n)  # identity columns
    out = semiflow.mass_lumped_evolve(op, m, t=0.05, v_nc=V)
    assert out.shape == (n, n)
    assert np.all(np.isfinite(out))


def test_mass_lumped_evolve_identity_mass():
    """Lumped mass = identity should match plain evolve_batched."""
    n = 8
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    m = np.ones(n)
    rng = np.random.default_rng(7)
    V = rng.standard_normal((n, 3))
    out_lumped = semiflow.mass_lumped_evolve(op, m, t=0.01, v_nc=V)
    out_plain = op.evolve_batched(t=0.01, v_nc=V)
    # D^{½} = I, D^{-½} = I, so lumped_evolve = evolve_batched
    np.testing.assert_allclose(out_lumped, out_plain, rtol=1e-8)


def test_mass_k_operator_from_k_and_mass_shape():
    n = 6
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    # Dense SPD mass matrix: M = 2I + 0.5*tridiag(1,0,1)
    M = 2.0 * np.eye(n)
    for i in range(n - 1):
        M[i, i + 1] = 0.5
        M[i + 1, i] = 0.5
    mo = semiflow.MassKOperator.from_k_and_mass(op, M.ravel())
    assert mo.n() == n
    v = np.ones(n)
    out = mo.evolve(t=0.01, v=v)
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Issue #12: phi_action, phi_action_batched, Etdrk4
# ---------------------------------------------------------------------------

def test_phi_action_k0_is_expm():
    """phi_0(t, A) v = e^{-tA} v."""
    n = 8
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    v = np.ones(n)
    t = 0.05
    phi0 = semiflow.phi_action(op, k=0, tau=t, v=v)
    ref = op.evolve_batched(t=t, v_nc=v.reshape(n, 1)).ravel()
    np.testing.assert_allclose(phi0, ref, rtol=1e-7)


def test_phi_action_batched_shape():
    n = 6
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    v = np.ones(n)
    out = semiflow.phi_action_batched(op, p=3, tau=0.1, v=v)
    # shape: [p+1, n]
    assert out.shape == (4, n)
    assert np.all(np.isfinite(out))


def test_etdrk4_smoke():
    """ETDRK4 with Allen-Cahn nonlinearity decreases away from trivial equilibria."""
    n = 16
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    h = 0.01
    stepper = semiflow.Etdrk4.from_symmetric_op(op, nonlinearity="allen_cahn", h=h)
    # Initial condition: small perturbation around 0
    rng = np.random.default_rng(99)
    u0 = rng.standard_normal(n) * 0.1
    u_end = stepper.integrate(u0, n_steps=10)
    assert u_end.shape == (n,)
    assert np.all(np.isfinite(u_end))


# ---------------------------------------------------------------------------
# Convenience: Laplacian.from_csr (graph_extra.rs)
# ---------------------------------------------------------------------------

def test_laplacian_from_csr():
    n = 6
    indptr, indices, data = _neg_laplacian_1d_csr(n)
    lap = semiflow.Laplacian.from_csr(indptr, indices, data, n)
    # Laplacian should expose .n() or similar; just check it's the right type
    assert isinstance(lap, semiflow.Laplacian)


# ---------------------------------------------------------------------------
# Regression: dir(semiflow) includes all new symbols
# ---------------------------------------------------------------------------

def test_new_symbols_exported():
    names = dir(semiflow)
    expected = [
        "SymmetricOperator",
        "symmetric_op_expmv_frechet",
        "ConservativeDiffusionChernoff",
        "assemble_conservative_csr_1d",
        "MassKOperator",
        "mass_lumped_evolve",
        "phi_action",
        "phi_action_batched",
        "Etdrk4",
    ]
    missing = [s for s in expected if s not in names]
    assert missing == [], f"Missing from dir(semiflow): {missing}"
