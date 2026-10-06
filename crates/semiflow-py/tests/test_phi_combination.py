"""ADR-0202 Wave 3: phi_combination, and the mass= kwarg / GeneralOperator on phi_action.

Oracles: dense augmented scipy.linalg.expm (skips without scipy) and a numpy-only
symmetric-eigenvalue closed form that always runs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import semiflow
from _spd_common import (
    _csr_from_rows,
    _dense,
    _op_1d,
    _rel,
    needs_scipy,
    scipy,
)

# ---------------------------------------------------------------------------
# φ-functions: phi_combination, mass= kwarg, GeneralOperator
# ---------------------------------------------------------------------------


def _phi_dense(g: np.ndarray, tau: float, k: int, v: np.ndarray) -> np.ndarray:
    """φ_k(τG) v from the dense augmented exponential (Sidje 1998); needs scipy."""
    n = g.shape[0]
    if k == 0:
        return scipy.linalg.expm(tau * g) @ v
    big = np.zeros((n + k, n + k))
    big[:n, :n] = tau * g
    big[:n, n] = v
    for j in range(k - 1):
        big[n + j, n + j + 1] = 1.0
    return scipy.linalg.expm(big)[:n, n + k - 1]


def _phi_eigh(a: np.ndarray, mass: np.ndarray, tau: float, k: int, v: np.ndarray):
    """numpy-only φ_k(−τM⁻¹A) v via the symmetric pencil (closed-form scalar φ_k)."""
    d = 1.0 / np.sqrt(mass)
    lam, q = np.linalg.eigh(d[:, None] * a * d[None, :])
    z = -tau * lam
    # closed-form recursion φ_{j}(z) = (φ_{j-1}(z) − 1/(j−1)!) / z
    phi = np.exp(z)
    for j in range(1, k + 1):
        phi = (phi - 1.0 / math.factorial(j - 1)) / z
    w = q.T @ (v / d)
    return d * (q @ (phi * w))


def test_phi_combination_numpy_eigh() -> None:
    n = 10
    op = _op_1d(n, reaction=0.5)
    mass = 1.0 + 2.0 * np.arange(n) / n
    tau = 0.05
    w = np.random.default_rng(1).uniform(-1.0, 1.0, (4, n))
    got = semiflow.phi_combination(op, tau, w, mass=mass)
    want = sum(
        (tau**k * _phi_eigh(_dense(op), mass, tau, k, w[k]) for k in range(4)),
        start=np.zeros(n),
    )
    assert _rel(got, want) <= 1e-9


@needs_scipy
@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_phi_combination_vs_expm(p: int) -> None:
    n = 10
    op = _op_1d(n, reaction=0.5)
    tau = 0.2
    w = np.random.default_rng(p).uniform(-1.0, 1.0, (p + 1, n))
    g = -_dense(op)
    want = sum(
        (tau**k * _phi_dense(g, tau, k, w[k]) for k in range(p + 1)), start=np.zeros(n)
    )
    got = semiflow.phi_combination(op, tau, w)
    assert _rel(got, want) <= 1e-12


@needs_scipy
def test_phi_combination_mass_and_general_operator() -> None:
    n = 10
    op = _op_1d(n, reaction=0.5)
    mass = 1.0 + 49.0 * (np.arange(n) / (n - 1)) ** 2
    tau = 0.3
    w = np.random.default_rng(5).uniform(-1.0, 1.0, (3, n))
    g = -_dense(op) / mass[:, None]
    want = sum((tau**k * _phi_dense(g, tau, k, w[k]) for k in range(3)), start=np.zeros(n))
    assert _rel(semiflow.phi_combination(op, tau, w, mass=mass), want) <= 1e-12

    # Upwind drift-diffusion: non-symmetric, GeneralOperator.
    a = _dense(op) + 0.3 * (np.eye(n) - np.eye(n, k=-1))
    indptr, indices, data = _csr_dense(a)
    gen = semiflow.GeneralOperator.from_csr(
        n, indptr.tolist(), indices.tolist(), data.tolist()
    )
    want_g = sum(
        (tau**k * _phi_dense(-a, tau, k, w[k]) for k in range(3)), start=np.zeros(n)
    )
    assert _rel(semiflow.phi_combination(gen, tau, w), want_g) <= 1e-12
    for k in range(4):
        got_k = semiflow.phi_action(gen, k, tau, w[0])
        assert _rel(got_k, _phi_dense(-a, tau, k, w[0])) <= 1e-12
    batched = semiflow.phi_action_batched(op, 3, tau, w[0], mass=mass)
    for k in range(4):
        assert _rel(batched[k], _phi_dense(g, tau, k, w[0])) <= 1e-12
        single = semiflow.phi_action(op, k, tau, w[0], mass=mass)
        assert np.array_equal(single, batched[k]) or _rel(single, batched[k]) <= 1e-13


def _csr_dense(a: np.ndarray):
    rows = [[(j, float(a[i, j])) for j in range(a.shape[1]) if a[i, j] != 0.0] for i in range(a.shape[0])]
    return _csr_from_rows(rows)
