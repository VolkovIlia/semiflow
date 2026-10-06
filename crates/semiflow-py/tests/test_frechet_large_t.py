"""
G_PY_FRECHET_LARGE_T (RELEASE_BLOCKING, ADR-0203, math §63.7).

``symmetric_op_expmv_frechet`` against a Daleckii-Krein eigen oracle
(``numpy.linalg.eigh`` + the cancellation-free form of §63.1.b) on the stiff
fixture F2 at ``lambda_max t in {1, 10, 1e2, 1e4, 1e6}``.

Bound per parameter: ``|g_k - g_k^ref| <= tau_k = (eps_Q + 2 n^2 u) G_k + eta N_k``
with ``tol = 1e-12`` and ``N_chain``, ``m_max`` from a Python replica of the
Rust ``graph_expmv_frechet_plan`` (same mesh, same substep rule, same Bessel
degree rule; the planner itself is deliberately not bound).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import semiflow

N = 12
FAST = 3e3  # W of F2: lambda_max / lambda_min in [5e5, 2e6] (asserted)
TOL = 1e-12
U = 2.0**-53
EPS_Q = 1.1e-14
Z_SAFE = 200.0
GRID = [1.0, 10.0, 1e2, 1e4, 1e6]
X8, W8 = np.polynomial.legendre.leggauss(8)
NODES = (X8 + 1) / 2


def _add_edge(mat: np.ndarray, i: int, j: int, w: float) -> None:
    mat[i, i] += w
    mat[j, j] += w
    mat[i, j] -= w
    mat[j, i] -= w


def stiff_contrast(n: int = N, fast: float = FAST, seed: int = 0) -> np.ndarray:
    """Fixture F2: two fast rings, one slow bridge, a Robin leak (SPD)."""
    rng = np.random.default_rng(seed)
    mat = np.zeros((n, n))
    h = n // 2
    for ring in (list(range(0, h)), list(range(h, n))):
        pairs = list(zip(ring[:-1], ring[1:])) + [(ring[0], ring[-1])]
        for i, j in pairs:
            _add_edge(mat, i, j, fast * rng.uniform(0.5, 1.5))
    _add_edge(mat, h - 1, h, 1.0)
    mat[0, 0] += 0.3
    return mat


def f1(t: float, la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """Cancellation-free divided difference of ``-t e^{-t x}`` (§63.1.b)."""
    lo, hi = np.minimum(la, lb), np.maximum(la, lb)
    th = t * (hi - lo)
    safe = np.where(th > 0, th, 1.0)
    phi = np.where(th > 0, -np.expm1(-th) / safe, 1.0)
    return -t * np.exp(-t * lo) * phi


def stencil(n: int, i: int, j: int) -> np.ndarray:
    m = np.zeros((n, n))
    m[i, j] = 1.0
    m[j, i] = 1.0
    return m


def oracle(lam, vec, u0, dj, t, entries):
    """``(g_k, G_k)`` of §63.1.b-c for symmetric entry directions."""
    al, be = vec.T @ dj, vec.T @ u0
    big_f = f1(t, lam[:, None], lam[None, :])
    g, gabs = [], []
    for i, j in entries:
        mt = vec.T @ stencil(len(lam), i, j) @ vec
        terms = al[:, None] * mt * be[None, :] * big_f
        g.append(terms.sum())
        gabs.append(np.abs(terms).sum())
    return np.array(g), np.array(gabs)


# --- Python replica of graph_expmv_frechet_plan (Chebyshev path) -----------------


def bessel_i(k: int, z: float) -> float:
    """Modified Bessel ``I_k(z)`` by its power series (as in the Rust core)."""
    half = z / 2.0
    term = half**k / math.factorial(k)
    total = term
    for m in range(1000):
        term *= half * half / ((m + 1) * (m + 1 + k))
        if total + term == total:
            break
        total += term
    return total


def cheb_degree(z: float, tol: float) -> int:
    """Smallest m >= 3 with ``e^{-z} I_{m+1}(z) <= tol/4`` (cap 200)."""
    m, em_z = 3, math.exp(-z)
    while m < 200 and em_z * bessel_i(m + 1, z) > tol / 4.0:
        m += 1
    return m


def action_cost(rho: float, tau: float, tol: float) -> tuple[int, int]:
    """``(substeps, degree)`` of one Chebyshev action (``graph_expmv_matvec_count``)."""
    z_total = tau * rho / 2.0
    s = 1 if z_total <= Z_SAFE else math.ceil(z_total / Z_SAFE)
    return s, cheb_degree(tau / s * rho / 2.0, tol)


def half_panels(half: float, rho: float) -> list[tuple[float, float]]:
    """§63.3 mesh by the multiply loop (no logarithm)."""
    d0 = min(half, 2.0 / rho) if rho > 0 else half
    edges = [0.0, d0]
    while edges[-1] < half:
        edges.append(min(edges[-1] * 1.5, half))
    return [(lo, hi - lo) for lo, hi in zip(edges[:-1], edges[1:])]


def plan_chain(rho: float, t: float, tol: float) -> tuple[int, int]:
    """``(N_chain, m_max)`` of the §63.5 sweep."""
    half = t / 2.0
    s0, m0 = action_cost(rho, half, tol)
    far, near_max, m_max, r_far = s0, 0, m0, half
    for lo, h in reversed(half_panels(half, rho)):
        r = lo + h * NODES
        s, m = action_cost(rho, r[0], tol)
        near, m_max = s, max(m_max, m)
        for q in range(1, 8):
            s, m = action_cost(rho, r[q] - r[q - 1], tol)
            near, m_max = near + s, max(m_max, m)
        near_max = max(near_max, near)
        for q in reversed(range(8)):
            step, r_far = r_far - r[q], r[q]
            if step > 0:
                s, m = action_cost(rho, step, tol)
                far, m_max = far + s, max(m_max, m)
    return max(far, near_max), m_max


def tau_bound(g_abs, norm_k, nd, nv, n_chain, m_max, row_nnz, rho_t, t):
    """``tau_k`` of §63.7.a."""
    eta = (
        2.0 * n_chain * (TOL + (row_nnz + 3.0) * m_max**2 * U)
        + TOL
        + (row_nnz + N) * U * rho_t
    )
    return (EPS_Q + 2.0 * N * N * U) * g_abs + eta * t * norm_k * nd * nv


def _operator(mat: np.ndarray) -> semiflow.SymmetricOperator:
    n = mat.shape[0]
    indptr, indices, data = [0], [], []
    for row in mat:
        for j, x in enumerate(row):
            if x != 0.0:
                indices.append(j)
                data.append(x)
        indptr.append(len(indices))
    return semiflow.SymmetricOperator.from_csr(
        np.array(indptr, dtype=np.int64),
        np.array(indices, dtype=np.int32),
        np.array(data),
        n,
        1e-10,
    )


def _entries() -> list[tuple[int, int]]:
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    return pairs + [(0, 0), (N // 2, N // 2)]


def f3_matrix(w: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Fixture F3 (Amendment 1): four triangle clusters, three bridges.

    Returns the Laplacian and the 15 edge pairs: 12 intra-cluster pairs, then the
    three bridges ``(3c+2, 3c+3)``. Deterministic, identical to the Rust gate.
    """
    mat = np.zeros((N, N))
    pairs: list[tuple[int, int]] = []
    for c in range(4):
        a, b, d = 3 * c, 3 * c + 1, 3 * c + 2
        for e, (i, j) in enumerate([(a, b), (b, d), (a, d)]):
            _add_edge(mat, i, j, w * (0.5 + ((7 * (3 * c + e)) % 11) / 10.0))
            pairs.append((i, j))
    for c, weight in enumerate([0.6, 1.0, 1.4]):
        _add_edge(mat, 3 * c + 2, 3 * c + 3, weight)
        pairs.append((3 * c + 2, 3 * c + 3))
    return mat, pairs


def f3_signals() -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(N)
    cluster = i // 3
    u0 = cluster - 1.5 + 0.1 * np.sin(1.7 * i + 0.3)
    dj = (-1.0) ** cluster + 0.1 * np.cos(2.3 * i)
    return u0, dj


def f3_point(target: float) -> tuple[float, float]:
    """Per-point ``(W, t)``: ``lambda_2 t = 1`` once ``target >= R1`` (Amendment 1)."""

    def spectrum(w: float) -> np.ndarray:
        return np.linalg.eigvalsh(f3_matrix(w)[0])

    lam = spectrum(1.0)
    if target < lam[-1] / lam[1]:
        return 1.0, target / lam[-1]
    lo, hi = 0.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lam = spectrum(math.exp(mid))
        lo, hi = (mid, hi) if lam[-1] / lam[1] < target else (lo, mid)
    w = math.exp(0.5 * (lo + hi))
    lam = spectrum(w)
    assert abs(lam[-1] / lam[1] / target - 1.0) <= 1e-3
    return w, 1.0 / lam[1]


def _check_point(mat, entries, u0, dj, t, lt, bridges):
    """Compare the binding with the oracle at one point.

    Returns ``(violations, informative-count-inside-bridges, max err/tau)``.
    """
    lam, vec = np.linalg.eigh(mat)
    row_nnz = int(np.max((mat != 0).sum(axis=1)))
    rho = float(np.max(np.abs(mat).sum(axis=1)))  # Gershgorin, as in Rust
    g_ref, g_abs = oracle(lam, vec, u0, dj, t, entries)
    n_chain, m_max = plan_chain(rho, t, TOL)
    tau = tau_bound(
        g_abs, np.ones(len(entries)), np.linalg.norm(dj), np.linalg.norm(u0),
        n_chain, m_max, row_nnz, rho * t, t,
    )
    got = semiflow.symmetric_op_expmv_frechet(
        _operator(mat), u0[:, None], dj[:, None], t=t, entries=entries, tol=TOL
    )
    err = np.abs(got - g_ref)
    informative = np.abs(g_ref) >= 1e3 * tau
    bad = np.flatnonzero(~(err <= tau))  # NaN-safe: NaN counts as a violation
    fails = [f"lt={lt:e} k={k} err={err[k]:.3e} > tau={tau[k]:.3e}" for k in bad]
    return fails, int(np.sum(informative[bridges])), float(np.max(err / tau))


def test_frechet_large_t_vs_eigh() -> None:
    """Carrier F2-entry: all entry pairs + two diagonals, >= 1 informative per point."""
    mat = stiff_contrast()
    lam = np.linalg.eigvalsh(mat)
    ratio = lam[-1] / lam[0]
    assert 5e5 <= ratio <= 2e6, f"F2 stiffness ratio {ratio:e} outside [5e5, 2e6]"
    rng = np.random.default_rng(22)
    u0, dj = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N)
    entries = _entries()
    failures = []
    for lt in GRID:
        fails, informative, worst = _check_point(
            mat, entries, u0, dj, lt / lam[-1], lt, slice(None)
        )
        print(f"F2-entry lt={lt:7.0e} max err/tau={worst:9.2e} informative={informative}")
        if informative < 1:
            fails.append(f"F2-entry lt={lt:e}: no informative parameter (vacuous)")
        failures += fails
    assert not failures, f"{len(failures)} violations, first: {failures[:5]}"


def test_frechet_large_t_f3_bridges() -> None:
    """Carrier F3-bridge (Amendment 1): >= 2 of 3 bridge pairs informative per point."""
    u0, dj = f3_signals()
    failures = []
    for lt in GRID:
        w, t = f3_point(lt)
        mat, pairs = f3_matrix(w)
        fails, informative, worst = _check_point(
            mat, pairs, u0, dj, t, lt, slice(12, 15)
        )
        print(
            f"F3-bridge lt={lt:7.0e} W={w:.4e} t={t:.4e} "
            f"max err/tau={worst:9.2e} informative bridges={informative}/3"
        )
        if informative < 2:
            fails.append(f"F3 lt={lt:e}: {informative}/3 bridge pairs informative (vacuous)")
        failures += fails
    assert not failures, f"{len(failures)} violations, first: {failures[:5]}"


def test_oracle_matches_scipy_expm_frechet() -> None:
    """Oracle self-check: the eigen oracle agrees with scipy to 1e-12 G_k (lt <= 1e2)."""
    sla = pytest.importorskip("scipy.linalg")
    mat = stiff_contrast()
    lam, vec = np.linalg.eigh(mat)
    rng = np.random.default_rng(22)
    u0, dj = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N)
    entries = _entries()
    for lt in (1.0, 10.0, 1e2):
        t = lt / lam[-1]
        g_ref, g_abs = oracle(lam, vec, u0, dj, t, entries)
        for k in (0, 7, 30, len(entries) - 1):
            i, j = entries[k]
            _, fr = sla.expm_frechet(-t * mat, -t * stencil(N, i, j))
            assert abs(dj @ fr @ u0 - g_ref[k]) <= 1e-12 * g_abs[k], (lt, k)


@pytest.mark.parametrize("bad_tol", [0.0, -1e-9, float("nan"), float("inf")])
def test_tol_keyword_is_validated(bad_tol: float) -> None:
    op = _operator(stiff_contrast())
    x = np.ones((N, 1))
    with pytest.raises(semiflow.SemiflowError, match="OutOfDomain"):
        semiflow.symmetric_op_expmv_frechet(op, x, x, t=0.1, entries=[(0, 1)], tol=bad_tol)


def test_tol_default_matches_explicit() -> None:
    op = _operator(stiff_contrast())
    rng = np.random.default_rng(5)
    u0, dj = rng.uniform(-1, 1, (N, 1)), rng.uniform(-1, 1, (N, 1))
    kw = {"t": 1e-4, "entries": [(0, 1), (0, 0)]}
    a = semiflow.symmetric_op_expmv_frechet(op, u0, dj, **kw)
    b = semiflow.symmetric_op_expmv_frechet(op, u0, dj, tol=1e-12, **kw)
    assert np.array_equal(a, b)
