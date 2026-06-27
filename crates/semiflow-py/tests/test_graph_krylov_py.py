"""
Parity tests for GraphKrylov (A1) and graph_expmv_frechet (A2) — ADR-0185.

Test plan (contract §5c + §5d):
1. forward_parity — batched [N,C] evolve_batched == column-by-column (0-ULP).
2. frechet_fd_triangle — graph_expmv_frechet matches central FD on a 3-node
   triangle graph; rel-err ≤ 1e-6 (mirrors Rust G_GRAPH_FRECHET_FD gate).
"""

from __future__ import annotations

import numpy as np
import pytest

import semiflow

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_NODES = 16          # path graph size for forward parity
T = 0.4               # evolution time
CHANNELS = [1, 4]     # batched widths to exercise


@pytest.fixture(scope="module")
def path_lap() -> semiflow.Laplacian:
    g = semiflow.Graph.path(N_NODES)
    return semiflow.Laplacian.combinatorial(g)


@pytest.fixture(scope="module")
def gk_cheb(path_lap: semiflow.Laplacian) -> semiflow.GraphKrylov:
    return semiflow.GraphKrylov(path_lap, tol=1e-10)


@pytest.fixture(scope="module")
def gk_lanczos(path_lap: semiflow.Laplacian) -> semiflow.GraphKrylov:
    return semiflow.GraphKrylov(path_lap, path="lanczos", tol=1e-10, m_max=18)


# ---------------------------------------------------------------------------
# 1. Forward parity — batched == per-column (0-ULP)
# ---------------------------------------------------------------------------

def _check_forward_parity(gk: semiflow.GraphKrylov, n_nodes: int, n_cols: int) -> None:
    """batched [N,C] == stacked per-column [N,1] calls, exactly (0-ULP)."""
    rng = np.random.default_rng(2025_06_26 + n_cols)
    X = rng.standard_normal((n_nodes, n_cols))

    batched = gk.evolve_batched(T, X)   # [N, C]
    assert batched.shape == (n_nodes, n_cols)

    for c in range(n_cols):
        col_in = X[:, c : c + 1]                      # [N, 1]
        col_out = gk.evolve_batched(T, col_in)[:, 0]  # [N]
        assert np.array_equal(batched[:, c], col_out), (
            f"Chebyshev batched≠single at column {c} (not 0-ULP)"
        )


@pytest.mark.parametrize("n_cols", CHANNELS)
def test_krylov_cheb_evolve_batched_parity(gk_cheb, n_cols):
    _check_forward_parity(gk_cheb, N_NODES, n_cols)


@pytest.mark.parametrize("n_cols", CHANNELS)
def test_krylov_lanczos_evolve_batched_parity(gk_lanczos, n_cols):
    _check_forward_parity(gk_lanczos, N_NODES, n_cols)


def test_krylov_n_nodes(path_lap):
    gk = semiflow.GraphKrylov(path_lap)
    assert gk.n_nodes() == N_NODES


def test_krylov_path_validation(path_lap):
    with pytest.raises(semiflow.SemiflowError, match="OutOfDomain"):
        semiflow.GraphKrylov(path_lap, path="invalid_path")


def test_krylov_tol_validation(path_lap):
    with pytest.raises(semiflow.SemiflowError, match="OutOfDomain"):
        semiflow.GraphKrylov(path_lap, tol=-1.0)


# ---------------------------------------------------------------------------
# 2. Fréchet FD triangle — grad ≤ 1e-6 rel-err vs central FD (G_GRAPH_FRECHET_FD)
# ---------------------------------------------------------------------------

def _build_triangle_lap(w01: float, w12: float, w02: float) -> semiflow.Laplacian:
    """Combinatorial Laplacian for N=3 triangle with given edge weights."""
    edges = np.array(
        [[0, 1, w01], [1, 2, w12], [0, 2, w02]],
        dtype=np.float64,
    )
    g = semiflow.Graph.from_edges(3, edges)
    return semiflow.Laplacian.combinatorial(g)


def _loss(gk: semiflow.GraphKrylov, u0_nc: np.ndarray, dj_nc: np.ndarray, t: float) -> float:
    """J = Σ_c dot(dj_c, e^{-tL} u0_c)."""
    out = gk.evolve_batched(t, u0_nc)  # [N, C]
    return float(np.sum(dj_nc * out))


def test_graph_expmv_frechet_fd_triangle():
    """
    graph_expmv_frechet gradient matches central FD on a triangle (N=3).

    Mirrors the Rust G_GRAPH_FRECHET_FD gate (§54.5, ADR-0185, rel-err ≤ 1e-6).
    """
    N = 3
    T_FD = 0.3
    EPS = 1e-5
    RTOL = 1e-6
    N_COLS = 2

    # Unit-weight triangle: edges (0,1), (1,2), (0,2)
    W0 = (1.0, 1.0, 1.0)
    params_list = [(0, 1), (1, 2), (0, 2)]

    rng = np.random.default_rng(54_05)
    u0 = rng.standard_normal((N, N_COLS))
    dj = rng.standard_normal((N, N_COLS))

    # Fréchet gradient (A2)
    lap_base = _build_triangle_lap(*W0)
    gk_base = semiflow.GraphKrylov(lap_base, tol=1e-12)
    grad_frechet = semiflow.graph_expmv_frechet(
        gk_base, u0, dj, t=T_FD, params=params_list
    )
    assert grad_frechet.shape == (len(params_list),)

    # Central finite difference for each edge
    grad_fd = np.zeros(len(params_list))
    w_vec = list(W0)
    for k in range(len(params_list)):
        w_plus = w_vec.copy()
        w_minus = w_vec.copy()
        w_plus[k] += EPS
        w_minus[k] -= EPS

        lap_p = _build_triangle_lap(*w_plus)
        lap_m = _build_triangle_lap(*w_minus)
        gk_p = semiflow.GraphKrylov(lap_p, tol=1e-12)
        gk_m = semiflow.GraphKrylov(lap_m, tol=1e-12)

        j_plus = _loss(gk_p, u0, dj, T_FD)
        j_minus = _loss(gk_m, u0, dj, T_FD)
        grad_fd[k] = (j_plus - j_minus) / (2.0 * EPS)

    np.testing.assert_allclose(
        grad_frechet, grad_fd,
        rtol=RTOL,
        atol=1e-8,
        err_msg=(
            f"graph_expmv_frechet vs FD: max rel-err = "
            f"{np.max(np.abs(grad_frechet - grad_fd) / (np.abs(grad_fd) + 1e-15)):.2e}"
        ),
    )


def test_graph_expmv_frechet_all_edges_rejected(path_lap):
    """'all_edges' string is explicitly rejected with OutOfDomain."""
    gk = semiflow.GraphKrylov(path_lap)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N_NODES, 2))
    with pytest.raises(semiflow.SemiflowError, match="OutOfDomain"):
        semiflow.graph_expmv_frechet(gk, X, X, t=0.1, params="all_edges")  # type: ignore[arg-type]  # intentional invalid-input test


def test_graph_expmv_frechet_shape_mismatch(path_lap):
    """u0 and dj shape mismatch raises OutOfDomain."""
    gk = semiflow.GraphKrylov(path_lap)
    rng = np.random.default_rng(2)
    u0 = rng.standard_normal((N_NODES, 2))
    dj = rng.standard_normal((N_NODES, 3))   # mismatched C
    with pytest.raises(semiflow.SemiflowError, match="OutOfDomain"):
        semiflow.graph_expmv_frechet(gk, u0, dj, t=0.1, params=[(0, 1)])
