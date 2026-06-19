"""Tests for MagnusGraphHeat6 (Magnus K=6 graph heat, Phase 5)."""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N = 32
T_FINAL = 0.25
N_STEPS = 20
RHO_BAR_MAX = 2.0  # spectral bound for P_n with unit weights


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_magnus6_from_graph_smoke():
    """MagnusGraphHeat6 constructed from Graph evolves without error."""
    g = rp.Graph.path(N)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    f0 = np.ones(N, dtype=np.float64)
    result = mgh6.evolve(T_FINAL, N_STEPS, f0)
    assert result.shape == (N,)
    assert np.all(np.isfinite(result))


def test_magnus6_from_laplacian_smoke():
    """MagnusGraphHeat6 constructed from Laplacian evolves without error."""
    g = rp.Graph.path(N)
    lap = rp.Laplacian.combinatorial(g)

    def lap_at_t(_t: float) -> rp.Laplacian:
        return lap

    mgh6 = rp.MagnusGraphHeat6(laplacian=lap, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    f0 = np.ones(N, dtype=np.float64)
    result = mgh6.evolve(T_FINAL, N_STEPS, f0)
    assert result.shape == (N,)
    assert np.all(np.isfinite(result))


def test_magnus6_no_graph_raises():
    """Neither graph nor laplacian raises SemiflowError(OutOfDomain)."""
    def lap_at_t(_t: float) -> rp.Graph:
        return rp.Graph.path(N)

    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.MagnusGraphHeat6(lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)


def test_magnus6_bad_rho_bar_raises():
    """rho_bar_max <= 0 raises SemiflowError(OutOfDomain)."""
    g = rp.Graph.path(N)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=0.0)


# ---------------------------------------------------------------------------
# Evolve tests
# ---------------------------------------------------------------------------


def test_magnus6_flat_ic_preserved():
    """Flat IC is in null-space of L_G; must be preserved under K=6."""
    g = rp.Graph.path(N)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    f0 = np.ones(N, dtype=np.float64)
    result = mgh6.evolve(T_FINAL, N_STEPS, f0)
    assert np.allclose(result, f0, atol=1e-12)


def test_magnus6_dissipation():
    """Heat evolution must not amplify sup norm."""
    g = rp.Graph.path(N)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    idx = np.arange(N, dtype=np.float64)
    f0 = np.exp(-((idx - N / 2.0) ** 2) / float(N))
    result = mgh6.evolve(T_FINAL, N_STEPS, f0)
    assert float(np.max(np.abs(result))) <= float(np.max(np.abs(f0))) + 1e-12


def test_magnus6_bad_t_raises():
    """t_final < 0 raises SemiflowError(OutOfDomain)."""
    g = rp.Graph.path(N)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        mgh6.evolve(-0.1, N_STEPS, np.ones(N, dtype=np.float64))


def test_magnus6_bad_n_steps_raises():
    """n_steps = 0 raises SemiflowError(OutOfDomain)."""
    g = rp.Graph.path(N)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        mgh6.evolve(T_FINAL, 0, np.ones(N, dtype=np.float64))


def test_magnus6_wrong_len_raises():
    """Wrong-length f0 raises SemiflowError(GridMismatch)."""
    g = rp.Graph.path(N)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    with pytest.raises(rp.SemiflowError, match="GridMismatch"):
        mgh6.evolve(T_FINAL, N_STEPS, np.ones(N + 1, dtype=np.float64))


def test_magnus6_zero_t():
    """evolve(0.0, 1) returns initial condition unchanged."""
    g = rp.Graph.path(N)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    idx = np.arange(N, dtype=np.float64)
    f0 = np.exp(-((idx - N / 2.0) ** 2) / float(N))
    result = mgh6.evolve(0.0, 1, f0)
    assert np.allclose(result, f0, atol=1e-14)


# ---------------------------------------------------------------------------
# Callback accepts Graph / Laplacian / GraphPath
# ---------------------------------------------------------------------------


def test_magnus6_callback_returns_laplacian():
    """lap_at_t can return Laplacian directly."""
    g = rp.Graph.path(N)
    lap = rp.Laplacian.combinatorial(g)

    def lap_at_t(_t: float) -> rp.Laplacian:
        return lap

    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    f0 = np.ones(N, dtype=np.float64)
    result = mgh6.evolve(T_FINAL, N_STEPS, f0)
    assert np.all(np.isfinite(result))


def test_magnus6_callback_returns_graph_path():
    """lap_at_t can return the legacy GraphPath."""
    gp = rp.GraphPath(N)

    def lap_at_t(_t: float) -> rp.GraphPath:
        return gp

    g_new = rp.Graph.path(N)
    mgh6 = rp.MagnusGraphHeat6(graph=g_new, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR_MAX)
    f0 = np.ones(N, dtype=np.float64)
    result = mgh6.evolve(T_FINAL, N_STEPS, f0)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Comparison with MagnusGraphHeat (K=4) for time-independent problem
# ---------------------------------------------------------------------------


def test_magnus6_vs_magnus4_agree():
    """K=6 and K=4 results agree within 1e-2 for time-independent problem."""
    g_new = rp.Graph.path(N)
    g_path = rp.GraphPath(N)

    def lap_at_t_6(_t: float) -> rp.Graph:
        return g_new

    def lap_at_t_4(_t: float) -> rp.GraphPath:
        return g_path

    mgh6 = rp.MagnusGraphHeat6(graph=g_new, lap_at_t=lap_at_t_6, rho_bar_max=RHO_BAR_MAX)
    mgh4 = rp.MagnusGraphHeat(graph=g_path, lap_at_t=lap_at_t_4, rho_bar_max=RHO_BAR_MAX)

    idx = np.arange(N, dtype=np.float64)
    f0 = np.exp(-((idx - N / 2.0) ** 2) / float(N))

    r6 = mgh6.evolve(T_FINAL, N_STEPS, f0)
    r4 = mgh4.evolve(T_FINAL, N_STEPS, f0)

    sup_diff = float(np.max(np.abs(r6 - r4)))
    print(f"K=6 vs K=4 sup_diff={sup_diff:.3e}")
    assert sup_diff < 1e-2, (
        f"MagnusGraphHeat6 and MagnusGraphHeat diverge: sup_diff={sup_diff:.3e}"
    )
