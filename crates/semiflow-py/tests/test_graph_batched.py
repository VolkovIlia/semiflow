"""Parity tests for Issue #10 batched evolve wrappers.

Verifies that ``evolve_batched(f0_2d)[:, c] == evolve(f0_2d[:, c])``
bit-exactly (numpy.array_equal, 0-ULP) for every kernel exposed in Step 2.

The batched layout is [N, C] (row-major).  The gather/scatter path must
produce the same floating-point result as calling the single-channel path
C times in ascending channel index (ADR-0184 D1–D4).
"""

from __future__ import annotations

import numpy as np
import pytest

import semiflow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_NODES = 32
N_STEPS = 8
T_FINAL = 0.25
RHO_BAR = float(N_NODES)  # safe Gershgorin bound for path graph

CHANNELS = [1, 4]


@pytest.fixture(scope="module")
def path_graph() -> semiflow.Graph:
    return semiflow.Graph.path(N_NODES)


@pytest.fixture(scope="module")
def laplacian(path_graph: semiflow.Graph) -> semiflow.Laplacian:
    return semiflow.Laplacian.combinatorial(path_graph)


def make_f0_batch(n_nodes: int, n_cols: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_nodes, n_cols))


def check_batched_parity(kernel, f0_batch: np.ndarray) -> None:
    """Assert evolve_batched == per-channel evolve for every column."""
    n_cols = f0_batch.shape[1]
    batched_out = kernel.evolve_batched(T_FINAL, N_STEPS, f0_batch)
    assert batched_out.shape == f0_batch.shape, (
        f"shape mismatch: got {batched_out.shape}, expected {f0_batch.shape}"
    )
    for c in range(n_cols):
        col_in = f0_batch[:, c]
        col_out_single = kernel.evolve(T_FINAL, N_STEPS, col_in)
        assert np.array_equal(batched_out[:, c], col_out_single), (
            f"column {c}: batched != single-channel (not 0-ULP identical)"
        )


# ---------------------------------------------------------------------------
# GraphHeat
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cols", CHANNELS)
def test_graph_heat_evolve_batched(path_graph, n_cols):
    kernel = semiflow.GraphHeat(graph=path_graph, rho_bar=RHO_BAR)
    f0 = make_f0_batch(N_NODES, n_cols)
    check_batched_parity(kernel, f0)


# ---------------------------------------------------------------------------
# MagnusGraphHeat
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cols", CHANNELS)
def test_magnus_graph_heat_evolve_batched(path_graph, n_cols):
    def lap_at_t(_t):
        return path_graph

    kernel = semiflow.MagnusGraphHeat(
        graph=path_graph,
        lap_at_t=lap_at_t,
        rho_bar_max=RHO_BAR,
    )
    f0 = make_f0_batch(N_NODES, n_cols)
    check_batched_parity(kernel, f0)


# ---------------------------------------------------------------------------
# GraphHeat4th
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cols", CHANNELS)
def test_graph_heat4th_evolve_batched(path_graph, n_cols):
    kernel = semiflow.GraphHeat4th(graph=path_graph, rho_bar=RHO_BAR)
    f0 = make_f0_batch(N_NODES, n_cols)
    check_batched_parity(kernel, f0)


# ---------------------------------------------------------------------------
# VarCoefGraphHeat
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cols", CHANNELS)
def test_varcoef_graph_heat_evolve_batched(path_graph, n_cols):
    a = np.ones(N_NODES) * 0.5
    kernel = semiflow.VarCoefGraphHeat(path_graph, a, rho_bar=RHO_BAR)
    f0 = make_f0_batch(N_NODES, n_cols)
    check_batched_parity(kernel, f0)


# ---------------------------------------------------------------------------
# GraphHeat6
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cols", CHANNELS)
def test_graph_heat6_evolve_batched(path_graph, n_cols):
    kernel = semiflow.GraphHeat6(graph=path_graph, rho_bar=RHO_BAR)
    f0 = make_f0_batch(N_NODES, n_cols)
    check_batched_parity(kernel, f0)


# ---------------------------------------------------------------------------
# MagnusGraphHeat6
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cols", CHANNELS)
def test_magnus_graph_heat6_evolve_batched(path_graph, n_cols):
    def lap_at_t(_t):
        return path_graph

    kernel = semiflow.MagnusGraphHeat6(
        graph=path_graph,
        lap_at_t=lap_at_t,
        rho_bar_max=RHO_BAR,
    )
    f0 = make_f0_batch(N_NODES, n_cols)
    check_batched_parity(kernel, f0)


# ---------------------------------------------------------------------------
# VarCoefMagnusGraph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cols", CHANNELS)
def test_varcoef_magnus_graph_evolve_batched(path_graph, n_cols):
    def lap_at_t(_t):
        return path_graph

    def a_at_t(_t):
        return np.ones(N_NODES) * 0.5

    kernel = semiflow.VarCoefMagnusGraph(
        N_NODES,
        lap_at_t=lap_at_t,
        a_at_t=a_at_t,
        rho_bar_max=RHO_BAR,
        a_sup_max=1.0,
    )
    f0 = make_f0_batch(N_NODES, n_cols)
    check_batched_parity(kernel, f0)


# ---------------------------------------------------------------------------
# GraphAdjointPresampled.evolve_state_adjoint_batched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cols", CHANNELS)
def test_adjoint_presampled_batched(path_graph, n_cols):
    def lap_at_t(_t):
        return path_graph

    adj = semiflow.GraphAdjointPresampled.from_presampled(
        graph=path_graph,
        lap_at_t=lap_at_t,
        rho_bar=RHO_BAR,
        n_steps=N_STEPS,
        t_horizon=T_FINAL,
    )
    lam_batch = make_f0_batch(N_NODES, n_cols, seed=7)
    batched_out = adj.evolve_state_adjoint_batched(lam_batch)
    assert batched_out.shape == lam_batch.shape
    for c in range(n_cols):
        single_out = adj.evolve_state_adjoint(lam_batch[:, c])
        assert np.array_equal(batched_out[:, c], single_out), (
            f"adjoint batched column {c}: not 0-ULP identical to single-channel"
        )


# ---------------------------------------------------------------------------
# edge_weight_grad_batched parity (Nit 4a)
# ---------------------------------------------------------------------------

def test_edge_weight_grad_batched_parity(path_graph):
    """edge_weight_grad_batched == ascending-index sum of per-channel edge_weight_grad (C=4)."""
    n_cols = 4
    u0_batch = make_f0_batch(N_NODES, n_cols, seed=11)
    dj_batch = make_f0_batch(N_NODES, n_cols, seed=22)

    batched = semiflow.edge_weight_grad_batched(
        graph=path_graph,
        u0_cols=u0_batch,
        dj_du_n_cols=dj_batch,
        t=T_FINAL,
        n_steps=N_STEPS,
        rho_bar=RHO_BAR,
        params="all_edges",
    )

    # Ascending-index sum of per-channel edge_weight_grad (must match batched exactly)
    summed = np.zeros_like(batched)
    for c in range(n_cols):
        summed += semiflow.edge_weight_grad(
            graph=path_graph,
            u0=u0_batch[:, c],
            dj_du_n=dj_batch[:, c],
            t=T_FINAL,
            n_steps=N_STEPS,
            rho_bar=RHO_BAR,
            params="all_edges",
        )

    assert np.array_equal(batched, summed), (
        "edge_weight_grad_batched not 0-ULP identical to ascending sum of per-channel grads"
    )


# ---------------------------------------------------------------------------
# VarCoef state-adjoint batched parity (Nit 4b)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# dtype="f32" rejection for evolve_batched (Nit 1)
# ---------------------------------------------------------------------------

def test_graph_heat_f32_rejects_evolve_batched(path_graph):
    """GraphHeat(dtype="f32").evolve_batched raises SemiflowError(kind=OutOfDomain)."""
    kernel = semiflow.GraphHeat(graph=path_graph, rho_bar=RHO_BAR, dtype="f32")
    f0 = make_f0_batch(N_NODES, 2)
    with pytest.raises(semiflow.SemiflowError, match="OutOfDomain"):
        kernel.evolve_batched(T_FINAL, N_STEPS, f0)


def test_magnus_graph_heat_f32_rejects_evolve_batched(path_graph):
    """MagnusGraphHeat(dtype="f32").evolve_batched raises SemiflowError(kind=OutOfDomain)."""
    def lap_at_t(_t):
        return path_graph

    kernel = semiflow.MagnusGraphHeat(
        graph=path_graph, lap_at_t=lap_at_t, rho_bar_max=RHO_BAR, dtype="f32"
    )
    f0 = make_f0_batch(N_NODES, 2)
    with pytest.raises(semiflow.SemiflowError, match="OutOfDomain"):
        kernel.evolve_batched(T_FINAL, N_STEPS, f0)


def test_varcoef_graph_heat_f32_rejects_evolve_batched(path_graph):
    """VarCoefGraphHeat(dtype="f32").evolve_batched raises SemiflowError(kind=OutOfDomain)."""
    a = np.ones(N_NODES) * 0.5
    kernel = semiflow.VarCoefGraphHeat(path_graph, a, rho_bar=RHO_BAR, dtype="f32")
    f0 = make_f0_batch(N_NODES, 2)
    with pytest.raises(semiflow.SemiflowError, match="OutOfDomain"):
        kernel.evolve_batched(T_FINAL, N_STEPS, f0)


def test_adjoint_presampled_varcoef_batched(path_graph):
    """VarCoef state-adjoint batched path == per-channel loop (C=4, 0-ULP)."""
    n_cols = 4

    def lap_at_t(_t):
        return path_graph

    def a_at_t(_t):
        return np.ones(N_NODES) * 0.5

    adj = semiflow.GraphAdjointPresampled.from_presampled(
        graph=path_graph,
        lap_at_t=lap_at_t,
        rho_bar=RHO_BAR,
        n_steps=N_STEPS,
        t_horizon=T_FINAL,
        a=a_at_t,
        kernel="varcoef_magnus_graph",
    )
    lam_batch = make_f0_batch(N_NODES, n_cols, seed=13)
    batched_out = adj.evolve_state_adjoint_batched(lam_batch)
    assert batched_out.shape == lam_batch.shape
    for c in range(n_cols):
        single_out = adj.evolve_state_adjoint(lam_batch[:, c])
        assert np.array_equal(batched_out[:, c], single_out), (
            f"varcoef adjoint batched column {c}: not 0-ULP identical to single-channel"
        )
