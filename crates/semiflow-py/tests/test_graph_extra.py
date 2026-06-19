"""Tests for Phase 5 graph extras: Graph, Laplacian, GraphHeat4th, VarCoefGraphHeat.

Also covers the extended GraphHeat dual-input constructor (graph= or laplacian=).
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N = 32
T_FINAL = 0.25
N_STEPS = 20
RHO_BAR = 2.0  # combinatorial P_n has spectral radius < 4


# ---------------------------------------------------------------------------
# Graph factory tests
# ---------------------------------------------------------------------------


def test_graph_path_n_nodes():
    """Graph.path(n) has n nodes."""
    g = rp.Graph.path(N)
    assert g.n_nodes == N


def test_graph_path_n_directed_edges():
    """P_n has 2*(n-1) directed half-edges."""
    g = rp.Graph.path(N)
    assert g.n_directed_edges == 2 * (N - 1)


def test_graph_cycle_n_nodes():
    """Graph.cycle(n) has n nodes."""
    g = rp.Graph.cycle(N)
    assert g.n_nodes == N


def test_graph_cycle_n_directed_edges():
    """C_n has 2*n directed half-edges."""
    g = rp.Graph.cycle(N)
    assert g.n_directed_edges == 2 * N


def test_graph_from_edges_triangle():
    """from_edges: triangle graph has 3 nodes, 6 directed edges (flat 1-D input)."""
    edges = np.array([0, 1, 1.0, 1, 2, 1.0, 0, 2, 1.0], dtype=np.float64)
    g = rp.Graph.from_edges(3, edges)
    assert g.n_nodes == 3
    assert g.n_directed_edges == 6


# ---------------------------------------------------------------------------
# GH-4: from_edges shape parity tests (issue #4)
# ---------------------------------------------------------------------------


def test_graph_from_edges_2d_array():
    """GH-4: (M,3) float64 2-D array is accepted (natural layout)."""
    edges_2d = np.array([[0, 1, 1.0], [1, 2, 1.0]], dtype=np.float64)
    g = rp.Graph.from_edges(4, edges_2d)
    assert g.n_nodes == 4
    assert g.n_directed_edges == 4  # 2 undirected → 4 directed


def test_graph_from_edges_2d_matches_flat():
    """GH-4: 2-D (M,3) and equivalent flat 1-D produce identical graphs."""
    edges_2d = np.array([[0, 1, 2.0], [1, 2, 3.0], [0, 2, 1.0]], dtype=np.float64)
    edges_flat = edges_2d.ravel()
    g2d = rp.Graph.from_edges(3, edges_2d)
    gfl = rp.Graph.from_edges(3, edges_flat)
    assert g2d.n_nodes == gfl.n_nodes
    assert g2d.n_directed_edges == gfl.n_directed_edges


def test_graph_from_edges_flat_still_works():
    """GH-4: flat 1-D path remains back-compatible."""
    edges = np.array([0, 1, 1.0, 1, 2, 1.0], dtype=np.float64)
    g = rp.Graph.from_edges(3, edges)
    assert g.n_nodes == 3
    assert g.n_directed_edges == 4


def test_graph_from_edges_wrong_cols_raises():
    """GH-4: (M,2) or (M,4) 2-D array raises TypeError with accurate message."""
    edges_m2 = np.array([[0, 1], [1, 2]], dtype=np.float64)
    with pytest.raises(TypeError, match=r"\(M, 3\)"):
        rp.Graph.from_edges(3, edges_m2)

    edges_m4 = np.array([[0, 1, 1.0, 99], [1, 2, 1.0, 99]], dtype=np.float64)
    with pytest.raises(TypeError, match=r"\(M, 3\)"):
        rp.Graph.from_edges(3, edges_m4)


def test_graph_from_edges_wrong_type_raises():
    """GH-4: passing a string or wrong-dtype array raises TypeError."""
    with pytest.raises(TypeError):
        rp.Graph.from_edges(3, "not an array")  # type: ignore[arg-type]


def test_graph_degree():
    """Path graph interior node has degree 2."""
    g = rp.Graph.path(N)
    assert g.degree(N // 2) == pytest.approx(2.0)


def test_graph_erdos_renyi_deterministic():
    """erdos_renyi with same seed is deterministic."""
    g1 = rp.Graph.erdos_renyi(20, 0.4, 42)
    g2 = rp.Graph.erdos_renyi(20, 0.4, 42)
    assert g1.n_nodes == g2.n_nodes
    assert g1.n_directed_edges == g2.n_directed_edges


# ---------------------------------------------------------------------------
# Laplacian tests
# ---------------------------------------------------------------------------


def test_laplacian_combinatorial_n_nodes():
    """Combinatorial Laplacian has correct n_nodes."""
    g = rp.Graph.path(N)
    lap = rp.Laplacian.combinatorial(g)
    assert lap.n_nodes == N


def test_laplacian_is_combinatorial():
    """combinatorial() sets is_combinatorial=True."""
    g = rp.Graph.path(N)
    lap = rp.Laplacian.combinatorial(g)
    assert lap.is_combinatorial
    assert not lap.is_normalized


def test_laplacian_is_normalized():
    """normalized() sets is_normalized=True."""
    g = rp.Graph.path(N)
    lap = rp.Laplacian.normalized(g)
    assert lap.is_normalized
    assert not lap.is_combinatorial


def test_laplacian_spectral_bound_positive():
    """spectral_bound must be > 0."""
    g = rp.Graph.path(N)
    lap = rp.Laplacian.combinatorial(g)
    assert lap.spectral_bound > 0.0


# ---------------------------------------------------------------------------
# GraphHeat4th tests
# ---------------------------------------------------------------------------


def test_graph_heat4_from_graph_smoke():
    """GraphHeat4th constructed from Graph evolves without error."""
    g = rp.Graph.path(N)
    gh4 = rp.GraphHeat4th(graph=g, rho_bar=RHO_BAR)
    f0 = np.ones(N, dtype=np.float64)
    result = gh4.evolve(T_FINAL, N_STEPS, f0)
    assert result.shape == (N,)
    assert np.all(np.isfinite(result))


def test_graph_heat4_from_laplacian_smoke():
    """GraphHeat4th constructed from Laplacian evolves without error."""
    g = rp.Graph.path(N)
    lap = rp.Laplacian.combinatorial(g)
    gh4 = rp.GraphHeat4th(laplacian=lap, rho_bar=RHO_BAR)
    f0 = np.ones(N, dtype=np.float64)
    result = gh4.evolve(T_FINAL, N_STEPS, f0)
    assert result.shape == (N,)
    assert np.all(np.isfinite(result))


def test_graph_heat4_flat_ic_preserved():
    """Flat IC is in null-space of L_G; must be preserved exactly."""
    g = rp.Graph.path(N)
    gh4 = rp.GraphHeat4th(graph=g, rho_bar=RHO_BAR)
    f0 = np.ones(N, dtype=np.float64)
    result = gh4.evolve(T_FINAL, N_STEPS, f0)
    assert np.allclose(result, f0, atol=1e-12)


def test_graph_heat4_dissipation():
    """Heat evolution must not amplify sup norm."""
    g = rp.Graph.path(N)
    gh4 = rp.GraphHeat4th(graph=g, rho_bar=RHO_BAR)
    idx = np.arange(N, dtype=np.float64)
    f0 = np.exp(-((idx - N / 2.0) ** 2) / float(N))
    result = gh4.evolve(T_FINAL, N_STEPS, f0)
    assert float(np.max(np.abs(result))) <= float(np.max(np.abs(f0))) + 1e-12


def test_graph_heat4_bad_rho_bar_raises():
    """rho_bar <= 0 raises SemiflowError(OutOfDomain)."""
    g = rp.Graph.path(N)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.GraphHeat4th(graph=g, rho_bar=0.0)


def test_graph_heat4_no_input_raises():
    """Neither graph nor laplacian raises SemiflowError(OutOfDomain)."""
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.GraphHeat4th(rho_bar=RHO_BAR)


def test_graph_heat4_wrong_len_raises():
    """Wrong-length f0 raises SemiflowError(GridMismatch)."""
    g = rp.Graph.path(N)
    gh4 = rp.GraphHeat4th(graph=g, rho_bar=RHO_BAR)
    with pytest.raises(rp.SemiflowError, match="GridMismatch"):
        gh4.evolve(T_FINAL, N_STEPS, np.ones(N + 1, dtype=np.float64))


def test_graph_heat4_zero_t():
    """evolve(0.0, 1) returns the initial condition."""
    g = rp.Graph.path(N)
    gh4 = rp.GraphHeat4th(graph=g, rho_bar=RHO_BAR)
    idx = np.arange(N, dtype=np.float64)
    f0 = np.exp(-((idx - N / 2.0) ** 2) / float(N))
    result = gh4.evolve(0.0, 1, f0)
    assert np.allclose(result, f0, atol=1e-14)


# ---------------------------------------------------------------------------
# VarCoefGraphHeat tests
# ---------------------------------------------------------------------------


def test_var_coef_smoke():
    """VarCoefGraphHeat.evolve smoke test with uniform a=1."""
    g = rp.Graph.path(N)
    a = np.ones(N, dtype=np.float64)
    vcgh = rp.VarCoefGraphHeat(g, a, rho_bar=RHO_BAR)
    f0 = np.ones(N, dtype=np.float64)
    result = vcgh.evolve(T_FINAL, N_STEPS, f0)
    assert result.shape == (N,)
    assert np.all(np.isfinite(result))


def test_var_coef_flat_ic():
    """Flat IC preserved under VarCoefGraphHeat (null-space)."""
    g = rp.Graph.path(N)
    a = np.ones(N, dtype=np.float64)
    vcgh = rp.VarCoefGraphHeat(g, a, rho_bar=RHO_BAR)
    f0 = np.ones(N, dtype=np.float64)
    result = vcgh.evolve(T_FINAL, N_STEPS, f0)
    assert np.allclose(result, f0, atol=1e-11)


def test_var_coef_dissipation():
    """VarCoefGraphHeat must not amplify sup norm."""
    g = rp.Graph.path(N)
    a = np.ones(N, dtype=np.float64)
    vcgh = rp.VarCoefGraphHeat(g, a, rho_bar=RHO_BAR)
    idx = np.arange(N, dtype=np.float64)
    f0 = np.exp(-((idx - N / 2.0) ** 2) / float(N))
    result = vcgh.evolve(T_FINAL, N_STEPS, f0)
    assert float(np.max(np.abs(result))) <= float(np.max(np.abs(f0))) + 1e-12


def test_var_coef_bad_rho_bar_raises():
    """rho_bar <= 0 raises SemiflowError(OutOfDomain)."""
    g = rp.Graph.path(N)
    a = np.ones(N, dtype=np.float64)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.VarCoefGraphHeat(g, a, rho_bar=0.0)


def test_var_coef_bad_a_raises():
    """a vector with non-positive entry raises SemiflowError(OutOfDomain)."""
    g = rp.Graph.path(N)
    a = np.ones(N, dtype=np.float64)
    a[5] = -0.1
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.VarCoefGraphHeat(g, a, rho_bar=RHO_BAR)


def test_var_coef_wrong_len_raises():
    """Wrong-length a raises SemiflowError(OutOfDomain)."""
    g = rp.Graph.path(N)
    a = np.ones(N + 1, dtype=np.float64)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.VarCoefGraphHeat(g, a, rho_bar=RHO_BAR)


def test_var_coef_wrong_f0_raises():
    """Wrong-length f0 raises SemiflowError(GridMismatch)."""
    g = rp.Graph.path(N)
    a = np.ones(N, dtype=np.float64)
    vcgh = rp.VarCoefGraphHeat(g, a, rho_bar=RHO_BAR)
    with pytest.raises(rp.SemiflowError, match="GridMismatch"):
        vcgh.evolve(T_FINAL, N_STEPS, np.ones(N + 1, dtype=np.float64))


# ---------------------------------------------------------------------------
# Extended GraphHeat dual-input constructor
# ---------------------------------------------------------------------------


def test_graph_heat_from_laplacian():
    """GraphHeat can be constructed from a Laplacian instead of GraphPath."""
    g = rp.Graph.path(N)
    lap = rp.Laplacian.combinatorial(g)
    gh = rp.GraphHeat(laplacian=lap, rho_bar=RHO_BAR)
    f0 = np.ones(N, dtype=np.float64)
    result = gh.evolve(T_FINAL, N_STEPS, f0)
    assert result.shape == (N,)
    assert np.allclose(result, f0, atol=1e-12)


def test_graph_heat_from_graph_new():
    """GraphHeat can be constructed from the new Graph class (keyword arg)."""
    g = rp.Graph.path(N)
    gh = rp.GraphHeat(graph=g, rho_bar=RHO_BAR)
    f0 = np.ones(N, dtype=np.float64)
    result = gh.evolve(T_FINAL, N_STEPS, f0)
    assert result.shape == (N,)
    assert np.allclose(result, f0, atol=1e-12)


# ---------------------------------------------------------------------------
# Issue #5 — Laplacian introspection: to_dense + CSR accessors
# ---------------------------------------------------------------------------


# Path-3 graph: nodes 0-1-2 with unit weights.
# Combinatorial Laplacian of P_3:
#   L = [[1, -1,  0],
#        [-1,  2, -1],
#        [0, -1,  1]]
_EXPECTED_PATH3_DENSE = np.array(
    [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]],
    dtype=np.float64,
)


def test_to_dense_path3_shape():
    """to_dense() shape is (n, n) for n=3."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    d = lap.to_dense()
    assert d.shape == (3, 3)


def test_to_dense_path3_dtype():
    """to_dense() dtype is float64."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    d = lap.to_dense()
    assert d.dtype == np.float64


def test_to_dense_path3_values():
    """to_dense() matches hand-computed combinatorial Laplacian of P_3."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    d = lap.to_dense()
    assert np.allclose(d, _EXPECTED_PATH3_DENSE, atol=1e-15)


def test_to_dense_cycle4_values():
    """to_dense() for C_4 combinatorial: degree 2 everywhere, off-diag -1 at edges."""
    # C_4: 0-1-2-3-0; L_ii=2, L_ij=-1 for edges, else 0
    expected = np.array(
        [
            [2.0, -1.0, 0.0, -1.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [-1.0, 0.0, -1.0, 2.0],
        ],
        dtype=np.float64,
    )
    g = rp.Graph.cycle(4)
    lap = rp.Laplacian.combinatorial(g)
    d = lap.to_dense()
    assert d.shape == (4, 4)
    assert np.allclose(d, expected, atol=1e-15)


def test_to_dense_normalized_path3():
    """to_dense() works for normalized Laplacian: diagonal entries are 1."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.normalized(g)
    d = lap.to_dense()
    assert d.shape == (3, 3)
    assert d.dtype == np.float64
    # Normalized Laplacian: diagonal = 1 everywhere (D^{-1/2} L D^{-1/2}; D_ii=deg)
    assert np.allclose(np.diag(d), np.ones(3), atol=1e-14)


def test_to_dense_is_copy():
    """to_dense() returns an independent copy; mutation does not affect Laplacian."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    d1 = lap.to_dense()
    d1[0, 0] = 999.0
    d2 = lap.to_dense()
    assert d2[0, 0] == pytest.approx(1.0)


def test_row_ptr_path3_dtype_shape():
    """row_ptr() dtype is int64 and length n+1=4 for n=3."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    rp_ = lap.row_ptr()
    assert rp_.dtype == np.int64
    assert rp_.shape == (4,)


def test_row_ptr_path3_values():
    """row_ptr() starts at 0, ends at nnz for P_3.

    P_3 Laplacian CSR stores all non-zeros including the diagonal:
    row 0: L_00, L_01          → 2 entries
    row 1: L_10, L_11, L_12   → 3 entries
    row 2: L_21, L_22          → 2 entries
    Total nnz = 7 = n + 2*(n-1) = 3 + 4.
    """
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    rp_ = lap.row_ptr()
    assert int(rp_[0]) == 0
    assert int(rp_[-1]) == 7


def test_col_idx_path3_dtype():
    """col_idx() dtype is int64."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    ci = lap.col_idx()
    assert ci.dtype == np.int64


def test_vals_path3_dtype():
    """vals() dtype is float64."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    v = lap.vals()
    assert v.dtype == np.float64


def test_csr_reconstructs_dense():
    """row_ptr + col_idx + vals reconstruct to_dense() exactly for P_5."""
    n = 5
    g = rp.Graph.path(n)
    lap = rp.Laplacian.combinatorial(g)
    dense = lap.to_dense()
    rp_ = lap.row_ptr()
    ci = lap.col_idx()
    v = lap.vals()
    # Reconstruct manually
    mat = np.zeros((n, n), dtype=np.float64)
    for row in range(n):
        for k in range(int(rp_[row]), int(rp_[row + 1])):
            mat[row, int(ci[k])] = v[k]
    assert np.allclose(mat, dense, atol=1e-15)


def test_csr_accessors_are_copies():
    """vals() and col_idx() returns are independent copies."""
    g = rp.Graph.path(3)
    lap = rp.Laplacian.combinatorial(g)
    v1 = lap.vals()
    v1[0] = 999.0
    v2 = lap.vals()
    # Original first entry should be the diagonal value (+1 or +2), not 999
    assert v2[0] != pytest.approx(999.0)


def test_row_ptr_monotone():
    """row_ptr() is non-strictly monotone for an arbitrary graph."""
    g = rp.Graph.erdos_renyi(20, 0.5, 42)
    lap = rp.Laplacian.combinatorial(g)
    rp_ = lap.row_ptr()
    diffs = np.diff(rp_.astype(np.int64))
    assert np.all(diffs >= 0)


def test_combined_combinatorial_and_normalized_csr():
    """CSR accessors work for both Laplacian kinds on C_5."""
    g = rp.Graph.cycle(5)
    for make in (rp.Laplacian.combinatorial, rp.Laplacian.normalized):
        lap = make(g)
        rp_ = lap.row_ptr()
        ci = lap.col_idx()
        v = lap.vals()
        n = lap.n_nodes
        assert rp_.shape == (n + 1,)
        assert ci.shape == v.shape
        # Reconstruct and compare to to_dense
        dense = lap.to_dense()
        mat = np.zeros((n, n), dtype=np.float64)
        for row in range(n):
            for k in range(int(rp_[row]), int(rp_[row + 1])):
                mat[row, int(ci[k])] = v[k]
        assert np.allclose(mat, dense, atol=1e-14)
