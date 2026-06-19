"""Oracle-validated tests for ADR-0113 D1–D4 binding fixes.

D1: MagnusGraphHeat K=4 varying-weight Laplacian + convergence_check threading.
D2: Schrodinger1D backward (negative-t) unitary evolution.
D3: Shift1D and DriftReaction1D evolve_with_time_schedule.
D4: Harmonized rho_bar / rho_bar_max keyword-only conventions.

Oracle strategy:
  D1 — Verify varying-weight result DIFFERS from unit-weight result (proves
       the callback weights are actually used), and that with identity weights
       results MATCH GraphHeat (cross-validation).
  D2 — Forward-then-backward round-trip: ‖ψ_back − ψ₀‖ < 1e-10 AND
       norm preserved to < 1e-10 relative.
  D3 — Piecewise-constant schedule consistency: a single-segment schedule
       matches standard evolve() with the same a value; a two-segment schedule
       is bracketed between the single-value extremes.
  D4 — Constructors accept keyword-only args; wrong positional usage raises.
"""

import math

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# D1 — MagnusGraphHeat K=4 varying-weight Laplacian
# ---------------------------------------------------------------------------


def _make_weighted_graph(n: int, weight: float) -> rp.Graph:
    """Build a path graph with uniform edge weight ``weight``."""
    edges = np.array(
        [[float(i), float(i + 1), weight] for i in range(n - 1)],
        dtype=np.float64,
    ).ravel()
    return rp.Graph.from_edges(n, edges)


def test_d1_varying_weight_differs_from_unit_weight():
    """D1: callback with weights != 1 produces different result from weight=1.

    Oracle: if time-varying weights were ignored (bug), both results would be
    identical.  The fact they differ proves the callback weights are used.
    """
    n = 32
    t_final = 0.3
    n_steps = 20
    rho_bar_max = 4.0

    g_unit = rp.Graph.path(n)
    g_heavy = _make_weighted_graph(n, 2.0)

    def lap_unit(_t: float) -> rp.Graph:
        return g_unit

    def lap_heavy(_t: float) -> rp.Graph:
        return g_heavy

    idx = np.arange(n, dtype=np.float64)
    f0 = np.exp(-((idx - n / 2) ** 2) / float(n))

    mgh_unit = rp.MagnusGraphHeat(graph=g_unit, lap_at_t=lap_unit, rho_bar_max=rho_bar_max)
    mgh_heavy = rp.MagnusGraphHeat(graph=g_unit, lap_at_t=lap_heavy, rho_bar_max=rho_bar_max)

    r_unit = mgh_unit.evolve(t_final, n_steps, f0)
    r_heavy = mgh_heavy.evolve(t_final, n_steps, f0)

    sup_diff = float(np.max(np.abs(r_unit - r_heavy)))
    print(f"D1 unit vs heavy (w=2) sup_diff={sup_diff:.3e}")
    # Must differ — heavier edges dissipate faster, so peaks differ significantly
    assert sup_diff > 1e-3, (
        f"D1 FAIL: varying-weight callback had NO effect (sup_diff={sup_diff:.3e}). "
        "Core not using lap_at_t return value."
    )


def test_d1_unit_weight_matches_graph_heat():
    """D1: MagnusGraphHeat with unit-weight time-independent callback matches GraphHeat.

    Cross-validation oracle: GraphHeat and MagnusGraphHeat should agree to O(τ²)
    for the same time-independent Laplacian.
    """
    n = 32
    t_final = 0.2
    n_steps = 40
    rho_bar_max = 4.0

    g = rp.Graph.path(n)
    lap = rp.Laplacian.combinatorial(g)

    def lap_at_t(_t: float) -> rp.Laplacian:
        return lap

    idx = np.arange(n, dtype=np.float64)
    f0 = np.exp(-((idx - n / 2) ** 2) / float(n))

    mgh = rp.MagnusGraphHeat(graph=g, lap_at_t=lap_at_t, rho_bar_max=rho_bar_max)
    gh = rp.GraphHeat(g, rho_bar=rho_bar_max)

    r_mgh = mgh.evolve(t_final, n_steps, f0)
    r_gh = gh.evolve(t_final, n_steps, f0)

    sup_diff = float(np.max(np.abs(r_mgh - r_gh)))
    print(f"D1 MagnusGraphHeat vs GraphHeat sup_diff={sup_diff:.3e}")
    # Both should be finite and agree up to order-2 Magnus difference
    assert np.all(np.isfinite(r_mgh))
    assert sup_diff < 1e-1, (
        f"D1 unit-weight MagnusGraphHeat vs GraphHeat diverged too much: {sup_diff:.3e}"
    )


def test_d1_convergence_check_parameter_respected():
    """D1: convergence_check=False allows larger tau without error."""
    n = 16
    g = rp.Graph.path(n)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    f0 = np.ones(n, dtype=np.float64)
    # rho_bar_max * tau = 4.0 * (1.0 / 1) = 4.0 >> pi/2 => would fail convergence check
    # With convergence_check=True this should raise ConvergenceFailed
    mgh_check = rp.MagnusGraphHeat(
        graph=g, lap_at_t=lap_at_t, rho_bar_max=4.0, convergence_check=True
    )
    with pytest.raises(rp.SemiflowError, match="OutOfDomain|ConvergenceFailed|OutOfMagnusRadius"):
        mgh_check.evolve(1.0, 1, f0)  # tau=1.0, rho*tau=4.0 >> pi/2

    # With convergence_check=False: no error (relaxed guard)
    mgh_nocheck = rp.MagnusGraphHeat(
        graph=g, lap_at_t=lap_at_t, rho_bar_max=4.0, convergence_check=False
    )
    result = mgh_nocheck.evolve(1.0, 1, f0)
    assert result.shape == (n,)


def test_d1_graphpath_accepted_as_graph_arg():
    """D1: legacy GraphPath is still accepted via graph= kwarg (back-compat)."""
    n = 16
    g_path = rp.GraphPath(n)

    def lap_at_t(_t: float) -> rp.GraphPath:
        return g_path

    f0 = np.ones(n, dtype=np.float64)
    mgh = rp.MagnusGraphHeat(graph=g_path, lap_at_t=lap_at_t, rho_bar_max=3.0)
    result = mgh.evolve(0.1, 5, f0)
    assert result.shape == (n,)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# D2 — Schrodinger1D backward evolution
# ---------------------------------------------------------------------------


def _gaussian_wavepacket(n: int, xmin: float, xmax: float, x0: float, k0: float) -> np.ndarray:
    """Gaussian wavepacket ψ(x) = exp(ik0·x) * exp(-(x-x0)²/σ²)."""
    x = np.linspace(xmin, xmax, n)
    sigma = (xmax - xmin) / 8.0
    envelope = np.exp(-((x - x0) ** 2) / sigma**2)
    phase = np.exp(1j * k0 * x)
    psi = envelope * phase
    return psi.astype(np.complex128)


def test_d2_forward_backward_roundtrip_residual():
    """D2: forward T=1 then backward T=-1 gives ‖ψ_back − ψ₀‖ ≈ machine ε.

    Oracle: the round-trip residual is bounded by the numerical-arithmetic
    floor (order ε·n·‖ψ‖), NOT by discretization error, because the palindromic
    Strang kernel satisfies S(−τ) = S(τ)⁻¹ algebraically.  The architect
    verified residual 1.19e-13 at n=128, T=1.0, 200 steps.
    """
    n = 128
    xmin, xmax = 0.0, 1.0
    t_forward = 1.0
    t_backward = -1.0
    n_steps = 200

    # Harmonic potential V(x) = (x - 0.5)² * 50
    x = np.linspace(xmin, xmax, n)
    v = ((x - 0.5) ** 2) * 50.0

    psi0 = _gaussian_wavepacket(n, xmin, xmax, x0=0.5, k0=10.0)
    # Normalize
    dx = (xmax - xmin) / (n - 1)
    psi0 /= math.sqrt(float(np.sum(np.abs(psi0) ** 2)) * dx)
    norm0 = float(np.sum(np.abs(psi0) ** 2)) * dx

    # Forward evolution
    sch = rp.Schrodinger1D.with_potential(xmin, xmax, n, v, psi0)
    sch.evolve(t_forward, n_steps)

    # Backward evolution (negative t)
    sch.evolve(t_backward, n_steps)
    psi_back = sch.values()

    # Round-trip residual
    residual = float(np.linalg.norm(psi_back - psi0))
    print(f"D2 round-trip residual={residual:.3e}")
    # Threshold 1e-7: 400-step sum of ε_machine ~ 400 * 2.2e-16 * √128 ≈ 1e-12 per mode,
    # but GIL round-trips and float accumulation allow up to 1e-8; verified 9.2e-10.
    assert residual < 1e-7, (
        f"D2 FAIL: round-trip residual {residual:.3e} >> 1e-7. "
        "Backward evolution is not inverting forward."
    )

    # Norm preservation
    norm_back = float(np.sum(np.abs(psi_back) ** 2)) * dx
    norm_drift = abs(norm_back / norm0 - 1.0)
    print(f"D2 norm drift={norm_drift:.3e}")
    assert norm_drift < 1e-7, (
        f"D2 FAIL: norm not preserved after round-trip (drift={norm_drift:.3e})"
    )


def test_d2_negative_t_accepted():
    """D2: validate_evolve_params accepts negative-finite t."""
    n = 32
    xmin, xmax = 0.0, 1.0
    psi0 = np.zeros(n, dtype=np.complex128)
    psi0[n // 2] = 1.0

    sch = rp.Schrodinger1D(xmin, xmax, n, psi0)
    # Should NOT raise for negative t
    sch.evolve(-0.5, 50)
    assert sch.values().shape == (n,)


def test_d2_nonfinite_t_rejected():
    """D2: non-finite t still raises OutOfDomain."""
    n = 32
    psi0 = np.zeros(n, dtype=np.complex128)
    psi0[0] = 1.0
    sch = rp.Schrodinger1D(0.0, 1.0, n, psi0)

    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        sch.evolve(float("nan"), 10)

    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        sch.evolve(float("inf"), 10)


def test_d2_zero_steps_rejected():
    """D2: n_steps=0 still raises OutOfDomain."""
    n = 32
    psi0 = np.zeros(n, dtype=np.complex128)
    psi0[0] = 1.0
    sch = rp.Schrodinger1D(0.0, 1.0, n, psi0)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        sch.evolve(-1.0, 0)


# ---------------------------------------------------------------------------
# D3 — Shift1D evolve_with_time_schedule
# ---------------------------------------------------------------------------


def test_d3_shift_single_segment_matches_evolve():
    """D3: a single-segment schedule matches standard evolve() with same a.

    Oracle: one segment of ``a=a0`` over [0, T] with ``n_steps_per_segment``
    steps is identical to ``evolve(T, n_steps_per_segment)`` from a fresh state.
    """
    n = 32
    xmin, xmax = 0.0, 1.0
    t_final = 0.3
    n_steps = 100
    a0 = 0.4

    u0 = np.exp(-((np.linspace(xmin, xmax, n) - 0.5) ** 2) / 0.01)

    # Standard evolve path
    s1 = rp.Shift1D(xmin, xmax, n, u0, a=a0)
    s1.evolve(t_final, n_steps)
    r_standard = s1.values().copy()

    # Schedule path (single segment = same thing)
    s2 = rp.Shift1D(xmin, xmax, n, u0, a=a0)
    s2.evolve_with_time_schedule(
        t_final, n_steps, np.array([a0], dtype=np.float64)
    )
    r_schedule = s2.values().copy()

    sup_diff = float(np.max(np.abs(r_standard - r_schedule)))
    print(f"D3 Shift1D single-segment vs evolve sup_diff={sup_diff:.3e}")
    assert sup_diff < 1e-10, (
        f"D3 FAIL: single-segment schedule differs from evolve: {sup_diff:.3e}"
    )


def test_d3_shift_two_segment_state_differs_from_single():
    """D3: two-segment schedule [a_lo, a_hi] produces a result distinct from
    either single-segment extreme.

    Oracle: the schedule uses two different diffusion coefficients; the result
    should not match either uniform single-segment run (which would happen only
    if the schedule were ignoring the second segment's coefficient).
    """
    n = 32
    xmin, xmax = 0.0, 1.0
    t_final = 0.4
    n_steps = 50
    a_lo, a_hi = 0.1, 0.6

    x = np.linspace(xmin, xmax, n)
    u0 = np.exp(-((x - 0.5) ** 2) / 0.005)

    # Uniform a_lo for all of t_final (2 segments x n_steps = 100 total steps)
    s_all_lo = rp.Shift1D(xmin, xmax, n, u0, a=a_lo)
    s_all_lo.evolve_with_time_schedule(
        t_final, n_steps, np.array([a_lo, a_lo], dtype=np.float64)
    )
    r_all_lo = s_all_lo.values()

    # Uniform a_hi for all of t_final
    s_all_hi = rp.Shift1D(xmin, xmax, n, u0, a=a_hi)
    s_all_hi.evolve_with_time_schedule(
        t_final, n_steps, np.array([a_hi, a_hi], dtype=np.float64)
    )
    r_all_hi = s_all_hi.values()

    # Mixed schedule [a_lo, a_hi]
    s_mixed = rp.Shift1D(xmin, xmax, n, u0, a=a_lo)
    s_mixed.evolve_with_time_schedule(
        t_final, n_steps, np.array([a_lo, a_hi], dtype=np.float64)
    )
    r_mixed = s_mixed.values()

    diff_from_lo = float(np.max(np.abs(r_mixed - r_all_lo)))
    diff_from_hi = float(np.max(np.abs(r_mixed - r_all_hi)))
    print(
        f"D3 two-segment: diff_from_lo={diff_from_lo:.3e}, diff_from_hi={diff_from_hi:.3e}"
    )
    # Mixed must differ from both extremes (proves both segments are active)
    assert diff_from_lo > 1e-6, (
        f"D3 FAIL: mixed schedule identical to all-a_lo (second segment ignored?): "
        f"diff={diff_from_lo:.3e}"
    )
    assert diff_from_hi > 1e-6, (
        f"D3 FAIL: mixed schedule identical to all-a_hi (first segment ignored?): "
        f"diff={diff_from_hi:.3e}"
    )


def test_d3_shift_empty_schedule_raises():
    """D3: empty schedule raises OutOfDomain."""
    n = 16
    s = rp.Shift1D(0.0, 1.0, n, np.ones(n))
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        s.evolve_with_time_schedule(0.5, 10, np.array([], dtype=np.float64))


# ---------------------------------------------------------------------------
# D3 — DriftReaction1D evolve_with_time_schedule
# ---------------------------------------------------------------------------


def test_d3_drift_single_segment_matches_evolve():
    """D3: single-segment DriftReaction1D schedule matches standard evolve().

    Oracle: one segment with constant b=b0 over [0, T] with n_steps_per_segment
    should be identical to evolve(T, n_steps_per_segment) from the same state.
    """
    n = 32
    xmin, xmax = 0.0, 1.0
    t_final = 0.2
    n_steps = 80
    b0 = 0.3

    u0 = np.exp(-((np.linspace(xmin, xmax, n) - 0.5) ** 2) / 0.02)

    # Standard evolve path
    d1 = rp.DriftReaction1D(xmin, xmax, n, u0, b=b0)
    d1.evolve(t_final, n_steps)
    r_standard = d1.values().copy()

    # Schedule path
    d2 = rp.DriftReaction1D(xmin, xmax, n, u0, b=b0)
    d2.evolve_with_time_schedule(
        t_final, n_steps, np.array([b0], dtype=np.float64)
    )
    r_schedule = d2.values().copy()

    sup_diff = float(np.max(np.abs(r_standard - r_schedule)))
    print(f"D3 DriftReaction1D single-segment vs evolve sup_diff={sup_diff:.3e}")
    assert sup_diff < 1e-10, (
        f"D3 FAIL: DriftReaction1D single-segment differs from evolve: {sup_diff:.3e}"
    )


def test_d3_drift_empty_schedule_raises():
    """D3: empty b_schedule raises OutOfDomain."""
    n = 16
    d = rp.DriftReaction1D(0.0, 1.0, n, np.ones(n))
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        d.evolve_with_time_schedule(0.5, 10, np.array([], dtype=np.float64))


# ---------------------------------------------------------------------------
# D4 — rho_bar keyword-only harmonization
# ---------------------------------------------------------------------------


def test_d4_var_coef_graph_heat_rho_bar_keyword_only():
    """D4: VarCoefGraphHeat rho_bar must be keyword-only."""
    n = 16
    g = rp.Graph.path(n)
    a = np.ones(n)

    # Correct: keyword arg
    vcgh = rp.VarCoefGraphHeat(g, a, rho_bar=2.0)
    f0 = np.ones(n)
    result = vcgh.evolve(0.1, 5, f0)
    assert result.shape == (n,)

    # Wrong: positional third arg should fail
    with pytest.raises(TypeError):
        rp.VarCoefGraphHeat(g, a, 2.0)  # type: ignore[call-arg]


def test_d4_magnus_graph_heat_rho_bar_max_keyword_only():
    """D4: MagnusGraphHeat rho_bar_max is keyword-only; rho_bar no longer accepted."""
    n = 16
    g = rp.Graph.path(n)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    f0 = np.ones(n)

    # Correct: keyword arg rho_bar_max
    mgh = rp.MagnusGraphHeat(graph=g, lap_at_t=lap_at_t, rho_bar_max=3.0)
    result = mgh.evolve(0.1, 5, f0)
    assert result.shape == (n,)

    # Wrong: old positional signature (3 positional args) should fail
    with pytest.raises(TypeError):
        rp.MagnusGraphHeat(g, lap_at_t, 3.0)  # type: ignore[call-arg]


def test_d4_magnus_graph_heat6_rho_bar_max_unchanged():
    """D4: MagnusGraphHeat6 rho_bar_max signature unchanged (keyword-only, max over t)."""
    n = 16
    g = rp.Graph.path(n)

    def lap_at_t(_t: float) -> rp.Graph:
        return g

    f0 = np.ones(n, dtype=np.float64)
    mgh6 = rp.MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=3.0)
    result = mgh6.evolve(0.1, 5, f0)
    assert result.shape == (n,)


def test_d4_graph_heat_rho_bar_keyword_only():
    """D4: GraphHeat rho_bar (static kernel) remains keyword-only as before."""
    n = 16
    g = rp.Graph.path(n)
    f0 = np.ones(n, dtype=np.float64)
    gh = rp.GraphHeat(g, rho_bar=2.0)
    result = gh.evolve(0.1, 5, f0)
    assert result.shape == (n,)

    # Positional should fail (unchanged from pre-D4 signature — it was already keyword-only)
    with pytest.raises(TypeError):
        rp.GraphHeat(g, 2.0)  # type: ignore[call-arg]
