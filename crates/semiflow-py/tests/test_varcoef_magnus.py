"""Tests for `VarCoefMagnusGraph` Python binding (v2.5 Phase 1, ADR-0063).

Mirrors `test_magnus6.py` pattern + parity vs MagnusGraphHeat (K=4) when
a(t) ≡ 1.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import semiflow as rp


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _make_constant(n: int):
    """Helpers for constant lap_at_t(t) and a_at_t(t)."""
    g = rp.Graph.path(n)

    def lap_at_t(_t: float):
        return g

    def a_at_t(_t: float):
        return np.ones(n)

    return g, lap_at_t, a_at_t


def test_construct_ok():
    n = 16
    _g, lap_at_t, a_at_t = _make_constant(n)
    vcm = rp.VarCoefMagnusGraph(
        n_nodes=n, lap_at_t=lap_at_t, a_at_t=a_at_t,
        rho_bar_max=4.0, a_sup_max=1.0,
    )
    assert vcm.n_nodes == n


def test_construct_rejects_zero_nodes():
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.VarCoefMagnusGraph(
            n_nodes=0, lap_at_t=lambda t: rp.Graph.path(1),
            a_at_t=lambda t: np.ones(0),
            rho_bar_max=4.0, a_sup_max=1.0,
        )


def test_construct_rejects_bad_rho():
    _g, lap_at_t, a_at_t = _make_constant(8)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.VarCoefMagnusGraph(
            n_nodes=8, lap_at_t=lap_at_t, a_at_t=a_at_t,
            rho_bar_max=-1.0, a_sup_max=1.0,
        )


def test_construct_rejects_bad_a_sup():
    _g, lap_at_t, a_at_t = _make_constant(8)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.VarCoefMagnusGraph(
            n_nodes=8, lap_at_t=lap_at_t, a_at_t=a_at_t,
            rho_bar_max=4.0, a_sup_max=0.0,
        )


# ---------------------------------------------------------------------------
# Basic evolution
# ---------------------------------------------------------------------------


def test_evolve_preserves_sum_constant_a():
    """With a ≡ 1 and combinatorial L_G, Σ f_i is conserved."""
    n = 16
    _g, lap_at_t, a_at_t = _make_constant(n)
    vcm = rp.VarCoefMagnusGraph(
        n_nodes=n, lap_at_t=lap_at_t, a_at_t=a_at_t,
        rho_bar_max=4.0, a_sup_max=1.0,
    )
    u0 = np.exp(-((np.arange(n) - n / 2.0) ** 2) / (n / 4.0))
    u1 = vcm.evolve(t_final=0.2, n_steps=10, f0=u0)
    assert abs(u1.sum() - u0.sum()) < 1e-10


def test_evolve_dissipates_variance():
    n = 32
    _g, lap_at_t, a_at_t = _make_constant(n)
    vcm = rp.VarCoefMagnusGraph(
        n_nodes=n, lap_at_t=lap_at_t, a_at_t=a_at_t,
        rho_bar_max=4.0, a_sup_max=1.0,
    )
    u0 = np.zeros(n)
    u0[n // 2] = 1.0
    u1 = vcm.evolve(t_final=0.5, n_steps=20, f0=u0)
    assert np.var(u1) < np.var(u0)


def test_evolve_invalid_f0_length():
    _g, lap_at_t, a_at_t = _make_constant(8)
    vcm = rp.VarCoefMagnusGraph(
        n_nodes=8, lap_at_t=lap_at_t, a_at_t=a_at_t,
        rho_bar_max=4.0, a_sup_max=1.0,
    )
    with pytest.raises(rp.SemiflowError, match="GridMismatch"):
        vcm.evolve(t_final=0.1, n_steps=5, f0=np.zeros(10))


# ---------------------------------------------------------------------------
# Convergence-radius violation
# ---------------------------------------------------------------------------


def test_radius_violation_raises():
    """tau * rho_bar * a_sup^2 >= π/2 ⇒ OutOfDomain (Magnus radius)."""
    n = 8
    _g, lap_at_t, a_at_t = _make_constant(n)
    vcm = rp.VarCoefMagnusGraph(
        n_nodes=n, lap_at_t=lap_at_t, a_at_t=a_at_t,
        rho_bar_max=4.0, a_sup_max=1.0,
    )
    # tau = pi/8 + 0.01 -> radius = 4 * 1 * tau > pi/2
    tau_bad = math.pi / 8.0 + 0.01
    u0 = np.zeros(n)
    # Note: OutOfMagnusRadius is mapped to SemiflowStatus::OutOfDomain by
    # error::from_core (see crates/semiflow-py/src/error.rs).
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        vcm.evolve(t_final=tau_bad, n_steps=1, f0=u0)


def test_radius_check_can_be_disabled():
    """With convergence_check=False the radius check is skipped."""
    n = 8
    _g, lap_at_t, a_at_t = _make_constant(n)
    vcm = rp.VarCoefMagnusGraph(
        n_nodes=n, lap_at_t=lap_at_t, a_at_t=a_at_t,
        rho_bar_max=4.0, a_sup_max=1.0,
        convergence_check=False,
    )
    tau_violating = math.pi / 8.0 + 0.01
    # Should NOT raise — caller takes responsibility.
    u0 = np.zeros(n)
    _ = vcm.evolve(t_final=tau_violating, n_steps=1, f0=u0)


# ---------------------------------------------------------------------------
# compute_rho_bar staticmethod
# ---------------------------------------------------------------------------


def test_compute_rho_bar_constant():
    n = 16
    _g, lap_at_t, a_at_t = _make_constant(n)
    rho, a_sup = rp.VarCoefMagnusGraph.compute_rho_bar(
        lap_at_t, a_at_t, 0.0, 1.0, n, n_samples=16,
    )
    assert abs(rho - 4.0) < 1e-9
    assert abs(a_sup - 1.0) < 1e-9


def test_compute_rho_bar_varying():
    """Time-varying weights — both estimates should track the maximum."""
    n = 16
    g = rp.Graph.path(n)

    def lap_at_t(_t: float):
        return g

    def a_at_t(t: float):
        return np.ones(n) * (1.0 + 0.5 * math.sin(math.pi * t))

    rho, a_sup = rp.VarCoefMagnusGraph.compute_rho_bar(
        lap_at_t, a_at_t, 0.0, 1.0, n, n_samples=32,
    )
    # a peaks at t=0.5 → a_max = 1.5 → a_sup = sqrt(1.5) ≈ 1.2247
    assert abs(a_sup - math.sqrt(1.5)) < 1e-2


# ---------------------------------------------------------------------------
# Parity vs MagnusGraphHeat (K=4 time-dep, no variable a)
# ---------------------------------------------------------------------------


def test_constant_a_parity_vs_magnus_k4():
    """With a(t) ≡ 1 the VarCoef Magnus must agree with the standard
    MagnusGraphHeat K=4 on the same problem."""
    n = 16
    g_path = rp.GraphPath(n)
    g_new = rp.Graph.path(n)

    # MagnusGraphHeat (v2.2) wants GraphPath; lap_at_t returns GraphPath.
    def lap_at_t_old(_t: float):
        return g_path

    # VarCoefMagnusGraph wants Graph; lap_at_t returns Graph.
    def lap_at_t_new(_t: float):
        return g_new

    def a_at_t(_t: float):
        return np.ones(n)

    mgh = rp.MagnusGraphHeat(graph=g_path, lap_at_t=lap_at_t_old, rho_bar_max=4.0)
    vcm = rp.VarCoefMagnusGraph(
        n_nodes=n, lap_at_t=lap_at_t_new, a_at_t=a_at_t,
        rho_bar_max=4.0, a_sup_max=1.0,
    )

    u0 = np.sin(2 * math.pi * np.arange(n) / n)
    t_final = 0.1
    n_steps = 20
    u_mgh = mgh.evolve(t_final=t_final, n_steps=n_steps, f0=u0)
    u_vcm = vcm.evolve(t_final=t_final, n_steps=n_steps, f0=u0)
    diff = float(np.max(np.abs(u_mgh - u_vcm)))
    # Tolerance per test_magnus6.py convention (graph cross-kernel parity).
    assert diff < 1e-2, f"constant-a parity diff = {diff:.3e}"
