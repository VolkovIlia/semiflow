"""Tests for `GraphHeat6` Python binding (v2.5 Phase 1, ADR-0062).

Mirrors `test_graph_extra.py::GraphHeat4th` pattern.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import semiflow as rp


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_graph_heat6_construct_from_graph():
    g = rp.Graph.path(16)
    gh6 = rp.GraphHeat6(graph=g, rho_bar=4.0)
    assert gh6.n_nodes == 16


def test_graph_heat6_construct_from_laplacian():
    g = rp.Graph.path(8)
    lap = rp.Laplacian.combinatorial(g)
    gh6 = rp.GraphHeat6(laplacian=lap, rho_bar=4.0)
    assert gh6.n_nodes == 8


def test_graph_heat6_rho_bar_validated():
    g = rp.Graph.path(8)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.GraphHeat6(graph=g, rho_bar=-1.0)


def test_graph_heat6_needs_graph_or_laplacian():
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.GraphHeat6(rho_bar=4.0)


# ---------------------------------------------------------------------------
# Basic semantics
# ---------------------------------------------------------------------------


def test_graph_heat6_preserves_sum():
    """Combinatorial Laplacian rows sum to 0, so Σ f_i is conserved."""
    n = 16
    g = rp.Graph.path(n)
    gh6 = rp.GraphHeat6(graph=g, rho_bar=4.0)
    u0 = np.exp(-((np.arange(n) - n / 2.0) ** 2) / (n / 4.0))
    u1 = gh6.evolve(t_final=0.5, n_steps=20, f0=u0)
    assert abs(u1.sum() - u0.sum()) < 1e-10


def test_graph_heat6_dissipates_variance():
    """Heat equation spreads the signal: variance must decrease."""
    n = 32
    g = rp.Graph.path(n)
    gh6 = rp.GraphHeat6(graph=g, rho_bar=4.0)
    u0 = np.zeros(n)
    u0[n // 2] = 1.0  # delta function
    u1 = gh6.evolve(t_final=1.0, n_steps=40, f0=u0)
    assert np.var(u1) < np.var(u0)


def test_graph_heat6_invalid_f0_length():
    g = rp.Graph.path(8)
    gh6 = rp.GraphHeat6(graph=g, rho_bar=4.0)
    with pytest.raises(rp.SemiflowError, match="GridMismatch"):
        gh6.evolve(t_final=0.1, n_steps=5, f0=np.zeros(10))


def test_graph_heat6_invalid_n_steps():
    g = rp.Graph.path(8)
    gh6 = rp.GraphHeat6(graph=g, rho_bar=4.0)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        gh6.evolve(t_final=0.1, n_steps=0, f0=np.zeros(8))


# ---------------------------------------------------------------------------
# Convergence: K=6 should out-perform K=4 at the same n_steps on smooth ICs
# ---------------------------------------------------------------------------


def test_graph_heat6_more_accurate_than_k4():
    """K=6 vs K=4 self-convergence diff at the same n_steps on a smooth IC."""
    n = 32
    g = rp.Graph.path(n)
    gh4 = rp.GraphHeat4th(graph=g, rho_bar=4.0)
    gh6 = rp.GraphHeat6(graph=g, rho_bar=4.0)
    u0 = np.cos(2 * math.pi * np.arange(n) / n)

    n_lo = 8
    n_hi = 16
    u4_lo = gh4.evolve(t_final=0.25, n_steps=n_lo, f0=u0)
    u4_hi = gh4.evolve(t_final=0.25, n_steps=n_hi, f0=u0)
    u6_lo = gh6.evolve(t_final=0.25, n_steps=n_lo, f0=u0)
    u6_hi = gh6.evolve(t_final=0.25, n_steps=n_hi, f0=u0)

    err4 = float(np.max(np.abs(u4_lo - u4_hi)))
    err6 = float(np.max(np.abs(u6_lo - u6_hi)))
    assert err6 < err4, f"K=6 self-conv diff {err6:.3e} should be < K=4 {err4:.3e}"


# ---------------------------------------------------------------------------
# Convergence: empirical order ~6 on cos IC
# ---------------------------------------------------------------------------


def test_graph_heat6_empirical_order():
    """Compare K=6 at n_steps={5,8,12} vs reference at n_steps=80."""
    n = 32
    g = rp.Graph.path(n)
    gh6 = rp.GraphHeat6(graph=g, rho_bar=4.0)
    u0 = np.cos(2 * math.pi * np.arange(n) / n)
    T = 1.0

    ref = gh6.evolve(t_final=T, n_steps=80, f0=u0)
    ns = [5, 8, 12]
    errs = []
    for ns_ in ns:
        u = gh6.evolve(t_final=T, n_steps=ns_, f0=u0)
        errs.append(float(np.max(np.abs(u - ref))))

    # log-log slope; expect ≤ −5 (order 6 with some floor).
    lx = np.log(ns)
    ly = np.log(errs)
    slope = float(np.polyfit(lx, ly, 1)[0])
    assert slope <= -5.0, f"K=6 empirical slope {slope:.3f} should be ≤ -5.0 (errs={errs})"
