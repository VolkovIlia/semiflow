"""Wave P1 smoke tests — ADR-0111 parity for 1-D diffusion completeness.

Covers:
  Heat1DZeta8   (M1) — order-8-temporal ζ⁸ kernel
  TruncatedExp1D     (M2) — K=4 truncated-exp, unit diffusion
  TruncatedExp4th1D  (M3) — 4th-order truncated-exp, unit diffusion
  Strang1D           (M4) — Strang operator-splitting D(τ/2)∘R(τ)∘D(τ/2)

Oracle for M1–M3 (pure diffusion, unit a=1):
  u(t, x) = exp(-x² / (1 + 4t)) / sqrt(1 + 4t)

Oracle for M4 (advection-diffusion, a=1, b=0.5, c=0):
  G3-strang oracle from math.md §9:
    u(x, t) = 1/sqrt(3) * exp(-(x + 0.5)² / 3)
  at t=1 with Gaussian IC exp(-x²) on [-5, 5], 128 nodes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Common fixture parameters
# ---------------------------------------------------------------------------

XMIN, XMAX, N = -5.0, 5.0, 128
T = 0.1
N_STEPS = 80


@pytest.fixture
def u0() -> np.ndarray:
    xs = np.linspace(XMIN, XMAX, N)
    return np.exp(-(xs**2))


def heat_oracle(xs: np.ndarray, t: float) -> np.ndarray:
    """Analytic heat kernel for u0=exp(-x²): u(t,x) = exp(-x²/(1+4t))/sqrt(1+4t)."""
    return np.exp(-(xs**2) / (1.0 + 4.0 * t)) / math.sqrt(1.0 + 4.0 * t)


# ---------------------------------------------------------------------------
# Heat1DZeta8 (M1)
# ---------------------------------------------------------------------------


class TestHeat1DZeta8:
    def test_construction(self, u0: np.ndarray) -> None:
        kern = rp.Heat1DZeta8(XMIN, XMAX, N, u0)
        assert kern is not None

    def test_order(self, u0: np.ndarray) -> None:
        assert rp.Heat1DZeta8(XMIN, XMAX, N, u0).order() == 8

    def test_len(self, u0: np.ndarray) -> None:
        assert len(rp.Heat1DZeta8(XMIN, XMAX, N, u0)) == N

    def test_initial_values(self, u0: np.ndarray) -> None:
        vals = rp.Heat1DZeta8(XMIN, XMAX, N, u0).values()
        assert vals.shape == (N,)
        assert np.allclose(vals, u0, atol=1e-14)

    def test_evolve_runs(self, u0: np.ndarray) -> None:
        kern = rp.Heat1DZeta8(XMIN, XMAX, N, u0)
        kern.evolve(T, N_STEPS)
        vals = kern.values()
        assert vals.shape == (N,)
        assert np.all(np.isfinite(vals))

    def test_evolve_accuracy(self, u0: np.ndarray) -> None:
        """sup_error < 5e-3 vs analytic oracle."""
        xs = np.linspace(XMIN, XMAX, N)
        expected = heat_oracle(xs, T)
        kern = rp.Heat1DZeta8(XMIN, XMAX, N, u0)
        kern.evolve(T, N_STEPS)
        sup_err = float(np.max(np.abs(kern.values() - expected)))
        assert sup_err < 5e-3, f"Heat1DZeta8 sup_error={sup_err:.3e} >= 5e-3"

    def test_nan_input_raises(self, u0: np.ndarray) -> None:
        bad = u0.copy()
        bad[3] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.Heat1DZeta8(XMIN, XMAX, N, bad)

    def test_invalid_grid_raises(self) -> None:
        with pytest.raises(rp.SemiflowError):
            rp.Heat1DZeta8(XMIN, XMAX, 2, np.ones(2))

    def test_cross_consistency_vs_zeta6(self, u0: np.ndarray) -> None:
        """ζ⁸ and ζ⁶ agree to < 1e-5 (both converge to same solution)."""
        kern8 = rp.Heat1DZeta8(XMIN, XMAX, N, u0)
        kern6 = rp.Heat1DZeta6(XMIN, XMAX, N, u0)
        kern8.evolve(T, N_STEPS)
        kern6.evolve(T, N_STEPS)
        diff = float(np.max(np.abs(kern8.values() - kern6.values())))
        assert diff < 1e-5, f"ζ⁸ vs ζ⁶ max_diff={diff:.3e}"


# ---------------------------------------------------------------------------
# TruncatedExp1D (M2)
# ---------------------------------------------------------------------------


class TestTruncatedExp1D:
    def test_construction(self, u0: np.ndarray) -> None:
        kern = rp.TruncatedExp1D(XMIN, XMAX, N, u0)
        assert kern is not None

    def test_order(self, u0: np.ndarray) -> None:
        assert rp.TruncatedExp1D(XMIN, XMAX, N, u0).order() == 2

    def test_len(self, u0: np.ndarray) -> None:
        assert len(rp.TruncatedExp1D(XMIN, XMAX, N, u0)) == N

    def test_evolve_runs(self, u0: np.ndarray) -> None:
        kern = rp.TruncatedExp1D(XMIN, XMAX, N, u0)
        kern.evolve(T, N_STEPS)
        assert np.all(np.isfinite(kern.values()))

    def test_evolve_accuracy(self, u0: np.ndarray) -> None:
        """TruncatedExp order-2 should match oracle to < 5e-3 at t=0.1, N=128."""
        xs = np.linspace(XMIN, XMAX, N)
        expected = heat_oracle(xs, T)
        kern = rp.TruncatedExp1D(XMIN, XMAX, N, u0)
        kern.evolve(T, N_STEPS)
        sup_err = float(np.max(np.abs(kern.values() - expected)))
        assert sup_err < 5e-3, f"TruncatedExp1D sup_error={sup_err:.3e} >= 5e-3"

    def test_nan_input_raises(self, u0: np.ndarray) -> None:
        bad = u0.copy()
        bad[0] = float("inf")
        with pytest.raises(rp.SemiflowError):
            rp.TruncatedExp1D(XMIN, XMAX, N, bad)

    def test_cross_consistency_vs_heat1d(self, u0: np.ndarray) -> None:
        """TruncatedExp1D and Heat1D agree to < 5e-3 (same order-2 kernel)."""
        kern_te = rp.TruncatedExp1D(XMIN, XMAX, N, u0)
        kern_h1 = rp.Heat1D(XMIN, XMAX, N, u0)
        kern_te.evolve(T, N_STEPS)
        kern_h1.evolve(T, N_STEPS)
        diff = float(np.max(np.abs(kern_te.values() - kern_h1.values())))
        assert diff < 5e-3, f"TruncatedExp1D vs Heat1D max_diff={diff:.3e}"


# ---------------------------------------------------------------------------
# TruncatedExp4th1D (M3)
# ---------------------------------------------------------------------------


class TestTruncatedExp4th1D:
    def test_construction(self, u0: np.ndarray) -> None:
        kern = rp.TruncatedExp4th1D(XMIN, XMAX, N, u0)
        assert kern is not None

    def test_order(self, u0: np.ndarray) -> None:
        assert rp.TruncatedExp4th1D(XMIN, XMAX, N, u0).order() == 2

    def test_len(self, u0: np.ndarray) -> None:
        assert len(rp.TruncatedExp4th1D(XMIN, XMAX, N, u0)) == N

    def test_evolve_runs(self, u0: np.ndarray) -> None:
        kern = rp.TruncatedExp4th1D(XMIN, XMAX, N, u0)
        kern.evolve(T, N_STEPS)
        assert np.all(np.isfinite(kern.values()))

    def test_evolve_accuracy(self, u0: np.ndarray) -> None:
        """TruncatedExp4th order-2 should match oracle to < 5e-3 at t=0.1."""
        xs = np.linspace(XMIN, XMAX, N)
        expected = heat_oracle(xs, T)
        kern = rp.TruncatedExp4th1D(XMIN, XMAX, N, u0)
        kern.evolve(T, N_STEPS)
        sup_err = float(np.max(np.abs(kern.values() - expected)))
        assert sup_err < 5e-3, f"TruncatedExp4th1D sup_error={sup_err:.3e} >= 5e-3"

    def test_nan_input_raises(self, u0: np.ndarray) -> None:
        bad = u0.copy()
        bad[10] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.TruncatedExp4th1D(XMIN, XMAX, N, bad)

    def test_cross_consistency_vs_truncated_exp1d(self, u0: np.ndarray) -> None:
        """TruncatedExp4th1D and TruncatedExp1D agree to < 5e-3."""
        kern4 = rp.TruncatedExp4th1D(XMIN, XMAX, N, u0)
        kern2 = rp.TruncatedExp1D(XMIN, XMAX, N, u0)
        kern4.evolve(T, N_STEPS)
        kern2.evolve(T, N_STEPS)
        diff = float(np.max(np.abs(kern4.values() - kern2.values())))
        assert diff < 5e-3, f"TruncExp4th vs TruncExp max_diff={diff:.3e}"


# ---------------------------------------------------------------------------
# Strang1D (M4)
# ---------------------------------------------------------------------------

# G3-strang oracle for advection-diffusion with unit diffusion a=1, b=0.5.
#
# PDE: du/dt = 1*d^2u/dx^2 + 0.5*du/dx, IC = exp(-x^2).
# Galilean substitution v(t,y) = u(t, y - bt/2a) reduces to pure heat
# with diffusion coefficient a. The exact solution is:
#
#   u(t, x) = 1/sqrt(1 + 4*a*t) * exp(-(x + b*t)^2 / (1 + 4*a*t))
#
# With a=1, b=0.5, t=1: denominator = 1 + 4 = 5, mean = -0.5.
#   u(1, x) = 1/sqrt(5) * exp(-(x + 0.5)^2 / 5)
#
# Domain [-5, 5] with N=128 is tight for t=1 — use larger N and larger tolerance
# than the Rust gate which uses N=100000 and [-10, 10].
STRANG_T = 1.0
STRANG_N = 256
STRANG_STEPS = 400


@pytest.fixture
def strang_u0() -> np.ndarray:
    xs = np.linspace(XMIN, XMAX, STRANG_N)
    return np.exp(-(xs**2))


def strang_oracle(xs: np.ndarray) -> np.ndarray:
    """Oracle: 1/sqrt(5) * exp(-(x+0.5)^2 / 5) at t=1, a=1, b=0.5, IC=exp(-x^2)."""
    denom = 1.0 + 4.0 * STRANG_T  # 1 + 4*a*t with a=1
    return np.exp(-((xs + 0.5 * STRANG_T) ** 2) / denom) / math.sqrt(denom)


class TestStrang1D:
    def test_construction(self, strang_u0: np.ndarray) -> None:
        kern = rp.Strang1D(XMIN, XMAX, STRANG_N, strang_u0)
        assert kern is not None

    def test_order(self, strang_u0: np.ndarray) -> None:
        assert rp.Strang1D(XMIN, XMAX, STRANG_N, strang_u0).order() == 2

    def test_len(self, strang_u0: np.ndarray) -> None:
        assert len(rp.Strang1D(XMIN, XMAX, STRANG_N, strang_u0)) == STRANG_N

    def test_initial_values(self, strang_u0: np.ndarray) -> None:
        vals = rp.Strang1D(XMIN, XMAX, STRANG_N, strang_u0).values()
        assert vals.shape == (STRANG_N,)
        assert np.allclose(vals, strang_u0, atol=1e-14)

    def test_evolve_runs(self, strang_u0: np.ndarray) -> None:
        kern = rp.Strang1D(XMIN, XMAX, STRANG_N, strang_u0)
        kern.evolve(STRANG_T, STRANG_STEPS)
        assert np.all(np.isfinite(kern.values()))

    def test_evolve_accuracy(self, strang_u0: np.ndarray) -> None:
        """Oracle: sup_error < 0.01 at t=1, a=1, b=0.5 (order-2, N=256)."""
        xs = np.linspace(XMIN, XMAX, STRANG_N)
        expected = strang_oracle(xs)
        kern = rp.Strang1D(XMIN, XMAX, STRANG_N, strang_u0)
        kern.evolve(STRANG_T, STRANG_STEPS)
        sup_err = float(np.max(np.abs(kern.values() - expected)))
        assert sup_err < 0.01, f"Strang1D sup_error={sup_err:.3e} >= 0.01"

    def test_nan_input_raises(self, strang_u0: np.ndarray) -> None:
        bad = strang_u0.copy()
        bad[5] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.Strang1D(XMIN, XMAX, STRANG_N, bad)

    def test_zero_drift_matches_heat1d(self, u0: np.ndarray) -> None:
        """Strang1D with b=0 should be close to Heat1D (no drift)."""
        kern_s = rp.Strang1D(XMIN, XMAX, N, u0, b=0.0)
        kern_h = rp.Heat1D(XMIN, XMAX, N, u0)
        kern_s.evolve(T, N_STEPS)
        kern_h.evolve(T, N_STEPS)
        diff = float(np.max(np.abs(kern_s.values() - kern_h.values())))
        assert diff < 5e-3, f"Strang1D(b=0) vs Heat1D max_diff={diff:.3e}"
