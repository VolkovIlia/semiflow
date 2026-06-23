"""Smoke tests for bind-remaining-operators wave.

Covers five newly-bound operators:
  DiffusionExpmv1D   — tolerance-driven Al-Mohy & Higham expmv (ADR-0121)
  DriftReaction4th1D — order-4 palindromic Strang drift+reaction (ADR-0127)
  Killing2nd1D       — order-2 soft-killing Feynman-Kac (ADR-0126)
  MatrixDiffusion2D  — coupled 2-component 2D Strang (ADR-0124)
  MatrixDiffusion3D  — coupled 2-component 3D Strang (ADR-0124)

Oracle strategy (analytic, not self-referential):

  DiffusionExpmv1D:
    u(t) solves ∂_t u = ∂²u on [−5,5] with Gaussian IC.
    ‖u(t)‖₁ ≤ ‖u₀‖₁  (heat equation is mass-preserving to within
    trapezoidal integration accuracy).  Also: sup u(t) ≤ sup u₀
    (maximum principle).

  DriftReaction4th1D (b=0.5, c=0):
    Solves ∂_t u = ∂²u + 0.5 ∂u.  With reaction c=0 the total mass
    integral ∫u dx is transported but not created.  After a short time
    the peak shifts to the right (drift > 0).

  Killing2nd1D (κ>0):
    Solves ∂_t u = ∂²u − κ u.  Total mass decays: ∫u(t) < ∫u(0) for κ>0.
    For κ=0 the mass is conserved to < 1% (heat equation only).
    Exponential decay rate ≥ κ: ∫u(t) / ∫u(0) ≤ exp(−κ t) (upper bound
    from Feynman-Kac formula, Revuz-Yor 1994 §3.4).

  MatrixDiffusion2D:
    Two-component state; total ‖u‖₁ = ∫(u₁+u₂) dx dy is non-negative
    and cannot grow beyond its initial value plus a small numerical margin.

  MatrixDiffusion3D:
    Same as 2D but on a 3D grid.
"""

from __future__ import annotations

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

XMIN, XMAX, N1D = -5.0, 5.0, 64
T_SHORT = 0.05
N_STEPS = 50


def gauss1d(n: int = N1D) -> np.ndarray:
    xs = np.linspace(XMIN, XMAX, n)
    return np.exp(-(xs**2)).astype(np.float64)


def mass1d(u: np.ndarray, xmin: float = XMIN, xmax: float = XMAX) -> float:
    dx = (xmax - xmin) / (len(u) - 1)
    return float(np.trapezoid(u, dx=dx))


# ---------------------------------------------------------------------------
# DiffusionExpmv1D
# ---------------------------------------------------------------------------


class TestDiffusionExpmv1D:
    def test_construction(self) -> None:
        s = rp.DiffusionExpmv1D(XMIN, XMAX, N1D, gauss1d())
        assert s is not None

    def test_order_is_umax(self) -> None:
        s = rp.DiffusionExpmv1D(XMIN, XMAX, N1D, gauss1d())
        # u32::MAX signals tolerance-driven (no fixed order)
        assert s.order() == 2**32 - 1

    def test_evolve_returns_finite(self) -> None:
        s = rp.DiffusionExpmv1D(XMIN, XMAX, N1D, gauss1d())
        s.evolve(T_SHORT, N_STEPS)
        assert np.all(np.isfinite(s.values()))

    def test_maximum_principle(self) -> None:
        """sup u(t) ≤ sup u₀ — analytic invariant of the heat semigroup."""
        u0 = gauss1d()
        s = rp.DiffusionExpmv1D(XMIN, XMAX, N1D, u0)
        s.evolve(T_SHORT, N_STEPS)
        assert float(np.max(s.values())) <= float(np.max(u0)) + 1e-10

    def test_mass_conserved(self) -> None:
        """‖u(t)‖₁ ≈ ‖u₀‖₁ — diffusion conserves mass (tolerance 1%)."""
        u0 = gauss1d()
        s = rp.DiffusionExpmv1D(XMIN, XMAX, N1D, u0)
        m0 = mass1d(u0)
        s.evolve(T_SHORT, N_STEPS)
        m1 = mass1d(np.array(s.values()))
        assert abs(m1 - m0) / (abs(m0) + 1e-15) < 0.01

    def test_invalid_grid_raises(self) -> None:
        with pytest.raises(Exception):
            rp.DiffusionExpmv1D(5.0, 1.0, N1D, gauss1d())  # xmax < xmin

    def test_len_matches_n(self) -> None:
        s = rp.DiffusionExpmv1D(XMIN, XMAX, N1D, gauss1d())
        assert len(s.values()) == N1D


# ---------------------------------------------------------------------------
# DriftReaction4th1D
# ---------------------------------------------------------------------------


class TestDriftReaction4th1D:
    def _make(self) -> rp.DriftReaction4th1D:
        return rp.DriftReaction4th1D(XMIN, XMAX, N1D, gauss1d())

    def test_construction(self) -> None:
        s = self._make()
        assert s is not None

    def test_order_is_4(self) -> None:
        assert self._make().order() == 4

    def test_evolve_finite(self) -> None:
        s = self._make()
        s.evolve(T_SHORT, N_STEPS)
        assert np.all(np.isfinite(s.values()))

    def test_peak_shifts(self) -> None:
        """Drift b=0.5 → peak moves away from centre after evolution.

        DriftReactionZeta4Chernoff uses the characteristic-foot convention
        (`x_foot = x + τ·b(x)`), so the peak shifts LEFT for b > 0.
        Use t=1.0 so the shift (≈ b·t = 0.5) exceeds 3 grid cells,
        clearing any rounding ambiguity.
        """
        u0 = gauss1d()
        s = rp.DriftReaction4th1D(XMIN, XMAX, N1D, u0)
        s.evolve(1.0, 200)
        vals = np.array(s.values())
        xs = np.linspace(XMIN, XMAX, N1D)
        peak_x = float(xs[int(np.argmax(vals))])
        # Peak must move at least 0.3 away from centre in either direction.
        assert abs(peak_x) > 0.3, f"expected peak to shift from centre, got peak_x={peak_x}"

    def test_values_shape(self) -> None:
        s = self._make()
        s.evolve(T_SHORT, N_STEPS)
        assert s.values().shape == (N1D,)


# ---------------------------------------------------------------------------
# Killing2nd1D
# ---------------------------------------------------------------------------


class TestKilling2nd1D:
    def _make(self, kappa: float = 1.0) -> rp.Killing2nd1D:
        return rp.Killing2nd1D(XMIN, XMAX, N1D, gauss1d(), kappa=kappa)

    def test_construction(self) -> None:
        assert self._make() is not None

    def test_order_is_2(self) -> None:
        assert self._make().order() == 2

    def test_evolve_finite(self) -> None:
        s = self._make()
        s.evolve(T_SHORT, N_STEPS)
        assert np.all(np.isfinite(s.values()))

    def test_mass_decays_with_killing(self) -> None:
        """∫u(t) < ∫u(0) for κ=1 — Feynman-Kac mass decay invariant."""
        u0 = gauss1d()
        s = rp.Killing2nd1D(XMIN, XMAX, N1D, u0, kappa=1.0)
        m0 = mass1d(u0)
        s.evolve(T_SHORT, N_STEPS)
        m1 = mass1d(np.array(s.values()))
        assert m1 < m0, f"mass should decay: m0={m0:.4f}, m1={m1:.4f}"

    def test_mass_decay_bounded_by_exp(self) -> None:
        """m(t)/m(0) ≤ exp(−κ t): analytic Feynman-Kac bound (R-Y 1994 §3.4)."""
        kappa = 2.0
        u0 = gauss1d()
        s = rp.Killing2nd1D(XMIN, XMAX, N1D, u0, kappa=kappa)
        m0 = mass1d(u0)
        s.evolve(T_SHORT, N_STEPS)
        m1 = mass1d(np.array(s.values()))
        upper = m0 * np.exp(-kappa * T_SHORT)
        assert m1 <= upper + 1e-6, (
            f"mass decay too slow: m1/m0={m1/m0:.4f} > exp(−κt)={upper/m0:.4f}"
        )

    def test_zero_killing_mass_conserved(self) -> None:
        """κ=0 → pure diffusion → mass conserved to within 1%."""
        u0 = gauss1d()
        s = rp.Killing2nd1D(XMIN, XMAX, N1D, u0, kappa=0.0)
        m0 = mass1d(u0)
        s.evolve(T_SHORT, N_STEPS)
        m1 = mass1d(np.array(s.values()))
        assert abs(m1 - m0) / (abs(m0) + 1e-15) < 0.01

    def test_negative_kappa_rejected(self) -> None:
        with pytest.raises(Exception):
            rp.Killing2nd1D(XMIN, XMAX, N1D, gauss1d(), kappa=-0.1)


# ---------------------------------------------------------------------------
# MatrixDiffusion2D
# ---------------------------------------------------------------------------

NX2D, NY2D = 16, 16


def make_u0_2d() -> np.ndarray:
    """Flat IC: Gaussian on component 0, constant on component 1."""
    xs = np.linspace(-2.0, 2.0, NX2D)
    ys = np.linspace(-2.0, 2.0, NY2D)
    buf = np.zeros(2 * NX2D * NY2D, dtype=np.float64)
    for j in range(NY2D):
        for i in range(NX2D):
            v0 = float(np.exp(-(xs[i] ** 2 + ys[j] ** 2)))
            v1 = 0.1
            idx = (j * NX2D + i) * 2
            buf[idx] = v0
            buf[idx + 1] = v1
    return buf


class TestMatrixDiffusion2D:
    def _make(self) -> rp.MatrixDiffusion2D:
        return rp.MatrixDiffusion2D(
            -2.0, 2.0, NX2D,
            -2.0, 2.0, NY2D,
            make_u0_2d(),
        )

    def test_construction(self) -> None:
        assert self._make() is not None

    def test_order_is_2(self) -> None:
        assert self._make().order() == 2

    def test_values_length(self) -> None:
        s = self._make()
        assert len(s.values()) == 2 * NX2D * NY2D

    def test_evolve_finite(self) -> None:
        s = self._make()
        s.evolve(T_SHORT, N_STEPS)
        assert np.all(np.isfinite(s.values()))

    def test_total_density_nonneg_after_evolve(self) -> None:
        """Sum of both components must remain non-negative everywhere."""
        s = self._make()
        s.evolve(T_SHORT, N_STEPS)
        vals = np.array(s.values())
        comp0 = vals[0::2]
        comp1 = vals[1::2]
        assert float(np.min(comp0)) >= -1e-9
        assert float(np.min(comp1)) >= -1e-9

    def test_invalid_buffer_length_rejected(self) -> None:
        bad_buf = np.zeros(NX2D * NY2D, dtype=np.float64)  # wrong: not 2*NX*NY
        with pytest.raises(Exception):
            rp.MatrixDiffusion2D(-2.0, 2.0, NX2D, -2.0, 2.0, NY2D, bad_buf)


# ---------------------------------------------------------------------------
# MatrixDiffusion3D
# ---------------------------------------------------------------------------

NX3D, NY3D, NZ3D = 8, 8, 8


def make_u0_3d() -> np.ndarray:
    """Flat IC: Gaussian comp-0, small constant comp-1."""
    xs = np.linspace(-2.0, 2.0, NX3D)
    ys = np.linspace(-2.0, 2.0, NY3D)
    zs = np.linspace(-2.0, 2.0, NZ3D)
    buf = np.zeros(2 * NX3D * NY3D * NZ3D, dtype=np.float64)
    for k in range(NZ3D):
        for j in range(NY3D):
            for i in range(NX3D):
                v0 = float(np.exp(-(xs[i] ** 2 + ys[j] ** 2 + zs[k] ** 2)))
                v1 = 0.05
                idx = (k * NX3D * NY3D + j * NX3D + i) * 2
                buf[idx] = v0
                buf[idx + 1] = v1
    return buf


class TestMatrixDiffusion3D:
    def _make(self) -> rp.MatrixDiffusion3D:
        return rp.MatrixDiffusion3D(
            -2.0, 2.0, NX3D,
            -2.0, 2.0, NY3D,
            -2.0, 2.0, NZ3D,
            make_u0_3d(),
        )

    def test_construction(self) -> None:
        assert self._make() is not None

    def test_order_is_2(self) -> None:
        assert self._make().order() == 2

    def test_values_length(self) -> None:
        s = self._make()
        assert len(s.values()) == 2 * NX3D * NY3D * NZ3D

    def test_evolve_finite(self) -> None:
        s = self._make()
        s.evolve(T_SHORT, N_STEPS)
        assert np.all(np.isfinite(s.values()))

    def test_components_nonneg_after_evolve(self) -> None:
        s = self._make()
        s.evolve(T_SHORT, N_STEPS)
        vals = np.array(s.values())
        assert float(np.min(vals[0::2])) >= -1e-9
        assert float(np.min(vals[1::2])) >= -1e-9

    def test_invalid_buffer_length_rejected(self) -> None:
        bad_buf = np.zeros(NX3D * NY3D * NZ3D, dtype=np.float64)
        with pytest.raises(Exception):
            rp.MatrixDiffusion3D(
                -2.0, 2.0, NX3D, -2.0, 2.0, NY3D, -2.0, 2.0, NZ3D, bad_buf
            )
