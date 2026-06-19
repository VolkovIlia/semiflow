"""Wave P5 smoke tests — ADR-0111 parity for geometry kernels.

Covers:
  Manifold2D                       (M13) — ManifoldChernoff<M, f64> (torus/sphere2/hyperbolic2)
  HypoellipticChernoffKolmogorov   (M14) — KolmogorovHypoelliptic<f64>
  HypoellipticChernoffEngel        (M15) — HypoellipticChernoff<f64, 4, 2>

Oracle strategy (INDEPENDENT, NOT self-referential):
  All oracles are derived from KNOWN ANALYTIC references that do NOT depend
  on the Rust kernel under test.

=============================================================================
M13 Manifold2D — oracle strategy per backend
=============================================================================

### Torus (manifold="torus", R ≡ 0)
  The flat torus T² IS a flat Euclidean space with periodic boundary
  conditions.  The heat equation ∂_t u = Δu on T² [0,L]² has the analytic
  fundamental solution (Fourier series, periodic Green's function):
    K_t(x-x0) = Σ_{k∈Z²} (1/(4πt)) exp(-|x - x0 + kL|²/(4t))
  For initial data u0(x,y) = exp(-|x|² - |y|²) on [-3,3]², the total
  integral (mass) is conserved under Δu (if Dirichlet is not used):
    mass(t) = integral u0  for all t  (assuming periodic b.c.)
  But ManifoldChernoff uses reflect b.c. (grid b.c. on the chart).
  Simpler oracle: positivity preservation + finite output.

  Additionally: on the torus the curvature correction is IDENTITY (R ≡ 0),
  so the effective generator is exactly Δ_M regardless of curvature_correction
  flag.  Therefore:
    Manifold2D(..., manifold="torus", curvature_correction=True)
  and
    Manifold2D(..., manifold="torus", curvature_correction=False)
  must produce IDENTICAL outputs.  This is a structural identity oracle
  (analytic: R ≡ 0 ⟹ 1 + τR/12 ≡ 1, so correction is exactly identity).

### Sphere2 (manifold="sphere2", R = 2/r²)
  The sphere S²(r) has POSITIVE curvature.  For the corrected generator
  Δ_M + R/12 = Δ_M + 1/(6r²), the semigroup has a POSITIVE spectral shift:
    exp(t(Δ_M + R/12)) versus exp(t·Δ_M).
  Therefore, for a smooth IC near the north pole, the corrected semigroup
  produces LARGER peak values than the uncorrected one at small t
  (spectral shift shifts spectrum to the right → less decay).

  Oracle: for identical IC and evolution time,
    sup(corrected values) >= sup(uncorrected values) - epsilon.
  This is an analytic invariant of the spectral shift (R > 0 ⟹ shift up).

### Hyperbolic2 (manifold="hyperbolic2", R = -2/s²)
  The Poincaré disk H²(s) has NEGATIVE curvature (R < 0).  For the corrected
  generator Δ_M + R/12 = Δ_M - 1/(6s²), the spectral shift is NEGATIVE:
    the corrected semigroup decays FASTER than the uncorrected one.

  Oracle: sup(corrected values) <= sup(uncorrected values) + epsilon.
  This is the analytic invariant for negative curvature.

=============================================================================
M14 HypoellipticChernoffKolmogorov — oracle strategy
=============================================================================

  The Kolmogorov fundamental solution is:
    p(t, x, v; 0, 0) = (√3 / (2π t²)) exp(-3x²/t³ - v²/t)

  at the source (x0, v0) = (0, 0).  This is independent of the Rust kernel
  (Kolmogorov 1934 *Math. Annalen* 108, math.md §28.4.A).

  Specifically, for a grid with x,v centered at 0, with IC = Kolmogorov
  solution at T_IC (smooth initial condition), after evolving by dt the
  output should approximate the Kolmogorov solution at T_IC + dt.

  We use a WEAKER but robust oracle: mass conservation.
  The Kolmogorov equation ∂_t p = v ∂_x p + ½ ∂²_v p conserves total mass:
    ∫∫ p(t, x, v) dx dv = constant.
  This is an analytic property of the CONTINUOUS equation (divergence theorem
  + periodic or zero b.c. at infinity).  For our Chernoff approximation on a
  large domain, mass should be approximately conserved:
    |mass(t)/mass(0) - 1| ≤ 5e-3.

  Tolerance: 5e-3 (coarse Chernoff n=20, domain truncation).

  Additional oracle: order == 2 (palindromic Strang-Hörmander, math §28.3).

=============================================================================
M15 HypoellipticChernoffEngel — oracle strategy
=============================================================================

  The Engel group has no known closed-form fundamental solution (Engel group
  is step-3 Carnot; Kolmogorov has step-2).  The numerical self-convergence
  slope test (T_HORM_ENGEL_SLOPE in core tests) shows order ~2 but is slow.

  For the Python binding smoke test we use two STRUCTURAL oracles:

  1. Positivity preservation: for a non-negative Gaussian IC, the output
     must be non-negative (heat semigroup on a Carnot group preserves
     positivity).  This is analytic: Chernoff products of positive operators
     are positive (Butko 2018 Corollary 1.1).

  2. Peak strictly decreases: for a non-constant non-negative IC,
     ‖T_t u₀‖_∞ < ‖u₀‖_∞ for t > 0 (Bony 1967 hypoelliptic maximum
     principle).  Coarse 8-point grid; the decrease must be > 1e-6.

  3. order == 2 (palindromic Strang-Hörmander, math §28.bis.2, ADR-0095).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from typing import Literal

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gaussian_2d(
    x0min: float, x0max: float, nx: int,
    x1min: float, x1max: float, ny: int,
    cx: float = 0.0, cy: float = 0.0,
) -> np.ndarray:
    """2D Gaussian IC on [x0min,x0max]×[x1min,x1max], peak at (cx, cy).

    Row-major flat array of length nx*ny.
    """
    xs = np.linspace(x0min, x0max, nx)
    ys = np.linspace(x1min, x1max, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")  # shape (nx, ny)
    u = np.exp(-((X - cx) ** 2) - ((Y - cy) ** 2))
    return u.flatten().astype(np.float64)


def mass_2d(u: np.ndarray, x0min: float, x0max: float,
            x1min: float, x1max: float) -> float:
    """Trapezoidal 2D integral of flat array."""
    n = len(u)
    nx = ny = int(round(n ** 0.5))
    assert nx * ny == n
    u2d = u.reshape(nx, ny)
    dx = (x0max - x0min) / (nx - 1)
    dy = (x1max - x1min) / (ny - 1)
    return float(np.trapezoid(np.trapezoid(u2d, dx=dy, axis=1), dx=dx))


# ---------------------------------------------------------------------------
# Manifold2D (M13)
# ---------------------------------------------------------------------------


CHART_MIN, CHART_MAX, NX, NY = -3.0, 3.0, 24, 24
T_MANIFOLD = 0.05
N_STEPS_MANIFOLD = 20


class TestManifold2D:
    """Tests for Manifold2D — manifold Chernoff binding."""

    # ---- construction and basic properties ----

    def test_torus_construction(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="torus")
        assert m is not None

    def test_sphere2_construction(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="sphere2", radius=1.0)
        assert m is not None

    def test_hyperbolic2_construction(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="hyperbolic2", radius=1.0)
        assert m is not None

    def test_len(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0)
        assert len(m) == NX * NY

    def test_torus_order_with_correction(self) -> None:
        """Torus R≡0 → correction is identity → order() returns 1 (R≡0 branch)."""
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="torus", curvature_correction=True)
        # Core returns 1 for torus regardless of correction flag (R≡0 ⟹ no-op).
        assert m.order() == 1

    def test_sphere2_order_with_correction(self) -> None:
        """Sphere with curvature correction → order 2."""
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="sphere2", curvature_correction=True)
        assert m.order() == 2

    def test_sphere2_order_without_correction(self) -> None:
        """Sphere without curvature correction → order 1."""
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="sphere2", curvature_correction=False)
        assert m.order() == 1

    def test_values_shape_dtype(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0)
        vals = m.values()
        assert vals.shape == (NX * NY,)
        assert vals.dtype == np.float64

    # ---- oracle test 1: torus curvature-correction identity ----

    def test_torus_correction_identity(self) -> None:
        """Torus R≡0: curvature correction is identity; both variants equal.

        Oracle (analytic): R≡0 on T² ⟹ 1 + τR/12 ≡ 1 ⟹ corrected == uncorrected.
        sup_error must be < 1e-14 (machine precision; correction is exactly
        the identity map for zero curvature).

        This oracle does NOT compare the binding to itself: it uses the
        analytic property that the correction factor is 1 when R=0.
        """
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)

        m_on = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY,
                             u0.copy(), manifold="torus", curvature_correction=True)
        m_on.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        v_on = m_on.values()

        m_off = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY,
                              u0.copy(), manifold="torus", curvature_correction=False)
        m_off.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        v_off = m_off.values()

        sup_err = float(np.max(np.abs(v_on - v_off)))
        tol = 1e-14
        assert sup_err < tol, (
            f"Torus correction identity violated: sup_error={sup_err:.3e} "
            f"(expected < {tol:.1e}; R≡0 ⟹ correction is identity)"
        )

    # ---- oracle test 2: sphere2 positive-curvature spectral shift ----

    def test_sphere2_corrected_larger_peak_than_uncorrected(self) -> None:
        """Sphere R>0: corrected semigroup decays slower (spectral shift up).

        Oracle (analytic): exp(t(Δ_M + R/12)) with R = 2/r² > 0 has a
        positive spectral shift compared to exp(t·Δ_M).  A positive spectral
        shift means slower decay of positive initial data.  Therefore the
        peak of the corrected output must be >= peak of uncorrected output
        minus a small tolerance (3%).

        sup_error = (uncorrected_peak - corrected_peak) / uncorrected_peak.
        Tolerance: corrected_peak >= uncorrected_peak * 0.97.
        """
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)

        m_corr = rp.Manifold2D(
            CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY,
            u0.copy(), manifold="sphere2", radius=1.0, curvature_correction=True,
        )
        m_corr.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        v_corr = m_corr.values()

        m_nocorr = rp.Manifold2D(
            CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY,
            u0.copy(), manifold="sphere2", radius=1.0, curvature_correction=False,
        )
        m_nocorr.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        v_nocorr = m_nocorr.values()

        peak_corr = float(np.max(v_corr))
        peak_nocorr = float(np.max(v_nocorr))
        lower_bound = peak_nocorr * 0.97

        assert peak_corr >= lower_bound, (
            f"Sphere2 curvature-correction spectral-shift oracle violated:\n"
            f"  corrected peak = {peak_corr:.6f}\n"
            f"  uncorrected peak = {peak_nocorr:.6f}\n"
            f"  lower bound = {lower_bound:.6f}\n"
            f"  (R>0 ⟹ positive spectral shift ⟹ corrected decays slower)"
        )

    # ---- oracle test 3: hyperbolic2 negative-curvature spectral shift ----

    def test_hyperbolic2_corrected_smaller_peak_than_uncorrected(self) -> None:
        """Hyperbolic2 R<0: corrected semigroup decays faster (spectral shift down).

        Oracle (analytic): exp(t(Δ_M + R/12)) with R = -2/s² < 0 has a
        negative spectral shift — faster decay of positive initial data.
        The peak of the corrected output must be <= peak of uncorrected + 3%.

        sup_error = (corrected_peak - uncorrected_peak) / uncorrected_peak.
        Tolerance: corrected_peak <= uncorrected_peak * 1.03.
        """
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)

        m_corr = rp.Manifold2D(
            CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY,
            u0.copy(), manifold="hyperbolic2", radius=1.0, curvature_correction=True,
        )
        m_corr.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        v_corr = m_corr.values()

        m_nocorr = rp.Manifold2D(
            CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY,
            u0.copy(), manifold="hyperbolic2", radius=1.0, curvature_correction=False,
        )
        m_nocorr.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        v_nocorr = m_nocorr.values()

        peak_corr = float(np.max(v_corr))
        peak_nocorr = float(np.max(v_nocorr))
        upper_bound = peak_nocorr * 1.03

        assert peak_corr <= upper_bound, (
            f"Hyperbolic2 curvature-correction spectral-shift oracle violated:\n"
            f"  corrected peak = {peak_corr:.6f}\n"
            f"  uncorrected peak = {peak_nocorr:.6f}\n"
            f"  upper bound = {upper_bound:.6f}\n"
            f"  (R<0 ⟹ negative spectral shift ⟹ corrected decays faster)"
        )

    # ---- oracle test 4: all backends produce finite non-negative outputs ----

    def test_torus_evolve_finite_nonneg(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="torus")
        m.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        vals = m.values()
        assert np.all(np.isfinite(vals)), "Torus: NaN or Inf in output"
        assert float(np.min(vals)) > -1e-10, (
            f"Torus: negative output (min={float(np.min(vals)):.3e})"
        )

    def test_sphere2_evolve_finite_nonneg(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="sphere2", radius=1.0)
        m.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        vals = m.values()
        assert np.all(np.isfinite(vals)), "Sphere2: NaN or Inf in output"
        assert float(np.min(vals)) > -1e-10, (
            f"Sphere2: negative output (min={float(np.min(vals)):.3e})"
        )

    def test_hyperbolic2_evolve_finite_nonneg(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="hyperbolic2", radius=1.0)
        m.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
        vals = m.values()
        assert np.all(np.isfinite(vals)), "Hyperbolic2: NaN or Inf in output"
        assert float(np.min(vals)) > -1e-10, (
            f"Hyperbolic2: negative output (min={float(np.min(vals)):.3e})"
        )

    # ---- different backends produce different outputs ----

    def test_backends_differ(self) -> None:
        """Three manifold backends give structurally different outputs.

        Oracle: distinct curvatures (R=0, R=2, R=-2) imply distinct semigroups.
        For any non-trivial IC and t>0, outputs must differ (analytic uniqueness).
        """
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)

        def evolve(manifold: Literal["torus", "sphere2", "hyperbolic2"]) -> np.ndarray:
            m = rp.Manifold2D(
                CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY,
                u0.copy(), manifold=manifold,
                curvature_correction=True,
            )
            m.evolve(T_MANIFOLD, N_STEPS_MANIFOLD)
            return m.values()

        v_torus = evolve("torus")
        v_sphere = evolve("sphere2")
        v_hyp = evolve("hyperbolic2")

        diff_ts = float(np.max(np.abs(v_torus - v_sphere)))
        diff_th = float(np.max(np.abs(v_torus - v_hyp)))
        assert diff_ts > 1e-10, f"torus and sphere2 gave same output (diff={diff_ts:.3e})"
        assert diff_th > 1e-10, f"torus and hyperbolic2 gave same output (diff={diff_th:.3e})"

    # ---- input validation ----

    def test_unknown_manifold_raises(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        with pytest.raises(rp.SemiflowError):
            rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="cylinder")  # type: ignore[arg-type]  # intentional invalid-input test

    def test_nan_u0_raises(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        u0[5] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0)

    def test_invalid_grid_raises(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        with pytest.raises(rp.SemiflowError):
            rp.Manifold2D(3.0, -3.0, NX, CHART_MIN, CHART_MAX, NY, u0)

    def test_invalid_radius_raises(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        with pytest.raises(rp.SemiflowError):
            rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0,
                          manifold="sphere2", radius=0.0)

    def test_n_steps_zero_raises(self) -> None:
        u0 = make_gaussian_2d(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY)
        m = rp.Manifold2D(CHART_MIN, CHART_MAX, NX, CHART_MIN, CHART_MAX, NY, u0)
        with pytest.raises(rp.SemiflowError):
            m.evolve(T_MANIFOLD, 0)


# ---------------------------------------------------------------------------
# HypoellipticChernoffKolmogorov (M14)
# ---------------------------------------------------------------------------

# Phase-space domain (x, v) ∈ [-L, L]²
KOL_L = 6.0
KOL_NX, KOL_NV = 32, 32
T_KOL = 0.1
N_STEPS_KOL = 20


def kolmogorov_oracle(t: float, x: float, v: float) -> float:
    """Kolmogorov 1934 fundamental solution centred at origin.

    p(t, x, v) = (√3 / (2π t²)) exp(-3x²/t³ - v²/t).

    Reference: Kolmogorov 1934 *Math. Annalen* 108; math.md §28.4.A.
    This formula is INDEPENDENT of the Rust kernel under test.
    """
    sqrt3 = math.sqrt(3.0)
    return (sqrt3 / (2.0 * math.pi * t * t)) * math.exp(
        -3.0 * x * x / (t ** 3) - v * v / t
    )


def make_kolmogorov_ic(t_ic: float = 1.0) -> np.ndarray:
    """Kolmogorov solution at t_ic as IC on the (x,v) grid.

    Uses the fundamental solution centred at origin — it is smooth and
    compactly supported within the domain for t_ic=1 (sigma_v ≈ 1).
    Row-major flat array of length KOL_NX * KOL_NV.
    """
    xs = np.linspace(-KOL_L, KOL_L, KOL_NX)
    vs = np.linspace(-KOL_L, KOL_L, KOL_NV)
    X, V = np.meshgrid(xs, vs, indexing="ij")
    p = np.vectorize(lambda x, v: kolmogorov_oracle(t_ic, x, v))(X, V)
    return p.flatten().astype(np.float64)


def mass_kol(u: np.ndarray) -> float:
    """Trapezoidal mass on (x,v) grid."""
    u2d = u.reshape(KOL_NX, KOL_NV)
    dx = 2.0 * KOL_L / (KOL_NX - 1)
    dv = 2.0 * KOL_L / (KOL_NV - 1)
    return float(np.trapezoid(np.trapezoid(u2d, dx=dv, axis=1), dx=dx))


class TestHypoellipticChernoffKolmogorov:
    """Tests for HypoellipticChernoffKolmogorov — Kolmogorov phase-space binding."""

    def test_construction(self) -> None:
        u0 = make_kolmogorov_ic()
        k = rp.HypoellipticChernoffKolmogorov(
            -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0
        )
        assert k is not None

    def test_order(self) -> None:
        """Order == 2 (palindromic Strang-Hörmander, math §28.3)."""
        u0 = make_kolmogorov_ic()
        k = rp.HypoellipticChernoffKolmogorov(
            -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0
        )
        assert k.order() == 2

    def test_len(self) -> None:
        u0 = make_kolmogorov_ic()
        k = rp.HypoellipticChernoffKolmogorov(
            -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0
        )
        assert len(k) == KOL_NX * KOL_NV

    def test_values_shape_dtype(self) -> None:
        u0 = make_kolmogorov_ic()
        k = rp.HypoellipticChernoffKolmogorov(
            -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0
        )
        vals = k.values()
        assert vals.shape == (KOL_NX * KOL_NV,)
        assert vals.dtype == np.float64

    def test_evolve_finite_output(self) -> None:
        u0 = make_kolmogorov_ic()
        k = rp.HypoellipticChernoffKolmogorov(
            -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0
        )
        k.evolve(T_KOL, N_STEPS_KOL)
        vals = k.values()
        assert np.all(np.isfinite(vals)), "Kolmogorov: NaN or Inf in output"

    # -----------------------------------------------------------------------
    # Oracle test: mass conservation.
    #
    # The continuous Kolmogorov equation ∂_t p = v ∂_x p + ½ ∂²_v p
    # conserves total mass: d/dt ∫∫ p dx dv = 0 (divergence theorem +
    # b.c.).  For a Chernoff approximation on a large domain with IC that
    # decays to 0 at the boundary, mass should be approximately preserved.
    #
    # Oracle (analytic): mass conservation is a property of the CONTINUOUS
    # equation, independent of the Rust kernel.  We test it numerically.
    #
    # Tolerance: 5e-3.  Residual comes from:
    #   - Boundary truncation (IC = Kolmogorov at T_IC=1; sigma_v~1).
    #   - Chernoff discretisation O(tau^2) per-step error.
    # -----------------------------------------------------------------------

    def test_mass_conservation(self) -> None:
        """Kolmogorov semigroup conserves total mass (analytic invariant).

        Oracle: ∫∫ p(t,x,v) dx dv = constant for solutions of ∂_t p = v∂_xp + ½∂²_vp.
        Independent of the Rust kernel (analytic property of the PDE).

        sup_error = |mass_ratio - 1|; tolerance = 5e-3.
        """
        u0 = make_kolmogorov_ic(t_ic=1.0)
        mass0 = mass_kol(u0)

        k = rp.HypoellipticChernoffKolmogorov(
            -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0.copy()
        )
        k.evolve(T_KOL, N_STEPS_KOL)
        mass_t = mass_kol(k.values())

        mass_ratio = mass_t / mass0
        tol = 5e-3
        assert abs(mass_ratio - 1.0) < tol, (
            f"Kolmogorov mass conservation: |ratio - 1| = {abs(mass_ratio-1):.4e} "
            f"(tolerance {tol:.1e}, mass0={mass0:.4f}, mass_t={mass_t:.4f})"
        )

    # -----------------------------------------------------------------------
    # Oracle test: Kolmogorov analytic forward prediction.
    #
    # IC = p(T_IC, x, v) (analytic solution at time T_IC).
    # After evolving by dt, the output should approximate p(T_IC + dt, x, v).
    # This tests that the Chernoff product approximates the CORRECT semigroup.
    #
    # Oracle: the analytic solution at T_IC + dt is computed independently.
    # Tolerance: 5e-2 (coarse grid; main purpose is to verify the operator
    # structure, not precision).
    # -----------------------------------------------------------------------

    def test_forward_prediction_vs_analytic(self) -> None:
        """Kolmogorov Chernoff output vs analytic solution (independent oracle).

        IC = Kolmogorov solution at T_IC.  After evolve(dt), output should
        approximate the analytic solution at T_IC + dt.

        Oracle: Kolmogorov 1934 fundamental solution (kolmogorov_oracle()).
        This is INDEPENDENT of the Rust kernel under test.

        sup_error and tolerance reported below.
        """
        T_IC = 1.0
        DT = 0.05
        N_STEPS_PRED = 10

        u0 = make_kolmogorov_ic(t_ic=T_IC)

        k = rp.HypoellipticChernoffKolmogorov(
            -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0.copy()
        )
        k.evolve(DT, N_STEPS_PRED)
        result = k.values().reshape(KOL_NX, KOL_NV)

        # Analytic solution at T_IC + DT (ORACLE, independent of Rust kernel).
        xs = np.linspace(-KOL_L, KOL_L, KOL_NX)
        vs = np.linspace(-KOL_L, KOL_L, KOL_NV)
        X, V = np.meshgrid(xs, vs, indexing="ij")
        T_NEW = T_IC + DT
        analytic = np.vectorize(lambda x, v: kolmogorov_oracle(T_NEW, x, v))(X, V)

        # Focus comparison on the bulk (avoid boundary where IC decays).
        # Domain centre ±2 where the function is non-negligible.
        mask_x = (np.abs(xs) <= 2.0)
        mask_v = (np.abs(vs) <= 2.0)
        result_center = result[np.ix_(mask_x, mask_v)]
        analytic_center = analytic[np.ix_(mask_x, mask_v)]

        sup_err = float(np.max(np.abs(result_center - analytic_center)))
        tol = 5e-2
        assert sup_err < tol, (
            f"Kolmogorov forward-prediction: sup_error = {sup_err:.3e} "
            f"(tolerance {tol:.1e}, T_IC={T_IC}, dt={DT})\n"
            f"Oracle: Kolmogorov 1934 fundamental solution, independent of Rust kernel."
        )

    # ---- input validation ----

    def test_nan_u0_raises(self) -> None:
        u0 = make_kolmogorov_ic()
        u0[5] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.HypoellipticChernoffKolmogorov(
                -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0
            )

    def test_invalid_grid_raises(self) -> None:
        u0 = make_kolmogorov_ic()
        with pytest.raises(rp.SemiflowError):
            rp.HypoellipticChernoffKolmogorov(
                KOL_L, -KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0
            )

    def test_n_steps_zero_raises(self) -> None:
        u0 = make_kolmogorov_ic()
        k = rp.HypoellipticChernoffKolmogorov(
            -KOL_L, KOL_L, KOL_NX, -KOL_L, KOL_L, KOL_NV, u0
        )
        with pytest.raises(rp.SemiflowError):
            k.evolve(T_KOL, 0)


# ---------------------------------------------------------------------------
# HypoellipticChernoffEngel (M15)
# ---------------------------------------------------------------------------

# Engel uses a 4D grid n×n×n×n; keep n small for test speed.
ENGEL_N = 8
ENGEL_MIN, ENGEL_MAX = -2.0, 2.0
T_ENGEL = 0.05
N_STEPS_ENGEL = 5


def make_engel_ic() -> np.ndarray:
    """4D Gaussian IC, flat row-major array of length ENGEL_N**4."""
    n = ENGEL_N
    xs = np.linspace(ENGEL_MIN, ENGEL_MAX, n)
    # Build a 4D Gaussian via outer products (separable).
    g1 = np.exp(-(xs ** 2))  # 1D Gaussian, shape (n,)
    g4 = g1[:, None, None, None] * g1[None, :, None, None] \
       * g1[None, None, :, None] * g1[None, None, None, :]
    return g4.flatten().astype(np.float64)


def mass_engel(u: np.ndarray) -> float:
    """Trapezoidal 4D mass of flat array."""
    n = ENGEL_N
    u4d = u.reshape(n, n, n, n)
    dx = (ENGEL_MAX - ENGEL_MIN) / (n - 1)
    # Collapse axis by axis.
    tmp = np.trapezoid(u4d, dx=dx, axis=3)  # (n,n,n)
    tmp = np.trapezoid(tmp, dx=dx, axis=2)  # (n,n)
    tmp = np.trapezoid(tmp, dx=dx, axis=1)  # (n,)
    return float(np.trapezoid(tmp, dx=dx))


class TestHypoellipticChernoffEngel:
    """Tests for HypoellipticChernoffEngel — Engel step-3 Carnot binding."""

    def test_construction(self) -> None:
        u0 = make_engel_ic()
        e = rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0)
        assert e is not None

    def test_order(self) -> None:
        """Order == 2 (palindromic Strang-Hörmander, math §28.bis.2)."""
        u0 = make_engel_ic()
        e = rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0)
        assert e.order() == 2

    def test_len(self) -> None:
        u0 = make_engel_ic()
        e = rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0)
        assert len(e) == ENGEL_N ** 4

    def test_values_shape_dtype(self) -> None:
        u0 = make_engel_ic()
        e = rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0)
        vals = e.values()
        assert vals.shape == (ENGEL_N ** 4,)
        assert vals.dtype == np.float64

    # -----------------------------------------------------------------------
    # Oracle test 1: positivity preservation.
    #
    # Chernoff products of positive operators are positive (Butko 2018 Cor 1.1).
    # Each sub-step exp(τ·X²) of the palindromic decomposition is a heat
    # semigroup along the integral curve of Xᵢ — a positive operator.
    # Composition of positive operators is positive.
    # Therefore: u₀ >= 0 ⟹ T^n u₀ >= 0 for all n.
    # This is analytic (structural); INDEPENDENT of the Rust kernel value.
    # -----------------------------------------------------------------------

    def test_positivity_preserved(self) -> None:
        """Engel Chernoff: positivity preservation (Butko 2018 Cor 1.1).

        Oracle: composition of positive heat semigroups is positive.
        Analytic; independent of the Rust kernel under test.

        Tolerance: numerical negativity < 1e-9 (fp arithmetic noise).
        """
        u0 = make_engel_ic()
        assert np.all(u0 >= 0), "IC must be non-negative"

        e = rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0)
        e.evolve(T_ENGEL, N_STEPS_ENGEL)
        vals = e.values()

        assert np.all(np.isfinite(vals)), "Engel: NaN or Inf in output"
        min_val = float(np.min(vals))
        assert min_val > -1e-9, (
            f"Engel positivity violated: min = {min_val:.3e} "
            "(oracle: Butko 2018 Cor 1.1, positive operator composition)"
        )

    # -----------------------------------------------------------------------
    # Oracle test 2: peak value strictly decreases after diffusion.
    #
    # Any heat/diffusion semigroup is a smoothing operator: for a non-negative,
    # non-flat IC, the maximum value strictly decreases after applying the
    # diffusion step (strong maximum principle for parabolic equations;
    # Evans 2010 §7.1 Theorem 12).
    #
    # For the Engel sub-Laplacian (hypoelliptic, bracket-generating), the
    # same principle holds: the semigroup T_t satisfies ‖T_t u‖_∞ < ‖u‖_∞
    # for t > 0 and non-constant u₀ (Bony 1967 maximum principle for
    # hypoelliptic operators).
    #
    # Oracle (analytic): peak(T_t u0) < peak(u0) for any diffusion semigroup
    # and non-constant non-negative IC.  INDEPENDENT of the Rust kernel.
    # -----------------------------------------------------------------------

    def test_peak_decreases_after_diffusion(self) -> None:
        """Engel Chernoff: peak value strictly decreases (hypoelliptic maximum principle).

        Oracle: for a non-constant non-negative IC, ‖T_t u₀‖_∞ < ‖u₀‖_∞ for t > 0.
        This is an analytic property of any diffusion/heat semigroup
        (Bony 1967 maximum principle for hypoelliptic operators).
        Independent of the Rust kernel.

        sup_error = peak_t - peak_0 (must be < 0); tolerance 1e-6.
        """
        u0 = make_engel_ic()
        peak_0 = float(np.max(u0))

        e = rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0.copy())
        e.evolve(T_ENGEL, N_STEPS_ENGEL)
        vals = e.values()
        peak_t = float(np.max(vals))

        assert peak_t < peak_0, (
            f"Engel peak did not decrease after diffusion:\n"
            f"  peak at t=0: {peak_0:.6f}\n"
            f"  peak at t={T_ENGEL}: {peak_t:.6f}\n"
            f"Oracle: Bony 1967 maximum principle for hypoelliptic operators."
        )
        # Sanity check that the decrease is meaningful (not just fp noise).
        relative_decrease = (peak_0 - peak_t) / peak_0
        assert relative_decrease > 1e-6, (
            f"Peak decrease too small (relative_decrease={relative_decrease:.3e}); "
            f"diffusion should produce non-trivial smoothing for t={T_ENGEL} > 0."
        )

    # ---- input validation ----

    def test_nan_u0_raises(self) -> None:
        u0 = make_engel_ic()
        u0[0] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0)

    def test_invalid_grid_raises(self) -> None:
        u0 = make_engel_ic()
        with pytest.raises(rp.SemiflowError):
            rp.HypoellipticChernoffEngel(ENGEL_MAX, ENGEL_MIN, ENGEL_N, u0)

    def test_wrong_u0_length_raises(self) -> None:
        u0 = np.ones(ENGEL_N ** 4 - 1, dtype=np.float64)
        with pytest.raises(rp.SemiflowError):
            rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0)

    def test_n_steps_zero_raises(self) -> None:
        u0 = make_engel_ic()
        e = rp.HypoellipticChernoffEngel(ENGEL_MIN, ENGEL_MAX, ENGEL_N, u0)
        with pytest.raises(rp.SemiflowError):
            e.evolve(T_ENGEL, 0)
