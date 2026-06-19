"""Wave P4 smoke tests — ADR-0111 parity for nonautonomous + subordinated kernels.

Covers:
  Howland1D     (M11) — HowlandLift<DiffusionChernoff> nonautonomous lift
  Subordinated1D (M12) — SubordinatedChernoff<DiffusionChernoff, Subordinator>
                          with "stable", "gamma", "inverse_gaussian" backends

Oracle strategy (independent, NOT self-referential):
  All oracles are derived from KNOWN ANALYTIC or ANALYTIC-LIMIT references
  that do NOT depend on the Rust kernel under test.

M11 Howland1D oracle — autonomous limit:
  The Howland lift wraps an autonomous (time-independent) DiffusionChernoff.
  For autonomous generators the lifted evolution at t = T equals the regular
  heat semigroup S(T) (Howland 1974, math §23.5):

    HowlandLift(∂², T, n_t).apply^{n_t-1}(u) = S(T) u

  The ORACLE is Heat1D (Wave P1-tested, independently validated).  We compare
  Howland1D.values() to Heat1D.values() after the same total evolution time T.
  This is NOT self-referential: Heat1D uses DiffusionChernoff directly
  (no Howland lift); Howland1D uses HowlandLift.  They share the same
  mathematical semigroup but follow entirely separate code paths.

  Tolerance: 5e-3.  Sources of discrepancy:
    - Howland uses n_steps = n_t - 1 (coarse temporal grid).
    - Heat1D uses n_steps = n_t - 1 too (for a fair comparison).
    - Remaining error is from the order-1 Howland approximation on the
      temporal axis (delta_s discretisation).

M12 Subordinated1D oracle — semigroup positivity + monotone decay:
  Subordinated semigroups inherit positivity preservation from the base
  heat semigroup (Schilling-Song-Vondraček 2012 §13):
    u₀ >= 0 → T^φ_t u₀ >= 0 for all t > 0.

  Additionally, the total L1-mass is NOT conserved (for strictly subordinated
  semigroups with φ(0)=0 but φ'(0+) = +∞, like stable and IG):
    ∫ T^φ_t u₀ dx < ∫ u₀ dx  for t > 0.

  For the Gamma subordinator (φ(λ) = log(1 + λ/c)), the subordinated
  semigroup is the relativistic heat semigroup.  For a unit Gaussian IC on a
  large domain, the total mass evolves as:
    mass(t) ≈ exp(−t · φ_c(0+)) · mass(0)
  but since φ_c(0) = 0 for all three backends, mass decay comes from the
  interaction of the spatial boundary and the semigroup, not from killing.
  We therefore use the WEAKER oracle: mass decreases or stays approximately
  constant.  The absence of mass growth (mass <= mass_0 * 1.01) is the
  independent test.

  For the stable backend we can also check the long-time anomalous diffusion:
  fractional heat spreads slower than standard heat.  At t=0.1, the sup-norm
  of the stable-subordinated solution must be LARGER than the standard heat
  solution (u spreads less → peak decays less for a Gaussian IC with α < 1).
  sup_norm(stable, t) >= sup_norm(heat, t) * (1 - small_tol).
  This is an analytic invariant of fractional calculus (Applebaum 2009 §3.3).

Additional analytic checks:
  - Each class reports order() == 1 (Butko 2018, Howland 1974).
  - Constructor validation: invalid inputs raise SemiflowError.
  - All three Subordinated1D backends produce finite, non-negative outputs
    for a non-negative IC (positivity preservation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------

XMIN, XMAX, N = -5.0, 5.0, 64
T = 0.1
N_STEPS = 50  # kept small so tests run fast; Howland uses n_t - 1 steps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_u0(xmin: float = XMIN, xmax: float = XMAX, n: int = N) -> np.ndarray:
    """Gaussian IC: u0(x) = exp(-x^2).  Non-negative, unit-peak."""
    xs = np.linspace(xmin, xmax, n)
    return np.exp(-(xs**2)).astype(np.float64)


def mass(u: np.ndarray, xmin: float, xmax: float) -> float:
    """Trapezoidal approximation of the integral."""
    dx = (xmax - xmin) / (len(u) - 1)
    return float(np.trapezoid(u, dx=dx))


# ---------------------------------------------------------------------------
# Howland1D (M11)
# ---------------------------------------------------------------------------


class TestHowland1D:
    """Tests for Howland1D — nonautonomous Howland-lift binding."""

    def test_construction(self) -> None:
        u0 = make_u0()
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=11, t_horizon=T)
        assert h is not None

    def test_len(self) -> None:
        u0 = make_u0()
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=11, t_horizon=T)
        assert len(h) == N

    def test_order(self) -> None:
        u0 = make_u0()
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=11, t_horizon=T)
        assert h.order() == 1

    def test_n_t_property(self) -> None:
        u0 = make_u0()
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=11, t_horizon=T)
        assert h.n_t() == 11

    def test_t_horizon_property(self) -> None:
        u0 = make_u0()
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=11, t_horizon=T)
        assert abs(h.t_horizon() - T) < 1e-15

    def test_delta_s_property(self) -> None:
        u0 = make_u0()
        n_t = 11
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=n_t, t_horizon=T)
        expected = T / (n_t - 1)
        assert abs(h.delta_s() - expected) < 1e-14

    def test_values_initial_shape_dtype(self) -> None:
        u0 = make_u0()
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=11, t_horizon=T)
        vals = h.values()
        assert vals.shape == (N,)
        assert vals.dtype == np.float64

    # -----------------------------------------------------------------------
    # Oracle test: autonomous-limit identity.
    #
    # For an autonomous DiffusionChernoff generator, the Howland-lifted
    # evolution at t = t_horizon equals the standard heat semigroup S(T).
    # ORACLE: Heat1D.evolve(T, n_steps).values() — uses a completely
    # different code path (direct ChernoffSemigroup, no HowlandLift).
    #
    # Both are evolved with the same n_steps = n_t - 1 = 50 Chernoff steps.
    # Tolerance 5e-3: the Howland lift is order-1 on the temporal axis
    # (delta_s grid), so per-step error is O(delta_s) = O(T/(n_t-1)).
    # With n_t=51, delta_s=0.002, total error ~ n_steps * delta_s^2 ≈ 1e-4.
    # We use the generous 5e-3 bound to cover all grid sizes robustly.
    # -----------------------------------------------------------------------

    def test_autonomous_limit_equals_heat1d(self) -> None:
        """Howland1D autonomous limit equals Heat1D oracle (independent paths).

        Oracle: Heat1D.evolve(T, n_steps) — unit diffusion, direct code path.
        Howland1D uses HowlandLift; Heat1D uses DiffusionChernoff directly.
        Both are mathematically equal for autonomous generators.

        sup_error and tolerance reported below.
        """
        n_t = N_STEPS + 1  # n_steps = n_t - 1 = N_STEPS internally

        u0 = make_u0()

        # Heat1D reference — direct DiffusionChernoff (ORACLE, independent).
        heat = rp.Heat1D(XMIN, XMAX, N, u0.copy())
        heat.evolve(T, N_STEPS)
        ref = heat.values()

        # Howland1D — HowlandLift code path (under test).
        howland = rp.Howland1D(XMIN, XMAX, N, u0.copy(), n_t=n_t, t_horizon=T)
        howland.evolve()
        result = howland.values()

        sup_err = float(np.max(np.abs(result - ref)))
        tol = 5e-3
        assert sup_err < tol, (
            f"Howland1D autonomous-limit: sup_error = {sup_err:.3e} "
            f"(tolerance {tol:.1e}, n_t={n_t}, T={T})"
        )

    def test_evolve_output_finite(self) -> None:
        """Evolved values are all finite."""
        u0 = make_u0()
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=11, t_horizon=T)
        h.evolve()
        vals = h.values()
        assert np.all(np.isfinite(vals)), "Howland1D evolve produced NaN or Inf"

    def test_evolve_output_non_negative(self) -> None:
        """Heat semigroup preserves non-negativity of a non-negative IC."""
        u0 = make_u0()
        assert np.all(u0 >= 0), "IC should be non-negative"
        h = rp.Howland1D(XMIN, XMAX, N, u0, n_t=21, t_horizon=T)
        h.evolve()
        vals = h.values()
        # Small numerical negativity from fp arithmetic is tolerated.
        assert np.min(vals) > -1e-10, (
            f"Howland1D: negative values in output (min={np.min(vals):.3e})"
        )

    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------

    def test_nan_u0_raises(self) -> None:
        u0 = make_u0()
        u0[5] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.Howland1D(XMIN, XMAX, N, u0)

    def test_invalid_n_t_one_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Howland1D(XMIN, XMAX, N, u0, n_t=1)

    def test_invalid_t_horizon_zero_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Howland1D(XMIN, XMAX, N, u0, t_horizon=0.0)

    def test_invalid_t_horizon_negative_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Howland1D(XMIN, XMAX, N, u0, t_horizon=-1.0)

    def test_invalid_grid_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Howland1D(5.0, -5.0, N, u0)  # xmin >= xmax


# ---------------------------------------------------------------------------
# Subordinated1D (M12)
# ---------------------------------------------------------------------------


class TestSubordinated1D:
    """Tests for Subordinated1D — subordinated Chernoff binding."""

    # --- Construction ---

    def test_stable_construction(self) -> None:
        u0 = make_u0()
        s = rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="stable", alpha=0.5)
        assert s is not None

    def test_gamma_construction(self) -> None:
        u0 = make_u0()
        s = rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="gamma", c=1.0)
        assert s is not None

    def test_ig_construction(self) -> None:
        u0 = make_u0()
        s = rp.Subordinated1D(
            XMIN, XMAX, N, u0, subordinator="inverse_gaussian", c=1.0
        )
        assert s is not None

    def test_len(self) -> None:
        u0 = make_u0()
        assert len(rp.Subordinated1D(XMIN, XMAX, N, u0)) == N

    def test_order(self) -> None:
        u0 = make_u0()
        s = rp.Subordinated1D(XMIN, XMAX, N, u0)
        assert s.order() == 1

    def test_values_initial_shape_dtype(self) -> None:
        u0 = make_u0()
        vals = rp.Subordinated1D(XMIN, XMAX, N, u0).values()
        assert vals.shape == (N,)
        assert vals.dtype == np.float64

    # -----------------------------------------------------------------------
    # Oracle test 1: positivity preservation for all backends.
    #
    # Subordinated semigroups inherit positivity from the base heat semigroup
    # (Schilling-Song-Vondraček 2012 §13 — any CBF subordinator preserves
    # positivity of the semigroup).  For u₀ >= 0, T^φ_t u₀ >= 0 at all t.
    # This is an analytic invariant INDEPENDENT of the Rust kernel.
    # -----------------------------------------------------------------------

    def _check_positivity(self, kern: rp.Subordinated1D) -> None:  # type: ignore[name-defined]
        kern.evolve(T, N_STEPS)
        vals = kern.values()
        assert np.all(np.isfinite(vals)), "Subordinated1D output contains NaN/Inf"
        assert np.min(vals) > -1e-10, (
            f"Subordinated1D: negative values (min={np.min(vals):.3e}); "
            "positivity preservation violated"
        )

    def test_stable_positivity_preserved(self) -> None:
        """Stable subordinator: positivity preserved (analytic invariant).

        Oracle: T^phi_t u0 >= 0 for any u0 >= 0 (Schilling-Song-Vondraček §13).
        This holds for ALL CBF subordinators, independently of the Rust kernel.
        """
        u0 = make_u0()
        s = rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="stable", alpha=0.5)
        self._check_positivity(s)

    def test_gamma_positivity_preserved(self) -> None:
        """Gamma subordinator: positivity preserved."""
        u0 = make_u0()
        s = rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="gamma", c=1.0)
        self._check_positivity(s)

    def test_ig_positivity_preserved(self) -> None:
        """Inverse-Gaussian subordinator: positivity preserved."""
        u0 = make_u0()
        s = rp.Subordinated1D(
            XMIN, XMAX, N, u0, subordinator="inverse_gaussian", c=1.0
        )
        self._check_positivity(s)

    # -----------------------------------------------------------------------
    # Oracle test 2: mass non-growth (stable subordinator).
    #
    # For the stable-subordinated semigroup T^alpha_t = exp(-t(-Delta)^alpha),
    # the semigroup is sub-Markovian (Phillips calculus): T^alpha_t 1 <= 1
    # (Applebaum 2009 §3.3, Kwasnicki 2017).  For a non-negative IC with
    # mass m0 = integral(u0 dx), we have mass(t) <= m0 for all t >= 0.
    # We verify: mass after evolve <= initial mass + 1% numerical tolerance.
    # -----------------------------------------------------------------------

    def test_stable_mass_non_growth(self) -> None:
        """Stable subordinated semigroup is sub-Markovian: mass <= initial mass.

        Oracle: T^alpha_t 1 <= 1 (sub-Markovian), so integral(T^alpha_t u0 dx)
        <= integral(u0 dx).  Independent of the Rust kernel (analytic invariant
        of the Phillips calculus + CBF Lévy subordinator).

        sup_error = mass ratio reported below; tolerance 1%.
        """
        u0 = make_u0()
        mass0 = mass(u0, XMIN, XMAX)
        s = rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="stable", alpha=0.5)
        s.evolve(T, N_STEPS)
        vals = s.values()
        mass_t = mass(vals, XMIN, XMAX)
        ratio = mass_t / mass0
        # Sub-Markovian: mass_t / mass0 <= 1 + 1% (numerical tolerance).
        assert ratio <= 1.01, (
            f"Subordinated1D (stable): mass ratio {ratio:.4f} > 1.01 "
            "(sub-Markovian oracle violated)"
        )

    # -----------------------------------------------------------------------
    # Oracle test 3: anomalous diffusion — slower spread for stable.
    #
    # Fractional diffusion exp(-t(-Δ)^α) with α < 1 spreads SLOWER than
    # standard diffusion exp(-t(-Δ)) (Metzler-Klafter 2000 Physics Reports).
    # For a Gaussian IC u0, the peak value (u at x=0) decays as:
    #   standard heat: peak(t) ≈ 1/sqrt(1+4t)
    #   stable frac.:  peak(t) > 1/sqrt(1+4t)  for α < 1
    # The inequality is analytic (slower spread → slower peak decay).
    # We compare the peak of Subordinated1D (stable, α=0.5) to
    # the standard heat oracle at t=0.1:
    #   heat oracle: peak = exp(0) / sqrt(1 + 4 * 0.1) = 1/sqrt(1.4)
    # Tolerance: 3%.
    # -----------------------------------------------------------------------

    def test_stable_slower_spread_than_heat(self) -> None:
        """Stable subordinated heat spreads slower than standard heat.

        Oracle: fractional diffusion peak decays slower than 1/sqrt(1+4t)
        for alpha < 1 (Metzler-Klafter 2000, analytic bound).
        The oracle for standard heat is: peak_heat = 1/sqrt(1 + 4*T).
        The subordinated peak at x=0 must be >= peak_heat * (1 - 0.03).

        sup_error = absolute difference from oracle lower bound.
        Tolerance: 3% of the heat oracle peak.
        """
        u0 = make_u0()
        # Standard heat oracle at t = T (analytic, independent of Rust kernel).
        heat_oracle_peak = 1.0 / math.sqrt(1.0 + 4.0 * T)

        # Subordinated1D stable evolve.
        s = rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="stable", alpha=0.5)
        s.evolve(T, N_STEPS)
        vals = s.values()

        # Peak is near x=0 (index N//2 for symmetric grid).
        idx_center = N // 2
        subord_peak = float(vals[idx_center])

        # Fractional diffusion must have a higher (or equal) peak than standard.
        lower_bound = heat_oracle_peak * 0.97
        assert subord_peak >= lower_bound, (
            f"Subordinated1D (stable): peak at x=0 = {subord_peak:.5f}, "
            f"expected >= {lower_bound:.5f} (heat oracle = {heat_oracle_peak:.5f}). "
            f"Fractional diffusion should spread slower than standard heat."
        )

    # -----------------------------------------------------------------------
    # Oracle test 4: different subordinators give different results.
    #
    # The three subordinators have different Laplace exponents phi(lambda),
    # so they generate genuinely different semigroups.  For any non-trivial IC
    # and t > 0, the outputs must differ (structural uniqueness).
    # -----------------------------------------------------------------------

    def test_different_subordinators_give_different_outputs(self) -> None:
        """Different subordinator strings produce structurally different outputs.

        Oracle: phi_stable != phi_gamma != phi_ig (analytic, different CBFs).
        Therefore the semigroups are different operators => different outputs
        for any non-trivial IC.  This is analytic, independent of Rust kernel.
        """
        u0 = make_u0()

        s_stable = rp.Subordinated1D(
            XMIN, XMAX, N, u0.copy(), subordinator="stable", alpha=0.5
        )
        s_stable.evolve(T, N_STEPS)

        s_gamma = rp.Subordinated1D(
            XMIN, XMAX, N, u0.copy(), subordinator="gamma", c=1.0
        )
        s_gamma.evolve(T, N_STEPS)

        s_ig = rp.Subordinated1D(
            XMIN, XMAX, N, u0.copy(), subordinator="inverse_gaussian", c=1.0
        )
        s_ig.evolve(T, N_STEPS)

        diff_sg = float(np.max(np.abs(s_stable.values() - s_gamma.values())))
        diff_si = float(np.max(np.abs(s_stable.values() - s_ig.values())))
        diff_gi = float(np.max(np.abs(s_gamma.values() - s_ig.values())))

        assert diff_sg > 1e-10, f"stable and gamma gave same output (diff={diff_sg:.3e})"
        assert diff_si > 1e-10, f"stable and IG gave same output (diff={diff_si:.3e})"
        assert diff_gi > 1e-10, f"gamma and IG gave same output (diff={diff_gi:.3e})"

    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------

    def test_unknown_subordinator_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="unknown_type")

    def test_invalid_alpha_zero_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="stable", alpha=0.0)

    def test_invalid_alpha_one_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="stable", alpha=1.0)

    def test_invalid_c_zero_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Subordinated1D(XMIN, XMAX, N, u0, subordinator="gamma", c=0.0)

    def test_invalid_n_nodes_zero_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Subordinated1D(XMIN, XMAX, N, u0, n_nodes=0)

    def test_invalid_n_nodes_over_cap_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.Subordinated1D(XMIN, XMAX, N, u0, n_nodes=33)

    def test_nan_u0_raises(self) -> None:
        u0 = make_u0()
        u0[5] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.Subordinated1D(XMIN, XMAX, N, u0)

    def test_n_steps_zero_raises(self) -> None:
        u0 = make_u0()
        s = rp.Subordinated1D(XMIN, XMAX, N, u0)
        with pytest.raises(rp.SemiflowError):
            s.evolve(T, 0)

    def test_negative_t_raises(self) -> None:
        u0 = make_u0()
        s = rp.Subordinated1D(XMIN, XMAX, N, u0)
        with pytest.raises(rp.SemiflowError):
            s.evolve(-0.1)
