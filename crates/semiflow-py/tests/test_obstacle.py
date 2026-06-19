"""Obstacle Chernoff smoke tests — v6.3.0 projective-splitting VI kernel (math §44).

Covers ObstacleChernoff (constant and array obstacle variants).

Oracle strategy:
  The fundamental invariant is the post-projection lower bound: after every
  Chernoff step ``V^{n+1} = Π_g(S(Δτ)Vⁿ)``, the result satisfies
  ``V(t) ≥ g`` elementwise (metric projection, Theorem 44.1).
  This is exact (algebraic) and independent of the time-stepping scheme.

Tests:
  (a) ``evolve`` result is >= obstacle floor elementwise (projection lower bound).
  (b) Result shape matches n.
  (c) Result dtype is float64.
  (d) Both ConstantObstacle (level=) and ArrayObstacle (obstacle_array=) variants.
  (e) Input validation: bad params raise SemiflowError.
  (f) ``evolve_active_set_adjoint`` raises Unsupported (DiffusionChernoff limitation).
"""

from __future__ import annotations

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------

XMIN, XMAX, N = 0.0, 1.0, 64
T = 0.05
N_STEPS = 50
LEVEL = 0.1  # constant obstacle floor


def make_u0() -> np.ndarray:
    """IC below the obstacle floor so projection effect is clearly visible."""
    xs = np.linspace(XMIN, XMAX, N)
    return (0.5 * np.sin(np.pi * xs) * 0.05).astype(np.float64)


def make_obstacle_array() -> np.ndarray:
    """Spatially varying floor: ramp from 0 to 0.2."""
    xs = np.linspace(XMIN, XMAX, N)
    return (0.2 * xs).astype(np.float64)


# ---------------------------------------------------------------------------
# Construction and shape
# ---------------------------------------------------------------------------


class TestObstacleChernoffConstruction:
    def test_constant_obstacle_constructs(self) -> None:
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        assert kern is not None

    def test_array_obstacle_constructs(self) -> None:
        u0 = make_u0()
        g = make_obstacle_array()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, obstacle_array=g)
        assert kern is not None

    def test_len_matches_n(self) -> None:
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        assert len(kern) == N

    def test_order_is_1(self) -> None:
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        assert kern.order() == 1

    def test_initial_values_shape(self) -> None:
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        vals = kern.values()
        assert vals.shape == (N,)
        assert vals.dtype == np.float64


# ---------------------------------------------------------------------------
# Oracle tests: projection lower bound V(t) >= g (analytic invariant)
# ---------------------------------------------------------------------------


class TestProjectionLowerBound:
    def test_constant_obstacle_lower_bound(self) -> None:
        """After evolve, all nodes satisfy V(t) >= level (Theorem 44.1).

        Oracle: Π_g(W) = max(W, g) guarantees V >= g elementwise.
        This is an algebraic post-step invariant — not a numerical approximation.
        """
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        result = kern.evolve(T, N_STEPS)
        min_val = float(np.min(result))
        assert min_val >= LEVEL - 1e-12, (
            f"ObstacleChernoff (constant): min(V(t)) = {min_val:.6f} < level={LEVEL}"
        )

    def test_array_obstacle_lower_bound(self) -> None:
        """After evolve with array obstacle, V(t) >= g[i] elementwise.

        Oracle: same algebraic invariant as constant case (Theorem 44.1).
        """
        u0 = make_u0()
        g = make_obstacle_array()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, obstacle_array=g)
        result = kern.evolve(T, N_STEPS)
        violations = result - g
        min_margin = float(np.min(violations))
        assert min_margin >= -1e-12, (
            f"ObstacleChernoff (array): min(V-g) = {min_margin:.6f} < 0"
        )

    def test_result_shape_matches_n(self) -> None:
        """evolve returns array of length n."""
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        result = kern.evolve(T, N_STEPS)
        assert result.shape == (N,), f"shape mismatch: {result.shape} != ({N},)"

    def test_result_dtype_float64(self) -> None:
        """evolve returns float64 array."""
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        result = kern.evolve(T, N_STEPS)
        assert result.dtype == np.float64

    def test_result_all_finite(self) -> None:
        """No NaN or Inf in evolve output."""
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        result = kern.evolve(T, N_STEPS)
        assert np.all(np.isfinite(result)), "evolve returned non-finite values"

    def test_values_matches_last_evolve(self) -> None:
        """values() returns the same array as the last evolve output."""
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        out = kern.evolve(T, N_STEPS)
        vals = kern.values()
        np.testing.assert_array_equal(out, vals)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_nan_u0_raises(self) -> None:
        u0 = make_u0()
        u0[5] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)

    def test_non_finite_level_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=float("nan"))

    def test_non_finite_a_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.ObstacleChernoff(XMIN, XMAX, N, u0, a=float("nan"), level=LEVEL)

    def test_negative_a_raises(self) -> None:
        u0 = make_u0()
        with pytest.raises(rp.SemiflowError):
            rp.ObstacleChernoff(XMIN, XMAX, N, u0, a=-1.0, level=LEVEL)

    def test_n_steps_zero_raises(self) -> None:
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        with pytest.raises(rp.SemiflowError):
            kern.evolve(T, 0)

    def test_negative_t_raises(self) -> None:
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        with pytest.raises(rp.SemiflowError):
            kern.evolve(-0.1)

    def test_array_obstacle_nan_raises(self) -> None:
        u0 = make_u0()
        g = make_obstacle_array()
        g[3] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.ObstacleChernoff(XMIN, XMAX, N, u0, obstacle_array=g)


# ---------------------------------------------------------------------------
# Adjoint limitation (documented Unsupported behaviour)
# ---------------------------------------------------------------------------


class TestAdjointLimitation:
    def test_evolve_active_set_adjoint_raises_unsupported(self) -> None:
        """DiffusionChernoff inner has no adjoint primitive (ADR-0114).

        evolve_active_set_adjoint must raise SemiflowError. The error message
        must mention "adjoint" to confirm the correct error path was hit.
        This is the documented limitation; callers should use the Adjoint wrapper
        for the self-adjoint forward-adjoint path.
        """
        u0 = make_u0()
        kern = rp.ObstacleChernoff(XMIN, XMAX, N, u0, level=LEVEL)
        lam = np.ones(N, dtype=np.float64)
        w_fwd = np.ones(N, dtype=np.float64) * 0.5
        with pytest.raises(rp.SemiflowError) as exc_info:
            kern.evolve_active_set_adjoint(w_fwd, lam, 0.01)
        # The error message must mention adjoint (confirms the right code path).
        msg = str(exc_info.value).lower()
        assert "adjoint" in msg, (
            f"Expected 'adjoint' in error message, got: {exc_info.value!r}"
        )


# ---------------------------------------------------------------------------
# Drift (b) and reaction (c) coefficient tests
# ---------------------------------------------------------------------------


class TestDriftReaction:
    """Tests for b,c kwargs added in v6.3.0 (generator L = a u_xx + b u_x + c u)."""

    def test_reaction_decay_c_negative(self) -> None:
        """With c < 0 the solution decays faster than the c=0 case.

        Oracle: for c = -r < 0 the ODE part gives exp(c·t) factor, so
        sum(u_c) < sum(u_0) (with the obstacle floor removed from accounting).
        """
        xmin, xmax, n_pts = -2.0, 2.0, 128
        xs = np.linspace(xmin, xmax, n_pts)
        u0 = np.exp(-(xs ** 2)).astype(np.float64)

        kern0 = rp.ObstacleChernoff(xmin, xmax, n_pts, u0.copy(), a=0.01, c=0.0, level=-1.0)
        kern_c = rp.ObstacleChernoff(xmin, xmax, n_pts, u0.copy(), a=0.01, c=-0.5, level=-1.0)

        T, steps = 1.0, 300
        r0 = kern0.evolve(T, steps)
        rc = kern_c.evolve(T, steps)

        assert rc.max() < r0.max(), (
            f"c=-0.5 should decay: max_c={rc.max():.4f} >= max_0={r0.max():.4f}"
        )

    def test_drift_shifts_peak(self) -> None:
        """With b ≠ 0 the peak of a Gaussian IC shifts in the drift direction.

        Oracle: drift `b u_x` in `L = a u_xx + b u_x` advects the peak by
        approximately `-b·T` over time T (characteristic flow).
        """
        xmin, xmax, n_pts = -3.0, 5.0, 256
        xs = np.linspace(xmin, xmax, n_pts)
        u0 = np.exp(-(xs ** 2)).astype(np.float64)

        T = 1.0
        steps = 500
        kern0 = rp.ObstacleChernoff(xmin, xmax, n_pts, u0.copy(), a=0.01, b=0.0, level=-1.0)
        kern_b = rp.ObstacleChernoff(xmin, xmax, n_pts, u0.copy(), a=0.01, b=0.5, level=-1.0)

        r0 = kern0.evolve(T, steps)
        rb = kern_b.evolve(T, steps)
        peak0 = xs[r0.argmax()]
        peakb = xs[rb.argmax()]
        shift = peakb - peak0
        # drift advects negatively for positive b (characteristics move right,
        # profile shifts left relative to b=0 stationary solution)
        assert abs(shift) > 0.1, f"Drift b=0.5 should shift peak, shift={shift:.3f}"

    def test_projection_floor_with_bc(self) -> None:
        """V(t) >= g still holds with b,c active (Theorem 44.1 invariant)."""
        xmin, xmax, n_pts = 0.0, 1.0, 64
        xs = np.linspace(xmin, xmax, n_pts)
        u0 = (0.05 * np.sin(np.pi * xs)).astype(np.float64)
        g = (0.1 * xs).astype(np.float64)

        kern = rp.ObstacleChernoff(xmin, xmax, n_pts, u0.copy(), a=0.5, b=0.2, c=-0.1,
                                   obstacle_array=g)
        result = kern.evolve(0.05, 50)
        violations = result - g
        assert np.min(violations) >= -1e-12, (
            f"Obstacle floor violated with b,c: min(V-g)={np.min(violations):.2e}"
        )

    def test_invalid_b_nan_raises(self) -> None:
        """Non-finite b raises SemiflowError."""
        u0 = np.ones(64, dtype=np.float64) * 0.5
        with pytest.raises(rp.SemiflowError):
            rp.ObstacleChernoff(0.0, 1.0, 64, u0, b=float("nan"), level=0.0)
