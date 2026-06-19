"""Tests for Heat1D.with_a_function (variable diffusion coefficient).

ADR-0034: variable a(x) via Python callables.

GIL note (ADR-0031 / ADR-0034): each callback re-acquires the GIL
(~2-5 µs on CPython 3.11), defeating the GIL-release optimisation.
These tests use small grids (n=64) to keep wall-clock reasonable.
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
XMIN = -4.0
XMAX = 4.0
N = 64
TAU = 0.01
N_STEPS = 1


def _gaussian(xs: np.ndarray) -> np.ndarray:
    return np.exp(-xs * xs)


# ---------------------------------------------------------------------------
# Test 1: constant a(x) = 1.0 matches unit-a constructor bit-for-bit
# ---------------------------------------------------------------------------

def test_var_a_constant_matches_unit() -> None:
    """Constant a(x)=1.0 via with_a_function must match Heat1D bit-exactly."""
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)

    h_var = rp.Heat1D.with_a_function(
        xmin=XMIN, xmax=XMAX, n=N,
        a=lambda _: 1.0,
        a_prime=lambda _: 0.0,
        a_double_prime=lambda _: 0.0,
        a_norm_bound=1.0,
        u0=u0,
    )
    h_unit = rp.Heat1D(xmin=XMIN, xmax=XMAX, n=N, u0=u0)

    h_var.evolve(TAU, n_steps=N_STEPS)
    h_unit.evolve(TAU, n_steps=N_STEPS)

    out_var = h_var.values()
    out_unit = h_unit.values()

    np.testing.assert_array_equal(
        out_var, out_unit,
        err_msg="with_a_function(a=1, a'=0, a''=0) must be bit-equal to Heat1D unit-a",
    )


# ---------------------------------------------------------------------------
# Test 2: actual variable a(x) produces finite positive output
# ---------------------------------------------------------------------------

def test_var_a_actual_variation() -> None:
    """a(x) = 1 + 0.1*x^2 with correct derivatives produces valid output."""
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)

    # a(x) = 1 + 0.1 * x^2  =>  a'(x) = 0.2*x,  a''(x) = 0.2
    # ||a||_inf on [-4, 4]: a(4) = 1 + 0.1*16 = 2.6
    h = rp.Heat1D.with_a_function(
        xmin=XMIN, xmax=XMAX, n=N,
        a=lambda x: 1.0 + 0.1 * x * x,
        a_prime=lambda x: 0.2 * x,
        a_double_prime=lambda _: 0.2,
        a_norm_bound=2.6,
        u0=u0,
    )

    h.evolve(TAU, n_steps=N_STEPS)
    out = h.values()

    assert out.shape == (N,), "output shape must match n"
    assert np.all(np.isfinite(out)), "output must be all-finite"
    # Gaussian initial datum + diffusion preserves positivity for short time
    assert np.all(out > 0), "positivity must be preserved for Gaussian + short diffusion"


# ---------------------------------------------------------------------------
# Test 3: non-callable coefficient raises SemiflowError at evolve time
# ---------------------------------------------------------------------------

def test_var_a_non_callable_raises_at_evolve() -> None:
    """Non-callable coefficient is silently stored; error surfaces at evolve time.

    The closure returns NAN when calling a non-callable; validate_a_x in the
    Chernoff kernel converts that to a DomainViolation (SemiflowError).
    """
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)

    # Construction succeeds (closures are only called during evolve)
    h = rp.Heat1D.with_a_function(
        xmin=XMIN, xmax=XMAX, n=N,
        a=42.0,  # not a callable — will error at evolve time
        a_prime=lambda _: 0.0,
        a_double_prime=lambda _: 0.0,
        a_norm_bound=1.0,
        u0=u0,
    )
    # evolve must raise because a(x_pre) = NAN fails validate_a_x
    with pytest.raises(rp.SemiflowError):
        h.evolve(TAU, n_steps=N_STEPS)


# ---------------------------------------------------------------------------
# Test 4: grid mismatch error is raised correctly
# ---------------------------------------------------------------------------

def test_var_a_grid_mismatch() -> None:
    """u0 length mismatch must raise SemiflowError(kind='GridMismatch')."""
    u0_wrong = np.ones(N + 1)

    with pytest.raises(rp.SemiflowError):
        rp.Heat1D.with_a_function(
            xmin=XMIN, xmax=XMAX, n=N,
            a=lambda _: 1.0,
            a_prime=lambda _: 0.0,
            a_double_prime=lambda _: 0.0,
            a_norm_bound=1.0,
            u0=u0_wrong,
        )
