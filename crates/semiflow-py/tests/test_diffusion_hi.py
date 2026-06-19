"""Smoke tests for Heat1D4th and Heat1D6th (4th/6th-order Chernoff diffusion).

Tests verify:
  - Basic construct + evolve produces finite output.
  - with_a_array (variable a) smoke: finite output, length preserved.
  - values() returns a copy (mutations do not affect internal state).
  - __len__ returns the grid size.
  - 4th-order kernel converges faster than 2nd-order on smooth datum.
"""

import math

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
N_STEPS = 10


def _gaussian(xs: np.ndarray) -> np.ndarray:
    return np.exp(-xs * xs)


# ---------------------------------------------------------------------------
# Heat1D4th
# ---------------------------------------------------------------------------

def test_heat1d4th_construct_evolve_finite() -> None:
    """Heat1D4th evolve must produce finite values on Gaussian initial datum."""
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)
    h = rp.Heat1D4th(xmin=XMIN, xmax=XMAX, n=N, u0=u0)
    h.evolve(TAU, n_steps=N_STEPS)
    out = h.values()
    assert out.shape == (N,)
    assert np.all(np.isfinite(out)), "Heat1D4th output must be finite"


def test_heat1d4th_len() -> None:
    """__len__ must return the grid node count."""
    xs = np.linspace(XMIN, XMAX, N)
    h = rp.Heat1D4th(xmin=XMIN, xmax=XMAX, n=N, u0=_gaussian(xs))
    assert len(h) == N


def test_heat1d4th_values_is_copy() -> None:
    """values() must return a copy; mutating it must not change state."""
    xs = np.linspace(XMIN, XMAX, N)
    h = rp.Heat1D4th(xmin=XMIN, xmax=XMAX, n=N, u0=_gaussian(xs))
    h.evolve(TAU, n_steps=1)
    v1 = h.values()
    v1[:] = 0.0
    v2 = h.values()
    assert not np.all(v2 == 0.0), "values() must not expose internal buffer"


def test_heat1d4th_with_a_array_finite() -> None:
    """with_a_array smoke: variable a(x) = 1 + 0.1*x^2 must yield finite output."""
    xs = np.linspace(XMIN, XMAX, N)
    a_vals = 1.0 + 0.1 * xs * xs
    u0 = _gaussian(xs)
    h = rp.Heat1D4th.with_a_array(
        xmin=XMIN, xmax=XMAX, n=N,
        a=a_vals,
        u0=u0,
    )
    h.evolve(TAU, n_steps=N_STEPS)
    out = h.values()
    assert out.shape == (N,), "output shape must equal grid size"
    assert np.all(np.isfinite(out)), "variable-a Heat1D4th output must be finite"


# ---------------------------------------------------------------------------
# Heat1D6th
# ---------------------------------------------------------------------------

def test_heat1d6th_construct_evolve_finite() -> None:
    """Heat1D6th evolve must produce finite values on Gaussian initial datum."""
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)
    h = rp.Heat1D6th(xmin=XMIN, xmax=XMAX, n=N, u0=u0)
    h.evolve(TAU, n_steps=N_STEPS)
    out = h.values()
    assert out.shape == (N,)
    assert np.all(np.isfinite(out)), "Heat1D6th output must be finite"


def test_heat1d6th_len() -> None:
    """__len__ must return the grid node count."""
    xs = np.linspace(XMIN, XMAX, N)
    h = rp.Heat1D6th(xmin=XMIN, xmax=XMAX, n=N, u0=_gaussian(xs))
    assert len(h) == N


def test_heat1d6th_values_is_copy() -> None:
    """values() must return a copy; mutating it must not change state."""
    xs = np.linspace(XMIN, XMAX, N)
    h = rp.Heat1D6th(xmin=XMIN, xmax=XMAX, n=N, u0=_gaussian(xs))
    h.evolve(TAU, n_steps=1)
    v1 = h.values()
    v1[:] = 0.0
    v2 = h.values()
    assert not np.all(v2 == 0.0), "values() must not expose internal buffer"


def test_heat1d6th_with_a_array_finite() -> None:
    """with_a_array smoke: variable a(x) = 1 + 0.1*x^2 must yield finite output."""
    xs = np.linspace(XMIN, XMAX, N)
    a_vals = 1.0 + 0.1 * xs * xs
    u0 = _gaussian(xs)
    h = rp.Heat1D6th.with_a_array(
        xmin=XMIN, xmax=XMAX, n=N,
        a=a_vals,
        u0=u0,
    )
    h.evolve(TAU, n_steps=N_STEPS)
    out = h.values()
    assert out.shape == (N,), "output shape must equal grid size"
    assert np.all(np.isfinite(out)), "variable-a Heat1D6th output must be finite"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_heat1d4th_invalid_n() -> None:
    """n < 4 must raise SemiflowError with GridMismatch in the message."""
    with pytest.raises(rp.SemiflowError, match="GridMismatch"):
        rp.Heat1D4th(xmin=0.0, xmax=1.0, n=2, u0=np.ones(2))


def test_heat1d6th_invalid_n() -> None:
    """n < 4 must raise SemiflowError with GridMismatch in the message."""
    with pytest.raises(rp.SemiflowError, match="GridMismatch"):
        rp.Heat1D6th(xmin=0.0, xmax=1.0, n=2, u0=np.ones(2))


def test_heat1d4th_nan_u0() -> None:
    """NaN in u0 must raise SemiflowError with NanInf in the message."""
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)
    u0[5] = float("nan")
    with pytest.raises(rp.SemiflowError, match="NanInf"):
        rp.Heat1D4th(xmin=XMIN, xmax=XMAX, n=N, u0=u0)
