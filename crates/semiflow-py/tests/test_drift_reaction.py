"""Smoke tests for DriftReaction1D (b(x) du/dx + c(x) u, order 2).

Tests verify:
  - Construct + evolve produces finite output with default coefficients.
  - with_arrays (variable b, c) smoke: finite output, length preserved.
  - values() returns a copy.
  - __len__ returns the grid size.
  - Invalid parameters raise SemiflowError with correct kind.
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
N_STEPS = 10


def _gaussian(xs: np.ndarray) -> np.ndarray:
    return np.exp(-xs * xs)


# ---------------------------------------------------------------------------
# Basic construction and evolve
# ---------------------------------------------------------------------------

def test_drift_reaction_construct_evolve_finite() -> None:
    """DriftReaction1D with default b=0.5, c=0 must produce finite output."""
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)
    dr = rp.DriftReaction1D(xmin=XMIN, xmax=XMAX, n=N, u0=u0)
    dr.evolve(TAU, n_steps=N_STEPS)
    out = dr.values()
    assert out.shape == (N,)
    assert np.all(np.isfinite(out)), "DriftReaction1D output must be finite"


def test_drift_reaction_custom_bc_finite() -> None:
    """DriftReaction1D with b=0.2, c=0.1 must produce finite output."""
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)
    dr = rp.DriftReaction1D(xmin=XMIN, xmax=XMAX, n=N, u0=u0, b=0.2, c=0.1)
    dr.evolve(TAU, n_steps=N_STEPS)
    out = dr.values()
    assert np.all(np.isfinite(out)), "Custom b/c DriftReaction1D output must be finite"


def test_drift_reaction_len() -> None:
    """__len__ must return the grid node count."""
    xs = np.linspace(XMIN, XMAX, N)
    dr = rp.DriftReaction1D(xmin=XMIN, xmax=XMAX, n=N, u0=_gaussian(xs))
    assert len(dr) == N


def test_drift_reaction_values_is_copy() -> None:
    """values() must return a copy; mutating it must not change state."""
    xs = np.linspace(XMIN, XMAX, N)
    dr = rp.DriftReaction1D(xmin=XMIN, xmax=XMAX, n=N, u0=_gaussian(xs))
    dr.evolve(TAU, n_steps=1)
    v1 = dr.values()
    v1[:] = 0.0
    v2 = dr.values()
    assert not np.all(v2 == 0.0), "values() must not expose internal buffer"


# ---------------------------------------------------------------------------
# with_arrays — variable coefficients
# ---------------------------------------------------------------------------

def test_drift_reaction_with_arrays_finite() -> None:
    """with_arrays smoke: b(x) = 0.5, c(x) = 0.1*sin(x) must yield finite output."""
    xs = np.linspace(XMIN, XMAX, N)
    b_vals = np.full(N, 0.5)
    c_vals = 0.1 * np.sin(xs)
    c_norm = 0.6
    u0 = _gaussian(xs)
    dr = rp.DriftReaction1D.with_arrays(
        xmin=XMIN, xmax=XMAX, n=N,
        b=b_vals,
        c=c_vals,
        c_norm_bound=c_norm,
        u0=u0,
    )
    dr.evolve(TAU, n_steps=N_STEPS)
    out = dr.values()
    assert out.shape == (N,), "output shape must equal grid size"
    assert np.all(np.isfinite(out)), "variable-coeff DriftReaction1D must be finite"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_drift_reaction_invalid_n() -> None:
    """n < 4 must raise SemiflowError with GridMismatch in the message."""
    with pytest.raises(rp.SemiflowError, match="GridMismatch"):
        rp.DriftReaction1D(xmin=0.0, xmax=1.0, n=2, u0=np.ones(2))


def test_drift_reaction_nan_u0() -> None:
    """NaN in u0 must raise SemiflowError with NanInf in the message."""
    xs = np.linspace(XMIN, XMAX, N)
    u0 = _gaussian(xs)
    u0[5] = float("nan")
    with pytest.raises(rp.SemiflowError, match="NanInf"):
        rp.DriftReaction1D(xmin=XMIN, xmax=XMAX, n=N, u0=u0)


def test_drift_reaction_bad_boundary() -> None:
    """Unrecognised boundary string must raise SemiflowError with OutOfDomain."""
    xs = np.linspace(XMIN, XMAX, N)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.DriftReaction1D(
            xmin=XMIN, xmax=XMAX, n=N,
            u0=_gaussian(xs),
            boundary="bogus",  # type: ignore[arg-type]
        )
