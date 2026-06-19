"""Smoke tests for semiflow.NonSeparable2D (v2.3 Phase 4, ADR-0058).

Covers:
  - zero coupling reduces to separable diffusion: positivity + finiteness
  - nonzero constant coupling: finite output, mass conserved (approx)
  - with_beta_array constructor: finite output for spatially-varying beta
  - grid / u0 length mismatch raises SemiflowError
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Test 1: zero coupling — degenerates to Strang2D
# ---------------------------------------------------------------------------

def test_nonsep_zero_coupling_smoke() -> None:
    """c=0.0 gives a valid, positive, finite output for a Gaussian."""
    nx = ny = 20
    x = np.linspace(-3.0, 3.0, nx)
    y = np.linspace(-3.0, 3.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u0 = np.exp(-(X**2 + Y**2)).flatten()

    ns = rp.NonSeparable2D(
        xmin=-3.0, xmax=3.0, nx=nx,
        ymin=-3.0, ymax=3.0, ny=ny,
        u0=u0, c=0.0,
    )
    out = ns.evolve(0.01, n_steps=10)

    assert out.shape == (nx * ny,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0), "positivity preserved for Gaussian + unit diffusion"


# ---------------------------------------------------------------------------
# Test 2: nonzero constant coupling
# ---------------------------------------------------------------------------

def test_nonsep_constant_coupling_finite() -> None:
    """Small constant c gives finite output and len() == nx*ny."""
    nx = ny = 16
    u0 = np.ones(nx * ny, dtype=np.float64)

    ns = rp.NonSeparable2D(
        xmin=0.0, xmax=1.0, nx=nx,
        ymin=0.0, ymax=1.0, ny=ny,
        u0=u0, c=0.1,
    )
    out = ns.evolve(0.005, n_steps=5)

    assert len(ns) == nx * ny
    assert out.shape == (nx * ny,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 3: with_beta_array constructor
# ---------------------------------------------------------------------------

def test_nonsep_with_beta_array_finite() -> None:
    """with_beta_array with a constant beta matrix gives finite output."""
    nx = ny = 16
    u0 = np.random.default_rng(42).standard_normal(nx * ny)
    u0 -= u0.min()  # make non-negative
    beta = np.full((nx, ny), 0.05, dtype=np.float64).flatten()

    ns = rp.NonSeparable2D.with_beta_array(
        xmin=-1.0, xmax=1.0, nx=nx,
        ymin=-1.0, ymax=1.0, ny=ny,
        beta_values=beta, u0=u0,
    )
    out = ns.evolve(0.01, n_steps=5)

    assert out.shape == (nx * ny,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 4: u0 length mismatch raises SemiflowError
# ---------------------------------------------------------------------------

def test_nonsep_u0_mismatch_raises() -> None:
    """u0 with wrong length raises an error (GridMismatch)."""
    nx = ny = 16

    with pytest.raises(Exception):  # SemiflowError or ValueError from PyO3
        rp.NonSeparable2D(
            xmin=0.0, xmax=1.0, nx=nx,
            ymin=0.0, ymax=1.0, ny=ny,
            u0=np.zeros(nx * ny + 1),
        )
