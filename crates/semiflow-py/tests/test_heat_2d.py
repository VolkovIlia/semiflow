"""Smoke tests for semiflow.Heat2D.

Covers:
  - unit Gaussian initial datum: positivity + finiteness
  - grid-mismatch error for wrong u0 length
  - constant initial datum preserved under symmetric heat equation
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Test 1: unit-a Gaussian smoke
# ---------------------------------------------------------------------------

def test_heat_2d_unit_smoke() -> None:
    """Unit a in both axes; Gaussian initial; check positivity + finite output."""
    nx = ny = 32
    h = rp.Heat2D(xmin=-4.0, xmax=4.0, nx=nx, ymin=-4.0, ymax=4.0, ny=ny)
    x = np.linspace(-4.0, 4.0, nx)
    y = np.linspace(-4.0, 4.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u0 = np.exp(-(X**2 + Y**2)).flatten()

    out = h.evolve(u0, tau=0.01, n_steps=1)

    assert out.shape == (nx * ny,), f"expected shape ({nx * ny},), got {out.shape}"
    assert np.all(np.isfinite(out)), "output must be all-finite"
    assert np.all(out > 0), "positivity must be preserved for Gaussian + short diffusion"


# ---------------------------------------------------------------------------
# Test 2: grid mismatch raises an error
# ---------------------------------------------------------------------------

def test_heat_2d_grid_mismatch() -> None:
    """u0 length != nx*ny raises a Python error."""
    h = rp.Heat2D(xmin=0.0, xmax=1.0, nx=4, ymin=0.0, ymax=1.0, ny=4)
    with pytest.raises(Exception):
        h.evolve(np.zeros(15), tau=0.01, n_steps=1)  # 15 != 16


# ---------------------------------------------------------------------------
# Test 3: constant initial value preserved
# ---------------------------------------------------------------------------

def test_heat_2d_oracle_constant() -> None:
    """Constant initial value preserved by symmetric heat eq up to discretisation noise."""
    h = rp.Heat2D(xmin=0.0, xmax=1.0, nx=8, ymin=0.0, ymax=1.0, ny=8)
    u0 = np.ones(64)
    out = h.evolve(u0, tau=0.001, n_steps=1)
    np.testing.assert_allclose(out, 1.0, atol=1e-10)
