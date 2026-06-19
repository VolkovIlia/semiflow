"""Smoke tests for semiflow.Heat3D.

Tests verify:
  - Basic positivity and finiteness for a 3D Gaussian initial datum (unit a).
  - Grid-size mismatch is rejected with an exception.
  - Constant initial datum is preserved (mass conservation of constant function).

Storage convention: x-fastest row-major (I-T1-3D).
  values[k * nx * ny + j * nx + i] ≈ u(x_i, y_j, z_k)
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_u0(
    nx: int, ny: int, nz: int,
    xmin: float = -2.0, xmax: float = 2.0,
    ymin: float = -2.0, ymax: float = 2.0,
    zmin: float = -2.0, zmax: float = 2.0,
) -> np.ndarray:
    """Return flat x-fastest Gaussian u0 = exp(-(x²+y²+z²))."""
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.exp(-(X**2 + Y**2 + Z**2)).flatten()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_heat_3d_unit_smoke():
    """Unit a in all three axes; 3D Gaussian; check positivity + finite output."""
    nx = ny = nz = 16
    h = rp.Heat3D(
        xmin=-2.0, xmax=2.0, nx=nx,
        ymin=-2.0, ymax=2.0, ny=ny,
        zmin=-2.0, zmax=2.0, nz=nz,
    )
    u0 = _make_gaussian_u0(nx, ny, nz)
    out = h.evolve(u0, tau=0.01, n_steps=1)
    assert out.shape == (nx * ny * nz,)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)


def test_heat_3d_grid_mismatch():
    """u0 length != nx*ny*nz raises an exception."""
    h = rp.Heat3D(
        xmin=0.0, xmax=1.0, nx=4,
        ymin=0.0, ymax=1.0, ny=4,
        zmin=0.0, zmax=1.0, nz=4,
    )
    wrong_u0 = np.zeros(63)  # 63 != 4*4*4=64
    with pytest.raises(Exception):
        h.evolve(wrong_u0, tau=0.01, n_steps=1)


def test_heat_3d_constant_preserved():
    """Constant initial datum is preserved to machine precision."""
    nx = ny = nz = 4
    h = rp.Heat3D(
        xmin=0.0, xmax=1.0, nx=nx,
        ymin=0.0, ymax=1.0, ny=ny,
        zmin=0.0, zmax=1.0, nz=nz,
    )
    u0 = np.ones(nx * ny * nz)
    out = h.evolve(u0, tau=0.001, n_steps=1)
    np.testing.assert_allclose(out, 1.0, atol=1e-10)


def test_heat_3d_output_shape():
    """Output shape matches nx*ny*nz for non-cubic grids."""
    nx, ny, nz = 6, 8, 5
    h = rp.Heat3D(
        xmin=-1.0, xmax=1.0, nx=nx,
        ymin=-1.0, ymax=1.0, ny=ny,
        zmin=-1.0, zmax=1.0, nz=nz,
    )
    u0 = np.ones(nx * ny * nz)
    out = h.evolve(u0, tau=0.01, n_steps=2)
    assert out.shape == (nx * ny * nz,)
    assert out.dtype == np.float64


def test_heat_3d_zero_n_steps_raises():
    """n_steps=0 raises an exception."""
    nx = ny = nz = 4
    h = rp.Heat3D(
        xmin=0.0, xmax=1.0, nx=nx,
        ymin=0.0, ymax=1.0, ny=ny,
        zmin=0.0, zmax=1.0, nz=nz,
    )
    u0 = np.ones(nx * ny * nz)
    with pytest.raises(Exception):
        h.evolve(u0, tau=0.01, n_steps=0)
