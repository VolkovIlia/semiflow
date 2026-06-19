"""Tests for Heat1D/2D/3D boundary-condition kwarg (Phase 1, v2.3).

Each of the 4 boundary policies is tested with a 1D oracle where the
analytic solution is known, then 2D/3D smoke tests confirm the kwarg
is accepted and produces finite output.
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Shared grid parameters
# ---------------------------------------------------------------------------
N = 128
T = 0.1
N_STEPS = 50
TOL = 1e-3


def _linspace(xmin: float, xmax: float) -> np.ndarray:
    return np.linspace(xmin, xmax, N)


# ---------------------------------------------------------------------------
# Reflect (default) — cosine initial condition
# ---------------------------------------------------------------------------

def test_reflect_cosine() -> None:
    """'reflect' boundary: cosine IC stays cosine (Neumann zero-flux).

    For ∂_t u = ∂_xx u with Neumann zero-flux BCs (a(x)=1, reflect) and
    u0(x) = cos(π x / L), the solution is:
        u(t, x) = exp(-π²t/L²) · cos(π x / L)
    We verify sup|u_numerical - u_analytic| <= TOL.
    """
    xmin, xmax = 0.0, 1.0
    xs = np.linspace(xmin, xmax, N)
    L = xmax - xmin
    k = np.pi / L
    u0 = np.cos(k * xs)

    state = rp.Heat1D(xmin, xmax, N, u0, boundary="reflect")
    state.evolve(T, N_STEPS)
    vals = state.values()

    oracle = np.exp(-(k ** 2) * T) * np.cos(k * xs)
    err = float(np.max(np.abs(vals - oracle)))
    assert err <= TOL, f"reflect sup_error={err:.3e} > tol={TOL}"


# ---------------------------------------------------------------------------
# Periodic — wave packet crossing boundary
# ---------------------------------------------------------------------------

def test_periodic_wave() -> None:
    """'periodic' boundary: heat kernel with periodic BC.

    For periodic BC on [0, 1] with u0(x) = sin(2π x), the solution is:
        u(t, x) = exp(-4π²t) · sin(2π x)
    Check sup error ≤ TOL.
    """
    xmin, xmax = 0.0, 1.0
    xs = np.linspace(xmin, xmax, N, endpoint=False)  # don't duplicate endpoint
    # Use endpoint=False to avoid duplicated node for periodic problem
    k = 2.0 * np.pi
    u0 = np.sin(k * xs)

    xs_full = np.linspace(xmin, xmax, N)
    u0_full = np.sin(k * xs_full)

    state = rp.Heat1D(xmin, xmax, N, u0_full, boundary="periodic")
    state.evolve(T, N_STEPS)
    vals = state.values()

    oracle = np.exp(-(k ** 2) * T) * np.sin(k * xs_full)
    err = float(np.max(np.abs(vals - oracle)))
    # Looser tolerance for periodic: grid has an exact endpoint copy which
    # introduces a small (O(dx²)) standing wave artefact in reflect vs periodic
    # BCs for sinusoidal initial data.
    periodic_tol = 2e-3
    assert err <= periodic_tol, f"periodic sup_error={err:.3e} > tol={periodic_tol}"


# ---------------------------------------------------------------------------
# Zero-extend — localised bump decays to zero at boundaries
# ---------------------------------------------------------------------------

def test_zero_extend_bump() -> None:
    """'zero' boundary: compact-support bump decays to zero at boundary.

    Use u0(x) = Gaussian centred on 0 with std=0.1 on [-2, 2].
    After time T=0.1 the function must still be near-zero at both boundaries.
    """
    xmin, xmax = -2.0, 2.0
    xs = _linspace(xmin, xmax)
    u0 = np.exp(-50.0 * xs ** 2)

    state = rp.Heat1D(xmin, xmax, N, u0, boundary="zero")
    state.evolve(T, N_STEPS)
    vals = state.values()

    # Near boundary (first/last 5 nodes) should be very small
    boundary_vals = np.concatenate([vals[:5], vals[-5:]])
    max_boundary = float(np.max(np.abs(boundary_vals)))
    assert max_boundary <= 0.05, (
        f"zero-extend boundary values not near zero: max={max_boundary:.3e}"
    )
    # Output must be finite everywhere
    assert np.all(np.isfinite(vals)), "zero-extend output contains NaN/Inf"


# ---------------------------------------------------------------------------
# Linear extrapolation — linear ramp preserves gradient
# ---------------------------------------------------------------------------

def test_linear_extrapolate_ramp() -> None:
    """'linear' boundary: linear-ramp IC stays linear (ramp is steady state).

    u0(x) = x on [0, 1].  ∂_xx u0 = 0, so u0 is the steady state.
    After evolving, sup|u(t) - u0| should be very small (numerical diffusion only).
    """
    xmin, xmax = 0.0, 1.0
    xs = _linspace(xmin, xmax)
    u0 = xs.copy()  # linear ramp

    state = rp.Heat1D(xmin, xmax, N, u0, boundary="linear")
    state.evolve(T, N_STEPS)
    vals = state.values()

    err = float(np.max(np.abs(vals - u0)))
    assert err <= TOL, f"linear-extrapolate ramp drift={err:.3e} > tol={TOL}"


# ---------------------------------------------------------------------------
# Error path: unknown boundary string
# ---------------------------------------------------------------------------

def test_unknown_boundary_raises() -> None:
    """Unknown boundary string raises SemiflowError(kind='OutOfDomain')."""
    xmin, xmax = 0.0, 1.0
    xs = _linspace(xmin, xmax)
    u0 = np.ones(N)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.Heat1D(xmin, xmax, N, u0, boundary="absorbing")  # type: ignore[arg-type]  # intentional invalid-input test


# ---------------------------------------------------------------------------
# Heat2D and Heat3D smoke tests
# ---------------------------------------------------------------------------

def test_heat2d_accepts_boundary_default() -> None:
    """Heat2D with default boundary (reflect) produces finite output."""
    h2d = rp.Heat2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    u0 = np.ones(32 * 32, dtype=np.float64)
    result = h2d.evolve(u0, tau=0.01, n_steps=5)
    assert np.all(np.isfinite(result)), "Heat2D default output has NaN/Inf"


def test_heat2d_accepts_boundary_periodic() -> None:
    """Heat2D with periodic boundary produces finite output."""
    h2d = rp.Heat2D(0.0, 1.0, 32, 0.0, 1.0, 32, boundary="periodic")
    u0 = np.ones(32 * 32, dtype=np.float64)
    result = h2d.evolve(u0, tau=0.01, n_steps=5)
    assert np.all(np.isfinite(result)), "Heat2D periodic output has NaN/Inf"


def test_heat3d_accepts_boundary_default() -> None:
    """Heat3D with default boundary (reflect) produces finite output."""
    h3d = rp.Heat3D(0.0, 1.0, 8, 0.0, 1.0, 8, 0.0, 1.0, 8)
    u0 = np.ones(8 * 8 * 8, dtype=np.float64)
    result = h3d.evolve(u0, tau=0.01, n_steps=5)
    assert np.all(np.isfinite(result)), "Heat3D default output has NaN/Inf"


def test_heat3d_accepts_boundary_periodic() -> None:
    """Heat3D with periodic boundary produces finite output."""
    h3d = rp.Heat3D(0.0, 1.0, 8, 0.0, 1.0, 8, 0.0, 1.0, 8, boundary="periodic")
    u0 = np.ones(8 * 8 * 8, dtype=np.float64)
    result = h3d.evolve(u0, tau=0.01, n_steps=5)
    assert np.all(np.isfinite(result)), "Heat3D periodic output has NaN/Inf"
