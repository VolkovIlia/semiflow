"""Smoke tests for semiflow.Adjoint (v2.3 Phase 4, ADR-0055).

Covers:
  - heat2 kernel: finite output, positivity preserved for Gaussian IC
  - heat4 / heat6 kernels: finite output
  - drift kernel: finite output
  - shift kernel: finite output
  - unknown kernel raises SemiflowError / ValueError
  - order() and is_self_adjoint() return sensible values
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

N = 64
XMIN, XMAX = -5.0, 5.0


def _gaussian_u0(n: int, xmin: float, xmax: float) -> np.ndarray:
    x = np.linspace(xmin, xmax, n)
    return np.exp(-(x**2))


# ---------------------------------------------------------------------------
# Test 1: heat2 kernel smoke
# ---------------------------------------------------------------------------

def test_adjoint_heat2_smoke() -> None:
    """heat2 (self-adjoint): Gaussian IC, positivity + finiteness."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    adj = rp.Adjoint(XMIN, XMAX, N, u0, kernel="heat2")

    out = adj.evolve(0.01, n_steps=10)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0), "Gaussian + pure diffusion must stay non-negative"
    assert adj.is_self_adjoint()
    assert adj.order() == 2
    assert len(adj) == N


# ---------------------------------------------------------------------------
# Test 2: heat4 kernel
# ---------------------------------------------------------------------------

def test_adjoint_heat4_finite() -> None:
    """heat4 (self-adjoint): finite output."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    adj = rp.Adjoint(XMIN, XMAX, N, u0, kernel="heat4")
    out = adj.evolve(0.01, n_steps=5)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))
    # Chernoff consistency order is 2 for all diffusion-type kernels;
    # spatial accuracy (4th-order FD stencil per ADR-0014) is separate.
    assert adj.order() == 2


# ---------------------------------------------------------------------------
# Test 3: heat6 kernel
# ---------------------------------------------------------------------------

def test_adjoint_heat6_finite() -> None:
    """heat6 (self-adjoint): finite output."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    adj = rp.Adjoint(XMIN, XMAX, N, u0, kernel="heat6")
    out = adj.evolve(0.01, n_steps=5)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))
    # Chernoff consistency order is 2 for all diffusion-type kernels;
    # spatial accuracy (6th-order FD stencil per ADR-0015) is separate.
    assert adj.order() == 2


# ---------------------------------------------------------------------------
# Test 4: drift kernel
# ---------------------------------------------------------------------------

def test_adjoint_drift_finite() -> None:
    """drift kernel (general adjoint): finite output."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    adj = rp.Adjoint(XMIN, XMAX, N, u0, kernel="drift")
    out = adj.evolve(0.01, n_steps=5)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))
    assert not adj.is_self_adjoint()


# ---------------------------------------------------------------------------
# Test 5: shift kernel
# ---------------------------------------------------------------------------

def test_adjoint_shift_finite() -> None:
    """shift kernel (general adjoint): finite output."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    adj = rp.Adjoint(XMIN, XMAX, N, u0, kernel="shift")
    out = adj.evolve(0.01, n_steps=5)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 6: unknown kernel raises
# ---------------------------------------------------------------------------

def test_adjoint_unknown_kernel_raises() -> None:
    """Unknown kernel string raises SemiflowError or ValueError."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    with pytest.raises(Exception):
        rp.Adjoint(XMIN, XMAX, N, u0, kernel="bogus")


# ---------------------------------------------------------------------------
# Test 7: values() accessor
# ---------------------------------------------------------------------------

def test_adjoint_values_matches_evolve() -> None:
    """values() returns current state matching the last evolve() return value."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    adj = rp.Adjoint(XMIN, XMAX, N, u0, kernel="heat2")

    # Before any evolve(): values() must equal u0 and have length n.
    v0 = adj.values()
    assert v0.shape == (N,), "values() before evolve must have length n"
    np.testing.assert_array_equal(v0, u0, err_msg="values() before evolve must match u0")

    # After evolve(): values() must match the returned array exactly.
    out = adj.evolve(0.01, n_steps=10)
    v1 = adj.values()
    assert v1.shape == (N,), "values() after evolve must have length n"
    np.testing.assert_array_equal(v1, out, err_msg="values() must match evolve() return")
