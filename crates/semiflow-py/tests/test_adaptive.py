"""Smoke tests for semiflow.AdaptivePI (v2.3 Phase 4, ADR-0044).

Covers:
  - heat2 kernel: finite output, len() correct
  - heat4 / heat6 kernels: finite output
  - drift kernel: finite output
  - shift kernel: finite output
  - invalid t raises SemiflowError / ValueError
  - unknown kernel raises SemiflowError / ValueError
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

def test_adaptive_heat2_smoke() -> None:
    """heat2 kernel: finite output; len() == N."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    api = rp.AdaptivePI(XMIN, XMAX, N, u0, kernel="heat2")
    out = api.evolve(0.01)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))
    assert len(api) == N


# ---------------------------------------------------------------------------
# Test 2: heat4 kernel
# ---------------------------------------------------------------------------

def test_adaptive_heat4_finite() -> None:
    """heat4 kernel: finite output."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    api = rp.AdaptivePI(XMIN, XMAX, N, u0, kernel="heat4")
    out = api.evolve(0.01)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 3: heat6 kernel
# ---------------------------------------------------------------------------

def test_adaptive_heat6_finite() -> None:
    """heat6 kernel: finite output."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    api = rp.AdaptivePI(XMIN, XMAX, N, u0, kernel="heat6")
    out = api.evolve(0.01)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 4: drift kernel
# ---------------------------------------------------------------------------

def test_adaptive_drift_finite() -> None:
    """drift kernel: finite output."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    api = rp.AdaptivePI(XMIN, XMAX, N, u0, kernel="drift")
    out = api.evolve(0.01)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 5: shift kernel
# ---------------------------------------------------------------------------

def test_adaptive_shift_finite() -> None:
    """shift kernel: finite output."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    api = rp.AdaptivePI(XMIN, XMAX, N, u0, kernel="shift")
    out = api.evolve(0.01)

    assert out.shape == (N,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 6: negative t raises
# ---------------------------------------------------------------------------

def test_adaptive_negative_t_raises() -> None:
    """t <= 0 must raise SemiflowError or ValueError."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    api = rp.AdaptivePI(XMIN, XMAX, N, u0)
    with pytest.raises(Exception):
        api.evolve(-1.0)


# ---------------------------------------------------------------------------
# Test 7: unknown kernel raises
# ---------------------------------------------------------------------------

def test_adaptive_unknown_kernel_raises() -> None:
    """Unknown kernel string raises SemiflowError or ValueError."""
    u0 = _gaussian_u0(N, XMIN, XMAX)
    with pytest.raises(Exception):
        rp.AdaptivePI(XMIN, XMAX, N, u0, kernel="bogus")
