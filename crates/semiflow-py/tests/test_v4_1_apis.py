"""Python parity tests for v4.1+ Rust APIs.

Tests cover:
  - heisenberg_heat_kernel module-level function
  - HypoellipticChernoffHeisenberg construction + order()
  - Heat1DZeta6 construction + order() + evolve accuracy
  - Heat1DZeta4 construction + order() + evolve accuracy
  - v7.0: with_quintic_sampling/with_cubic_sampling REMOVED (ADR-0109)
"""

import math

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# heisenberg_heat_kernel
# ---------------------------------------------------------------------------


def test_heisenberg_heat_kernel_at_origin() -> None:
    """p_h(0, 0, 0) ≈ 1/(2h²) per on-diagonal analytic identity.

    The 32-pt Gauss-Legendre quadrature on [-16/h, +16/h] approximates the
    full integral to ~0.24% error (systematic tail truncation, not a bug).
    Gate: relative error < 5e-3, matching the Rust `heisenberg_kernel_on_diagonal`
    test (math.md §28 AMENDMENT 2, heisenberg_kernel.rs line 233).
    """
    h = 1.0
    expected = 1.0 / (2.0 * h**2)   # = 0.5 at h=1
    actual = rp.heisenberg_heat_kernel(h, 0.0, 0.0, 0.0)
    rel_err = abs(actual - expected) / expected
    assert rel_err < 5e-3, (
        f"heisenberg_heat_kernel at origin: got {actual:.6e}, expected ~{expected:.6e}, "
        f"rel_err={rel_err:.3e}"
    )


def test_heisenberg_heat_kernel_zero_for_nonpositive_h() -> None:
    """h <= 0 returns 0.0 per convention (undefined for non-positive step)."""
    assert rp.heisenberg_heat_kernel(0.0, 0.5, 0.5, 0.5) == 0.0
    assert rp.heisenberg_heat_kernel(-1.0, 0.5, 0.5, 0.5) == 0.0


def test_heisenberg_heat_kernel_symmetry() -> None:
    """p_h(x, y, t) = p_h(-x, -y, -t) for symmetric Heisenberg kernel."""
    h = 0.5
    x, y, tc = 0.3, 0.4, 0.1
    v1 = rp.heisenberg_heat_kernel(h, x, y, tc)
    v2 = rp.heisenberg_heat_kernel(h, -x, -y, -tc)
    assert abs(v1 - v2) < 1e-10, f"Symmetry broken: p({x},{y},{tc})={v1}, p(-x,-y,-t)={v2}"


def test_heisenberg_heat_kernel_positive() -> None:
    """Kernel value must be non-negative for physical interpretation."""
    h = 0.5
    for x in [0.0, 0.2, -0.2]:
        for y in [0.0, 0.1, -0.1]:
            val = rp.heisenberg_heat_kernel(h, x, y, 0.0)
            assert val >= 0.0, f"Negative kernel at ({x},{y},0): {val}"


# ---------------------------------------------------------------------------
# HypoellipticChernoffHeisenberg
# ---------------------------------------------------------------------------


def test_hypoelliptic_heisenberg_construction() -> None:
    """new_heisenberg() should construct without error."""
    chernoff = rp.HypoellipticChernoffHeisenberg()
    assert chernoff is not None


def test_hypoelliptic_heisenberg_order() -> None:
    """order() must return 2 (palindromic Strang-Hörmander, ADR-0087)."""
    chernoff = rp.HypoellipticChernoffHeisenberg()
    assert chernoff.order() == 2


def test_hypoelliptic_heisenberg_kernel_method() -> None:
    """kernel() convenience method delegates to heisenberg_heat_kernel."""
    chernoff = rp.HypoellipticChernoffHeisenberg()
    h = 1.0
    direct = rp.heisenberg_heat_kernel(h, 0.0, 0.0, 0.0)
    via_method = chernoff.kernel(h, 0.0, 0.0, 0.0)
    assert direct == via_method, f"kernel() mismatch: {direct} != {via_method}"


def test_hypoelliptic_heisenberg_repr() -> None:
    """repr() should include the class name."""
    chernoff = rp.HypoellipticChernoffHeisenberg()
    r = repr(chernoff)
    assert "HypoellipticChernoffHeisenberg" in r


# ---------------------------------------------------------------------------
# Heat1DZeta6
# ---------------------------------------------------------------------------

XMIN, XMAX, N = -5.0, 5.0, 64
T = 0.1
N_STEPS = 50


@pytest.fixture
def u0_small() -> np.ndarray:
    xs = np.linspace(XMIN, XMAX, N)
    return np.exp(-(xs**2))


def test_heat1d_zeta6_construction(u0_small: np.ndarray) -> None:
    """Heat1DZeta6 should construct without error on valid inputs."""
    kern = rp.Heat1DZeta6(XMIN, XMAX, N, u0_small)
    assert kern is not None


def test_heat1d_zeta6_order(u0_small: np.ndarray) -> None:
    """order() must return 6 for ζ⁶ kernel (ADR-0086 rung K=3)."""
    kern = rp.Heat1DZeta6(XMIN, XMAX, N, u0_small)
    assert kern.order() == 6


def test_heat1d_zeta6_len(u0_small: np.ndarray) -> None:
    """__len__() should return the number of grid nodes."""
    kern = rp.Heat1DZeta6(XMIN, XMAX, N, u0_small)
    assert len(kern) == N


def test_heat1d_zeta6_initial_values(u0_small: np.ndarray) -> None:
    """values() before evolve should match u0."""
    kern = rp.Heat1DZeta6(XMIN, XMAX, N, u0_small)
    vals = kern.values()
    assert isinstance(vals, np.ndarray)
    assert vals.shape == (N,)
    assert np.allclose(vals, u0_small, atol=1e-14)


def test_heat1d_zeta6_evolve_runs(u0_small: np.ndarray) -> None:
    """evolve() should complete without error."""
    kern = rp.Heat1DZeta6(XMIN, XMAX, N, u0_small)
    kern.evolve(T, N_STEPS)
    vals = kern.values()
    assert vals.shape == (N,)
    assert np.all(np.isfinite(vals))


def test_heat1d_zeta6_evolve_accuracy(u0_small: np.ndarray) -> None:
    """ζ⁶ should approximate heat kernel within 5e-3 at t=0.1, n=64."""
    # Oracle: heat kernel u(t, x) = 1/sqrt(1+4t) * exp(-x²/(1+4t))
    xs = np.linspace(XMIN, XMAX, N)
    expected = np.exp(-xs**2 / (1.0 + 4.0 * T)) / math.sqrt(1.0 + 4.0 * T)
    kern = rp.Heat1DZeta6(XMIN, XMAX, N, u0_small)
    kern.evolve(T, N_STEPS)
    vals = kern.values()
    sup_err = float(np.max(np.abs(vals - expected)))
    assert sup_err < 5e-3, f"Heat1DZeta6 sup_error={sup_err:.3e} >= 5e-3"


def test_heat1d_zeta6_bad_input_nan(u0_small: np.ndarray) -> None:
    """NaN in u0 should raise SemiflowError."""
    bad = u0_small.copy()
    bad[5] = float("nan")
    with pytest.raises(rp.SemiflowError):
        rp.Heat1DZeta6(XMIN, XMAX, N, bad)


def test_heat1d_zeta6_bad_grid() -> None:
    """Invalid grid (n < 4) should raise SemiflowError."""
    with pytest.raises(rp.SemiflowError):
        rp.Heat1DZeta6(XMIN, XMAX, 2, np.ones(2))


# ---------------------------------------------------------------------------
# Heat1DZeta4
# ---------------------------------------------------------------------------


def test_heat1d_zeta4_construction(u0_small: np.ndarray) -> None:
    """Heat1DZeta4 should construct without error on valid inputs."""
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    assert kern is not None


def test_heat1d_zeta4_order(u0_small: np.ndarray) -> None:
    """order() must return 4 for ζ⁴ kernel (ADR-0086 Path β)."""
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    assert kern.order() == 4


def test_heat1d_zeta4_len(u0_small: np.ndarray) -> None:
    """__len__() should return the number of grid nodes."""
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    assert len(kern) == N


def test_heat1d_zeta4_evolve_runs(u0_small: np.ndarray) -> None:
    """evolve() should complete without error."""
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    kern.evolve(T, N_STEPS)
    vals = kern.values()
    assert vals.shape == (N,)
    assert np.all(np.isfinite(vals))


def test_heat1d_zeta4_evolve_accuracy(u0_small: np.ndarray) -> None:
    """ζ⁴ should approximate heat kernel within 5e-3 at t=0.1, n=64."""
    xs = np.linspace(XMIN, XMAX, N)
    expected = np.exp(-xs**2 / (1.0 + 4.0 * T)) / math.sqrt(1.0 + 4.0 * T)
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    kern.evolve(T, N_STEPS)
    vals = kern.values()
    sup_err = float(np.max(np.abs(vals - expected)))
    assert sup_err < 5e-3, f"Heat1DZeta4 sup_error={sup_err:.3e} >= 5e-3"


# ---------------------------------------------------------------------------
# Sampling API: v7.0 — with_quintic_sampling / with_cubic_sampling REMOVED
# (ADR-0109 QuinticHermite removal clock fulfilled; see docs/migration/v6-to-v7.md)
# Replacement tests verify that with_quintic_sampling does NOT exist on the class.
# ---------------------------------------------------------------------------


def test_heat1d_zeta4_no_quintic_sampling_attr(u0_small: np.ndarray) -> None:
    """with_quintic_sampling must NOT exist on Heat1DZeta4 at v7.0 (API removed)."""
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    assert not hasattr(kern, "with_quintic_sampling"), (
        "with_quintic_sampling must be removed at v7.0 (ADR-0109)"
    )


def test_heat1d_zeta4_no_cubic_sampling_attr(u0_small: np.ndarray) -> None:
    """with_cubic_sampling must NOT exist on Heat1DZeta4 at v7.0 (API removed)."""
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    assert not hasattr(kern, "with_cubic_sampling"), (
        "with_cubic_sampling must be removed at v7.0 (ADR-0109)"
    )


def test_heat1d_zeta4_default_evolve_runs(u0_small: np.ndarray) -> None:
    """Default construction + evolve() should succeed and yield finite values."""
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    kern.evolve(T, N_STEPS)
    vals = kern.values()
    assert np.all(np.isfinite(vals)), "Non-finite values after default evolve"


def test_heat1d_zeta4_order_is_4(u0_small: np.ndarray) -> None:
    """order() must return 4 for the ζ⁴ kernel."""
    kern = rp.Heat1DZeta4(XMIN, XMAX, N, u0_small)
    assert kern.order() == 4
