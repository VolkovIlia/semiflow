"""Tests for Schrodinger1D — Schrödinger equation `iψ_t = (−Δ + V(x))ψ`.

Verifies:
1. Construction smoke: free-particle Gaussian wavepacket constructs and evolves.
2. Unitarity (free particle): norm preserved to 6 decimal places.
3. Unitarity (harmonic oscillator): norm preserved under V(x) = 0.5 x².
4. Free-particle dispersion oracle: wavepacket centroid moves at group velocity.
5. values() ↔ values_parts() consistency: same data, different view.
6. from_parts == complex constructor: bit-identical state.
7. Boundary kwarg roundtrip: all 4 policies accepted; unknown raises.
8. with_potential zero-V matches default constructor within 1 ULP.
9. GIL release: SIGINT mid-evolve surfaces correctly (ADR-0031).
"""

import math
import os
import signal
import sys
import threading
import time

import numpy as np
import pytest
from numpy.typing import NDArray

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
XMIN = -10.0
XMAX = 10.0
N = 512
X0 = -2.0
SIGMA = 1.0
K0 = 2.0


def _wavepacket(xs: np.ndarray, x0: float, sigma: float, k0: float) -> NDArray[np.complex128]:
    """Gaussian wavepacket ψ(x) = exp(-(x-x0)²/2σ²) · exp(ikx) normalised."""
    psi = np.exp(-(xs - x0) ** 2 / (2.0 * sigma ** 2)) * np.exp(1j * k0 * xs)
    dx = (xs[-1] - xs[0]) / (len(xs) - 1)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm


def _make_free_sch(xs: np.ndarray) -> rp.Schrodinger1D:
    """Build a free-particle Schrodinger1D with normalised Gaussian IC."""
    psi0 = _wavepacket(xs, X0, SIGMA, K0)
    return rp.Schrodinger1D(xmin=XMIN, xmax=XMAX, n=N, psi0=psi0)


# ---------------------------------------------------------------------------
# 1. Construction smoke
# ---------------------------------------------------------------------------

def test_construction_smoke() -> None:
    """Free-particle Gaussian wavepacket constructs and values() is finite."""
    xs = np.linspace(XMIN, XMAX, N)
    sch = _make_free_sch(xs)
    psi_t = sch.values()
    assert psi_t.shape == (N,), "values() shape must equal n"
    assert psi_t.dtype == np.complex128, "values() dtype must be complex128"
    assert np.all(np.isfinite(psi_t.real)) and np.all(np.isfinite(psi_t.imag))
    assert len(sch) == N


# ---------------------------------------------------------------------------
# 2. Unitarity (free particle)
# ---------------------------------------------------------------------------

def test_unitarity_free_particle() -> None:
    """Free-particle evolution preserves ‖ψ‖² to 6 decimal places."""
    xs = np.linspace(XMIN, XMAX, N)
    sch = _make_free_sch(xs)
    norm0 = sch.norm_squared()
    sch.evolve(t=0.5, n_steps=200)
    norm_t = sch.norm_squared()
    ratio = norm_t / norm0
    assert abs(ratio - 1.0) < 1e-6, f"norm not preserved: ratio={ratio:.8f}"


# ---------------------------------------------------------------------------
# 3. Unitarity (harmonic oscillator)
# ---------------------------------------------------------------------------

def test_unitarity_harmonic_oscillator() -> None:
    """Harmonic oscillator evolution preserves ‖ψ‖² to 6 decimal places."""
    xs = np.linspace(XMIN, XMAX, N)
    psi0 = _wavepacket(xs, X0, SIGMA, K0)
    v_vals = 0.5 * xs ** 2
    sch = rp.Schrodinger1D.with_potential(
        xmin=XMIN, xmax=XMAX, n=N, v=v_vals, psi0=psi0,
    )
    norm0 = sch.norm_squared()
    sch.evolve(t=1.0, n_steps=500)
    norm_t = sch.norm_squared()
    ratio = norm_t / norm0
    assert abs(ratio - 1.0) < 1e-6, f"harmonic oscillator norm not preserved: ratio={ratio:.8f}"


# ---------------------------------------------------------------------------
# 4. Free-particle dispersion oracle
# ---------------------------------------------------------------------------

def test_free_particle_dispersion_oracle() -> None:
    """Free-particle centroid moves at group velocity v_g = 2*k0 within 15%.

    For `iψ_t = −Δψ`, a plane wave exp(ikx) satisfies ω(k) = k², so the
    group velocity is v_g = dω/dk = 2k.  This is the standard dispersion for
    the free Schrödinger equation with the convention `iψ_t = −Δψ` (no
    prefactor of 1/2m).  With k0_disp = +1.0 and t = 0.5 the centroid moves
    to x0 + 2*k0*t = 0 + 2*1*0.5 = 1.0.
    """
    xs = np.linspace(XMIN, XMAX, N)
    x0_initial = 0.0
    k0_disp = 1.0  # positive k0 → positive group velocity v_g = 2*k0 = +2
    psi0 = _wavepacket(xs, x0_initial, SIGMA, k0_disp)
    sch = rp.Schrodinger1D(xmin=XMIN, xmax=XMAX, n=N, psi0=psi0)
    t = 0.5
    sch.evolve(t=t, n_steps=200)
    psi_t = sch.values()
    dx = (XMAX - XMIN) / (N - 1)
    prob = np.abs(psi_t) ** 2
    centroid = np.sum(xs * prob) * dx / (np.sum(prob) * dx)
    # v_g = 2*k0 (standard iψ_t = −Δψ dispersion relation).
    expected_x = x0_initial + 2.0 * k0_disp * t  # = 0 + 2.0 * 0.5 = 1.0
    assert abs(centroid - expected_x) < 0.15, (
        f"centroid={centroid:.3f}, expected≈{expected_x:.3f}"
    )


# ---------------------------------------------------------------------------
# 5. values() ↔ values_parts() consistency
# ---------------------------------------------------------------------------

def test_values_parts_consistency() -> None:
    """values() and values_parts() return numerically identical data."""
    xs = np.linspace(XMIN, XMAX, N)
    sch = _make_free_sch(xs)
    sch.evolve(t=0.3, n_steps=100)
    psi_c = sch.values()
    re_p, im_p = sch.values_parts()
    assert np.array_equal(psi_c.real, re_p), "values().real != values_parts()[0]"
    assert np.array_equal(psi_c.imag, im_p), "values().imag != values_parts()[1]"


# ---------------------------------------------------------------------------
# 6. from_parts == complex constructor
# ---------------------------------------------------------------------------

def test_from_parts_equals_complex_constructor() -> None:
    """from_parts gives bit-identical state to the complex constructor."""
    xs = np.linspace(XMIN, XMAX, N)
    psi0 = _wavepacket(xs, X0, SIGMA, K0)
    sch_complex = rp.Schrodinger1D(xmin=XMIN, xmax=XMAX, n=N, psi0=psi0)
    sch_parts = rp.Schrodinger1D.from_parts(
        XMIN, XMAX, N, psi0.real.copy(), psi0.imag.copy(),
    )
    v_complex = sch_complex.values()
    v_parts = sch_parts.values()
    assert np.array_equal(v_complex, v_parts), "from_parts state != complex constructor"


# ---------------------------------------------------------------------------
# 7. Boundary kwarg roundtrip
# ---------------------------------------------------------------------------

def test_boundary_all_accepted() -> None:
    """All 4 recognised boundary policies are accepted."""
    xs = np.linspace(XMIN, XMAX, N)
    psi0 = _wavepacket(xs, X0, SIGMA, K0)
    for policy in ("reflect", "periodic", "zero", "linear"):
        sch = rp.Schrodinger1D(xmin=XMIN, xmax=XMAX, n=N, psi0=psi0, boundary=policy)
        assert len(sch) == N, f"boundary={policy}: len mismatch"


def test_boundary_unknown_raises() -> None:
    """Unknown boundary policy raises SemiflowError(kind='OutOfDomain')."""
    xs = np.linspace(XMIN, XMAX, N)
    psi0 = _wavepacket(xs, X0, SIGMA, K0)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        rp.Schrodinger1D(xmin=XMIN, xmax=XMAX, n=N, psi0=psi0, boundary="bogus")  # type: ignore[arg-type]  # intentional invalid-input test


# ---------------------------------------------------------------------------
# 8. with_potential zero-V matches default constructor
# ---------------------------------------------------------------------------

def test_zero_potential_matches_free_particle() -> None:
    """with_potential(V=0) produces identical initial state as default ctor."""
    xs = np.linspace(XMIN, XMAX, N)
    psi0 = _wavepacket(xs, X0, SIGMA, K0)
    v_zeros = np.zeros(N)
    sch_default = rp.Schrodinger1D(xmin=XMIN, xmax=XMAX, n=N, psi0=psi0.copy())
    sch_zero_v = rp.Schrodinger1D.with_potential(
        xmin=XMIN, xmax=XMAX, n=N, v=v_zeros, psi0=psi0.copy(),
    )
    v_def = sch_default.values()
    v_zv = sch_zero_v.values()
    # Initial states should be bit-identical before any evolution.
    assert np.array_equal(v_def, v_zv), "zero-V potential ctor differs from free particle"


# ---------------------------------------------------------------------------
# 9. GIL release (SIGINT test, ADR-0031)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT semantics differ on Windows")
def test_evolve_releases_gil() -> None:
    """SIGINT during py.detach surfaces as a received signal after re-acquisition.

    Protocol mirrors test_heat.py's test_evolve_handles_sigint (ADR-0031 F6.2).
    1. Install a non-raising SIGINT handler that records the signal.
    2. Start evolve in a background thread with large n_steps (>100ms).
    3. Main thread sends SIGINT after 50ms.
    4. Assert the signal was received.
    """
    xs = np.linspace(XMIN, XMAX, N)
    sch = _make_free_sch(xs)
    sigint_received: list = []
    original_handler = signal.getsignal(signal.SIGINT)

    def _flag_handler(signum: int, _frame: object) -> None:
        del _frame
        sigint_received.append(signum)

    signal.signal(signal.SIGINT, _flag_handler)
    exception_in_thread: list = []

    def run_evolve() -> None:
        try:
            sch.evolve(t=5.0, n_steps=10_000)
        except Exception as exc:  # noqa: BLE001
            exception_in_thread.append(exc)

    thread = threading.Thread(target=run_evolve, daemon=True)
    thread.start()
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGINT)
    thread.join(timeout=30.0)
    signal.signal(signal.SIGINT, original_handler)

    assert not thread.is_alive(), "evolve did not finish within 30s"
    assert not exception_in_thread, f"evolve raised unexpectedly: {exception_in_thread}"
    assert sigint_received, (
        "SIGINT was NOT received during py.detach. GIL release may be broken."
    )
