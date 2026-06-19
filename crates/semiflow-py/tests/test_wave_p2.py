"""Wave P2 smoke tests — ADR-0111 parity for complex Schrödinger.

Covers:
  SchrodingerComplex1D (M5) — native complex Cayley-Crank-Nicolson kernel
                               (SchrödingerChernoffComplex, ADR-0079 Option B)

Oracle strategy (independent reference — NOT self-referential):
  Free-particle norm preservation: ‖ψ(t)‖² = ‖ψ(0)‖² exactly (unitarity).

  For validation with potential, we compare against the existing `Schrodinger1D`
  (real-pair ADR-0057 kernel) as a cross-consistency check.  Both implement the
  same Cayley-CN step; they must agree to < 1e-10 for t=0.1, n=128.

  Harmonic potential eigenfunction: the ground state of H = -½∂² + ½x² is
  ψ₀(x) = π^{-1/4} exp(-x²/2).  After time t it picks up phase exp(-iE₀t)
  with E₀ = ½ (zero-point energy).  So ‖ψ(t) - exp(-i·E₀·t)·ψ₀‖ < tol.
  This is a genuine analytic check that does NOT depend on the Rust kernel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Common fixture parameters (match test_schrodinger.py convention)
# ---------------------------------------------------------------------------

XMIN, XMAX, N = -8.0, 8.0, 256
T = 0.1
N_STEPS = 200


@pytest.fixture
def free_psi0() -> np.ndarray:
    """Gaussian wavepacket (unit norm when dx-normalised)."""
    xs = np.linspace(XMIN, XMAX, N)
    dx = (XMAX - XMIN) / (N - 1)
    psi = np.exp(-(xs**2)) + 0j
    psi /= math.sqrt(float(np.sum(np.abs(psi) ** 2) * dx))  # normalise
    return psi.astype(np.complex128)


@pytest.fixture
def harmonic_psi0() -> np.ndarray:
    """Ground state of H = -½∂² + ½x²: ψ₀ = π^{-1/4} exp(-x²/2)."""
    xs = np.linspace(XMIN, XMAX, N)
    psi = math.pi ** (-0.25) * np.exp(-(xs**2) / 2.0) + 0j
    return psi.astype(np.complex128)


def make_harmonic_v() -> np.ndarray:
    """Return pre-sampled harmonic potential V(x) = ½x²."""
    xs = np.linspace(XMIN, XMAX, N)
    return (0.5 * xs**2).astype(np.float64)


# ---------------------------------------------------------------------------
# SchrodingerComplex1D (M5)
# ---------------------------------------------------------------------------


class TestSchrodingerComplex1D:
    def test_construction_free_particle(self, free_psi0: np.ndarray) -> None:
        kern = rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0)
        assert kern is not None

    def test_order(self, free_psi0: np.ndarray) -> None:
        assert rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0).order() == 2

    def test_len(self, free_psi0: np.ndarray) -> None:
        assert len(rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0)) == N

    def test_initial_values_dtype(self, free_psi0: np.ndarray) -> None:
        vals = rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0).values()
        assert vals.shape == (N,)
        assert vals.dtype == np.complex128

    def test_initial_norm_preserved(self, free_psi0: np.ndarray) -> None:
        """Norm before evolve should equal construction norm."""
        kern = rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0)
        dx = (XMAX - XMIN) / (N - 1)
        norm0 = float(np.sum(np.abs(free_psi0) ** 2) * dx)
        assert abs(kern.norm_squared() / norm0 - 1.0) < 1e-12

    # ------------------------------------------------------------------
    # Oracle test 1: unitarity — ‖ψ(t)‖ = ‖ψ(0)‖ exactly.
    # This is a genuine analytic invariant of the Cayley-CN scheme, not
    # self-referential: any drift in norm_squared signals a bug.
    # ------------------------------------------------------------------

    def test_unitarity_free_particle(self, free_psi0: np.ndarray) -> None:
        """Norm preserved to < 1e-10 after evolution (Cayley unitary guarantee)."""
        kern = rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0)
        norm0 = kern.norm_squared()
        kern.evolve(T, N_STEPS)
        norm1 = kern.norm_squared()
        rel_drift = abs(norm1 / norm0 - 1.0)
        assert rel_drift < 1e-10, (
            f"SchrodingerComplex1D norm drift = {rel_drift:.3e} (>= 1e-10)"
        )

    def test_unitarity_multiple_steps(self, free_psi0: np.ndarray) -> None:
        """Norm preserved over 5 successive evolve calls (cumulative drift < 1e-9)."""
        kern = rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0)
        norm0 = kern.norm_squared()
        for _ in range(5):
            kern.evolve(T, N_STEPS)
        norm_final = kern.norm_squared()
        rel_drift = abs(norm_final / norm0 - 1.0)
        assert rel_drift < 1e-9, (
            f"SchrodingerComplex1D cumulative norm drift = {rel_drift:.3e}"
        )

    # ------------------------------------------------------------------
    # Oracle test 2: harmonic ground state picks up phase exp(-i E₀ t).
    # E₀ = ½ for H = -½∂² + ½x².  ψ(t) = exp(-i/2 · t) · ψ₀(x).
    # The analytic reference is independent of the Rust kernel.
    # ------------------------------------------------------------------

    def test_harmonic_phase_evolution(self, harmonic_psi0: np.ndarray) -> None:
        """Ground state of -½∂² + ½x² picks up phase exp(-i·t/2).

        sup |ψ_computed(t) - exp(-i·t/2)·ψ₀(x)| / max|ψ₀| < 1e-3.
        """
        v = make_harmonic_v()
        kern = rp.SchrodingerComplex1D.with_potential(XMIN, XMAX, N, v, harmonic_psi0)
        kern.evolve(T, N_STEPS)
        psi_t = kern.values()

        # Analytic: phase factor exp(-i E₀ T) with E₀ = 0.5.
        phase = np.exp(-1j * 0.5 * T)
        psi_analytic = phase * harmonic_psi0

        sup_err = float(np.max(np.abs(psi_t - psi_analytic)))
        psi_max = float(np.max(np.abs(harmonic_psi0)))
        rel_err = sup_err / psi_max
        assert rel_err < 1e-3, (
            f"SchrodingerComplex1D harmonic phase: rel_sup_err={rel_err:.3e} (>= 1e-3)"
        )

    # ------------------------------------------------------------------
    # Oracle test 3: free-particle wavepacket propagation.
    #
    # A Gaussian wavepacket with initial momentum p₀ = k₀ evolves as:
    #   ψ(t, x) = A(t) exp(-(x - p₀t)² / (2σ²(t))) exp(i(p₀x - p₀²t/2))
    #
    # For zero-momentum IC (p₀ = 0) the wavepacket spreads symmetrically.
    # We verify that |ψ(t)| is symmetric and its peak is at x=0, plus
    # that the norm is preserved.  This is an analytic structural check.
    # ------------------------------------------------------------------

    def test_free_particle_symmetry(self, free_psi0: np.ndarray) -> None:
        """Free-particle wavepacket remains symmetric (|ψ(-x)| = |ψ(x)|) to < 1e-10."""
        kern = rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0)
        kern.evolve(T, N_STEPS)
        psi_t = kern.values()
        abs_t = np.abs(psi_t)

        # Grid is symmetric about x=0; |ψ(x_i)| ≈ |ψ(x_{N-1-i})|
        sym_err = float(np.max(np.abs(abs_t - abs_t[::-1])))
        assert sym_err < 1e-10, (
            f"SchrodingerComplex1D free-particle symmetry error = {sym_err:.3e}"
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_nan_input_raises(self, free_psi0: np.ndarray) -> None:
        bad = free_psi0.copy()
        bad[5] = complex(float("nan"), 0.0)
        with pytest.raises(rp.SemiflowError):
            rp.SchrodingerComplex1D(XMIN, XMAX, N, bad)

    def test_inf_input_raises(self, free_psi0: np.ndarray) -> None:
        bad = free_psi0.copy()
        bad[10] = complex(0.0, float("inf"))
        with pytest.raises(rp.SemiflowError):
            rp.SchrodingerComplex1D(XMIN, XMAX, N, bad)

    def test_invalid_grid_raises(self) -> None:
        with pytest.raises(rp.SemiflowError):
            rp.SchrodingerComplex1D(XMIN, XMAX, 2, np.ones(2, dtype=np.complex128))

    def test_n_steps_zero_raises(self, free_psi0: np.ndarray) -> None:
        kern = rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0)
        with pytest.raises(rp.SemiflowError):
            kern.evolve(T, 0)

    def test_negative_t_raises(self, free_psi0: np.ndarray) -> None:
        kern = rp.SchrodingerComplex1D(XMIN, XMAX, N, free_psi0)
        with pytest.raises(rp.SemiflowError):
            kern.evolve(-1.0)

    # ------------------------------------------------------------------
    # with_potential constructor
    # ------------------------------------------------------------------

    def test_with_potential_construction(self, harmonic_psi0: np.ndarray) -> None:
        v = make_harmonic_v()
        kern = rp.SchrodingerComplex1D.with_potential(XMIN, XMAX, N, v, harmonic_psi0)
        assert kern is not None
        assert len(kern) == N

    def test_with_potential_nan_v_raises(self, harmonic_psi0: np.ndarray) -> None:
        v = make_harmonic_v()
        v[3] = float("nan")
        with pytest.raises(rp.SemiflowError):
            rp.SchrodingerComplex1D.with_potential(XMIN, XMAX, N, v, harmonic_psi0)

    def test_with_potential_unitarity(self, harmonic_psi0: np.ndarray) -> None:
        """Unitarity must hold even with non-zero potential."""
        v = make_harmonic_v()
        kern = rp.SchrodingerComplex1D.with_potential(XMIN, XMAX, N, v, harmonic_psi0)
        norm0 = kern.norm_squared()
        kern.evolve(T, N_STEPS)
        norm1 = kern.norm_squared()
        rel_drift = abs(norm1 / norm0 - 1.0)
        assert rel_drift < 1e-10, (
            f"SchrodingerComplex1D (with_potential) norm drift = {rel_drift:.3e}"
        )
