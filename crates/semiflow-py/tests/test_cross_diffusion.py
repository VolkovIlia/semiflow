"""Cross-diffusion: a ``K×K`` ``diffusivity`` matrix (ADR-0213)."""

from __future__ import annotations

import numpy as np
import pytest

import semiflow  # pyright: ignore[reportMissingImports]

D = [[1.0, 0.5], [0.2, 0.6]]  # non-symmetric, eigenvalues (8 ± √14)/10
N = 64
XMAX = 1.0 - 1.0 / N  # periodic: node N is node 0, period 1


def zero(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.zeros_like(u)


def run_1d(diffusivity: object, u0: np.ndarray, t: float = 0.02, n_steps: int = 40) -> np.ndarray:
    rd = semiflow.ReactionDiffusion1D(0.0, XMAX, N, u0, diffusivity=diffusivity, reaction=zero,
                                      boundary="periodic")
    rd.evolve(t, n_steps=n_steps)
    return rd.values()


def test_fourier_mode_matches_closed_form_and_modal_reference() -> None:
    x = np.linspace(0.0, XMAX, N)
    s = np.sin(2 * np.pi * x)
    c = np.array([1.0, 0.3])
    got = run_1d(D, np.outer(c, s))
    lam, p = np.linalg.eig(np.array(D))
    assert np.all(np.isreal(lam))
    lam, p = np.real(lam), np.real(p)
    m = np.linalg.solve(p, c)
    # e^{-t k^2 D} c through D = P diag(lam) P^-1 (no SciPy in the wheel tests).
    exact = np.outer(p @ (np.exp(-0.02 * (2 * np.pi) ** 2 * lam) * m), s)
    assert np.max(np.abs(got - exact)) <= 1e-4
    # The same engines mode by mode, mixed with numpy's eigenvectors: identical
    # up to rounding (the mixing commutes with the linear scalar engines).
    modes = [run_1d([float(lk)], s[None, :])[0] for lk in lam]
    ref = sum(np.outer(p[:, k] * m[k], modes[k]) for k in range(2))
    assert np.max(np.abs(got - ref)) <= 1e-13


def test_diagonal_matrix_equals_the_vector_form() -> None:
    x = np.linspace(0.0, XMAX, N)
    u0 = np.stack([np.cos(2 * np.pi * x), 0.5 + np.sin(4 * np.pi * x)])
    vec = run_1d([1.0, 0.5], u0)
    mat = run_1d([[1.0, 0.0], [0.0, 0.5]], u0)
    assert np.max(np.abs(vec - mat)) <= 1e-15


def test_cross_diffusion_with_reaction_in_2d_and_richardson() -> None:
    n = 16
    u0 = np.stack([np.ones((n, n)), 0.1 * np.ones((n, n))])
    u0[1, 6:10, 6:10] = 0.5
    rd = semiflow.ReactionDiffusion2D(0.0, 1.0, n, 0.0, 1.0, n, u0,
                                      diffusivity=[[2e-3, 1e-3], [0.0, 1e-3]],
                                      reaction="gray_scott", params={"feed": 0.04, "kill": 0.06})
    rd.evolve(1.0, n_steps=10, richardson=2)
    v = rd.values()
    assert v.shape == (2, n, n) and np.all(np.isfinite(v))
    assert rd.time == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("matrix", "why"),
    [
        ([[1.0, -2.0], [2.0, 1.0]], "complex"),
        ([[1.0, 2.0], [2.0, 1.0]], "negative"),
        ([[1.0, 1.0], [0.0, 1.0]], "defective"),
        ([[1.0, 1.0], [0.0, 1.0 + 1e-9]], "ill-conditioned"),
        ([[1.0, 0.5, 0.0], [0.2, 0.6, 0.0], [0.0, 0.0, 1.0]], "K×K"),
        ([[1.0, 0.5], [0.2]], "K×K"),
    ],
)
def test_rejected_matrices(matrix: list[list[float]], why: str) -> None:
    u0 = np.ones((2, N))
    with pytest.raises(semiflow.SemiflowError, match=why):
        semiflow.ReactionDiffusion1D(0.0, XMAX, N, u0, diffusivity=matrix, reaction=zero)
