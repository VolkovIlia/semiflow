"""ReactionDiffusion1D/2D/3D and richardson_weights (ADR-0208/0209).

G_PY_SEMILIN: the Python surface of the semilinear reaction–diffusion systems:
exact Fisher–KPP front, callable vs built-in kinetics, one callback per RK stage,
spatially uniform data reduces to the exact logistic ODE in 1-D/2-D/3-D,
Richardson extrapolation, and error paths.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import semiflow


def fisher_wave(t: float, x: np.ndarray) -> np.ndarray:
    """Ablowitz–Zeppetella exact front of u_t = u_xx + u(1 − u)."""
    c = 5.0 / math.sqrt(6.0)
    return (1.0 + np.exp((x - c * t) / math.sqrt(6.0))) ** -2


def test_fisher_front_converges() -> None:
    x = np.linspace(-60.0, 60.0, 1201)
    errs = []
    for n in (10, 20, 40):
        rd = semiflow.ReactionDiffusion1D(
            -60.0, 60.0, 1201, fisher_wave(0.0, x)[None, :],
            diffusivity=[1.0], reaction="fisher_kpp",
        )
        rd.evolve(2.0, n_steps=n)
        assert rd.values().shape == (1, 1201)
        errs.append(np.max(np.abs(rd.values()[0] - fisher_wave(2.0, x))))
    assert errs[2] <= 1e-6, errs
    assert math.log2(errs[0] / errs[1]) >= 1.8 and math.log2(errs[1] / errs[2]) >= 1.8, errs


def test_callable_matches_builtin_and_counts_calls() -> None:
    n = 129
    x = np.linspace(-5.0, 5.0, n)
    u0 = np.stack([1.0 - 0.5 * np.exp(-x**2), 0.25 * np.exp(-x**2)])
    feed, kill = 0.04, 0.06
    calls = []

    def gray_scott(t, xs, u):
        calls.append(t)
        assert xs.shape == (1, n) and u.shape == (2, n)
        uv2 = u[0] * u[1] * u[1]
        return np.stack([feed * (1.0 - u[0]) - uv2, uv2 - (feed + kill) * u[1]])

    kw = dict(diffusivity=[0.02, 0.01])
    a = semiflow.ReactionDiffusion1D(-5.0, 5.0, n, u0, reaction="gray_scott",
                                     params={"feed": feed, "kill": kill}, **kw)
    b = semiflow.ReactionDiffusion1D(-5.0, 5.0, n, u0, reaction=gray_scott, **kw)
    a.evolve(5.0, n_steps=10)
    b.evolve(5.0, n_steps=10)
    assert np.max(np.abs(a.values() - b.values())) <= 1e-12
    # n steps merge into n + 1 reaction flows of 4 RK4 stages each.
    assert len(calls) == 4 * 11
    assert a.time == b.time == 5.0 and a.species == 2 and len(a) == n


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_uniform_state_is_exact_logistic(dim: int) -> None:
    shape = [9] * dim
    u0 = np.full([1, *shape], 0.1)
    lims = []
    for _ in range(dim):
        lims += [0.0, 1.0, 9]
    cls = {1: semiflow.ReactionDiffusion1D, 2: semiflow.ReactionDiffusion2D,
           3: semiflow.ReactionDiffusion3D}[dim]
    rd = cls(*lims, u0, diffusivity=[0.3], reaction="fisher_kpp",
             params={"rate": 2.0, "capacity": 1.0}, boundary="periodic")
    rd.evolve(1.5, n_steps=7)
    exact = 1.0 / (1.0 + 9.0 * math.exp(-2.0 * 1.5))
    assert rd.values().shape == (1, *shape)
    assert np.max(np.abs(rd.values() - exact)) <= 1e-12


def test_richardson_raises_the_order() -> None:
    a = 0.5
    x = np.linspace(-10.0, 10.0, 801)

    def source(t, xs, u):
        return -(4 * a * xs[0] ** 2 - 2 * a + 1) * np.exp(-t) * np.exp(-xs[0] ** 2)[None, :]

    errs = {}
    for levels in (1, 2):
        rd = semiflow.ReactionDiffusion1D(-10.0, 10.0, 801, np.exp(-x**2)[None, :],
                                          diffusivity=[a], reaction=source)
        rd.evolve(1.0, n_steps=20, richardson=levels)
        errs[levels] = np.max(np.abs(rd.values()[0] - math.exp(-1.0) * np.exp(-x**2)))
    assert errs[2] * 50 <= errs[1], errs


def test_richardson_weights() -> None:
    assert semiflow.richardson_weights(1, 2) == pytest.approx([-1.0, 2.0])
    assert semiflow.richardson_weights(1, 3) == pytest.approx([0.5, -4.0, 4.5])
    with pytest.raises(semiflow.SemiflowError):
        semiflow.richardson_weights(0, 2)


def test_error_paths() -> None:
    u0 = np.ones((2, 16))
    mk = semiflow.ReactionDiffusion1D
    with pytest.raises(semiflow.SemiflowError, match="shape"):
        mk(0.0, 1.0, 17, u0, diffusivity=[1.0, 1.0], reaction="gray_scott")
    with pytest.raises(semiflow.SemiflowError, match="diffusivity"):
        mk(0.0, 1.0, 16, u0, diffusivity=[1.0], reaction="gray_scott")
    with pytest.raises(semiflow.SemiflowError, match="unknown reaction"):
        mk(0.0, 1.0, 16, u0, diffusivity=[1.0, 1.0], reaction="grey_scott")
    with pytest.raises(semiflow.SemiflowError, match="unknown parameter"):
        mk(0.0, 1.0, 16, u0, diffusivity=[1.0, 1.0], reaction="gray_scott", params={"feeed": 1.0})
    with pytest.raises(semiflow.SemiflowError, match="species"):
        mk(0.0, 1.0, 16, u0, diffusivity=[1.0, 1.0], reaction="nagumo")
    with pytest.raises(TypeError):
        mk(0.0, 1.0, 16, u0, diffusivity=[1.0, 1.0], reaction=3)

    def boom(t, x, u):
        raise ZeroDivisionError("from the callback")

    rd = mk(0.0, 1.0, 16, u0, diffusivity=[1.0, 1.0], reaction=boom)
    with pytest.raises(ZeroDivisionError, match="from the callback"):
        rd.evolve(0.1, n_steps=2)
    assert rd.time == 0.0  # failed evolve leaves the state untouched

    rd = mk(0.0, 1.0, 16, u0, diffusivity=[1.0, 1.0], reaction=lambda t, x, u: u[:1])
    with pytest.raises(semiflow.SemiflowError, match="expected K"):
        rd.evolve(0.1, n_steps=2)
    rd = mk(0.0, 1.0, 16, u0, diffusivity=[1.0, 1.0], reaction="brusselator")
    with pytest.raises(semiflow.SemiflowError):
        rd.evolve(0.1, n_steps=2, richardson=7)


def test_heat1d_richardson() -> None:
    """Heat1D.evolve(richardson=2) beats the plain run on the Gaussian heat kernel."""
    x = np.linspace(-12.0, 12.0, 1201)
    exact = np.exp(-x**2 / (1.0 + 4.0)) / math.sqrt(1.0 + 4.0)  # u_t = u_xx, t = 1
    errs = {}
    for levels in (1, 2):
        h = semiflow.Heat1D(-12.0, 12.0, 1201, np.exp(-x**2))
        h.evolve(1.0, n_steps=10, richardson=levels)
        errs[levels] = np.max(np.abs(np.asarray(h.values()) - exact))
    assert errs[2] * 10 <= errs[1], errs


def test_zero_diffusivity_species_does_not_move() -> None:
    """D = 0 is legal (e.g. the FitzHugh–Nagumo recovery variable)."""
    x = np.linspace(-5.0, 5.0, 101)
    u0 = np.stack([np.exp(-x**2), np.cos(x)])
    rd = semiflow.ReactionDiffusion1D(-5.0, 5.0, 101, u0, diffusivity=[1.0, 0.0],
                                      reaction="linear", params={"matrix": [[0.0, 0.0], [0.0, 0.0]]})
    rd.evolve(0.5, n_steps=10)
    out = rd.values()
    assert np.max(np.abs(out[1] - u0[1])) <= 1e-14
    assert np.max(np.abs(out[0] - u0[0])) > 1e-3  # species 0 did diffuse
