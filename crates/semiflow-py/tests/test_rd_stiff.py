"""Stiff kinetics: ``reaction_solver="rosenbrock"`` (adaptive ROS2, ADR-0212)."""

from __future__ import annotations

import numpy as np
import pytest

import semiflow  # pyright: ignore[reportMissingImports]

# Robertson's chemistry at t = 40: SciPy Radau, rtol 1e-13, atol 1e-20.
ROBERTSON_T40 = np.array([0.7158270687194084, 9.185534764557822e-06, 0.28416374574582987])


def robertson(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    a, b, c = u
    return np.stack([-0.04 * a + 1e4 * b * c, 0.04 * a - 1e4 * b * c - 3e7 * b * b, 3e7 * b * b])


def robertson_jac(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    _, b, c = u
    z = np.zeros_like(b)
    return np.array([
        [z - 0.04, 1e4 * c, 1e4 * b],
        [z + 0.04, -1e4 * c - 6e7 * b, -1e4 * b],
        [z, 6e7 * b, z],
    ])


def uniform_robertson(**kw: object) -> np.ndarray:
    u0 = np.stack([np.ones(5), np.zeros(5), np.zeros(5)])
    rd = semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, diffusivity=[1.0, 1.0, 1.0],
                                      reaction=robertson, **kw)
    rd.evolve(40.0, n_steps=1)
    return rd.values()[:, 2]


@pytest.mark.parametrize("with_jacobian", [False, True])
def test_robertson_meets_rtol(with_jacobian: bool) -> None:
    for rtol in (1e-3, 1e-5):
        got = uniform_robertson(reaction_solver="rosenbrock", rtol=rtol, atol=1e-12,
                                jacobian=robertson_jac if with_jacobian else None)
        rel = np.max(np.abs(got - ROBERTSON_T40) / ROBERTSON_T40)
        assert rel <= 10 * rtol, (rtol, rel)
        assert abs(got.sum() - 1.0) <= 1e-12


def test_stiff_linear_rk4_blows_up_rosenbrock_does_not() -> None:
    a, b = (-1e6 - 1.0) / 2.0, (-1e6 + 1.0) / 2.0

    def linear(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.stack([a * u[0] + b * u[1], b * u[0] + a * u[1]])

    u0 = np.stack([np.ones(5), 0.3 * np.ones(5)])
    kw = dict(diffusivity=[0.0, 0.0], reaction=linear)
    exact = 0.35 * np.exp(-1.0) * np.array([1.0, -1.0])  # slow mode only
    rk4 = semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, **kw)
    try:
        rk4.evolve(1.0, n_steps=10)
        assert np.max(np.abs(rk4.values()[:, 0] - exact)) > 1e6
    except semiflow.SemiflowError:
        pass  # a non-finite result is reported: also a failure of RK4
    ros = semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, reaction_solver="rosenbrock",
                                       rtol=1e-8, atol=1e-14, **kw)
    ros.evolve(1.0, n_steps=1)
    assert np.max(np.abs(ros.values()[:, 0] - exact)) <= 1e-7


def test_non_autonomous_forcing_and_nan_kinetics() -> None:
    """Forcing on the slow mode (time enters stage 2); NaN kinetics fail at once."""
    a, b = (-1e6 - 1.0) / 2.0, (-1e6 + 1.0) / 2.0

    def forced(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.stack([a * u[0] + b * u[1] + np.cos(t), b * u[0] + a * u[1] - np.cos(t)])

    u0 = np.stack([np.ones(5), 0.3 * np.ones(5)])
    rd = semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, diffusivity=[0.0, 0.0], reaction=forced,
                                      reaction_solver="rosenbrock", rtol=1e-6, atol=1e-14)
    rd.evolve(1.0, n_steps=1)
    p = np.exp(-1e6) * 0.65
    m = np.exp(-1.0) * 0.35 + (np.cos(1.0) + np.sin(1.0) - np.exp(-1.0)) / 2.0
    assert np.max(np.abs(rd.values()[:, 0] - np.array([p + m, p - m]))) <= 1e-5

    def root(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return -np.sqrt(u)

    bad = semiflow.ReactionDiffusion1D(0.0, 1.0, 5, -np.ones((1, 5)) * 1e-12, diffusivity=[0.0],
                                       reaction=root, reaction_solver="rosenbrock")
    with pytest.raises(semiflow.SemiflowError):
        bad.evolve(1.0, n_steps=1)


def test_builtin_model_with_rosenbrock_matches_rk4() -> None:
    n = 101
    x = np.linspace(-5.0, 5.0, n)
    u0 = np.stack([1.0 - 0.5 * np.exp(-x**2), 0.25 * np.exp(-x**2)])
    kw = dict(diffusivity=[1e-2, 5e-3], reaction="gray_scott",
              params={"feed": 0.04, "kill": 0.06})
    ref = semiflow.ReactionDiffusion1D(-5.0, 5.0, n, u0, substeps=40, **kw)
    ros = semiflow.ReactionDiffusion1D(-5.0, 5.0, n, u0, reaction_solver="rosenbrock",
                                       rtol=1e-9, atol=1e-12, **kw)
    for rd in (ref, ros):
        rd.evolve(5.0, n_steps=20)
    assert np.max(np.abs(ros.values() - ref.values())) <= 1e-7


def test_option_errors_and_callback_exceptions() -> None:
    u0 = np.ones((1, 5))
    with pytest.raises(semiflow.SemiflowError):
        semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, diffusivity=[1.0], reaction="nagumo",
                                     reaction_solver="implicit")
    with pytest.raises(semiflow.SemiflowError):
        semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, diffusivity=[1.0], reaction="nagumo",
                                     jacobian=lambda t, x, u: u)
    with pytest.raises(TypeError):
        semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, diffusivity=[1.0],
                                     reaction=lambda t, x, u: -u, jacobian=3)
    rd = semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, diffusivity=[1.0], reaction="nagumo",
                                      reaction_solver="rosenbrock", rtol=2.0)
    with pytest.raises(semiflow.SemiflowError):
        rd.evolve(1.0)

    def bad_jac(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise ValueError("jacobian boom")

    rd = semiflow.ReactionDiffusion1D(0.0, 1.0, 5, u0, diffusivity=[1.0],
                                      reaction=lambda t, x, u: -u, jacobian=bad_jac,
                                      reaction_solver="rosenbrock")
    with pytest.raises(ValueError, match="jacobian boom"):
        rd.evolve(1.0)
