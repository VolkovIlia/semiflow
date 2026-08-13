"""Acceptance tests for the #17 / #21 / #26 issue campaign (ADR-0190, ADR-0191).

Three groups:

``TestAnisotropicMoment``
    The reproduction from issue #17, run against the built wheel. A Gaussian
    under a constant tensor must gain exactly ``2·a·t`` of variance per axis,
    and that must not drift as ``n_steps`` grows.

``TestNDLayout``
    The layout half of #17 — the report's "axis mixing" was a C-order ``ravel()``
    meeting the library's x-fastest flat layout. Passing the 2-D array directly
    is now the safe route and must agree with an explicit Fortran ravel.

``TestHeat2DVarAConstantCoeff``
    The constant-``a`` regression oracle. The variable-``a`` order question is
    left OPEN and skipped with its evidence recorded — see the skip reason.
"""

import numpy as np
import pytest

from semiflow import AnisotropicShiftND2, Heat2D, Heat2DVarA


# ---------------------------------------------------------------------------
# Issue #17 — second moment
# ---------------------------------------------------------------------------

NX = 96
LO, HI = -8.0, 8.0
VAR0 = 0.5
T_FINAL = 0.5


def _grid_1d():
    return np.linspace(LO, HI, NX)


def _run_nd2(a_tensor, n_steps):
    """Evolve a unit Gaussian and return (dVar_x, dVar_y, dCov)."""
    x = _grid_1d()
    # indexing="ij" -> X[i, j] = x[i]; the kernel wants x fastest, so pass the
    # 2-D arrays directly and let the binding ravel them.
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0 = np.exp(-(X**2 + Y**2) / (2 * VAR0))

    a_flat = np.tile(
        [a_tensor[0, 0], a_tensor[0, 1], a_tensor[1, 0], a_tensor[1, 1]], NX * NX
    )
    k = AnisotropicShiftND2(NX, NX, LO, HI, LO, HI, a_flat)
    k.set_state(u0)
    k.evolve(T_FINAL, n_steps)
    u = k.values_2d()

    m = u.sum()
    ex, ey = (X * u).sum() / m, (Y * u).sum() / m
    vx = ((X - ex) ** 2 * u).sum() / m - VAR0
    vy = ((Y - ey) ** 2 * u).sum() / m - VAR0
    cxy = ((X - ex) * (Y - ey) * u).sum() / m
    return vx, vy, cxy


class TestAnisotropicMoment:
    """Issue #17: variance gain must be 2·a·t and flat in n_steps."""

    @pytest.mark.parametrize("n_steps", [100, 400, 1600])
    def test_isotropic_variance_is_step_count_flat(self, n_steps):
        exact = 2.0 * 1.0 * T_FINAL
        vx, vy, cxy = _run_nd2(np.eye(2), n_steps)
        # Pre-ADR-0190 this returned 1.2113 / 2.2449 / 4.4901 for exact = 1.0.
        assert abs(vx - exact) / exact < 2e-2, f"dVar_x={vx} vs {exact}"
        assert abs(vy - exact) / exact < 2e-2, f"dVar_y={vy} vs {exact}"
        assert abs(cxy) < 2e-2

    def test_diagonal_tensor_does_not_mix_axes(self):
        vx, vy, _ = _run_nd2(np.diag([1.0, 0.5]), 400)
        assert abs(vx - 1.0) / 1.0 < 2e-2, f"x axis got {vx}, expected 1.0"
        assert abs(vy - 0.5) / 0.5 < 2e-2, f"y axis got {vy}, expected 0.5"

    def test_off_diagonal_drives_covariance(self):
        _, _, cxy = _run_nd2(np.array([[1.0, 0.4], [0.4, 1.0]]), 400)
        assert abs(cxy - 0.4) / 0.4 < 2e-2, f"dCov={cxy}, expected 0.4"


class TestNDLayout:
    """Issue #17 secondary: the flat layout is x-fastest (Fortran for (nx, ny))."""

    def test_2d_array_matches_explicit_fortran_ravel(self):
        n = 16
        rng = np.random.default_rng(0)
        u = rng.random((n, n))
        a_flat = np.tile([1.0, 0.0, 0.0, 0.5], n * n)

        k1 = AnisotropicShiftND2(n, n, -2.0, 2.0, -2.0, 2.0, a_flat)
        k1.set_state(u)
        k2 = AnisotropicShiftND2(n, n, -2.0, 2.0, -2.0, 2.0, a_flat)
        k2.set_state(np.ravel(u, order="F"))
        assert np.array_equal(k1.values(), k2.values())

    def test_values_2d_round_trips(self):
        n = 12
        rng = np.random.default_rng(1)
        u = rng.random((n, n))
        a_flat = np.tile([1.0, 0.0, 0.0, 1.0], n * n)
        k = AnisotropicShiftND2(n, n, -1.0, 1.0, -1.0, 1.0, a_flat)
        k.set_state(u)
        assert np.allclose(k.values_2d(), u)

    def test_c_order_ravel_is_the_transpose_trap(self):
        """A C-order ravel of an asymmetric field genuinely differs — the trap is real."""
        n = 16
        rng = np.random.default_rng(2)
        u = rng.random((n, n))
        a_flat = np.tile([1.0, 0.0, 0.0, 1.0], n * n)
        k1 = AnisotropicShiftND2(n, n, -2.0, 2.0, -2.0, 2.0, a_flat)
        k1.set_state(u)
        k2 = AnisotropicShiftND2(n, n, -2.0, 2.0, -2.0, 2.0, a_flat)
        k2.set_state(u.ravel())  # C order — the reporter's mistake
        assert not np.allclose(k1.values(), k2.values())

    def test_boundary_kwarg_is_accepted_and_changes_the_answer(self):
        n = 16
        a_flat = np.tile([1.0, 0.0, 0.0, 1.0], n * n)
        x = np.linspace(-2.0, 2.0, n)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u0 = np.exp(-(X**2 + Y**2))

        out = {}
        for policy in ("reflect", "zero", "periodic"):
            k = AnisotropicShiftND2(
                n, n, -2.0, 2.0, -2.0, 2.0, a_flat, boundary=policy
            )
            k.set_state(u0)
            k.evolve(0.4, 40)
            out[policy] = k.values()
            assert np.all(np.isfinite(out[policy]))
        # ADR-0190: the N-D sampler used to clamp and ignore the policy entirely,
        # so all three would have been byte-identical.
        assert not np.allclose(out["reflect"], out["zero"])


# ---------------------------------------------------------------------------
# Heat2DVarA — variable-a convergence order
# ---------------------------------------------------------------------------


class TestHeat2DVarAConstantCoeff:
    """Regression only. The variable-`a` order question is OPEN — see below."""

    N = 48

    def test_constant_a_still_matches_heat2d(self):
        ones = np.ones(self.N)
        x = np.linspace(0.0, 1.0, self.N)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u0 = np.ravel(np.sin(np.pi * X) * np.sin(np.pi * Y), order="F")
        var_a = Heat2DVarA(0.0, 1.0, self.N, 0.0, 1.0, self.N, ones, ones)
        plain = Heat2D(0.0, 1.0, self.N, 0.0, 1.0, self.N)
        got = var_a.evolve(u0, 0.01, 20)
        want = plain.evolve(u0, 0.01, 20)
        assert np.max(np.abs(got - want)) < 1e-10

    @pytest.mark.skip(
        reason="OPEN (ADR-0190): Heat2DVarA passes a'=a''=0 to a divergence-form "
        "kernel while advertising the non-divergence operator a_x(x)*u_xx across "
        "all three binding surfaces. A 1-D A/B measured slope ~1.09 with zeroed "
        "derivatives and ~1.03 with derived ones -- both at the O(tau^1) global "
        "ceiling documented in ADR-0112 AM2 / math.md 9.2.3.B, so self-convergence "
        "does not discriminate. Needs an analytic oracle per candidate operator "
        "plus an architect decision on which PDE Heat2DVarA is meant to solve."
    )
    def test_variable_a_self_convergence_slope_is_two(self):
        raise AssertionError("see skip reason")
