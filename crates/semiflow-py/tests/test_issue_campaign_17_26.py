"""Acceptance tests for the #17 / #21 / #26 issue campaign (ADR-0191, ADR-0192).

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

from semiflow import (
    AdaptivePI,
    AnisotropicShiftND2,
    Heat2D,
    Heat2DVarA,
    Shift1D,
    SemiflowError,
    shift1d_coeff_grad,
)


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
        # Pre-ADR-0191 this returned 1.2113 / 2.2449 / 4.4901 for exact = 1.0.
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
        # ADR-0191: the N-D sampler used to clamp and ignore the policy entirely,
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
        reason="OPEN (ADR-0191): Heat2DVarA passes a'=a''=0 to a divergence-form "
        "kernel while advertising the non-divergence operator a_x(x)*u_xx across "
        "all three binding surfaces. A 1-D A/B measured slope ~1.09 with zeroed "
        "derivatives and ~1.03 with derived ones -- both at the O(tau^1) global "
        "ceiling documented in ADR-0112 AM2 / math.md 9.2.3.B, so self-convergence "
        "does not discriminate. Needs an analytic oracle per candidate operator "
        "plus an architect decision on which PDE Heat2DVarA is meant to solve."
    )
    def test_variable_a_self_convergence_slope_is_two(self):
        raise AssertionError("see skip reason")


# ---------------------------------------------------------------------------
# Issue #22 — AdaptivePI over variable-coefficient kernels
# ---------------------------------------------------------------------------


class TestAdaptivePIWithArrays:
    """AdaptivePI.with_arrays reaches Black-Scholes-type generators (#22).

    The ``kernel=`` menu on ``__init__`` only reaches constant-coefficient
    kernels — its ``"shift"`` arm hard-codes ``a=0.5, b=0, c=0`` — so a
    variable-coefficient generator had no adaptive path at all.
    """

    @staticmethod
    def _black_scholes(n=257, s_max=300.0, sigma=0.25, r=0.03, strike=100.0):
        """½σ²S²·u_SS + rS·u_S − r·u on a uniform S grid, with a call payoff."""
        s = np.linspace(0.0, s_max, n)
        a = 0.5 * sigma**2 * s**2
        b = r * s
        c = np.full(n, -r)
        u0 = np.maximum(s - strike, 0.0)
        return s, a, b, c, u0

    def test_with_arrays_runs_and_preserves_shape(self):
        s, a, b, c, u0 = self._black_scholes()
        pi = AdaptivePI.with_arrays(
            0.0, 300.0, len(s), a, b, c, 0.03, u0, tol_abs=1e-6, tol_rel=1e-4
        )
        out = pi.evolve(0.5)
        assert len(out) == len(s)
        assert np.all(np.isfinite(out))

    def test_with_arrays_actually_uses_the_coefficients(self):
        """A different sigma must give a different answer — the arrays are live."""
        s, a1, b, c, u0 = self._black_scholes(sigma=0.25)
        _, a2, _, _, _ = self._black_scholes(sigma=0.50)
        o1 = AdaptivePI.with_arrays(0.0, 300.0, len(s), a1, b, c, 0.03, u0).evolve(0.5)
        o2 = AdaptivePI.with_arrays(0.0, 300.0, len(s), a2, b, c, 0.03, u0).evolve(0.5)
        assert not np.allclose(o1, o2)

    def test_tighter_tolerance_does_not_change_the_answer_much(self):
        """Adaptivity is doing its job: the answer is tolerance-converged."""
        s, a, b, c, u0 = self._black_scholes()
        loose = AdaptivePI.with_arrays(
            0.0, 300.0, len(s), a, b, c, 0.03, u0, tol_abs=1e-6, tol_rel=1e-4
        ).evolve(0.5)
        tight = AdaptivePI.with_arrays(
            0.0, 300.0, len(s), a, b, c, 0.03, u0, tol_abs=1e-9, tol_rel=1e-7
        ).evolve(0.5)
        rel = np.max(np.abs(loose - tight)) / max(np.max(np.abs(tight)), 1.0)
        assert rel < 5e-2, f"loose vs tight differ by {rel:.3e}"

    def test_bad_coefficient_length_raises(self):
        s, a, b, c, u0 = self._black_scholes(n=64)
        with pytest.raises(SemiflowError):
            AdaptivePI.with_arrays(0.0, 300.0, 64, a[:-1], b, c, 0.03, u0)


# ---------------------------------------------------------------------------
# Issue #23 — coefficient schedules for a, b AND c
# ---------------------------------------------------------------------------


class TestCoefficientSchedule:
    """evolve_with_coefficient_schedule: b/c schedules and array entries (#23)."""

    N = 129
    XMIN, XMAX = -3.0, 3.0

    def _fresh(self):
        x = np.linspace(self.XMIN, self.XMAX, self.N)
        u0 = np.exp(-(x**2))
        return x, Shift1D(self.XMIN, self.XMAX, self.N, u0, a=0.5, b=0.0, c=0.0)

    def test_scalar_a_schedule_matches_the_legacy_entry_point(self):
        """New method reduces to the old one when only `a` varies."""
        sched = [0.4, 0.6, 0.5]
        _, k_new = self._fresh()
        k_new.evolve_with_coefficient_schedule(0.3, 25, sched)
        _, k_old = self._fresh()
        k_old.evolve_with_time_schedule(0.3, 25, np.array(sched))
        assert np.allclose(k_new.values(), k_old.values(), rtol=0, atol=1e-12)

    def test_c_schedule_is_live(self):
        """Time-varying killing must change the answer (the #23 motivation)."""
        _, k0 = self._fresh()
        k0.evolve_with_coefficient_schedule(0.3, 25, [0.5, 0.5, 0.5])
        _, k1 = self._fresh()
        k1.evolve_with_coefficient_schedule(
            0.3, 25, [0.5, 0.5, 0.5], c_schedule=[-0.5, -1.0, -2.0]
        )
        assert not np.allclose(k0.values(), k1.values())
        # Killing removes mass monotonically.
        assert k1.values().sum() < k0.values().sum()

    def test_array_entries_inside_a_schedule(self):
        """Almgren-Chriss shape: killing linear in p with a time-varying slope."""
        x, k = self._fresh()
        slopes = [0.2, 0.6, 1.2]
        c_sched = [-(s * np.abs(x)) for s in slopes]
        k.evolve_with_coefficient_schedule(
            0.3, 25, [0.5, 0.5, 0.5], c_schedule=c_sched
        )
        out = k.values()
        assert np.all(np.isfinite(out))
        # Space-varying killing must break the symmetry differently than a
        # constant one: compare against the scalar-c control.
        _, k2 = self._fresh()
        k2.evolve_with_coefficient_schedule(
            0.3, 25, [0.5, 0.5, 0.5], c_schedule=[-0.2, -0.6, -1.2]
        )
        assert not np.allclose(out, k2.values())

    def test_mixed_scalar_and_array_entries(self):
        x, k = self._fresh()
        k.evolve_with_coefficient_schedule(
            0.2, 20,
            [0.5, 0.4 + 0.1 * np.abs(x)],       # scalar, then array
            b_schedule=[0.1 * x, 0.0],           # array, then scalar
        )
        assert np.all(np.isfinite(k.values()))

    def test_subsequent_evolve_keeps_the_final_segment_coefficients(self):
        """The legacy method silently reverts to construction-time `a`; this one does not.

        After a schedule ending at ``a = 2.0``, a follow-up ``evolve`` must
        behave like a kernel with ``a = 2.0`` — not like the ``a = 0.5`` the
        object was constructed with.
        """
        _, k = self._fresh()
        k.evolve_with_coefficient_schedule(0.2, 20, [2.0, 2.0])
        mid = k.values().copy()
        k.evolve(0.2, 20)
        got = k.values().copy()

        # Reference: a fresh kernel at the schedule's final `a`, same state,
        # same tau and step count.
        want = Shift1D(self.XMIN, self.XMAX, self.N, mid, a=2.0, b=0.0, c=0.0)
        want.evolve(0.2, 20)
        assert np.allclose(got, want.values(), rtol=1e-12, atol=1e-14)

        # Negative control: the construction-time coefficient gives a different
        # answer, so the assertion above is not vacuous.
        stale = Shift1D(self.XMIN, self.XMAX, self.N, mid, a=0.5, b=0.0, c=0.0)
        stale.evolve(0.2, 20)
        assert not np.allclose(got, stale.values())

    def test_schedule_length_mismatch_raises(self):
        _, k = self._fresh()
        with pytest.raises(SemiflowError):
            k.evolve_with_coefficient_schedule(0.3, 10, [0.5, 0.5], c_schedule=[0.1])

    def test_wrong_array_length_raises(self):
        _, k = self._fresh()
        with pytest.raises(SemiflowError):
            k.evolve_with_coefficient_schedule(
                0.3, 10, [0.5], c_schedule=[np.zeros(self.N - 1)]
            )


# ---------------------------------------------------------------------------
# Issue #19 — batched multi-channel evolve for 1-D grid kernels
# ---------------------------------------------------------------------------


class TestShift1DBatched:
    """Shift1D.evolve_batched: [N, C] in, [N, C] out, 0-ULP vs the loop (#19)."""

    N = 129
    XMIN, XMAX = -4.0, 4.0

    def _kernel(self, u0):
        x = np.linspace(self.XMIN, self.XMAX, self.N)
        a = 0.3 + 0.2 * np.abs(np.tanh(0.5 * x))
        b = 0.15 * np.sin(x)
        c = -0.05 * np.log1p(x**2)
        return Shift1D.with_arrays(
            self.XMIN, self.XMAX, self.N, a, b, c, 0.7, u0
        )

    def _strip(self, n_cols):
        """A strike strip: same generator, different initial conditions."""
        x = np.linspace(self.XMIN, self.XMAX, self.N)
        return np.stack(
            [np.exp(-((x - (0.3 * c - 0.6)) ** 2)) + 0.01 * c for c in range(n_cols)],
            axis=1,
        )

    def test_batched_is_bit_identical_to_the_python_loop(self):
        """The point of the gate: batching must not perturb a single bit."""
        for n_cols in (1, 3, 8):
            u0_nc = self._strip(n_cols)
            batched = self._kernel(u0_nc[:, 0]).evolve_batched(0.05, u0_nc, 7)
            assert batched.shape == (self.N, n_cols)
            for c in range(n_cols):
                k = self._kernel(u0_nc[:, c].copy())
                k.evolve(0.05, 7)
                want = k.values()
                assert np.array_equal(
                    batched[:, c].view(np.int64), want.view(np.int64)
                ), f"n_cols={n_cols} c={c}: batched != sequential bit-for-bit"

    def test_batched_does_not_mutate_the_object(self):
        u0_nc = self._strip(4)
        k = self._kernel(u0_nc[:, 0])
        before = k.values().copy()
        k.evolve_batched(0.05, u0_nc, 5)
        assert np.array_equal(k.values(), before)

    def test_wrong_row_count_raises(self):
        u0_nc = self._strip(3)
        k = self._kernel(u0_nc[:, 0])
        with pytest.raises(SemiflowError):
            k.evolve_batched(0.05, u0_nc[:-1, :], 5)

    def test_nan_input_raises(self):
        u0_nc = self._strip(3).copy()
        u0_nc[5, 1] = np.nan
        k = self._kernel(u0_nc[:, 0])
        with pytest.raises(SemiflowError):
            k.evolve_batched(0.05, u0_nc, 5)


# ---------------------------------------------------------------------------
# Issue #21 — full-grid a_x(x,y) / a_y(x,y)
# ---------------------------------------------------------------------------


class TestHeat2DVarAGridArrays:
    """Heat2DVarA.with_grid_arrays: transverse-varying coefficients (#21)."""

    N = 33

    @staticmethod
    def _flat(nx, ny, f):
        """Build a full-grid array in the library's x-fastest layout."""
        out = np.empty(nx * ny)
        for j in range(ny):
            for i in range(nx):
                out[j * nx + i] = f(i / (nx - 1), j / (ny - 1))
        return out

    def _ic(self):
        x = np.linspace(0.0, 1.0, self.N)
        X, Y = np.meshgrid(x, x, indexing="ij")
        return np.ravel(np.exp(-40.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)), order="F")

    def test_separable_input_reproduces_the_per_axis_constructor(self):
        """Reduction anchor: a_x(x,y)=a_x(x) must match the old constructor."""
        n = self.N
        ax_prof = 1.0 + 0.3 * np.sin(np.pi * np.linspace(0, 1, n))
        ay_prof = 1.0 + 0.3 * np.cos(np.pi * np.linspace(0, 1, n))
        old = Heat2DVarA(0.0, 1.0, n, 0.0, 1.0, n, ax_prof, ay_prof)
        new = Heat2DVarA.with_grid_arrays(
            0.0, 1.0, n, 0.0, 1.0, n,
            self._flat(n, n, lambda x, y: 1.0 + 0.3 * np.sin(np.pi * x)),
            self._flat(n, n, lambda x, y: 1.0 + 0.3 * np.cos(np.pi * y)),
        )
        u0 = self._ic()
        assert np.allclose(
            old.evolve(u0, 0.001, 20), new.evolve(u0, 0.001, 20), rtol=0, atol=1e-12
        )

    def test_transverse_variation_is_expressible_and_differs(self):
        """The whole point: a_x depending on y, which the old API cannot say."""
        n = self.N
        k = Heat2DVarA.with_grid_arrays(
            0.0, 1.0, n, 0.0, 1.0, n,
            self._flat(n, n, lambda x, y: 1.0 + 0.3 * np.sin(2 * np.pi * y)),
            self._flat(n, n, lambda x, y: 1.0 + 0.3 * np.cos(2 * np.pi * x)),
        )
        u0 = self._ic()
        got = k.evolve(u0, 0.001, 20)
        assert np.all(np.isfinite(got))

        # The best separable stand-in (the transverse means, both 1.0) differs.
        ones = np.ones(n)
        sep = Heat2DVarA(0.0, 1.0, n, 0.0, 1.0, n, ones, ones).evolve(u0, 0.001, 20)
        assert not np.allclose(got, sep, atol=1e-8)

    def test_validation(self):
        n = self.N
        good = self._flat(n, n, lambda x, y: 1.0)
        with pytest.raises(SemiflowError):  # wrong length
            Heat2DVarA.with_grid_arrays(0.0, 1.0, n, 0.0, 1.0, n, good[:-1], good)
        bad = good.copy()
        bad[10] = 0.0
        with pytest.raises(SemiflowError):  # non-positive
            Heat2DVarA.with_grid_arrays(0.0, 1.0, n, 0.0, 1.0, n, bad, good)
        nan = good.copy()
        nan[3] = np.nan
        with pytest.raises(SemiflowError):
            Heat2DVarA.with_grid_arrays(0.0, 1.0, n, 0.0, 1.0, n, good, nan)


# ---------------------------------------------------------------------------
# Issue #25 — gradients w.r.t. Shift1D coefficient fields
# ---------------------------------------------------------------------------


class TestShift1DCoeffGrad:
    """shift1d_coeff_grad: dJ/da_i, dJ/db_i, dJ/dc_i (#25)."""

    N = 24
    XMIN, XMAX = -2.0, 2.0

    def _setup(self):
        x = np.linspace(self.XMIN, self.XMAX, self.N)
        a = 0.25 + 0.15 * np.abs(np.sin(np.arange(self.N) * 0.31))
        b = 0.10 * np.sin(np.arange(self.N) * 0.17)
        c = -0.05 * np.cos(np.arange(self.N) * 0.23)
        u0 = np.sin(1.7 * x) + 0.5 * np.cos(0.7 * x) + 1.0
        return a, b, c, u0

    def _solve(self, a, b, c, u0, t, n_steps):
        k = Shift1D.with_arrays(
            self.XMIN, self.XMAX, self.N, a, b, c, float(np.max(np.abs(c))), u0
        )
        k.evolve(t, n_steps)
        return k.values()

    @pytest.mark.parametrize("wrt", ["a", "b", "c"])
    def test_gradient_matches_finite_differences(self, wrt):
        a, b, c, u0 = self._setup()
        t, n_steps = 0.024, 6
        un = self._solve(a, b, c, u0, t, n_steps)
        grad = shift1d_coeff_grad(
            self.XMIN, self.XMAX, self.N, a, b, c, u0, un, t, n_steps, wrt=wrt
        )
        assert grad.shape == (self.N,)

        eps = 1e-6
        fd = np.zeros(self.N)
        loss = lambda A, B, C: 0.5 * np.sum(self._solve(A, B, C, u0, t, n_steps) ** 2)
        for i in range(self.N):
            arrs = {"a": a, "b": b, "c": c}
            plus = {k: v.copy() for k, v in arrs.items()}
            minus = {k: v.copy() for k, v in arrs.items()}
            plus[wrt][i] += eps
            minus[wrt][i] -= eps
            fd[i] = (loss(plus["a"], plus["b"], plus["c"])
                     - loss(minus["a"], minus["b"], minus["c"])) / (2 * eps)
        scale = np.max(np.abs(fd))
        assert scale > 1e-6, "FD gradient is identically zero"
        assert np.max(np.abs(grad - fd)) / scale < 1e-5, (
            f"wrt={wrt}: max|adjoint-fd|/max|fd| = "
            f"{np.max(np.abs(grad - fd)) / scale:.3e}"
        )

    def test_degenerate_a_is_refused_for_wrt_a_only(self):
        a, b, c, u0 = self._setup()
        a[3] = 0.0
        with pytest.raises(SemiflowError):
            shift1d_coeff_grad(
                self.XMIN, self.XMAX, self.N, a, b, c, u0, u0, 0.01, 2, wrt="a"
            )
        # b and c carry no sqrt(tau/a) factor and stay available.
        out = shift1d_coeff_grad(
            self.XMIN, self.XMAX, self.N, a, b, c, u0, u0, 0.01, 2, wrt="c"
        )
        assert np.all(np.isfinite(out))

    def test_bad_wrt_and_lengths_raise(self):
        a, b, c, u0 = self._setup()
        with pytest.raises(SemiflowError):
            shift1d_coeff_grad(
                self.XMIN, self.XMAX, self.N, a, b, c, u0, u0, 0.01, 2, wrt="d"
            )
        with pytest.raises(SemiflowError):
            shift1d_coeff_grad(
                self.XMIN, self.XMAX, self.N, a[:-1], b, c, u0, u0, 0.01, 2, wrt="a"
            )
