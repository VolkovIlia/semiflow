"""Smoke tests for v9 S³ PyO3 bindings: TtState, TtEvolver, TtCoupledEvolver,
MeasureState, GridlessEvolver, VarCoefTtEvolver.

Exercises every class instantiation + one evolution call and checks that the
result is finite (sanity gate only; numerical accuracy is validated by the
Rust test suite in tt_chernoff_tests.rs / gridless_tests.rs).
"""

import math
import numpy as np
import pytest
import semiflow
from semiflow import (
    TtState,
    TtEvolver,
    TtCoupledEvolver,
    MeasureState,
    GridlessEvolver,
    VarCoefTtEvolver,
    SemiflowError,
)


# ---------------------------------------------------------------------------
# TtState
# ---------------------------------------------------------------------------

class TestTtState:
    def _make(self, d=2, n=8):
        """Rank-1 separable IC: uniform-1 slices, length n, d axes."""
        slices = [np.ones(n, dtype=np.float64) for _ in range(d)]
        return TtState(slices)

    def test_construction(self):
        s = self._make()
        assert s.ndim() == 2
        assert s.peak_rank() >= 1
        assert s.storage_size() > 0

    def test_n_j(self):
        s = self._make(d=3, n=6)
        for j in range(3):
            assert s.n_j(j) == 6

    def test_n_j_out_of_range(self):
        s = self._make(d=2)
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            s.n_j(99)

    def test_inner_separable(self):
        n = 8
        s = self._make(d=2, n=n)
        funcs = [np.ones(n, dtype=np.float64) for _ in range(2)]
        v = s.inner_separable(funcs)
        assert math.isfinite(v)
        assert v > 0.0

    def test_inner_separable_length_mismatch(self):
        s = self._make(d=2, n=8)
        with pytest.raises(SemiflowError, match="GridMismatch"):
            # wrong functional list length (3 != 2)
            s.inner_separable([np.ones(8)] * 3)

    def test_inner_separable_axis_mismatch(self):
        s = self._make(d=2, n=8)
        with pytest.raises(SemiflowError, match="GridMismatch"):
            # axis 1 has wrong length
            s.inner_separable([np.ones(8), np.ones(5)])

    def test_empty_slices_rejected(self):
        with pytest.raises(SemiflowError, match="GridMismatch"):
            TtState([])

    def test_nanf_in_slice_rejected(self):
        slices = [np.array([1.0, float("nan")])]
        with pytest.raises(SemiflowError, match="NanInf"):
            TtState(slices)


# ---------------------------------------------------------------------------
# TtEvolver
# ---------------------------------------------------------------------------

class TestTtEvolver:
    def _make_evolver(self, d=2, a=0.5):
        return TtEvolver(
            a=[a] * d,
            b=[0.0] * d,
            c=0.0,
            dom_min=[-3.0] * d,
            dom_max=[3.0] * d,
            eps_round=1e-10,
        )

    def _make_state(self, d=2, n=16):
        # Gaussian-shaped IC on each axis
        xs = np.linspace(-3.0, 3.0, n)
        g = np.exp(-xs**2)
        return TtState([g.copy() for _ in range(d)])

    def test_construction(self):
        ev = self._make_evolver()
        assert ev.ndim() == 2

    def test_evolve_returns_finite(self):
        d = 2
        ev = self._make_evolver(d=d)
        state = self._make_state(d=d, n=16)
        ev.evolve(state, t_final=0.05, n_steps=4)
        # Check via inner product with uniform functional
        n = state.n_j(0)
        v = state.inner_separable([np.ones(n)] * d)
        assert math.isfinite(v)

    def test_evolve_zero_steps_rejected(self):
        ev = self._make_evolver()
        state = self._make_state()
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ev.evolve(state, t_final=0.1, n_steps=0)

    def test_evolve_negative_t_rejected(self):
        ev = self._make_evolver()
        state = self._make_state()
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ev.evolve(state, t_final=-0.1, n_steps=4)

    def test_evolve_ndim_mismatch(self):
        ev = self._make_evolver(d=2)
        state = self._make_state(d=3)  # ndim=3 != ev.ndim()=2
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ev.evolve(state, t_final=0.1, n_steps=2)

    def test_nan_a_rejected(self):
        with pytest.raises(SemiflowError, match="NanInf"):
            TtEvolver(
                a=[float("nan")],
                b=[0.0],
                c=0.0,
                dom_min=[-1.0],
                dom_max=[1.0],
                eps_round=1e-10,
            )

    def test_negative_a_rejected(self):
        with pytest.raises(SemiflowError, match="NanInf"):
            TtEvolver(
                a=[-0.1],
                b=[0.0],
                c=0.0,
                dom_min=[-1.0],
                dom_max=[1.0],
                eps_round=1e-10,
            )


# ---------------------------------------------------------------------------
# TtCoupledEvolver
# ---------------------------------------------------------------------------

class TestTtCoupledEvolver:
    def _make_state(self, d=2, n=16):
        xs = np.linspace(-3.0, 3.0, n)
        g = np.exp(-xs**2)
        return TtState([g.copy() for _ in range(d)])

    def _evolver(self, d=2, coupling=("None",), a=0.5):
        return TtCoupledEvolver(
            a=[a] * d,
            b=[0.0] * d,
            c=0.0,
            coupling=coupling,
            dom_min=[-3.0] * d,
            dom_max=[3.0] * d,
            eps_round=1e-10,
        )

    def test_none_coupling(self):
        ev = self._evolver(coupling=("None",))
        assert ev.ndim() == 2

    def test_tridiagonal_evolve_finite(self):
        d = 3
        ev = self._evolver(d=d, coupling=("Tridiagonal", 0.3))
        state = self._make_state(d=d)
        ev.evolve(state, t_final=0.05, n_steps=4)
        n = state.n_j(0)
        v = state.inner_separable([np.ones(n)] * d)
        assert math.isfinite(v)

    def test_pairs_evolve_finite(self):
        d = 2
        # a=0.5 for both, rho=0.3 → det = 0.25 - 0.09 = 0.16 > 0 ✓
        ev = TtCoupledEvolver(
            a=[0.5, 0.5],
            b=[0.0, 0.0],
            c=0.0,
            coupling=("Pairs", [(0, 1, 0.3)]),
            dom_min=[-3.0, -3.0],
            dom_max=[3.0, 3.0],
            eps_round=1e-10,
        )
        state = self._make_state(d=d)
        ev.evolve(state, t_final=0.05, n_steps=4)
        v = state.inner_separable([np.ones(state.n_j(0))] * d)
        assert math.isfinite(v)

    def test_drift_wall(self):
        """b != 0 must raise OutOfDomain."""
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            TtCoupledEvolver(
                a=[0.5, 0.5],
                b=[0.1, 0.0],  # drift != 0
                c=0.0,
                coupling=("None",),
                dom_min=[-3.0, -3.0],
                dom_max=[3.0, 3.0],
                eps_round=1e-10,
            )

    def test_non_adjacent_pair_wall(self):
        """Pair (0, 2) is not adjacent — must raise OutOfDomain."""
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            TtCoupledEvolver(
                a=[0.5, 0.5, 0.5],
                b=[0.0, 0.0, 0.0],
                c=0.0,
                coupling=("Pairs", [(0, 2, 0.3)]),
                dom_min=[-3.0] * 3,
                dom_max=[3.0] * 3,
                eps_round=1e-10,
            )

    def test_non_spd_block_wall(self):
        """rho=1.0 with a=[0.5,0.5] gives det=0 — must raise OutOfDomain."""
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            TtCoupledEvolver(
                a=[0.5, 0.5],
                b=[0.0, 0.0],
                c=0.0,
                coupling=("Pairs", [(0, 1, 1.0)]),
                dom_min=[-3.0, -3.0],
                dom_max=[3.0, 3.0],
                eps_round=1e-10,
            )

    def test_unknown_coupling_tag(self):
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            self._evolver(coupling=("Dense", 0.5))


# ---------------------------------------------------------------------------
# MeasureState
# ---------------------------------------------------------------------------

class TestMeasureState:
    def _dirac(self, x=0.0, w=1.0):
        return MeasureState(
            positions=np.array([x], dtype=np.float64),
            weights=np.array([w], dtype=np.float64),
            dim=1,
        )

    def test_construction(self):
        ms = self._dirac()
        assert ms.n_diracs() == 1

    def test_total_variation(self):
        ms = self._dirac(w=2.0)
        assert abs(ms.total_variation() - 2.0) < 1e-12

    def test_second_moment(self):
        ms = self._dirac(x=3.0, w=1.0)
        assert abs(ms.second_moment() - 9.0) < 1e-12

    def test_marginal(self):
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        wts = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        ms = MeasureState(positions=pos, weights=wts, dim=1)
        out_pos, out_wts = ms.marginal(axis=0)
        assert len(out_pos) == 3
        assert abs(out_wts.sum() - 1.0) < 1e-12

    def test_dim_unsupported(self):
        with pytest.raises(SemiflowError, match="Unsupported"):
            MeasureState(
                positions=np.array([0.0, 1.0]),
                weights=np.array([0.5, 0.5]),
                dim=2,
            )

    def test_empty_rejected(self):
        with pytest.raises(SemiflowError, match="GridMismatch"):
            MeasureState(
                positions=np.array([], dtype=np.float64),
                weights=np.array([], dtype=np.float64),
                dim=1,
            )

    def test_nan_rejected(self):
        with pytest.raises(SemiflowError, match="NanInf"):
            MeasureState(
                positions=np.array([float("nan")]),
                weights=np.array([1.0]),
                dim=1,
            )

    def test_marginal_axis_out_of_range(self):
        ms = self._dirac()
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ms.marginal(axis=1)


# ---------------------------------------------------------------------------
# GridlessEvolver
# ---------------------------------------------------------------------------

class TestGridlessEvolver:
    def _make(self, a=0.5, b=0.0, c=0.0, cap=64):
        return GridlessEvolver(a=a, b=b, c=c, voronoi_cap=cap)

    def _dirac_ms(self, x=0.0, w=1.0):
        return MeasureState(
            positions=np.array([x], dtype=np.float64),
            weights=np.array([w], dtype=np.float64),
            dim=1,
        )

    def test_construction(self):
        ev = self._make()
        assert ev is not None  # constructor did not raise

    def test_evolve_finite(self):
        ev = self._make(a=0.5)
        ms = self._dirac_ms()
        ev.evolve(ms, t_final=0.1, n_steps=4)
        assert math.isfinite(ms.total_variation())
        assert math.isfinite(ms.second_moment())
        # Second moment should grow (diffusion spreading)
        assert ms.second_moment() >= 0.0

    def test_apply_single_step(self):
        ev = self._make(a=0.5)
        src = self._dirac_ms(x=0.0, w=1.0)
        dst = self._dirac_ms(x=99.0, w=0.0)  # will be overwritten
        ev.apply(tau=0.1, src=src, dst=dst)
        tv = dst.total_variation()
        assert math.isfinite(tv)

    def test_evolve_zero_steps_rejected(self):
        ev = self._make()
        ms = self._dirac_ms()
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ev.evolve(ms, t_final=0.1, n_steps=0)

    def test_evolve_negative_t_rejected(self):
        ev = self._make()
        ms = self._dirac_ms()
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ev.evolve(ms, t_final=-0.1, n_steps=4)

    def test_negative_a_rejected(self):
        with pytest.raises(SemiflowError, match="NanInf"):
            GridlessEvolver(a=-0.1, b=0.0, c=0.0)

    def test_nan_b_rejected(self):
        with pytest.raises(SemiflowError, match="NanInf"):
            GridlessEvolver(a=0.5, b=float("nan"), c=0.0)

    def test_zero_cap_rejected(self):
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            GridlessEvolver(a=0.5, b=0.0, c=0.0, voronoi_cap=0)

    def test_gaussian_background_stub(self):
        ev = GridlessEvolver(a=0.5, b=0.0, c=0.0, gaussian_background=True)
        ms = self._dirac_ms()
        ev.evolve(ms, t_final=0.1, n_steps=2)
        assert math.isfinite(ms.total_variation())


# ---------------------------------------------------------------------------
# VarCoefTtEvolver
# ---------------------------------------------------------------------------

class TestVarCoefTtEvolver:
    """Smoke tests for VarCoefTtEvolver (ADR-0178, issue #2)."""

    N = 16
    D = 3
    XS = None  # set up in helpers

    def _xs(self):
        return np.linspace(-3.0, 3.0, self.N)

    def _const_coef(self, val):
        return [np.full(self.N, val) for _ in range(self.D)]

    def _zero_coef(self):
        return [np.zeros(self.N) for _ in range(self.D)]

    def _make_ev(self, a_val=0.5, b_val=0.0, v_val=0.0):
        dom = [(-3.0, 3.0)] * self.D
        return VarCoefTtEvolver(
            a_axis=self._const_coef(a_val),
            b_axis=self._const_coef(b_val),
            v_axis=self._const_coef(v_val),
            domain=dom,
            eps_round=1e-10,
        )

    def _make_state(self):
        xs = self._xs()
        g = np.exp(-xs ** 2)
        return TtState([g.copy() for _ in range(self.D)])

    # -----------------------------------------------------------------------
    # Oracle 1: rank-1 IC stays rank-1 after evolve (§52.10d bond-preserving).
    # The additive-separable per-axis step operates on mode slices one at a time;
    # no entanglement is introduced, so a rank-1 IC stays rank-1.
    # -----------------------------------------------------------------------
    def test_rank1_preserved_and_finite(self):
        """Rank-1 IC stays rank-1; all values finite (§52.10d invariant)."""
        ev = self._make_ev()
        state = self._make_state()
        ev.evolve(state, t_final=0.1, n_steps=4)
        assert state.peak_rank() == 1, f"rank grew to {state.peak_rank()} (expected 1)"
        sz = state.storage_size()
        assert sz > 0 and math.isfinite(sz), "storage_size not finite positive"
        n = state.n_j(0)
        v = state.inner_separable([np.ones(n)] * self.D)
        assert math.isfinite(v), f"inner_separable returned non-finite {v}"

    # -----------------------------------------------------------------------
    # Oracle 2: ndim attribute matches constructed dimension.
    # -----------------------------------------------------------------------
    def test_ndim(self):
        ev = self._make_ev()
        assert ev.ndim() == self.D

    # -----------------------------------------------------------------------
    # Validation: zero n_steps rejected.
    # -----------------------------------------------------------------------
    def test_zero_steps_rejected(self):
        ev = self._make_ev()
        state = self._make_state()
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ev.evolve(state, t_final=0.1, n_steps=0)

    # -----------------------------------------------------------------------
    # Validation: negative t_final rejected.
    # -----------------------------------------------------------------------
    def test_negative_t_rejected(self):
        ev = self._make_ev()
        state = self._make_state()
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ev.evolve(state, t_final=-0.1, n_steps=4)

    # -----------------------------------------------------------------------
    # Validation: ndim mismatch between evolver and state.
    # -----------------------------------------------------------------------
    def test_ndim_mismatch_rejected(self):
        ev = self._make_ev()               # D=3
        state_2d = TtState([np.ones(self.N)] * 2)  # d=2
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            ev.evolve(state_2d, t_final=0.1, n_steps=2)

    # -----------------------------------------------------------------------
    # Fail-loud: a_axis with non-positive entry triggers VarCoefOutOfClass
    # → OutOfDomain at Python boundary.
    # -----------------------------------------------------------------------
    def test_nonpositive_a_rejected(self):
        a_bad = [np.full(self.N, -0.1)] + [np.full(self.N, 0.5)] * (self.D - 1)
        with pytest.raises(SemiflowError, match="OutOfDomain"):
            VarCoefTtEvolver(
                a_axis=a_bad,
                b_axis=self._zero_coef(),
                v_axis=self._zero_coef(),
                domain=[(-3.0, 3.0)] * self.D,
                eps_round=1e-10,
            )

    # -----------------------------------------------------------------------
    # Fail-loud: NaN in b_axis.
    # -----------------------------------------------------------------------
    def test_nan_b_rejected(self):
        b_bad = [np.zeros(self.N) for _ in range(self.D)]
        b_bad[0][3] = float("nan")
        with pytest.raises(SemiflowError, match="NanInf"):
            VarCoefTtEvolver(
                a_axis=self._const_coef(0.5),
                b_axis=b_bad,
                v_axis=self._zero_coef(),
                domain=[(-3.0, 3.0)] * self.D,
                eps_round=1e-10,
            )

    # -----------------------------------------------------------------------
    # Fail-loud: empty a_axis (d=0).
    # -----------------------------------------------------------------------
    def test_empty_axis_rejected(self):
        with pytest.raises(SemiflowError):
            VarCoefTtEvolver(
                a_axis=[],
                b_axis=[],
                v_axis=[],
                domain=[],
                eps_round=1e-10,
            )
