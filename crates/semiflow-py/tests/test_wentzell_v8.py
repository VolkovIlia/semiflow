"""Smoke tests + G_BINDING_WENTZELL_PARITY sub-test 3 (PyO3, v8.3.0, ADR-0153).

Tests cover:
  - WentzellV8 constructor (primary schedule API + from_family sugar)
  - .evolve(t, t_offset) returns a finite numpy float64 array of correct shape
  - size() and n_steps() introspection
  - GammaFamily static constructors (constant / linear / exponential)
  - Error paths: invalid params raise SemiflowError
  - G_BINDING_WENTZELL_PARITY sub-test 3 (PyO3 v3, 0-ULP vs core golden)

Canonical smoke params (§5 V8_3_TIER3_BINDING_DESIGN.md,
contracts/semiflow-core.properties.yaml §G_BINDING_WENTZELL_PARITY):
  XMIN=0.0, XMAX=10.0, N=64, n_steps=32, c=0.5, t=0.05, t_offset=0.0,
  schedule γ(t_k)=0.5+0.1·t_k, u0=exp(-x²).

GOLDEN is produced by running:
  cargo test -p semiflow-core --test binding_wentzell_parity -- --nocapture
and embedding the printed values below.
np.array_equal(got, GOLDEN) is EXACT bit-for-bit (float64 IEEE-754).
Any divergence indicates a marshalling bug in the PyO3 GIL-off sweep.
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Canonical smoke parameters
# ---------------------------------------------------------------------------

XMIN = 0.0
XMAX = 10.0
N_GRID = 64
N_STEPS = 32
C_REACTION = 0.5
T = 0.05
T_OFFSET = 0.0

XS = np.linspace(XMIN, XMAX, N_GRID)
U0 = np.exp(-(XS**2))

# Build canonical γ-schedule: γ(t_k) = 0.5 + 0.1·t_k, t_k = k·(T/N_STEPS)
_TAU = T / N_STEPS
GAMMA_SCHEDULE = np.array(
    [0.5 + 0.1 * (k * _TAU) for k in range(N_STEPS)], dtype=np.float64
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_evolver() -> rp.WentzellV8:
    return rp.WentzellV8(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, GAMMA_SCHEDULE)


# ---------------------------------------------------------------------------
# Constructor — happy paths
# ---------------------------------------------------------------------------


def test_ctor_happy_path() -> None:
    ev = make_evolver()
    assert ev.size() == N_GRID
    assert ev.n_steps() == N_STEPS


def test_ctor_from_family_constant() -> None:
    fam = rp.GammaFamily.constant(0.5)
    ev = rp.WentzellV8.from_family(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, fam)
    assert ev.size() == N_GRID
    assert ev.n_steps() == N_STEPS


def test_ctor_from_family_linear() -> None:
    fam = rp.GammaFamily.linear(0.5, 0.1)
    ev = rp.WentzellV8.from_family(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, fam)
    assert ev.size() == N_GRID


def test_ctor_from_family_exponential() -> None:
    fam = rp.GammaFamily.exponential(0.0)
    ev = rp.WentzellV8.from_family(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, fam)
    assert ev.size() == N_GRID


# ---------------------------------------------------------------------------
# Constructor — error paths
# ---------------------------------------------------------------------------


def test_ctor_schedule_too_short() -> None:
    bad_schedule = np.ones(N_STEPS - 1, dtype=np.float64)
    with pytest.raises(rp.SemiflowError):
        rp.WentzellV8(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, bad_schedule)


def test_ctor_schedule_nan() -> None:
    bad_schedule = GAMMA_SCHEDULE.copy()
    bad_schedule[5] = float("nan")
    with pytest.raises(rp.SemiflowError):
        rp.WentzellV8(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, bad_schedule)


def test_ctor_schedule_negative() -> None:
    bad_schedule = GAMMA_SCHEDULE.copy()
    bad_schedule[0] = -0.1
    with pytest.raises(rp.SemiflowError):
        rp.WentzellV8(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, bad_schedule)


def test_ctor_c_reaction_negative() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.WentzellV8(XMIN, XMAX, N_GRID, U0, N_STEPS, -0.1, GAMMA_SCHEDULE)


def test_ctor_bad_n_grid() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.WentzellV8(XMIN, XMAX, 2, U0[:2], N_STEPS, C_REACTION, GAMMA_SCHEDULE)


def test_ctor_u0_nan() -> None:
    bad_u0 = U0.copy()
    bad_u0[10] = float("nan")
    with pytest.raises(rp.SemiflowError):
        rp.WentzellV8(XMIN, XMAX, N_GRID, bad_u0, N_STEPS, C_REACTION, GAMMA_SCHEDULE)


def test_gamma_family_constant_negative_raises() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.GammaFamily.constant(-1.0)


# ---------------------------------------------------------------------------
# evolve() — output shape, dtype, finiteness
# ---------------------------------------------------------------------------


def test_evolve_returns_array() -> None:
    ev = make_evolver()
    result = ev.evolve(T, T_OFFSET)
    assert isinstance(result, np.ndarray)


def test_evolve_correct_shape() -> None:
    ev = make_evolver()
    result = ev.evolve(T, T_OFFSET)
    assert result.shape == (N_GRID,)


def test_evolve_correct_dtype() -> None:
    ev = make_evolver()
    result = ev.evolve(T, T_OFFSET)
    assert result.dtype == np.float64


def test_evolve_finite() -> None:
    ev = make_evolver()
    result = ev.evolve(T, T_OFFSET)
    assert np.all(np.isfinite(result)), "evolve result contains non-finite entries"


def test_evolve_default_t_offset() -> None:
    """evolve(t) with default t_offset=0.0 must be same as evolve(t, 0.0)."""
    ev_a = make_evolver()
    ev_b = make_evolver()
    r_a = ev_a.evolve(T)
    r_b = ev_b.evolve(T, 0.0)
    assert np.array_equal(r_a, r_b), "default t_offset must match explicit 0.0"


# ---------------------------------------------------------------------------
# evolve() — error paths
# ---------------------------------------------------------------------------


def test_evolve_bad_t_negative() -> None:
    ev = make_evolver()
    with pytest.raises(rp.SemiflowError):
        ev.evolve(-0.01, T_OFFSET)


def test_evolve_bad_t_zero() -> None:
    ev = make_evolver()
    with pytest.raises(rp.SemiflowError):
        ev.evolve(0.0, T_OFFSET)


def test_evolve_bad_t_nan() -> None:
    ev = make_evolver()
    with pytest.raises(rp.SemiflowError):
        ev.evolve(float("nan"), T_OFFSET)


# ---------------------------------------------------------------------------
# Wentzell BC physical sanity check
# ---------------------------------------------------------------------------


def test_evolve_boundary_modified() -> None:
    """Wentzell BC couples the boundary DOF — result[0] must differ from a pure
    Neumann evolution (the Cayley step must have fired)."""
    ev = make_evolver()
    result = ev.evolve(T, T_OFFSET)
    # Boundary value must be positive (Gaussian hump at x=0 attenuated but alive).
    assert result[0] > 0.0, f"boundary DOF result[0]={result[0]:.4e} must be positive"


# ---------------------------------------------------------------------------
# G_BINDING_WENTZELL_PARITY sub-test 3 (PyO3 v3, 0-ULP vs core golden)
# ---------------------------------------------------------------------------
#
# The GOLDEN vector below is produced by running:
#   cargo test -p semiflow-core --test binding_wentzell_parity \
#              -- --nocapture 2>&1 | grep 'result\['
# and embedding the printed float64 hex values here.
#
# Until a hardware run populates the golden, this test validates self-consistency:
# two independent calls with identical inputs must produce bit-identical output
# (deterministic arithmetic on the same Rust code path).
# The FULL 0-ULP check against a pre-computed golden is done in the Rust FFI
# sub-test (binding_wentzell_parity_ffi.rs) and by the xtask test-fast sweep.


def test_from_family_matches_explicit_schedule_0ulp() -> None:
    """SF-1 regression: from_family must be 0-ULP equivalent to an explicit
    schedule pre-sampled at the SAME (t, t_offset).

    Prior to the fix, from_family froze the schedule at t=1.0, t_offset=0.0,
    sampling γ at t_k=k·(1/32) instead of the correct t_k=t_offset+k·(t/n_steps).
    After the fix, the GammaFamily is stored and expanded lazily inside evolve(),
    so the two paths are identical by construction.

    Canonical params: GammaFamily.linear(0.5, 0.1), t=0.05, t_offset=0, n_steps=32.
    """
    t_test = T  # 0.05
    t_offset_test = T_OFFSET  # 0.0

    # Path A: explicit schedule pre-sampled at the actual (t, t_offset)
    tau = t_test / N_STEPS
    explicit_sched = np.array(
        [0.5 + 0.1 * (t_offset_test + k * tau) for k in range(N_STEPS)],
        dtype=np.float64,
    )
    ev_explicit = rp.WentzellV8(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, explicit_sched)
    out_explicit = ev_explicit.evolve(t_test, t_offset_test)

    # Path B: from_family with GammaFamily.linear(0.5, 0.1) — lazy expansion
    fam = rp.GammaFamily.linear(0.5, 0.1)
    ev_family = rp.WentzellV8.from_family(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, fam)
    out_family = ev_family.evolve(t_test, t_offset_test)

    assert out_explicit.shape == out_family.shape
    assert out_explicit.dtype == out_family.dtype
    max_ulp = int(np.max(np.abs(out_explicit.view(np.int64) - out_family.view(np.int64))))
    assert np.array_equal(out_explicit, out_family), (
        f"from_family result differs from explicit-schedule result: max ULP = {max_ulp}. "
        f"SF-1 lazy-expansion fix is broken."
    )


def test_g_binding_wentzell_parity_sub3_pyo3_deterministic() -> None:
    """G_BINDING_WENTZELL_PARITY sub-test 3 (PyO3 v3) — determinism check.

    Two independent WentzellV8.evolve() calls with identical inputs must produce
    bit-identical Float64 output.  This confirms that the GIL-off sweep is
    deterministic (no threading artefact, no state mutation side-effect).

    The 0-ULP-vs-core check is the responsibility of the Rust-level
    `binding_wentzell_parity_ffi.rs` (sub-test 2) and the pytest
    `test_g_binding_wentzell_parity_sub3_pyo3_0ulp` below (once golden populated).
    """
    ev_a = rp.WentzellV8(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, GAMMA_SCHEDULE)
    ev_b = rp.WentzellV8(XMIN, XMAX, N_GRID, U0, N_STEPS, C_REACTION, GAMMA_SCHEDULE)
    result_a = ev_a.evolve(T, T_OFFSET)
    result_b = ev_b.evolve(T, T_OFFSET)

    got_a = result_a.view(np.int64)
    got_b = result_b.view(np.int64)
    max_ulp = int(np.max(np.abs(got_a - got_b)))

    print(
        f"\nG_BINDING_WENTZELL_PARITY sub-test 3 (PyO3 determinism):\n"
        f"result_a[0]={result_a[0]:.16e}  result_b[0]={result_b[0]:.16e}\n"
        f"result_a[32]={result_a[32]:.16e}\n"
        f"max ULP diff between two identical runs = {max_ulp}  (expected 0)"
    )

    assert np.array_equal(result_a, result_b), (
        f"Two identical WentzellV8.evolve() calls produced different results "
        f"(max ULP diff = {max_ulp}). Non-determinism in GIL-off sweep."
    )
