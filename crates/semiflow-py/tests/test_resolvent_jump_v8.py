"""Smoke tests for ResolventJumpV8 (v8.1.0 F2, ADR-0138, ADR-0134).

Tests cover:
  - ResolventJumpV8 constructor (happy + error paths)
  - .jump(t, g) returns a finite numpy float64 array of correct shape
  - size() and m_nodes() introspection
  - Error paths: invalid params raise SemiflowError
  - G_BINDING_RESOLVENT_JUMP_PARITY sub-test 3 (PyO3 v3, 0-ULP vs core golden)

Canonical smoke params (§1.1 V8_1_TIER3_BINDING_DESIGN.md,
contracts/semiflow-core.properties.yaml §G_BINDING_RESOLVENT_JUMP_PARITY):
  XMIN=-10.0, XMAX=10.0, N=64, M_NODES=16, T=0.5, u0=exp(-x²),
  unit diffusion a=1, DEFAULT grid.

GOLDEN_JUMP produced by:
  cargo test --package semiflow-core --test binding_resolvent_jump_parity \
             --features slow-tests -- --nocapture
(crates/semiflow-core/tests/binding_resolvent_jump_parity.rs,
 verified against M_ref=40 self-convergence at ‖jump_M16−jump_M40‖∞ ~3e-8.)
np.array_equal(got, GOLDEN_JUMP) is EXACT bit-for-bit (float64 IEEE-754).
Any divergence indicates a marshalling bug in the PyO3 layer.
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Canonical smoke parameters (§1.1 design doc)
# ---------------------------------------------------------------------------

XMIN = -10.0
XMAX = 10.0
N_GRID = 64
M_NODES = 16
T = 0.5  # LARGE step — the whole point of F2

XS = np.linspace(XMIN, XMAX, N_GRID)
U0 = np.exp(-(XS**2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_evolver(
    n_grid: int = N_GRID, m_nodes: int = M_NODES
) -> rp.ResolventJumpV8:
    return rp.ResolventJumpV8(XMIN, XMAX, n_grid, m_nodes)


# ---------------------------------------------------------------------------
# Constructor — happy paths
# ---------------------------------------------------------------------------


def test_ctor_happy_path() -> None:
    ev = make_evolver()
    assert ev.size() == N_GRID
    assert ev.m_nodes() == M_NODES


def test_ctor_custom_m_nodes() -> None:
    ev = rp.ResolventJumpV8(XMIN, XMAX, N_GRID, 32)
    assert ev.m_nodes() == 32


# ---------------------------------------------------------------------------
# Constructor — error paths
# ---------------------------------------------------------------------------


def test_ctor_bad_m_nodes_too_small() -> None:
    """m_nodes < 6 must raise SemiflowError (OutOfDomain)."""
    with pytest.raises(rp.SemiflowError):
        rp.ResolventJumpV8(XMIN, XMAX, N_GRID, 3)


def test_ctor_bad_n_grid_too_small() -> None:
    """n_grid < 4 must raise SemiflowError (GridMismatch)."""
    with pytest.raises(rp.SemiflowError):
        rp.ResolventJumpV8(XMIN, XMAX, 2, M_NODES)


def test_ctor_bad_domain_reversed() -> None:
    """domain_lo >= domain_hi must raise SemiflowError."""
    with pytest.raises(rp.SemiflowError):
        rp.ResolventJumpV8(10.0, -10.0, N_GRID, M_NODES)


# ---------------------------------------------------------------------------
# jump() — output shape, dtype, finiteness
# ---------------------------------------------------------------------------


def test_jump_returns_array() -> None:
    ev = make_evolver()
    result = ev.jump(T, U0)
    assert isinstance(result, np.ndarray)


def test_jump_correct_shape() -> None:
    ev = make_evolver()
    result = ev.jump(T, U0)
    assert result.shape == (N_GRID,)


def test_jump_correct_dtype() -> None:
    ev = make_evolver()
    result = ev.jump(T, U0)
    assert result.dtype == np.float64


def test_jump_finite() -> None:
    ev = make_evolver()
    result = ev.jump(T, U0)
    assert np.all(np.isfinite(result)), "jump result contains non-finite entries"


# ---------------------------------------------------------------------------
# jump() — error paths
# ---------------------------------------------------------------------------


def test_jump_bad_t_negative() -> None:
    ev = make_evolver()
    with pytest.raises(rp.SemiflowError):
        ev.jump(-1.0, U0)


def test_jump_bad_t_zero() -> None:
    ev = make_evolver()
    with pytest.raises(rp.SemiflowError):
        ev.jump(0.0, U0)


def test_jump_bad_t_nan() -> None:
    ev = make_evolver()
    with pytest.raises(rp.SemiflowError):
        ev.jump(float("nan"), U0)


def test_jump_g_wrong_length() -> None:
    """Passing g with len != n_grid must raise SemiflowError (GridMismatch)."""
    ev = make_evolver()
    bad_g = np.ones(N_GRID + 5)
    with pytest.raises(rp.SemiflowError):
        ev.jump(T, bad_g)


# ---------------------------------------------------------------------------
# G_BINDING_RESOLVENT_JUMP_PARITY sub-test 3 (PyO3 v3, 0-ULP vs core golden)
# ---------------------------------------------------------------------------
#
# Canonical params (contracts/semiflow-core.properties.yaml §G_BINDING_RESOLVENT_JUMP_PARITY):
#   XMIN=-10.0, XMAX=10.0, N=64, M_NODES=16, T=0.5, u0=exp(-x²),
#   unit diffusion a=1, DEFAULT grid.
#
# GOLDEN_JUMP produced by:
#   cargo test --package semiflow-core --test binding_resolvent_jump_parity \
#              --features slow-tests -- --nocapture
# (binding_resolvent_jump_parity.rs `canonical_resolvent_jump_core`,
#  verified against M_ref=40 self-convergence.)
#
# np.array_equal(got, GOLDEN_JUMP) is EXACT bit-for-bit (float64 IEEE-754).
# Any divergence indicates a marshalling bug in the PyO3 layer.

# fmt: off
GOLDEN_JUMP = np.array([
    -1.6604697501999292e-8, -1.8196979968145994e-8, -2.1043919050724678e-8,
    -2.4511947707652132e-8, -2.7750673863565958e-8, -2.9755645262172097e-8,
    -2.9178481838373031e-8, -2.3064492315000520e-8, -1.8294588627155877e-9,
     6.8098941268099657e-8,  3.0025203991446158e-7,  1.0590372225268815e-6,
     3.4540117153368524e-6,  1.0688839435941088e-5,  3.1520824771074648e-5,
     8.8557240011859345e-5,  2.3674745541133347e-4,  6.0139395379888003e-4,
     1.4494198768605999e-3,  3.3092310985592330e-3,  7.1464104426002132e-3,
     1.4574931605988384e-2,  2.8029691282033423e-2,  5.0754194890130812e-2,
     8.6404743263586714e-2,  1.3810723156721388e-1,  2.0699074294690928e-1,
     2.9055989251532677e-1,  3.8161744643560219e-1,  4.6855398374232571e-1,
     5.3745717006837412e-1,  5.7568582771510446e-1,  5.7568582771510468e-1,
     5.3745717006837446e-1,  4.6855398374232632e-1,  3.8161744643560264e-1,
     2.9055989251532705e-1,  2.0699074294690953e-1,  1.3810723156721413e-1,
     8.6404743263586894e-2,  5.0754194890130937e-2,  2.8029691282033503e-2,
     1.4574931605988432e-2,  7.1464104426002383e-3,  3.3092310985592456e-3,
     1.4494198768606056e-3,  6.0139395379888252e-4,  2.3674745541133445e-4,
     8.8557240011859724e-5,  3.1520824771074743e-5,  1.0688839435941097e-5,
     3.4540117153368545e-6,  1.0590372225268821e-6,  3.0025203991446105e-7,
     6.8098941268099895e-8, -1.8294588627148139e-9, -2.3064492315000017e-8,
    -2.9178481838372693e-8, -2.9755645262171945e-8, -2.7750673863565958e-8,
    -2.4511947707652185e-8, -2.1043919050724764e-8, -1.8196979968146083e-8,
    -1.6604697501999388e-8,
], dtype=np.float64)
# fmt: on


def test_g_binding_resolvent_jump_parity_sub3_pyo3_0ulp() -> None:
    """G_BINDING_RESOLVENT_JUMP_PARITY sub-test 3 (PyO3 v3).

    Calls ResolventJumpV8.jump() with the canonical params and asserts the
    returned numpy array is byte-identical (0 ULP) to the core golden.

    How called: semiflow.ResolventJumpV8 (PyO3 pyclass, GIL-release via
    py.detach around the M-node TWS contour solve, ADR-0031).
    The jump() return is a numpy float64 array unmarshalled from the Rust
    Vec<f64> via .to_pyarray().

    np.array_equal performs element-wise equality including sign of zero —
    exactly the 0-ULP contract.
    """
    ev = rp.ResolventJumpV8(XMIN, XMAX, N_GRID, M_NODES)
    result = ev.jump(T, U0)

    # Compute ULP diagnostics.
    got = result.view(np.int64)
    want = GOLDEN_JUMP.view(np.int64)
    ulp_diff = int(np.max(np.abs(got - want)))

    print(
        f"\nG_BINDING_RESOLVENT_JUMP_PARITY sub-test 3 (PyO3 v3):\n"
        f"How called: rp.ResolventJumpV8 (PyO3 pyclass, GIL-released .jump())\n"
        f"jump: max ULP diff = {ulp_diff}  (expected 0)\n"
        f"jump[32] = {result[32]:.16e}  (golden = {GOLDEN_JUMP[32]:.16e})"
    )

    assert np.array_equal(result, GOLDEN_JUMP), (
        f"PyO3 jump is NOT byte-identical to core golden (max ULP diff = {ulp_diff})"
    )
