"""Smoke tests for ComplexTripleJumpV8 (v8.1.0 F4, ADR-0138, ADR-0136 Amdt 2).

Tests cover:
  - ComplexTripleJumpV8 constructor (happy + error paths)
  - .apply_real(tau, u0) returns numpy float64 array of correct length
  - .verify_gamma_star() staticmethod
  - .size() introspection
  - Error paths: invalid params raise SemiflowError
  - G_BINDING_CARNOT_CPLX3_PARITY sub-test 2 (PyO3 0-ULP vs core golden)

Canonical smoke params (§1.4 V8_1_TIER3_BINDING_DESIGN.md,
contracts/semiflow-core.properties.yaml §G_BINDING_CARNOT_CPLX3_PARITY):
  D=5, domain=[-1.5,1.5] per axis, n_per_axis=5 (5^5=3125 pts),
  tau=0.02, u0(x)=exp(-(x0²+x1²)/2) (Gaussian IC).

GOLDEN produced by:
  cargo test --package semiflow-core --test binding_carnot_cplx3_parity \\
             --features slow-tests -- --nocapture
(crates/semiflow-core/tests/binding_carnot_cplx3_parity.rs, γ⋆ anchor PASS.)
np.array_equal(got, GOLDEN) is EXACT bit-for-bit (float64 IEEE-754).
Any divergence indicates a marshalling bug in the PyO3 layer.
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Canonical smoke parameters (§1.4 design doc)
# ---------------------------------------------------------------------------

DOMAIN_LO = -1.5
DOMAIN_HI = 1.5
N_PER_AXIS = 5
TAU = 0.02
N_TOTAL = N_PER_AXIS ** 5  # 3125

# Gaussian IC: exp(-(x0²+x1²)/2) on a 5-D grid.
# IMPORTANT: Use Fortran-order (order='F') to match Rust's axis-0-fastest
# enumerate_nd ordering (flat = k0 + k1*n + k2*n^2 + ..., axis-0 fastest).
_xs = np.linspace(DOMAIN_LO, DOMAIN_HI, N_PER_AXIS)
_mg = np.meshgrid(*([_xs] * 5), indexing="ij")
U0 = np.exp(-(_mg[0] ** 2 + _mg[1] ** 2) * 0.5).ravel(order="F").astype(np.float64)
del _xs, _mg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kernel() -> rp.ComplexTripleJumpV8:
    return rp.ComplexTripleJumpV8(DOMAIN_LO, DOMAIN_HI, N_PER_AXIS)


# ---------------------------------------------------------------------------
# Constructor — happy paths
# ---------------------------------------------------------------------------


def test_ctor_happy_path() -> None:
    k = make_kernel()
    assert isinstance(k, rp.ComplexTripleJumpV8)


def test_ctor_asymmetric_domain() -> None:
    k = rp.ComplexTripleJumpV8(-2.0, 2.0, 4)
    assert isinstance(k, rp.ComplexTripleJumpV8)


# ---------------------------------------------------------------------------
# Constructor — error paths
# ---------------------------------------------------------------------------


def test_ctor_nan_lo_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.ComplexTripleJumpV8(float("nan"), 1.5, 5)


def test_ctor_inf_hi_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.ComplexTripleJumpV8(-1.5, float("inf"), 5)


def test_ctor_lo_ge_hi_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.ComplexTripleJumpV8(1.5, -1.5, 5)


def test_ctor_n_per_axis_too_small_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.ComplexTripleJumpV8(-1.5, 1.5, 3)


# ---------------------------------------------------------------------------
# verify_gamma_star — staticmethod
# ---------------------------------------------------------------------------


def test_verify_gamma_star_true() -> None:
    """γ⋆ must satisfy the cubic residual check."""
    assert rp.ComplexTripleJumpV8.verify_gamma_star() is True


# ---------------------------------------------------------------------------
# apply_real() — output shape, dtype, finiteness
# ---------------------------------------------------------------------------


def test_apply_real_returns_ndarray() -> None:
    k = make_kernel()
    out = k.apply_real(TAU, U0)
    assert isinstance(out, np.ndarray)


def test_apply_real_output_length() -> None:
    k = make_kernel()
    out = k.apply_real(TAU, U0)
    assert len(out) == N_TOTAL


def test_apply_real_output_dtype() -> None:
    k = make_kernel()
    out = k.apply_real(TAU, U0)
    assert out.dtype == np.float64


def test_apply_real_output_finite() -> None:
    k = make_kernel()
    out = k.apply_real(TAU, U0)
    assert np.all(np.isfinite(out)), "apply_real() output contains non-finite values"


def test_apply_real_max_positive() -> None:
    """Real projection of a heat flow is positive for positive IC."""
    k = make_kernel()
    out = k.apply_real(TAU, U0)
    assert float(np.max(out)) > 0.0


def test_apply_real_tau_zero() -> None:
    """tau=0 is allowed (returns approximately u0 for small tau)."""
    k = make_kernel()
    out = k.apply_real(0.0, U0)
    assert len(out) == N_TOTAL
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# apply_real() — error paths
# ---------------------------------------------------------------------------


def test_apply_real_bad_tau_negative_raises() -> None:
    k = make_kernel()
    with pytest.raises(rp.SemiflowError):
        k.apply_real(-0.01, U0)


def test_apply_real_bad_tau_nan_raises() -> None:
    k = make_kernel()
    with pytest.raises(rp.SemiflowError):
        k.apply_real(float("nan"), U0)


def test_apply_real_wrong_u0_length_raises() -> None:
    k = make_kernel()
    bad_u0 = np.ones(100, dtype=np.float64)
    with pytest.raises(rp.SemiflowError):
        k.apply_real(TAU, bad_u0)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_size() -> None:
    k = make_kernel()
    assert k.size() == N_TOTAL


# ---------------------------------------------------------------------------
# G_BINDING_CARNOT_CPLX3_PARITY sub-test 2 (PyO3 0-ULP vs core golden)
# ---------------------------------------------------------------------------
#
# Canonical params (contracts/semiflow-core.properties.yaml
# §G_BINDING_CARNOT_CPLX3_PARITY):
#   D=5, domain=[-1.5,1.5], n_per_axis=5, tau=0.02.
#
# GOLDEN_FIRST8 / GOLDEN_LAST4 produced by:
#   cargo test --package semiflow-core --test binding_carnot_cplx3_parity \
#              --features slow-tests -- --nocapture
# (binding_carnot_cplx3_parity.rs, γ⋆ cubic residual PASS.)
#
# Spot-check strategy (first 8 + last 4 of 3125-value output) to avoid
# bloating the test file with the full golden.

_GOLDEN_FIRST8 = np.array(
    [
        0.010387378510976555,
        0.021655022523433723,
        0.028161470148009148,
        0.021655022523433733,
        0.010387378510976566,
        0.02230484212130334,
        0.046499880467253486,
        0.06047119066972382,
    ],
    dtype=np.float64,
)

_GOLDEN_LAST4 = np.array(
    [
        0.021655022523433723,
        0.028161470148009158,
        0.021655022523433723,
        0.01038737851097657,
    ],
    dtype=np.float64,
)


def test_g_binding_carnot_cplx3_parity_sub2_pyo3_0ulp() -> None:
    """G_BINDING_CARNOT_CPLX3_PARITY sub-test 2 (PyO3 0-ULP vs core golden).

    Calls ComplexTripleJumpV8.apply_real() with the canonical params and verifies:
    1. Output length == 3125.
    2. All values finite and max > 0.
    3. verify_gamma_star() == True (γ⋆ cubic residual, INDEPENDENT anchor).
    4. Spot-check first 8 and last 4 values are byte-identical (0-ULP) to the
       core golden printed by binding_carnot_cplx3_parity.rs.
    5. Two independent PyO3 calls produce identical output (byte-exact).

    How called: semiflow.ComplexTripleJumpV8 (PyO3 pyclass, GIL-released via
    py.detach around the triple complex Strang sweep, ADR-0031).
    """
    # Anchor 1: γ⋆ residual (INDEPENDENT of apply_real).
    assert rp.ComplexTripleJumpV8.verify_gamma_star(), "γ⋆ cubic residual check FAILED"

    k = make_kernel()
    got = k.apply_real(TAU, U0)

    assert len(got) == N_TOTAL, f"length mismatch: {len(got)} != {N_TOTAL}"
    assert np.all(np.isfinite(got)), "output contains non-finite values"
    assert float(np.max(got)) > 0.0, "output max should be > 0"

    # Spot-check first 8 entries.
    first8_ulp = int(
        np.max(
            np.abs(got[:8].view(np.int64) - _GOLDEN_FIRST8.view(np.int64))
        )
    )
    last4_ulp = int(
        np.max(
            np.abs(got[-4:].view(np.int64) - _GOLDEN_LAST4.view(np.int64))
        )
    )

    print(
        f"\nG_BINDING_CARNOT_CPLX3_PARITY sub-test 2 (PyO3):\n"
        f"How called: rp.ComplexTripleJumpV8 (GIL-released .apply_real())\n"
        f"first-8 max ULP diff = {first8_ulp}  (expected 0)\n"
        f"last-4  max ULP diff = {last4_ulp}   (expected 0)\n"
        f"got[0..8] = {got[:8]}\n"
        f"got[-4:]  = {got[-4:]}"
    )

    assert np.array_equal(got[:8], _GOLDEN_FIRST8), (
        f"PyO3 first-8 NOT byte-identical to core golden "
        f"(max ULP diff = {first8_ulp})\n"
        f"got:    {got[:8]}\n"
        f"golden: {_GOLDEN_FIRST8}"
    )
    assert np.array_equal(got[-4:], _GOLDEN_LAST4), (
        f"PyO3 last-4 NOT byte-identical to core golden "
        f"(max ULP diff = {last4_ulp})\n"
        f"got:    {got[-4:]}\n"
        f"golden: {_GOLDEN_LAST4}"
    )

    # Idempotency: two identical calls must return byte-identical results.
    got2 = k.apply_real(TAU, U0)
    assert np.array_equal(got, got2), "Two PyO3 calls produced different results"
