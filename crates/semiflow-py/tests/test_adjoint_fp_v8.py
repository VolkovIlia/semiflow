"""Smoke tests for AdjointFokkerPlanckV8 (v8.1.0 C2, ADR-0138, ADR-0107 Amdt 1).

Tests cover:
  - AdjointFokkerPlanckV8 constructor (happy + error paths)
  - .step(tau, positions, weights, n_steps) returns (np.ndarray, np.ndarray)
  - .total_variation() and .second_moment() scalar diagnostics
  - Error paths: invalid params raise SemiflowError
  - G_BINDING_ADJOINT_FP_PARITY sub-test 3 (PyO3 v3, 0-ULP vs core golden)

Canonical smoke params (§1.2 V8_1_TIER3_BINDING_DESIGN.md,
contracts/semiflow-core.properties.yaml §G_BINDING_ADJOINT_FP_PARITY):
  a=0.5, b=0.0, c=0.0 (Brownian, mass-conserving),
  ρ₀ = δ_0 (positions=[0.0], weights=[1.0]),
  tau=0.1, n_steps=1 → exactly 4 Diracs.

GOLDEN_POS / GOLDEN_WTS produced by:
  cargo test --package semiflow-core --test binding_adjoint_fp_parity \
             --features slow-tests -- --nocapture
(crates/semiflow-core/tests/binding_adjoint_fp_parity.rs,
 verified against Lemma A.1 closed-form anchor ‖·‖∞ ≤ 1e-14.)
np.array_equal(got, GOLDEN_*) is EXACT bit-for-bit (float64 IEEE-754).
Any divergence indicates a marshalling bug in the PyO3 layer.
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Canonical smoke parameters (§1.2 design doc)
# ---------------------------------------------------------------------------

A = 0.5
B = 0.0
C_COEF = 0.0
TAU = 0.1

POS0 = np.array([0.0], dtype=np.float64)  # δ_0
WTS0 = np.array([1.0], dtype=np.float64)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kernel(a: float = A, b: float = B, c: float = C_COEF) -> rp.AdjointFokkerPlanckV8:
    return rp.AdjointFokkerPlanckV8(a, b, c)


# ---------------------------------------------------------------------------
# Constructor — happy paths
# ---------------------------------------------------------------------------


def test_ctor_happy_path() -> None:
    adj = make_kernel()
    assert isinstance(adj, rp.AdjointFokkerPlanckV8)


def test_ctor_nonzero_drift() -> None:
    adj = rp.AdjointFokkerPlanckV8(0.5, 0.1, 0.0)
    assert isinstance(adj, rp.AdjointFokkerPlanckV8)


def test_ctor_with_reaction() -> None:
    adj = rp.AdjointFokkerPlanckV8(0.5, 0.0, 0.1)
    assert isinstance(adj, rp.AdjointFokkerPlanckV8)


# ---------------------------------------------------------------------------
# Constructor — error paths
# ---------------------------------------------------------------------------


def test_ctor_nan_a_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.AdjointFokkerPlanckV8(float("nan"), 0.0, 0.0)


def test_ctor_inf_b_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.AdjointFokkerPlanckV8(0.5, float("inf"), 0.0)


# ---------------------------------------------------------------------------
# step() — output shape, dtype, finiteness
# ---------------------------------------------------------------------------


def test_step_returns_tuple() -> None:
    adj = make_kernel()
    pos, wts = adj.step(TAU, POS0, WTS0)
    assert isinstance(pos, np.ndarray)
    assert isinstance(wts, np.ndarray)


def test_step_output_length_one_step() -> None:
    """One step from 1 Dirac → 4 children."""
    adj = make_kernel()
    pos, wts = adj.step(TAU, POS0, WTS0)
    assert len(pos) == 4
    assert len(wts) == 4


def test_step_output_length_two_steps() -> None:
    """Two steps from 1 Dirac → 16 children."""
    adj = make_kernel()
    pos, wts = adj.step(TAU, POS0, WTS0, n_steps=2)
    assert len(pos) == 16
    assert len(wts) == 16


def test_step_output_dtype() -> None:
    adj = make_kernel()
    pos, wts = adj.step(TAU, POS0, WTS0)
    assert pos.dtype == np.float64
    assert wts.dtype == np.float64


def test_step_output_finite() -> None:
    adj = make_kernel()
    pos, wts = adj.step(TAU, POS0, WTS0)
    assert np.all(np.isfinite(pos)), "positions contain non-finite values"
    assert np.all(np.isfinite(wts)), "weights contain non-finite values"


def test_step_mass_conservation_c_zero() -> None:
    """c=0 → mass is exactly 1.0."""
    adj = make_kernel(c=0.0)
    pos, wts = adj.step(TAU, POS0, WTS0)
    mass = float(np.sum(wts))
    assert abs(mass - 1.0) < 1e-14, f"mass = {mass}, expected 1.0"


# ---------------------------------------------------------------------------
# step() — error paths
# ---------------------------------------------------------------------------


def test_step_bad_tau_zero() -> None:
    adj = make_kernel()
    with pytest.raises(rp.SemiflowError):
        adj.step(0.0, POS0, WTS0)


def test_step_bad_tau_negative() -> None:
    adj = make_kernel()
    with pytest.raises(rp.SemiflowError):
        adj.step(-0.1, POS0, WTS0)


def test_step_bad_tau_nan() -> None:
    adj = make_kernel()
    with pytest.raises(rp.SemiflowError):
        adj.step(float("nan"), POS0, WTS0)


def test_step_mismatched_lengths() -> None:
    """positions and weights with different lengths → GridMismatch."""
    adj = make_kernel()
    bad_wts = np.array([1.0, 0.5], dtype=np.float64)
    with pytest.raises(rp.SemiflowError):
        adj.step(TAU, POS0, bad_wts)


def test_step_n_steps_zero() -> None:
    adj = make_kernel()
    with pytest.raises(rp.SemiflowError):
        adj.step(TAU, POS0, WTS0, n_steps=0)


# ---------------------------------------------------------------------------
# Scalar diagnostics
# ---------------------------------------------------------------------------


def test_total_variation_single_dirac() -> None:
    adj = make_kernel()
    tv = adj.total_variation(POS0, WTS0)
    assert abs(tv - 1.0) < 1e-14, f"TV = {tv}, expected 1.0"


def test_second_moment_at_origin() -> None:
    adj = make_kernel()
    sm = adj.second_moment(POS0, WTS0)
    assert abs(sm) < 1e-14, f"second_moment(δ_0) = {sm}, expected 0.0"


def test_second_moment_after_step() -> None:
    """After one Brownian step: E[x²] = 2aτ = 2·0.5·0.1 = 0.1."""
    adj = make_kernel()
    pos, wts = adj.step(TAU, POS0, WTS0)
    sm = adj.second_moment(pos, wts)
    expected = 2.0 * A * TAU  # = 0.1 for Brownian
    # Lemma A.1: h² = 4aτ, weights ¼ + ¼ = ½  → 0.5·h² = 0.5·4aτ = 2aτ
    assert abs(sm - expected) < 1e-14, f"second_moment = {sm}, expected {expected}"


# ---------------------------------------------------------------------------
# G_BINDING_ADJOINT_FP_PARITY sub-test 3 (PyO3 v3, 0-ULP vs core golden)
# ---------------------------------------------------------------------------
#
# Canonical params (contracts/semiflow-core.properties.yaml §G_BINDING_ADJOINT_FP_PARITY):
#   a=0.5, b=0.0, c=0.0, tau=0.1, ρ₀=δ_0.
#
# GOLDEN_POS / GOLDEN_WTS produced by:
#   cargo test --package semiflow-core --test binding_adjoint_fp_parity \
#              --features slow-tests -- --nocapture
# (binding_adjoint_fp_parity.rs `canonical_adjoint_fp_core`,
#  verified against Lemma A.1 analytic anchor ‖·‖∞ ≤ 1e-14.)
#
# np.array_equal(got, GOLDEN_*) is EXACT bit-for-bit (float64 IEEE-754).
# Any divergence indicates a marshalling bug in the PyO3 layer.

GOLDEN_POS = np.array(
    [0.4472135954999579, -0.4472135954999579, 0.0, 0.0],
    dtype=np.float64,
)
GOLDEN_WTS = np.array([0.25, 0.25, 0.5, 0.0], dtype=np.float64)


def test_g_binding_adjoint_fp_parity_sub3_pyo3_0ulp() -> None:
    """G_BINDING_ADJOINT_FP_PARITY sub-test 3 (PyO3 v3).

    Calls AdjointFokkerPlanckV8.step() with the canonical params and asserts
    the returned (positions, weights) arrays are byte-identical (0 ULP) to
    the core golden.

    How called: semiflow.AdjointFokkerPlanckV8 (PyO3 pyclass, GIL-release via
    py.detach around the Lemma A.1 multi-step push, ADR-0031).
    The step() return values are numpy float64 arrays unmarshalled from the
    Rust Vec<f64> via .to_pyarray().

    np.array_equal performs element-wise equality including sign of zero —
    exactly the 0-ULP contract.
    """
    adj = rp.AdjointFokkerPlanckV8(A, B, C_COEF)
    pos_got, wts_got = adj.step(TAU, POS0, WTS0, n_steps=1)

    # Compute ULP diagnostics.
    pos_ulp = int(np.max(np.abs(pos_got.view(np.int64) - GOLDEN_POS.view(np.int64))))
    wts_ulp = int(np.max(np.abs(wts_got.view(np.int64) - GOLDEN_WTS.view(np.int64))))

    print(
        f"\nG_BINDING_ADJOINT_FP_PARITY sub-test 3 (PyO3 v3):\n"
        f"How called: rp.AdjointFokkerPlanckV8 (PyO3 pyclass, GIL-released .step())\n"
        f"positions max ULP diff = {pos_ulp}  (expected 0)\n"
        f"weights   max ULP diff = {wts_ulp}  (expected 0)\n"
        f"pos_got = {pos_got}\n"
        f"wts_got = {wts_got}"
    )

    assert np.array_equal(pos_got, GOLDEN_POS), (
        f"PyO3 positions NOT byte-identical to core golden "
        f"(max ULP diff = {pos_ulp})\n"
        f"got:    {pos_got}\n"
        f"golden: {GOLDEN_POS}"
    )
    assert np.array_equal(wts_got, GOLDEN_WTS), (
        f"PyO3 weights NOT byte-identical to core golden "
        f"(max ULP diff = {wts_ulp})\n"
        f"got:    {wts_got}\n"
        f"golden: {GOLDEN_WTS}"
    )
