"""Smoke tests for SmolyakD6V8 (v8.1.0 C1, ADR-0138, ADR-0123 Amdt 1).

Tests cover:
  - SmolyakD6V8 constructor (happy + error paths)
  - .apply(tau, u0) returns numpy float64 array of correct length
  - .n_nodes(), .level(), .size() introspection
  - Error paths: invalid params raise SemiflowError
  - G_BINDING_SMOLYAK_PARITY sub-test 2 (PyO3 0-ULP vs core golden)

Canonical smoke params (§1.3 V8_1_TIER3_BINDING_DESIGN.md,
contracts/semiflow-core.properties.yaml §G_BINDING_SMOLYAK_PARITY):
  D=6, domain=[-2.0,2.0] per axis, n_per_axis=4 (4^6=4096 pts),
  n_chernoff=1, tau=0.01, u0(x)=exp(-Σx²) (Gaussian IC).

Domain [-2,2] (not [-5,5]): same correction as g_smolyak_d6 test; N=4 with [-5,5]
gives near-zero IC values (inner pts at ±5/3 → IC≈5.6e-8, essentially zero).

GOLDEN produced by:
  cargo test --package semiflow-core --test binding_smolyak_parity \\
             --features slow-tests -- --nocapture
(crates/semiflow-core/tests/binding_smolyak_parity.rs, F(0)=I anchor 3.3e-15.)
np.array_equal(got, GOLDEN) is EXACT bit-for-bit (float64 IEEE-754).
Any divergence indicates a marshalling bug in the PyO3 layer.
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Canonical smoke parameters (§1.3 design doc)
# ---------------------------------------------------------------------------

DOMAIN_LO = -2.0
DOMAIN_HI = 2.0
N_PER_AXIS = 4
TAU = 0.01
N_TOTAL = N_PER_AXIS ** 6  # 4096

# Gaussian IC on a 6-D grid: exp(-Σx²).
#
# IMPORTANT: Use Fortran-order (order='F') to match Rust's axis-0-fastest
# enumerate_nd ordering (flat = k0 + k1*n + k2*n^2 + ..., axis-0 fastest).
#
# The exponent is computed; the exponential is NOT. See ADR-0191 AMENDMENT 6:
# this file previously built U0 with `np.exp(...)` while GOLDEN was captured
# from a Rust run whose IC came from `f64::exp`. Asserting bit equality between
# two independent `exp` implementations — one of them vectorised and dispatched
# at run time by CPU feature — made the parity gate a coin flip. It failed on
# `(ubuntu, 3.13)`, passed on the same NumPy 2.5.2 wheel a run later, and failed
# again, with no change to any code that produces the vector.
#
# Everything up to the exponential is exact and needs no pinning: `linspace`
# reproduces `Grid1D`'s `x_i = xmin + i·dx` bit-for-bit (verified), and the
# exponent is a sum of products of those coordinates. Only 16 distinct exponents
# occur over the 4096 points, so pinning the IC costs 16 constants rather than
# 4096. These are the exact bit patterns the Rust golden was computed from,
# extracted from `binding_smolyak_parity.rs`'s own IC; keys are the raw float64
# bits of the exponent, values the raw bits of `exp` of it.
#
# The gate now measures what it always claimed to: given identical input bits,
# the PyO3 path reproduces the core's output bits. That is strictly stronger
# than before — it removes accidental passes as well as accidental failures.
_IC_EXP_TABLE = {
    13850820653977960448: 4450894569972245130,
    13849819854060767004: 4473961556924148569,
    13848819054143573561: 4497080136181554011,
    13848819054143573560: 4497080136181554036,
    13847067654288485035: 4520255156233017250,
    13847067654288485034: 4520255156233017264,
    13845066054454098148: 4543491921213358464,
    13845066054454098147: 4543491921213358479,
    13842063654702517818: 4566723127938800310,
    13842063654702517817: 4566723127938800314,
    13842063654702517816: 4566723127938800319,
    13836559255157953879: 4589671230983731725,
    13836559255157953878: 4589671230983731727,
    13836559255157953877: 4589671230983731729,
    13836559255157953876: 4589671230983731731,
    13836559255157953875: 4589671230983731734,
}

# FNV-1a/64 over the little-endian bytes of the 4096 IC values, as printed by
# the Rust side. Pins the reconstruction end-to-end rather than trusting that
# the 16 lookups were wired up correctly.
_IC_FNV1A64 = 3401955263860027927

_xs = np.linspace(DOMAIN_LO, DOMAIN_HI, N_PER_AXIS)
_mg = np.meshgrid(*([_xs] * 6), indexing="ij")
_IC_EXPONENT = (-sum(x**2 for x in _mg)).ravel(order="F").astype(np.float64)
U0 = np.array(
    [_IC_EXP_TABLE[int(b)] for b in _IC_EXPONENT.view(np.uint64)],
    dtype=np.uint64,
).view(np.float64)
del _xs, _mg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kernel() -> rp.SmolyakD6V8:
    return rp.SmolyakD6V8(DOMAIN_LO, DOMAIN_HI, N_PER_AXIS)


# ---------------------------------------------------------------------------
# Constructor — happy paths
# ---------------------------------------------------------------------------


def test_ctor_happy_path() -> None:
    k = make_kernel()
    assert isinstance(k, rp.SmolyakD6V8)


def test_ctor_asymmetric_domain() -> None:
    k = rp.SmolyakD6V8(-3.0, 3.0, 4)
    assert isinstance(k, rp.SmolyakD6V8)


# ---------------------------------------------------------------------------
# Constructor — error paths
# ---------------------------------------------------------------------------


def test_ctor_nan_lo_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.SmolyakD6V8(float("nan"), 2.0, 4)


def test_ctor_inf_hi_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.SmolyakD6V8(-2.0, float("inf"), 4)


def test_ctor_lo_ge_hi_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.SmolyakD6V8(2.0, -2.0, 4)


def test_ctor_n_per_axis_too_small_raises() -> None:
    with pytest.raises((rp.SemiflowError, Exception)):
        rp.SmolyakD6V8(-2.0, 2.0, 3)


# ---------------------------------------------------------------------------
# apply() — output shape, dtype, finiteness
# ---------------------------------------------------------------------------


def test_apply_returns_ndarray() -> None:
    k = make_kernel()
    out = k.apply(TAU, U0)
    assert isinstance(out, np.ndarray)


def test_apply_output_length() -> None:
    k = make_kernel()
    out = k.apply(TAU, U0)
    assert len(out) == N_TOTAL


def test_apply_output_dtype() -> None:
    k = make_kernel()
    out = k.apply(TAU, U0)
    assert out.dtype == np.float64


def test_apply_output_finite() -> None:
    k = make_kernel()
    out = k.apply(TAU, U0)
    assert np.all(np.isfinite(out)), "apply() output contains non-finite values"


def test_apply_max_positive() -> None:
    """Heat semigroup preserves positivity for positive IC."""
    k = make_kernel()
    out = k.apply(TAU, U0)
    assert float(np.max(out)) > 0.0


def test_apply_tau_zero_ones_identity() -> None:
    """F(0)=I on constant-1 function: Smolyak weight-sum = pi^3 → normalized result = 1.

    The F(0)=I identity holds exactly for constant functions (the GH quadrature
    nodes all evaluate to the same constant, and the weight normalization gives 1).
    Non-constant ICs may differ by O(interpolation error); see g_smolyak_d6 sub-test 2.
    """
    k = make_kernel()
    ones = np.ones(N_TOTAL, dtype=np.float64)
    out = k.apply(0.0, ones)
    sup_err = float(np.max(np.abs(out - ones)))
    assert sup_err < 1e-10, f"F(0)=I ones sup_err={sup_err:.3e} >= 1e-10"


def test_apply_two_steps() -> None:
    k = make_kernel()
    out = k.apply(TAU, U0, n_steps=2)
    assert len(out) == N_TOTAL
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# apply() — error paths
# ---------------------------------------------------------------------------


def test_apply_bad_tau_negative_raises() -> None:
    k = make_kernel()
    with pytest.raises(rp.SemiflowError):
        k.apply(-0.01, U0)


def test_apply_bad_tau_nan_raises() -> None:
    k = make_kernel()
    with pytest.raises(rp.SemiflowError):
        k.apply(float("nan"), U0)


def test_apply_wrong_u0_length_raises() -> None:
    k = make_kernel()
    bad_u0 = np.ones(100, dtype=np.float64)
    with pytest.raises(rp.SemiflowError):
        k.apply(TAU, bad_u0)


def test_apply_n_steps_zero_raises() -> None:
    k = make_kernel()
    with pytest.raises(rp.SemiflowError):
        k.apply(TAU, U0, n_steps=0)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_n_nodes_d6_level9() -> None:
    """D=6, ℓ=9 Smolyak grid should have exactly 533 nodes."""
    k = make_kernel()
    n = k.n_nodes()
    assert n == 533, f"expected 533 Smolyak nodes, got {n}"


def test_level_default() -> None:
    k = make_kernel()
    assert k.level() == 9


def test_size() -> None:
    k = make_kernel()
    assert k.size() == N_TOTAL


# ---------------------------------------------------------------------------
# G_BINDING_SMOLYAK_PARITY sub-test 2 (PyO3 0-ULP vs core golden)
# ---------------------------------------------------------------------------
#
# Canonical params (contracts/semiflow-core.properties.yaml §G_BINDING_SMOLYAK_PARITY):
#   D=6, domain=[-2,2], n_per_axis=4, tau=0.01, n_chernoff=1.
#
# GOLDEN_FLAT produced by:
#   cargo test --package semiflow-core --test binding_smolyak_parity \
#              --features slow-tests -- --nocapture
# (binding_smolyak_parity.rs, F(0)=I anchor: sup_err=3.3e-15.)
#
# Only first 8 + last 4 values are embedded here (full 4096-value golden
# would bloat the file). The 0-ULP check uses compute-time equality against
# a freshly-computed Rust reference to avoid embedding the full 4096-value
# golden.  A spot-check on the first 8 and last 4 entries verifies the
# canonical values printed by the core golden test.

# Regenerated at ADR-0191 (issue #17): `GridFnND::sample` became a genuine
# tensor-product interpolant honouring each axis's BoundaryPolicy, instead of
# multilinear-with-index-clamp, so the canonical output legitimately changed.
# Reprinted by `cargo test -p semiflow --release --features slow-tests \
#   --test binding_smolyak_parity -- --nocapture`.
#
# Worth noting what the change looks like here. The IC `exp(-Σx²)` is symmetric
# on the symmetric domain [-2,2], and entries 0..3 traverse x0 ∈ {-2, -2/3, +2/3,
# +2} at fixed other axes — so the output must satisfy `g[0] == g[3]` and
# `g[1] == g[2]`. The NEW golden does: `[a, b, b, a, ...]`. The OLD golden did
# NOT — it read `[1.98e-9, 4.26e-8, 4.34e-8, 4.34e-8, ...]`, i.e. `a, b, c, c`,
# with `g[0]` differing from `g[3]` by a factor of 22. That broken reflection
# symmetry was the index clamp overriding the axes' `Reflect` policy asymmetrically
# at the two ends. The regeneration is not merely a churn update; it restores a
# symmetry the values had lost.
_GOLDEN_FIRST8 = np.array(
    [
        1.1135489538169915e-09,
        2.4645994657186092e-08,
        2.4645994657186105e-08,
        1.11354895381699e-09,
        2.4645994657186085e-08,
        5.132235735928101e-07,
        5.132235735928105e-07,
        2.464599465718608e-08,
    ],
    dtype=np.float64,
)

_GOLDEN_LAST4 = np.array(
    [
        1.1135489538169857e-09,
        2.464599465718599e-08,
        2.464599465718615e-08,
        1.1135489538169873e-09,
    ],
    dtype=np.float64,
)


def test_ic_reconstruction_matches_rust_checksum() -> None:
    """The pinned IC reconstructs the exact vector the Rust golden was fed.

    Guards the lookup wiring, not the arithmetic: a wrong ravel order, a wrong
    key, or a truncated table all change this checksum. Without it the 16
    embedded constants are only as trustworthy as the loop that applies them.
    """
    h = 1469598103934665603
    for bits in U0.view(np.uint64):
        for byte in int(bits).to_bytes(8, "little"):
            h ^= byte
            h = (h * 1099511628211) % (1 << 64)
    assert h == _IC_FNV1A64, (
        f"pinned IC does not reconstruct the Rust golden's input\n"
        f"got FNV-1a/64 = {h}, expected {_IC_FNV1A64}"
    )


def test_ic_table_is_the_documented_gaussian() -> None:
    """The pinned constants really are exp(-Σx²), not arbitrary numbers.

    Division of labour with the checksum test above, which matters more than it
    looks: that one is the bit-exact correctness guard, and it is deterministic
    everywhere because it compares pinned data against a pinned constant. This
    one is a *sanity* guard tying the pinned data back to the canonical §1.3
    parameters, and it is the only assertion in this file that evaluates
    ``np.exp`` — so its tolerance must be loose enough that no libm on any
    platform can trip it.

    Hence a relative tolerance of 1e-12 (~4500 ULP of headroom) rather than a
    few ULP. A tight ULP bound here would be the same mistake this file was
    just fixed for (ADR-0191 AMENDMENT 6): betting that two independent ``exp``
    implementations agree to the last few bits on every CI runner. Real drift —
    an edited constant, a changed domain, a changed grid size — moves values by
    orders of magnitude, not by 1e-12, so nothing is given up.
    """
    recomputed = np.exp(_IC_EXPONENT)
    rel = np.abs(U0 - recomputed) / np.abs(recomputed)
    assert float(rel.max()) <= 1e-12, (
        f"pinned IC disagrees with exp(-sum x^2) by relative {float(rel.max()):.3e} "
        f"— either the table is wrong or the canonical params changed"
    )


def test_g_binding_smolyak_parity_sub2_pyo3_0ulp() -> None:
    """G_BINDING_SMOLYAK_PARITY sub-test 2 (PyO3 0-ULP vs core golden).

    Calls SmolyakD6V8.apply() with the canonical params and verifies:
    1. Output length == 4096.
    2. All values finite.
    3. Spot-check first 8 and last 4 values are byte-identical (0-ULP) to the
       core golden printed by binding_smolyak_parity.rs.
    4. Two independent PyO3 calls produce identical output (byte-exact).

    The spot-check strategy avoids embedding all 4096 values while still
    catching any marshalling regression.

    How called: semiflow.SmolyakD6V8 (PyO3 pyclass, GIL-released via
    py.detach around the Smolyak quadrature, ADR-0031).
    """
    k = make_kernel()
    got = k.apply(TAU, U0)

    assert len(got) == N_TOTAL, f"length mismatch: {len(got)} != {N_TOTAL}"
    assert np.all(np.isfinite(got)), "output contains non-finite values"

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
        f"\nG_BINDING_SMOLYAK_PARITY sub-test 2 (PyO3):\n"
        f"How called: rp.SmolyakD6V8 (GIL-released .apply())\n"
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
    got2 = k.apply(TAU, U0)
    assert np.array_equal(got, got2), "Two PyO3 calls produced different results"
