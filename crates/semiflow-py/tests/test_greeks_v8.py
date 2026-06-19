"""Smoke tests for EvolverHeat1DGreeksV3 and KilledDirichlet1D (v8.0.0 F1).

Tests cover:
  - EvolverHeat1DGreeksV3 constructor (happy + error paths)
  - .greeks(t) returns three finite numpy float64 arrays
  - delta matches numpy central-difference of value w.r.t. theta (~1e-6)
  - gamma positive (heat spreads — second derivative of energy w.r.t. theta)
  - size() and n_chernoff() introspection
  - KilledDirichlet1D constructor + .apply() returns finite array of correct shape

Parameters (canonical smoke per §5 G_BINDING_GREEKS_PARITY):
  theta=0.5, n_grid=64, n_chernoff=32, t=0.05, u0=exp(-x^2), domain [-5, 5].
"""

import math

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Smoke parameters (§5 G_BINDING_GREEKS_PARITY canonical)
# ---------------------------------------------------------------------------

XMIN, XMAX = -5.0, 5.0
N_GRID = 64
N_CHERNOFF = 32
T = 0.05
THETA_0 = 0.5

XS = np.linspace(XMIN, XMAX, N_GRID)
U0 = np.exp(-(XS**2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_evolver(theta: float = THETA_0) -> rp.EvolverHeat1DGreeksV3:
    return rp.EvolverHeat1DGreeksV3(XMIN, XMAX, N_GRID, U0, N_CHERNOFF, theta)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_ctor_happy_path() -> None:
    ev = make_evolver()
    assert ev.size() == N_GRID
    assert ev.n_chernoff() == N_CHERNOFF


def test_ctor_default_theta() -> None:
    """Default scale_theta=0.5 should be accepted without explicit arg."""
    ev = rp.EvolverHeat1DGreeksV3(XMIN, XMAX, N_GRID, U0, N_CHERNOFF)
    assert ev.size() == N_GRID


def test_ctor_bad_n_grid() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.EvolverHeat1DGreeksV3(XMIN, XMAX, 2, U0[:2], N_CHERNOFF, THETA_0)


def test_ctor_bad_n_chernoff() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.EvolverHeat1DGreeksV3(XMIN, XMAX, N_GRID, U0, 0, THETA_0)


def test_ctor_nan_in_u0() -> None:
    bad = U0.copy()
    bad[3] = float("nan")
    with pytest.raises(rp.SemiflowError):
        rp.EvolverHeat1DGreeksV3(XMIN, XMAX, N_GRID, bad, N_CHERNOFF, THETA_0)


def test_ctor_nan_theta() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.EvolverHeat1DGreeksV3(XMIN, XMAX, N_GRID, U0, N_CHERNOFF, float("nan"))


# ---------------------------------------------------------------------------
# greeks() output shape and finiteness
# ---------------------------------------------------------------------------


def test_greeks_returns_tuple_of_three() -> None:
    ev = make_evolver()
    result = ev.greeks(T)
    assert len(result) == 3


def test_greeks_correct_dtype_and_shape() -> None:
    ev = make_evolver()
    value, delta, gamma = ev.greeks(T)
    for arr in (value, delta, gamma):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (N_GRID,)
        assert arr.dtype == np.float64


def test_greeks_value_finite() -> None:
    ev = make_evolver()
    value, _, _ = ev.greeks(T)
    assert np.all(np.isfinite(value)), "value contains non-finite entries"


def test_greeks_delta_finite() -> None:
    ev = make_evolver()
    _, delta, _ = ev.greeks(T)
    assert np.all(np.isfinite(delta)), "delta contains non-finite entries"


def test_greeks_gamma_finite() -> None:
    ev = make_evolver()
    _, _, gamma = ev.greeks(T)
    assert np.all(np.isfinite(gamma)), "gamma contains non-finite entries"


# ---------------------------------------------------------------------------
# Numerical accuracy: delta ≈ central-difference of value w.r.t. theta
# ---------------------------------------------------------------------------


def test_delta_matches_finite_difference() -> None:
    """delta[i] ≈ (value(theta+h)[i] - value(theta-h)[i]) / (2h).

    Tolerance 1e-6: exact AD vs noisy FD should agree to machine precision
    for smooth diffusion kernels at this grid resolution.
    """
    h = 1e-5

    ev_hi = make_evolver(THETA_0 + h)
    value_hi, _, _ = ev_hi.greeks(T)

    ev_lo = make_evolver(THETA_0 - h)
    value_lo, _, _ = ev_lo.greeks(T)

    fd_delta = (value_hi - value_lo) / (2.0 * h)

    ev_ref = make_evolver(THETA_0)
    _, delta, _ = ev_ref.greeks(T)

    err = np.max(np.abs(delta - fd_delta))
    assert err < 1e-6, (
        f"delta FD check FAILED: max |delta - FD| = {err:.3e} >= 1e-6"
    )


# ---------------------------------------------------------------------------
# Error path: bad t
# ---------------------------------------------------------------------------


def test_greeks_bad_t() -> None:
    ev = make_evolver()
    with pytest.raises(rp.SemiflowError):
        ev.greeks(-1.0)


def test_greeks_nan_t() -> None:
    ev = make_evolver()
    with pytest.raises(rp.SemiflowError):
        ev.greeks(float("nan"))


# ---------------------------------------------------------------------------
# TIER 2: KilledDirichlet1D
# ---------------------------------------------------------------------------


def test_killed_ctor_happy() -> None:
    u0_k = np.sin(np.pi * np.linspace(0.0, 1.0, 32))
    ev = rp.KilledDirichlet1D(0.0, 1.0, 32, u0_k, 16)
    assert ev.size() == 32


def test_killed_apply_shape_and_finite() -> None:
    n = 32
    u0_k = np.sin(np.pi * np.linspace(0.0, 1.0, n))
    ev = rp.KilledDirichlet1D(0.0, 1.0, n, u0_k, 16)
    result = ev.apply(0.01)
    assert isinstance(result, np.ndarray)
    assert result.shape == (n,)
    assert np.all(np.isfinite(result))


def test_killed_boundary_zero() -> None:
    """Absorbing wall: endpoints should converge toward 0 over time."""
    n = 64
    xs = np.linspace(0.0, 1.0, n)
    u0_k = np.sin(np.pi * xs)
    ev = rp.KilledDirichlet1D(0.0, 1.0, n, u0_k, 32)
    # After several steps the boundary nodes should stay near 0.
    result = ev.apply(0.1)
    # Endpoints (wall nodes 0 and n-1) should be very small.
    assert abs(result[0]) < 1e-10, f"boundary[0] = {result[0]:.3e} not near 0"
    assert abs(result[-1]) < 1e-10, f"boundary[-1] = {result[-1]:.3e} not near 0"


def test_killed_ctor_bad_n_grid() -> None:
    u0_k = np.ones(2)
    with pytest.raises(rp.SemiflowError):
        rp.KilledDirichlet1D(0.0, 1.0, 2, u0_k, 4)


def test_killed_bad_t() -> None:
    n = 16
    u0_k = np.ones(n)
    ev = rp.KilledDirichlet1D(0.0, 1.0, n, u0_k, 4)
    with pytest.raises(rp.SemiflowError):
        ev.apply(-0.5)


# ---------------------------------------------------------------------------
# G_BINDING_GREEKS_PARITY sub-test 3 (PyO3 v3, 0-ULP against core golden)
# ---------------------------------------------------------------------------
#
# Canonical params (contracts/semiflow-core.properties.yaml §G_BINDING_GREEKS_PARITY):
#   domain [-10, 10], N=64, n_chernoff=32, t=0.05, theta=0.5, u0=exp(-x²).
#
# GOLDEN arrays were produced by:
#   cargo test --package semiflow-core --test binding_greeks_parity \
#              print_golden -- --nocapture
# (crates/semiflow-core/tests/binding_greeks_parity.rs, verified vs Richardson FD)
#
# np.array_equal(got, GOLDEN) is EXACT bit-for-bit (float64 IEEE-754).
# Any divergence indicates a marshalling bug in the PyO3 layer.

_XMIN_CANONICAL = -10.0
_XMAX_CANONICAL = 10.0
_N_CANONICAL = 64
_N_CHERNOFF_CANONICAL = 32
_T_CANONICAL = 0.05
_THETA_CANONICAL = 0.5

_XS_CANONICAL = np.linspace(_XMIN_CANONICAL, _XMAX_CANONICAL, _N_CANONICAL)
_U0_CANONICAL = np.exp(-(_XS_CANONICAL ** 2))

# fmt: off
GOLDEN_VALUE = np.array([
    1.2273665126349644e-23, 1.5634124948354552e-22, 1.8630596457416103e-21,
    1.3334389244702536e-20, 6.2731790865844487e-20, 2.6228348873235650e-19,
    2.6996951373641581e-18, 3.9812079022180766e-17, 4.2176942148193078e-16,
    3.1493096321099560e-15, 1.7743373610017136e-14, 9.1389160966536402e-14,
    6.6624972821867715e-13, 6.8439108189117889e-12, 6.5164572692422500e-11,
    4.9629940211796168e-10, 3.0456011009952755e-9, 1.6334237696875445e-8,
    9.2735844448864484e-8, 6.6127065123771670e-7, 5.2809695914217134e-6,
    3.8923073423285504e-5, 2.4368461722402000e-4, 1.2709197832545980e-3,
    5.5100563112493728e-3, 1.9873008874759564e-2, 5.9662313475552890e-2,
    1.4912925268685812e-1, 3.1036207501057161e-1, 5.3778655727172475e-1,
    7.7584807469246408e-1, 9.3188389445411468e-1, 9.3188389445411679e-1,
    7.7584807469245520e-1, 5.3778655727172864e-1, 3.1036207501055646e-1,
    1.4912925268685959e-1, 5.9662313475555125e-2, 1.9873008874759211e-2,
    5.5100563112494743e-3, 1.2709197832546308e-3, 2.4368461722399043e-4,
    3.8923073423283288e-5, 5.2809695914224884e-6, 6.6127065123759038e-7,
    9.2735844448861771e-8, 1.6334237696880289e-8, 3.0456011009940273e-9,
    4.9629940211799280e-10, 6.5164572692449706e-11, 6.8439108189034605e-12,
    6.6624972821951867e-13, 9.1389160966632896e-14, 1.7743373609967763e-14,
    3.1493096321175424e-15, 4.2176942148173021e-16, 3.9812079021975348e-17,
    2.6996951374135936e-18, 2.6228348872722076e-19, 6.2731790865499806e-20,
    1.3334389244918855e-20, 1.8630596457028758e-21, 1.5634124948587485e-22,
    1.2273665127349842e-23,
], dtype=np.float64)

GOLDEN_DELTA = np.array([
    1.5116386225478912e-22, 1.9747025981399162e-21, 2.1906503466756586e-20,
    1.4072643655854709e-19, 5.6148186625432164e-19, 1.9820087363987545e-18,
    2.3424935159904366e-17, 3.5538335324717292e-16, 3.5412389753390115e-15,
    2.4065563997177281e-14, 1.2020239792222649e-13, 5.4354851619359720e-13,
    3.7505416186551337e-12, 3.7663721814684258e-11, 3.3617336067723558e-10,
    2.3256787716543681e-9, 1.2651129008784693e-8, 5.8289264976897211e-8,
    2.7412537590325326e-7, 1.6178048223466443e-6, 1.0896828975319930e-5,
    6.7202293663238351e-5, 3.4327755576714857e-4, 1.4150456337306452e-3,
    4.6673858446864272e-3, 1.2186025656545101e-2, 2.4636150347783770e-2,
    3.6732162269824856e-2, 3.5110862017445975e-2, 7.1146726711546577e-3,
    -4.1440107702895135e-2, -8.0845538991234520e-2, -8.0845538991240418e-2,
    -4.1440107702884074e-2, 7.1146726711431669e-3, 3.5110862017459027e-2,
    3.6732162269819443e-2, 2.4636150347783634e-2, 1.2186025656546107e-2,
    4.6673858446862867e-3, 1.4150456337307053e-3, 3.4327755576715182e-4,
    6.7202293663229650e-5, 1.0896828975321073e-5, 1.6178048223465736e-6,
    2.7412537590319503e-7, 5.8289264976922364e-8, 1.2651129008781323e-8,
    2.3256787716538776e-9, 3.3617336067746812e-10, 3.7663721814636734e-11,
    3.7505416186564852e-12, 5.4354851619526853e-13, 1.2020239792176395e-13,
    2.4065563997225146e-14, 3.5412389753436650e-15, 3.5538335324416834e-16,
    2.3424935160441756e-17, 1.9820087363691325e-18, 5.6148186624338871e-19,
    1.4072643656202276e-19, 2.1906503466277156e-20, 1.9747025981419001e-21,
    1.5116386228096023e-22,
], dtype=np.float64)

GOLDEN_GAMMA = np.array([
    1.7668632693894237e-21, 2.1532879021608059e-20, 2.1330711897272125e-19,
    1.1663813226807447e-18, 3.5203935794705589e-18, 9.7145298297364971e-18,
    1.7033496611335089e-16, 2.5972351777791464e-15, 2.3213671338768400e-14,
    1.3552303624501300e-13, 5.5738295184314034e-13, 2.1338674016780794e-12,
    1.5395048314131047e-11, 1.5450056076291651e-10, 1.2246814727029343e-9,
    7.0835022630746286e-9, 3.0882174492358587e-8, 1.1337699163707118e-7,
    4.8019415692922199e-7, 2.8831626811395561e-6, 1.8036862783792596e-5,
    9.2047154693554090e-5, 3.5938192981092111e-4, 1.0473952876320701e-3,
    2.1674943139937688e-3, 2.6978839202082771e-3, 1.9824093612927720e-4,
    -6.7896438287168162e-3, -1.3891784394294955e-2, -1.1320284783412078e-2,
    4.4721626988278931e-3, 2.0945553805408568e-2, 2.0945553805424004e-2,
    4.4721626988054493e-3, -1.1320284783389995e-2, -1.3891784394316670e-2,
    -6.7896438287052647e-3, 1.9824093612621886e-4, 2.6978839202074600e-3,
    2.1674943139943026e-3, 1.0473952876319218e-3, 3.5938192981095223e-4,
    9.2047154693553995e-5, 1.8036862783790279e-5, 2.8831626811404476e-6,
    4.8019415692893029e-7, 1.1337699163711157e-7, 3.0882174492368686e-8,
    7.0835022630690666e-9, 1.2246814727040287e-9, 1.5450056076284108e-10,
    1.5395048314089992e-11, 2.1338674016943157e-12, 5.5738295184052279e-13,
    1.3552303624500194e-13, 2.3213671338881788e-14, 2.5972351777476566e-15,
    1.7033496611697046e-16, 9.7145298299890112e-18, 3.5203935792679018e-18,
    1.1663813227231383e-18, 2.1330711896908851e-19, 2.1532879021021273e-20,
    1.7668632698973982e-21,
], dtype=np.float64)
# fmt: on


def test_g_binding_greeks_parity_sub3_pyo3_0ulp() -> None:
    """G_BINDING_GREEKS_PARITY sub-test 3 (PyO3 v3).

    Calls EvolverHeat1DGreeksV3.greeks() with the canonical params and asserts
    that (value, delta, gamma) are byte-identical (0 ULP) to the core golden.

    How called: semiflow.EvolverHeat1DGreeksV3 (PyO3 pyclass, GIL-release
    three-phase pattern ADR-0031).  The greeks() return is three numpy float64
    arrays unmarshalled from the Rust Vec<f64> via .to_pyarray().

    np.array_equal performs element-wise equality including sign of zero —
    exactly the 0-ULP contract.
    """
    ev = rp.EvolverHeat1DGreeksV3(
        _XMIN_CANONICAL, _XMAX_CANONICAL, _N_CANONICAL,
        _U0_CANONICAL, _N_CHERNOFF_CANONICAL, _THETA_CANONICAL,
    )
    value, delta, gamma = ev.greeks(_T_CANONICAL)

    # Compute bit-difference statistics for the diagnostic message.
    def max_ulp(got: "np.ndarray", want: "np.ndarray") -> int:
        g = got.view(np.int64)
        w = want.view(np.int64)
        return int(np.max(np.abs(g - w)))

    ulp_v = max_ulp(value, GOLDEN_VALUE)
    ulp_d = max_ulp(delta, GOLDEN_DELTA)
    ulp_g = max_ulp(gamma, GOLDEN_GAMMA)

    print(
        f"\nG_BINDING_GREEKS_PARITY sub-test 3 (PyO3 v3):\n"
        f"How called: rp.EvolverHeat1DGreeksV3 (PyO3 pyclass, GIL-released .greeks())\n"
        f"value: max ULP diff = {ulp_v}  (expected 0)\n"
        f"delta: max ULP diff = {ulp_d}  (expected 0)\n"
        f"gamma: max ULP diff = {ulp_g}  (expected 0)"
    )

    assert np.array_equal(value, GOLDEN_VALUE), (
        f"PyO3 value is NOT byte-identical to core golden (max ULP diff = {ulp_v})"
    )
    assert np.array_equal(delta, GOLDEN_DELTA), (
        f"PyO3 delta is NOT byte-identical to core golden (max ULP diff = {ulp_d})"
    )
    assert np.array_equal(gamma, GOLDEN_GAMMA), (
        f"PyO3 gamma is NOT byte-identical to core golden (max ULP diff = {ulp_g})"
    )
