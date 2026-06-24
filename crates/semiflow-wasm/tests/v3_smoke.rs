//! v3.0 WASM smoke tests (ADR-0076, Wave F, Approach A).
//!
//! Tests `EvolverHeat1DUnitV3` and `GrowthV3` — the new v3 JS classes.
//!
//! ## `G_binding_parity` sub-test 5 (WASM v3 ⇔ v2)
//!
//! Verifies that the v3 WASM surface produces byte-identical output to the
//! v2 `Heat1D` surface on the same Gaussian initial condition.  Per ADR-0076
//! §`G_binding_parity`: the binding redesign is a pure pass-through to the same
//! Rust core; zero ULP error is achievable and required.
//!
//! ## Parameters (mirror Wave D / Wave E smoke suites)
//!
//! - Domain `[-10, 10]`, `n_grid = 1000`, `n_chernoff = 100`.
//! - Initial condition: `u₀(x) = exp(−x²)`.
//! - `t = 1.0`.
//! - Oracle: `u(1, x) = exp(−x²/5) / sqrt(5)` (Gaussian heat kernel).
//! - Gate: `sup_error < 5e-4`.

#![cfg(target_arch = "wasm32")]
// wasm32 is 32-bit; usize→u32 and i→f64 casts are exact on this target.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use js_sys::{Float64Array, Reflect};
use semiflow_wasm::{EvolverHeat1DGreeksV3, EvolverHeat1DUnitV3, GrowthV3, Heat1D};
use wasm_bindgen_test::*;

// ---------------------------------------------------------------------------
// Test constants
// ---------------------------------------------------------------------------

const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
const N: usize = 1000;
const T: f64 = 1.0;
const N_CHERNOFF: usize = 100;
const TOL: f64 = 5e-4;

/// Per-array sup relative-error gate for the WASM Greeks parity sub-test 4
/// (ADR-0183). Native↔wasm32 libm `exp()` differs in the last ULP; over 32
/// Chernoff steps + the hyper-dual chain rule the measured divergence is
/// ≤ 6.1e-11 relative. The 1e-9 gate gives ≈150× headroom while staying tight
/// enough that any marshalling bug (order-1 relative) is caught.
const TOL_REL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `Float64Array` of length `n` from a closure.
fn make_f64_array(n: usize, f: impl Fn(usize) -> f64) -> Float64Array {
    let buf: Vec<f64> = (0..n).map(f).collect();
    let arr = Float64Array::new_with_length(n as u32);
    arr.copy_from(&buf);
    arr
}

/// Build the Gaussian initial condition `u₀(x) = exp(−x²)`.
fn make_u0(n: usize) -> Float64Array {
    make_f64_array(n, |i| {
        let x = XMIN + (XMAX - XMIN) * (i as f64) / ((n - 1) as f64);
        (-x * x).exp()
    })
}

/// Evaluate the Gaussian heat-kernel oracle `u(1, x) = exp(−x²/5) / sqrt(5)`.
fn oracle_vals(n: usize) -> Vec<f64> {
    let s = 5.0_f64;
    let denom = s.sqrt();
    (0..n)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((n - 1) as f64);
            (-x * x / s).exp() / denom
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Constructor smoke
// ---------------------------------------------------------------------------

/// `EvolverHeat1DUnitV3::new` with valid arguments must succeed.
#[wasm_bindgen_test]
fn evolver_v3_construct_ok() {
    let u0 = make_u0(64);
    let ev = EvolverHeat1DUnitV3::new(-1.0, 1.0, 64, &u0, 32)
        .expect("EvolverHeat1DUnitV3::new(-1,1,64,u0,32)");
    assert_eq!(ev.size(), 64, "size() must equal n_grid");
    assert_eq!(
        ev.n_chernoff(),
        32,
        "n_chernoff() must equal constructor arg"
    );
}

/// `EvolverHeat1DUnitV3::new` with `u0.length != n_grid` must throw `GridMismatch`.
#[wasm_bindgen_test]
fn evolver_v3_length_mismatch_errors() {
    let u0 = make_u0(32); // length 32, but n_grid = 64
    let err = EvolverHeat1DUnitV3::new(-1.0, 1.0, 64, &u0, 32)
        .err()
        .expect("expected GridMismatch");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "GridMismatch", "got kind={kind}");
}

/// `EvolverHeat1DUnitV3::new` with `n_chernoff = 0` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn evolver_v3_n_chernoff_zero_errors() {
    let u0 = make_u0(32);
    let err = EvolverHeat1DUnitV3::new(-1.0, 1.0, 32, &u0, 0)
        .err()
        .expect("expected OutOfDomain for n_chernoff=0");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain", "got kind={kind}");
}

/// `EvolverHeat1DUnitV3::new` with NaN in `u0` must throw `NanInf`.
#[wasm_bindgen_test]
fn evolver_v3_nan_u0_errors() {
    let buf = {
        let arr = Float64Array::new_with_length(32);
        // Set one element to NaN via JS.
        arr.set_index(0, f64::NAN);
        arr
    };
    let err = EvolverHeat1DUnitV3::new(-1.0, 1.0, 32, &buf, 16)
        .err()
        .expect("expected NanInf for NaN in u0");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "NanInf", "got kind={kind}");
}

// ---------------------------------------------------------------------------
// Numerical accuracy smoke
// ---------------------------------------------------------------------------

/// Gaussian heat-kernel smoke: `sup_error < 5e-4` (mirrors Wave D/E gate).
#[wasm_bindgen_test]
fn evolver_v3_gaussian_smoke() {
    let u0 = make_u0(N);
    let mut ev =
        EvolverHeat1DUnitV3::new(XMIN, XMAX, N, &u0, N_CHERNOFF).expect("EvolverHeat1DUnitV3::new");

    let out = Float64Array::new_with_length(N as u32);
    ev.evolve_into(T, &out).expect("evolveInto");

    let mut got = vec![0.0f64; N];
    out.copy_to(&mut got);

    let oracle = oracle_vals(N);
    let sup_err = got
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    wasm_bindgen_test::console_log!("evolver_v3 sup_error = {sup_err:.6e}");

    assert!(
        sup_err < TOL,
        "sup_error {sup_err:.3e} >= {TOL} (expected ≈ 1.46e-6)"
    );
}

/// `evolveInto` with negative `t` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn evolver_v3_negative_t_errors() {
    let u0 = make_u0(64);
    let mut ev =
        EvolverHeat1DUnitV3::new(-1.0, 1.0, 64, &u0, 32).expect("EvolverHeat1DUnitV3::new");
    let out = Float64Array::new_with_length(64);
    let err = ev
        .evolve_into(-1.0, &out)
        .expect_err("expected OutOfDomain for t<0");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain", "got kind={kind}");
}

/// `evolveInto` with wrong `out` length must throw `GridMismatch`.
#[wasm_bindgen_test]
fn evolver_v3_out_length_mismatch_errors() {
    let u0 = make_u0(64);
    let mut ev =
        EvolverHeat1DUnitV3::new(-1.0, 1.0, 64, &u0, 32).expect("EvolverHeat1DUnitV3::new");
    let out = Float64Array::new_with_length(32); // wrong length
    let err = ev
        .evolve_into(0.1, &out)
        .expect_err("expected GridMismatch for out.length mismatch");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "GridMismatch", "got kind={kind}");
}

// ---------------------------------------------------------------------------
// GrowthV3 attributes
// ---------------------------------------------------------------------------

/// `growth()` for unit diffusion must return `multiplier = 1.0`, `omega = 0.0`.
#[wasm_bindgen_test]
fn growth_v3_unit_diffusion_attributes() {
    let u0 = make_u0(32);
    let ev = EvolverHeat1DUnitV3::new(-1.0, 1.0, 32, &u0, 16).expect("EvolverHeat1DUnitV3::new");
    let g: GrowthV3 = ev.growth();
    assert_eq!(
        g.multiplier(),
        1.0,
        "multiplier must be 1.0 for unit diffusion"
    );
    assert_eq!(g.omega(), 0.0, "omega must be 0.0 for unit diffusion");
}

// ---------------------------------------------------------------------------
// values() copy semantics
// ---------------------------------------------------------------------------

/// `values()` must return a copy equal to `out` after `evolveInto`.
#[wasm_bindgen_test]
fn evolver_v3_values_copy_semantics() {
    let u0 = make_u0(64);
    let mut ev =
        EvolverHeat1DUnitV3::new(-1.0, 1.0, 64, &u0, 32).expect("EvolverHeat1DUnitV3::new");

    let out = Float64Array::new_with_length(64);
    ev.evolve_into(0.05, &out).expect("evolveInto");

    let v = ev.values();
    assert_eq!(v.length(), 64, "values() length must equal size()");

    // `values()` and `out` must be element-wise identical.
    let mut from_v = vec![0.0f64; 64];
    v.copy_to(&mut from_v);
    let mut from_out = vec![0.0f64; 64];
    out.copy_to(&mut from_out);
    let max_diff = from_v
        .iter()
        .zip(from_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert_eq!(max_diff, 0.0, "values() and out must be byte-identical");
}

// ---------------------------------------------------------------------------
// G_binding_parity sub-test 5 (WASM v3 ⇔ v2 bit-identical)
// ---------------------------------------------------------------------------

/// G_binding_parity sub-test 5: v3 `EvolverHeat1DUnitV3` output must be
/// byte-identical to v2 `Heat1D` output on the same inputs.
///
/// Per ADR-0076 §G_binding_parity: the binding redesign is a pure pass-through
/// to the same Rust core; zero ULP error is required.
#[wasm_bindgen_test]
fn g_binding_parity_sub5_wasm_v3_vs_v2() {
    // --- v2 path (Heat1D) ---
    let u0_arr = make_u0(N);
    let mut v2_state = Heat1D::new(XMIN, XMAX, N, &u0_arr).expect("Heat1D::new (v2)");
    v2_state.evolve(T, N_CHERNOFF).expect("Heat1D::evolve (v2)");
    let v2_result_arr = v2_state.values();
    let mut v2_result = vec![0.0f64; N];
    v2_result_arr.copy_to(&mut v2_result);

    // --- v3 path (EvolverHeat1DUnitV3) ---
    let u0_arr2 = make_u0(N);
    let mut v3_ev =
        EvolverHeat1DUnitV3::new(XMIN, XMAX, N, &u0_arr2, N_CHERNOFF).expect("v3 Evolver::new");
    let v3_out = Float64Array::new_with_length(N as u32);
    v3_ev.evolve_into(T, &v3_out).expect("v3 evolveInto");
    let mut v3_result = vec![0.0f64; N];
    v3_out.copy_to(&mut v3_result);

    // Gate: byte-identical (0 ULP per ADR-0076 §G_binding_parity).
    let max_diff = v2_result
        .iter()
        .zip(v3_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    wasm_bindgen_test::console_log!(
        "G_binding_parity sub-test 5: WASM v3 vs v2 max_diff = {max_diff:.3e}"
    );

    assert_eq!(
        max_diff, 0.0,
        "WASM v3 output differs from v2 output by {max_diff:.3e} (expected 0 ULP)"
    );
}

// ---------------------------------------------------------------------------
// EvolverHeat1DGreeksV3 smoke (v8.0.0, G_BINDING_GREEKS_PARITY sub-test 4)
// ---------------------------------------------------------------------------

/// Greeks constructor: valid args must succeed and size/nChernoff match.
#[wasm_bindgen_test]
fn greeks_v3_construct_ok() {
    let u0 = make_u0(64);
    let ev = EvolverHeat1DGreeksV3::new(-5.0, 5.0, 64, &u0, 32, 0.5)
        .expect("EvolverHeat1DGreeksV3::new");
    assert_eq!(ev.size(), 64);
    assert_eq!(ev.n_chernoff(), 32);
}

/// Greeks constructor: `u0.length != n_grid` must throw `GridMismatch`.
#[wasm_bindgen_test]
fn greeks_v3_length_mismatch_errors() {
    let u0 = make_u0(32);
    let err = EvolverHeat1DGreeksV3::new(-5.0, 5.0, 64, &u0, 32, 0.5)
        .err()
        .expect("expected GridMismatch");
    let kind = Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "GridMismatch");
}

/// Greeks constructor: `n_chernoff == 0` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn greeks_v3_n_chernoff_zero_errors() {
    let u0 = make_u0(32);
    let err = EvolverHeat1DGreeksV3::new(-5.0, 5.0, 32, &u0, 0, 0.5)
        .err()
        .expect("expected OutOfDomain");
    let kind = Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain");
}

/// Greeks constructor: NaN in u0 must throw `NanInf`.
#[wasm_bindgen_test]
fn greeks_v3_nan_u0_errors() {
    let arr = Float64Array::new_with_length(32);
    arr.set_index(0, f64::NAN);
    let err = EvolverHeat1DGreeksV3::new(-5.0, 5.0, 32, &arr, 16, 0.5)
        .err()
        .expect("expected NanInf");
    let kind = Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "NanInf");
}

/// Greeks: `t < 0` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn greeks_v3_negative_t_errors() {
    let u0 = make_u0(64);
    let mut ev = EvolverHeat1DGreeksV3::new(-5.0, 5.0, 64, &u0, 32, 0.5)
        .expect("EvolverHeat1DGreeksV3::new");
    let err = ev.greeks(-1.0).err().expect("expected OutOfDomain");
    let kind = Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain");
}

/// G_BINDING_GREEKS_PARITY sub-test 4 (WASM):
/// value/delta/gamma must all be finite, and delta must match a central-
/// difference of value w.r.t. θ to within 1e-6 (§5, design doc).
#[wasm_bindgen_test]
fn greeks_v3_smoke_finite_and_fd_match() {
    const NG: usize = 64;
    const NC: usize = 32;
    const THETA: f64 = 0.5;
    const T: f64 = 0.05;
    const H: f64 = 1e-5;
    const TOL_DELTA_FD: f64 = 1e-6;

    let u0 = make_u0(NG);

    // --- hyper-dual sweep ---
    let mut ev = EvolverHeat1DGreeksV3::new(-5.0, 5.0, NG, &u0, NC, THETA)
        .expect("EvolverHeat1DGreeksV3::new");
    let result = ev.greeks(T).expect("greeks(T)");

    let value_arr: Float64Array = Reflect::get(&result, &"value".into())
        .map(|v| v.into())
        .expect("result.value");
    let delta_arr: Float64Array = Reflect::get(&result, &"delta".into())
        .map(|v| v.into())
        .expect("result.delta");
    let gamma_arr: Float64Array = Reflect::get(&result, &"gamma".into())
        .map(|v| v.into())
        .expect("result.gamma");

    let n = NG as u32;
    assert_eq!(value_arr.length(), n, "value length");
    assert_eq!(delta_arr.length(), n, "delta length");
    assert_eq!(gamma_arr.length(), n, "gamma length");

    let mut value = vec![0.0f64; NG];
    let mut delta = vec![0.0f64; NG];
    let mut gamma = vec![0.0f64; NG];
    value_arr.copy_to(&mut value);
    delta_arr.copy_to(&mut delta);
    gamma_arr.copy_to(&mut gamma);

    // All three buffers must be finite.
    for i in 0..NG {
        assert!(value[i].is_finite(), "value[{i}] not finite");
        assert!(delta[i].is_finite(), "delta[{i}] not finite");
        assert!(gamma[i].is_finite(), "gamma[{i}] not finite");
    }

    // Central-difference oracle for delta w.r.t. θ.
    let val_hi = evolve_at_theta(&u0, THETA + H, NG, NC, T);
    let val_lo = evolve_at_theta(&u0, THETA - H, NG, NC, T);
    let max_fd_err = value
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let fd = (val_hi[i] - val_lo[i]) / (2.0 * H);
            (delta[i] - fd).abs()
        })
        .fold(0.0f64, f64::max);

    wasm_bindgen_test::console_log!(
        "greeks_v3_smoke: max |delta - delta_fd| = {max_fd_err:.3e} (tol {TOL_DELTA_FD:.0e})"
    );
    assert!(
        max_fd_err < TOL_DELTA_FD,
        "delta vs FD: max_err {max_fd_err:.3e} >= {TOL_DELTA_FD:.0e}"
    );
}

// ---------------------------------------------------------------------------
// FD helper — evolve a u0 at a given θ, return value vector
// ---------------------------------------------------------------------------

/// Run a scalar (f64) evolve at diffusion scale `theta` for `t` time.
///
/// Used only as the FD oracle in the Greeks smoke test; duplicates the
/// per-crate build pattern (no shared util, ADR-0028 Amendment 2).
fn evolve_at_theta(
    u0: &Float64Array,
    theta: f64,
    n_grid: usize,
    n_chernoff: usize,
    t: f64,
) -> Vec<f64> {
    use semiflow_wasm::EvolverHeat1DUnitV3;
    // Rescale a unit-diffusion evolver: DiffusionChernoff uses a=1.0, but the
    // heat at scale θ satisfies u_t = θ·u_xx; equivalently advance time by θ·t.
    // Since EvolverHeat1DUnitV3 hard-codes a=1, we evolve for θ·t instead.
    let mut ev = EvolverHeat1DUnitV3::new(-5.0, 5.0, n_grid, u0, n_chernoff)
        .expect("EvolverHeat1DUnitV3::new (FD)");
    let out = Float64Array::new_with_length(n_grid as u32);
    ev.evolve_into(theta * t, &out).expect("evolveInto (FD)");
    let mut buf = vec![0.0f64; n_grid];
    out.copy_to(&mut buf);
    buf
}

// ---------------------------------------------------------------------------
// G_BINDING_GREEKS_PARITY sub-test 4 (WASM v3, tolerance-bounded vs core golden)
// ---------------------------------------------------------------------------
//
// Canonical params (contracts/semiflow-core.properties.yaml §G_BINDING_GREEKS_PARITY):
//   domain [-10, 10], N=64, n_chernoff=32, t=0.05, theta=0.5, u0=exp(-x²).
//
// GOLDEN arrays are the CORE library's hyper-dual sweep — an oracle independent
// of this WASM SUT. They are produced by `print_golden_full` in
//   crates/semiflow/tests/binding_greeks_parity.rs
// built SCALAR (no SIMD) so the golden uses the same strict scalar IEEE-754 f64
// arithmetic family as wasm32:
//   cargo test -p semiflow --no-default-features --features std \
//              --test binding_greeks_parity print_golden_full -- --nocapture
// The core test verifies this golden against a Richardson 4-/6-point FD oracle
// (sub-test 1, g_binding_greeks_parity_core_golden).
//
// WASM parity is TOLERANCE-BOUNDED, not byte-identical (ADR-0183): native↔wasm32
// libm exp() differs in the last ULP; over 32 Chernoff steps + the hyper-dual
// chain rule the measured divergence is ≤ 6.1e-11 relative (but ~285k ULP on the
// ~1e-23 Gaussian tail, where ULP is a misleading scale). The 1e-9 relative gate
// still catches any marshalling bug (order-1 relative), which is the gate's job.
// FFI and PyO3 sub-tests stay 0-ULP (native, same libm as core).

/// Scalar core golden — value array (semiflow hyper-dual sweep, no-SIMD build).
const GOLDEN_VALUE: [f64; 64] = [
    1.2273665126349644e-23,
    1.5634124948354552e-22,
    1.8630596457416103e-21,
    1.3334389244702536e-20,
    6.2731790865844487e-20,
    2.6228348873235650e-19,
    2.6996951373641581e-18,
    3.9812079022180766e-17,
    4.2176942148193078e-16,
    3.1493096321099560e-15,
    1.7743373610017136e-14,
    9.1389160966536402e-14,
    6.6624972821867715e-13,
    6.8439108189117889e-12,
    6.5164572692422500e-11,
    4.9629940211796168e-10,
    3.0456011009952755e-9,
    1.6334237696875445e-8,
    9.2735844448864484e-8,
    6.6127065123771670e-7,
    5.2809695914217134e-6,
    3.8923073423285504e-5,
    2.4368461722402000e-4,
    1.2709197832545980e-3,
    5.5100563112493728e-3,
    1.9873008874759564e-2,
    5.9662313475552890e-2,
    1.4912925268685812e-1,
    3.1036207501057161e-1,
    5.3778655727172475e-1,
    7.7584807469246408e-1,
    9.3188389445411468e-1,
    9.3188389445411679e-1,
    7.7584807469245520e-1,
    5.3778655727172864e-1,
    3.1036207501055646e-1,
    1.4912925268685959e-1,
    5.9662313475555125e-2,
    1.9873008874759211e-2,
    5.5100563112494743e-3,
    1.2709197832546308e-3,
    2.4368461722399043e-4,
    3.8923073423283288e-5,
    5.2809695914224884e-6,
    6.6127065123759038e-7,
    9.2735844448861771e-8,
    1.6334237696880289e-8,
    3.0456011009940273e-9,
    4.9629940211799280e-10,
    6.5164572692449706e-11,
    6.8439108189034605e-12,
    6.6624972821951867e-13,
    9.1389160966632896e-14,
    1.7743373609967763e-14,
    3.1493096321175424e-15,
    4.2176942148173021e-16,
    3.9812079021975348e-17,
    2.6996951374135936e-18,
    2.6228348872722076e-19,
    6.2731790865499806e-20,
    1.3334389244918855e-20,
    1.8630596457028758e-21,
    1.5634124948587485e-22,
    1.2273665127349842e-23,
];
/// Scalar core golden — delta array (∂/∂θ, no-SIMD build).
const GOLDEN_DELTA: [f64; 64] = [
    1.5116386225478912e-22,
    1.9747025981399162e-21,
    2.1906503466756586e-20,
    1.4072643655854709e-19,
    5.6148186625432164e-19,
    1.9820087363987545e-18,
    2.3424935159904366e-17,
    3.5538335324717292e-16,
    3.5412389753390115e-15,
    2.4065563997177281e-14,
    1.2020239792222649e-13,
    5.4354851619359720e-13,
    3.7505416186551337e-12,
    3.7663721814684258e-11,
    3.3617336067723558e-10,
    2.3256787716543681e-9,
    1.2651129008784693e-8,
    5.8289264976897211e-8,
    2.7412537590325326e-7,
    1.6178048223466443e-6,
    1.0896828975319930e-5,
    6.7202293663238351e-5,
    3.4327755576714857e-4,
    1.4150456337306452e-3,
    4.6673858446864272e-3,
    1.2186025656545101e-2,
    2.4636150347783770e-2,
    3.6732162269824856e-2,
    3.5110862017445975e-2,
    7.1146726711546577e-3,
    -4.1440107702895135e-2,
    -8.0845538991234520e-2,
    -8.0845538991240418e-2,
    -4.1440107702884074e-2,
    7.1146726711431669e-3,
    3.5110862017459027e-2,
    3.6732162269819443e-2,
    2.4636150347783634e-2,
    1.2186025656546107e-2,
    4.6673858446862867e-3,
    1.4150456337307053e-3,
    3.4327755576715182e-4,
    6.7202293663229650e-5,
    1.0896828975321073e-5,
    1.6178048223465736e-6,
    2.7412537590319503e-7,
    5.8289264976922364e-8,
    1.2651129008781323e-8,
    2.3256787716538776e-9,
    3.3617336067746812e-10,
    3.7663721814636734e-11,
    3.7505416186564852e-12,
    5.4354851619526853e-13,
    1.2020239792176395e-13,
    2.4065563997225146e-14,
    3.5412389753436650e-15,
    3.5538335324416834e-16,
    2.3424935160441756e-17,
    1.9820087363691325e-18,
    5.6148186624338871e-19,
    1.4072643656202276e-19,
    2.1906503466277156e-20,
    1.9747025981419001e-21,
    1.5116386228096023e-22,
];
/// Scalar core golden — gamma array (∂²/∂θ², no-SIMD build).
const GOLDEN_GAMMA: [f64; 64] = [
    1.7668632693894237e-21,
    2.1532879021608059e-20,
    2.1330711897272125e-19,
    1.1663813226807447e-18,
    3.5203935794705589e-18,
    9.7145298297364971e-18,
    1.7033496611335089e-16,
    2.5972351777791464e-15,
    2.3213671338768400e-14,
    1.3552303624501300e-13,
    5.5738295184314034e-13,
    2.1338674016780794e-12,
    1.5395048314131047e-11,
    1.5450056076291651e-10,
    1.2246814727029343e-9,
    7.0835022630746286e-9,
    3.0882174492358587e-8,
    1.1337699163707118e-7,
    4.8019415692922199e-7,
    2.8831626811395561e-6,
    1.8036862783792596e-5,
    9.2047154693554090e-5,
    3.5938192981092111e-4,
    1.0473952876320701e-3,
    2.1674943139937688e-3,
    2.6978839202082771e-3,
    1.9824093612927720e-4,
    -6.7896438287168162e-3,
    -1.3891784394294955e-2,
    -1.1320284783412078e-2,
    4.4721626988278931e-3,
    2.0945553805408568e-2,
    2.0945553805424004e-2,
    4.4721626988054493e-3,
    -1.1320284783389995e-2,
    -1.3891784394316670e-2,
    -6.7896438287052647e-3,
    1.9824093612621886e-4,
    2.6978839202074600e-3,
    2.1674943139943026e-3,
    1.0473952876319218e-3,
    3.5938192981095223e-4,
    9.2047154693553995e-5,
    1.8036862783790279e-5,
    2.8831626811404476e-6,
    4.8019415692893029e-7,
    1.1337699163711157e-7,
    3.0882174492368686e-8,
    7.0835022630690666e-9,
    1.2246814727040287e-9,
    1.5450056076284108e-10,
    1.5395048314089992e-11,
    2.1338674016943157e-12,
    5.5738295184052279e-13,
    1.3552303624500194e-13,
    2.3213671338881788e-14,
    2.5972351777476566e-15,
    1.7033496611697046e-16,
    9.7145298299890112e-18,
    3.5203935792679018e-18,
    1.1663813227231383e-18,
    2.1330711896908851e-19,
    2.1532879021021273e-20,
    1.7668632698973982e-21,
];

/// G_BINDING_GREEKS_PARITY sub-test 4 (WASM v3).
///
/// Calls `EvolverHeat1DGreeksV3::new` + `.greeks(0.05)` via wasm_bindgen with
/// the canonical params (domain [-10, 10], N=64, n_chernoff=32, theta=0.5,
/// u0=exp(-x²)).  Asserts that value, delta, and gamma Float64Arrays match the
/// SCALAR CORE GOLDEN (an oracle independent of this WASM SUT) to ≤ 1e-9 per-
/// array sup relative error.
///
/// Why tolerance, not 0-ULP (ADR-0183): native↔wasm32 libm `exp()` differs in
/// the last ULP; over 32 Chernoff steps + the hyper-dual chain rule this is
/// ≤ 6.1e-11 relative — physically irreducible. The 1e-9 gate still catches any
/// marshalling bug (wrong index / transposed array / sign flip → order-1
/// relative). FFI/PyO3 sub-tests remain 0-ULP (native, same libm as core).
///
/// How called: `EvolverHeat1DGreeksV3` (wasm_bindgen `#[wasm_bindgen]` class,
/// WASM module compiled with panic=abort per ADR-0028 Amendment 1).
#[wasm_bindgen_test]
fn g_binding_greeks_parity_sub4_wasm_tol() {
    // Canonical params: domain [-10, 10], N=64, n_chernoff=32, theta=0.5.
    const NG: usize = 64;
    const NC: usize = 32;
    const THETA: f64 = 0.5;
    const TVAL: f64 = 0.05;
    const XMIN_C: f64 = -10.0;
    const XMAX_C: f64 = 10.0;

    // Build u0 = exp(-x²) on [-10, 10] with N=64 nodes.
    let u0_arr = make_f64_array(NG, |i| {
        let x = XMIN_C + (XMAX_C - XMIN_C) * (i as f64) / ((NG - 1) as f64);
        (-x * x).exp()
    });

    let mut ev = EvolverHeat1DGreeksV3::new(XMIN_C, XMAX_C, NG, &u0_arr, NC, THETA)
        .expect("EvolverHeat1DGreeksV3::new (canonical)");
    let result = ev.greeks(TVAL).expect("greeks(0.05)");

    // Extract Float64Arrays from the {value, delta, gamma} JS object.
    let value_arr: Float64Array = Reflect::get(&result, &"value".into())
        .map(|v| v.into())
        .expect("result.value");
    let delta_arr: Float64Array = Reflect::get(&result, &"delta".into())
        .map(|v| v.into())
        .expect("result.delta");
    let gamma_arr: Float64Array = Reflect::get(&result, &"gamma".into())
        .map(|v| v.into())
        .expect("result.gamma");

    assert_eq!(value_arr.length() as usize, NG);
    assert_eq!(delta_arr.length() as usize, NG);
    assert_eq!(gamma_arr.length() as usize, NG);

    let mut value = vec![0.0f64; NG];
    let mut delta = vec![0.0f64; NG];
    let mut gamma = vec![0.0f64; NG];
    value_arr.copy_to(&mut value);
    delta_arr.copy_to(&mut delta);
    gamma_arr.copy_to(&mut gamma);

    // Per-array sup RELATIVE error vs the scalar core golden (ADR-0183).
    // ULP comparison is unusable across native↔wasm32: the irreducible libm
    // exp() gap is ~6e-11 relative but amplifies to ~285k ULP on the ~1e-23
    // Gaussian tail, where one mantissa bit is many ULP yet negligible.
    let max_rel = |got: &[f64], want: &[f64]| -> f64 {
        got.iter()
            .zip(want.iter())
            .map(|(&g, &w)| {
                let denom = if w == 0.0 { 1.0 } else { w.abs() };
                (g - w).abs() / denom
            })
            .fold(0.0f64, f64::max)
    };

    let rel_v = max_rel(&value, &GOLDEN_VALUE);
    let rel_d = max_rel(&delta, &GOLDEN_DELTA);
    let rel_g = max_rel(&gamma, &GOLDEN_GAMMA);

    wasm_bindgen_test::console_log!(
        "G_BINDING_GREEKS_PARITY sub-test 4 (WASM v3):\n\
         How called: EvolverHeat1DGreeksV3 (wasm_bindgen, Node)\n\
         oracle: scalar core golden (semiflow, --no-default-features --features std)\n\
         value: max rel err = {:.3e}  (gate <= {:.0e})\n\
         delta: max rel err = {:.3e}  (gate <= {:.0e})\n\
         gamma: max rel err = {:.3e}  (gate <= {:.0e})",
        rel_v,
        TOL_REL,
        rel_d,
        TOL_REL,
        rel_g,
        TOL_REL,
    );

    assert!(
        rel_v <= TOL_REL,
        "WASM value diverges from scalar core golden by rel err {rel_v:.3e} > {TOL_REL:.0e}"
    );
    assert!(
        rel_d <= TOL_REL,
        "WASM delta diverges from scalar core golden by rel err {rel_d:.3e} > {TOL_REL:.0e}"
    );
    assert!(
        rel_g <= TOL_REL,
        "WASM gamma diverges from scalar core golden by rel err {rel_g:.3e} > {TOL_REL:.0e}"
    );
}
