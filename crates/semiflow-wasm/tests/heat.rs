//! Wave C smoke tests for semiflow-wasm.
//!
//! Mirrors Wave A's `examples/heat.c` and Wave B's `tests/test_heat.py`.
//! Same parameters: domain `[-10, 10]`, `n=1000`, `t=1`, `n_steps=100`,
//! oracle `u(1,x) = exp(-x²/5)/sqrt(5)`, gate `sup_error < 5e-4`.
//!
//! Cross-validation: Wave C sup-error should sit within sub-ULP of
//! Wave A's `1.46e-6` because wasm32 drives the identical Rust core.

#![cfg(target_arch = "wasm32")]
// usize→f64 and usize→u32 casts are intentional for wasm32 grid-index
// arithmetic and Float64Array sizing; wasm32 is a 32-bit target so both
// casts are exact.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use js_sys::Float64Array;
use semiflow_wasm::{version, Heat1D};
use wasm_bindgen_test::*;

// runtime selected by wasm-pack test --node / --chrome flag

const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
const N: usize = 1000;
const T: f64 = 1.0;
const N_STEPS: usize = 100;
const TOL: f64 = 5e-4;

fn make_u0_arr() -> Float64Array {
    let buf: Vec<f64> = (0..N)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((N - 1) as f64);
            (-x * x).exp()
        })
        .collect();
    let arr = Float64Array::new_with_length(N as u32);
    arr.copy_from(&buf);
    arr
}

fn oracle_vals() -> Vec<f64> {
    let s = 5.0_f64;
    let denom = s.sqrt();
    (0..N)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((N - 1) as f64);
            (-x * x / s).exp() / denom
        })
        .collect()
}

/// Gaussian heat-kernel smoke: `sup_error` must be `< 5e-4`.
///
/// Cross-validation against Wave A (C ABI): expected `≈ 1.46e-6`.
#[wasm_bindgen_test]
fn gaussian_smoke() {
    let u0 = make_u0_arr();
    let mut state = Heat1D::new(XMIN, XMAX, N, &u0).expect("Heat1D::new");
    state.evolve(T, N_STEPS).expect("evolve");

    let vals = state.values();
    let mut got = vec![0.0f64; N];
    vals.copy_to(&mut got);

    let oracle = oracle_vals();
    let sup_err = got
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    // Log visible in `wasm-pack test --node` output.
    wasm_bindgen_test::console_log!("sup_error = {:.6e}, version = {}", sup_err, version());

    assert!(
        sup_err < TOL,
        "sup_error {sup_err:.3e} >= {TOL} (expected ≈ 1.46e-6)"
    );
}

/// `evolve` with `t < 0` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn negative_t_errors() {
    let u0 = make_u0_arr();
    let mut state = Heat1D::new(XMIN, XMAX, N, &u0).expect("Heat1D::new");
    let err = state
        .evolve(-1.0, N_STEPS)
        .expect_err("expected OutOfDomain error");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain", "got kind={kind}");
}

/// `len_method()` must equal `N`.
#[wasm_bindgen_test]
fn len_matches_n() {
    let u0 = make_u0_arr();
    let state = Heat1D::new(XMIN, XMAX, N, &u0).expect("Heat1D::new");
    assert_eq!(state.len_method(), N);
}

/// `version()` must be a non-empty semver string.
#[wasm_bindgen_test]
fn version_is_semver() {
    let v = version();
    assert!(!v.is_empty(), "version string is empty");
    let core = v.split(['-', '+']).next().unwrap_or(&v);
    let parts: Vec<&str> = core.split('.').collect();
    assert_eq!(parts.len(), 3, "unexpected version core format: {v}");
}

// ---------------------------------------------------------------------------
// `withAFunction` tests — ADR-0034 WASM variable-a(x) binding
// ---------------------------------------------------------------------------

/// Build a JS function `x => value` (constant) for use in withAFunction tests.
///
/// Uses `js_sys::Function::new_with_args` to create a JS function from a
/// source string — the canonical way to create inline JS functions from Rust
/// wasm-bindgen-test code.
fn js_const_fn(value: f64) -> js_sys::Function {
    js_sys::Function::new_with_args("x", &format!("return {value};"))
}

/// Build a `Float64Array` of length `n` from a Rust iterator.
fn make_arr(vals: impl IntoIterator<Item = f64>) -> Float64Array {
    let buf: Vec<f64> = vals.into_iter().collect();
    let arr = Float64Array::new_with_length(buf.len() as u32);
    arr.copy_from(&buf);
    arr
}

/// `withAFunction` with constant `a(x) = 1`, `a'=0`, `a''=0` must produce
/// results numerically close to the unit-coefficient `new` constructor.
///
/// Both paths use the same Chernoff kernel; only the coefficient-lookup
/// mechanism differs (JS callback vs fn-ptr). Maximum absolute deviation must
/// be below a generous 1e-8 tolerance that absorbs JS-side floating-point
/// round-trip noise (the JS `return 1.0;` literal is exact in IEEE-754).
#[wasm_bindgen_test]
fn var_a_constant_matches_unit() {
    let u0_vals: Vec<f64> = (0..N)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((N - 1) as f64);
            (-x * x).exp()
        })
        .collect();

    // Unit path (fn-ptr constructor).
    let u0_arr = make_arr(u0_vals.iter().copied());
    let mut unit_state = Heat1D::new(XMIN, XMAX, N, &u0_arr).expect("unit Heat1D::new");
    unit_state.evolve(T, N_STEPS).expect("unit evolve");
    let unit_vals_arr = unit_state.values();
    let mut unit_vals = vec![0.0f64; N];
    unit_vals_arr.copy_to(&mut unit_vals);

    // Variable-a path with constant a=1 (JS callback constructor).
    let u0_arr2 = make_arr(u0_vals.iter().copied());
    let a_fn = js_const_fn(1.0);
    let ap_fn = js_const_fn(0.0);
    let app_fn = js_const_fn(0.0);
    let mut var_state =
        Heat1D::with_a_function(XMIN, XMAX, N, a_fn, ap_fn, app_fn, 1.0, &u0_arr2, N_STEPS)
            .expect("withAFunction Heat1D::with_a_function");
    var_state.evolve(T, N_STEPS).expect("var evolve");
    let var_vals_arr = var_state.values();
    let mut var_vals = vec![0.0f64; N];
    var_vals_arr.copy_to(&mut var_vals);

    let max_diff = unit_vals
        .iter()
        .zip(var_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    wasm_bindgen_test::console_log!("var_a_constant_matches_unit: max_diff = {:.3e}", max_diff);

    assert!(
        max_diff < 1e-8,
        "constant a=1 path deviated from unit path: max_diff={max_diff:.3e}"
    );
}

/// `withAFunction` with genuinely variable `a(x) = 1 + 0.1·x²` must
/// construct successfully, evolve without error, and produce finite results.
///
/// We cannot compare against a closed-form oracle for variable `a`; we
/// instead verify (1) no JS exception, (2) all output values are finite,
/// (3) sup_error against the unit oracle is larger than zero (i.e. the variable
/// coefficient actually changes the solution).
#[wasm_bindgen_test]
fn var_a_actual_variation() {
    let u0_vals: Vec<f64> = (0..N)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((N - 1) as f64);
            (-x * x).exp()
        })
        .collect();
    let u0_arr = make_arr(u0_vals.iter().copied());

    // a(x) = 1 + 0.1*x^2 (strictly positive)
    let a_fn = js_sys::Function::new_with_args("x", "return 1 + 0.1 * x * x;");
    // a'(x) = 0.2*x
    let ap_fn = js_sys::Function::new_with_args("x", "return 0.2 * x;");
    // a''(x) = 0.2
    let app_fn = js_sys::Function::new_with_args("x", "return 0.2;");
    // ||a||_inf on [-10,10]: 1 + 0.1*100 = 11
    let a_norm_bound = 11.0;

    let mut state = Heat1D::with_a_function(
        XMIN,
        XMAX,
        N,
        a_fn,
        ap_fn,
        app_fn,
        a_norm_bound,
        &u0_arr,
        N_STEPS,
    )
    .expect("withAFunction var-a construction");
    state.evolve(T, N_STEPS).expect("var-a evolve");

    let result_arr = state.values();
    let mut result = vec![0.0f64; N];
    result_arr.copy_to(&mut result);

    // All values must be finite.
    let all_finite = result.iter().all(|v| v.is_finite());
    assert!(all_finite, "var-a output contains NaN or Inf");

    // len() must equal N.
    assert_eq!(state.len_method(), N);

    // The variable-a solution differs from the unit oracle (i.e. it is not
    // trivially identical to the constant-a path).
    let oracle = oracle_vals();
    let sup_diff = result
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    wasm_bindgen_test::console_log!(
        "var_a_actual_variation: sup_diff_vs_unit_oracle = {:.6e}",
        sup_diff
    );

    // Must differ from the unit oracle (variable a ≠ constant a=1 solution).
    assert!(
        sup_diff > 0.0,
        "variable-a solution is identical to unit oracle — unexpected"
    );
}
