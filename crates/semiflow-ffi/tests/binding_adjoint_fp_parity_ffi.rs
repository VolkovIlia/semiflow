//! `G_BINDING_ADJOINT_FP_PARITY` — sub-test 2 (FFI v3, 0-ULP against core golden).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0138, ADR-0107 Amdt 1):
//!   Call `smf_adjoint_fp_new_brownian_1d_v3` and `smf_adjoint_fp_step_v3`
//!   at the CANONICAL smoke params (§1.2 `V8_1_TIER3_BINDING_DESIGN.md)`:
//!     a=0.5, b=0.0, c=0.0 (Brownian), `ρ₀=δ_0`, tau=0.1, `n_steps=1`.
//!   Assert that the returned (positions, weights) buffers are byte-identical
//!   (0 ULP) to the CORE GOLDEN — the values produced directly by `semiflow-core`.
//!
//! ## Why this is GENUINE
//!
//! The FFI path crosses an `extern "C"` boundary + `Box<AdjointFpInnerV3>`
//! construction + `smf_adjoint_fp_step_v3` + flat buffer copy.
//! Any marshalling bug (wrong pointer arithmetic, extra copy, off-by-one)
//! would show up as a non-zero ULP divergence from the core golden.
//! The core golden is independently produced by
//! `crates/semiflow-core/tests/binding_adjoint_fp_parity.rs`, not this file.

#![allow(unsafe_code)]
// Binding layer: allows for FFI/PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_wrap, clippy::unreadable_literal)]

use semiflow_ffi::{
    smf_adjoint_fp_free_v3, smf_adjoint_fp_new_brownian_1d_v3, smf_adjoint_fp_step_v3,
    SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters (§1.2, V8_1_TIER3_BINDING_DESIGN.md)
// ---------------------------------------------------------------------------

const A: f64 = 0.5;
const B: f64 = 0.0;
const C_COEF: f64 = 0.0;
const TAU: f64 = 0.1;
const MAX_STEPS: usize = 1;
/// Output buffer capacity = `4^max_steps` * `n_in` = 4 * 1 = 4.
const OUT_CAP: usize = 4;

// ---------------------------------------------------------------------------
// Core golden (from crates/semiflow-core/tests/binding_adjoint_fp_parity.rs
// `canonical_adjoint_fp_core`; verified against Lemma A.1 analytic anchor).
// h = 2√(aτ) = 2√(0.05) = 0.4472135954999579 (exact IEEE-754 double).
// ---------------------------------------------------------------------------

const GOLDEN_POS: [f64; 4] = [0.4472135954999579, -0.4472135954999579, 0.0, 0.0];
const GOLDEN_WTS: [f64; 4] = [0.25, 0.25, 0.5, 0.0];

// ---------------------------------------------------------------------------
// G_BINDING_ADJOINT_FP_PARITY sub-test 2: FFI v3 byte-identical to core golden
// ---------------------------------------------------------------------------

/// `G_BINDING_ADJOINT_FP_PARITY` sub-test 2 (FFI v3, 0-ULP).
///
/// Calls `smf_adjoint_fp_new_brownian_1d_v3` + `smf_adjoint_fp_step_v3`
/// from Rust (same mechanics as a C caller). Asserts that the output
/// (positions, weights) buffers are byte-identical (0 ULP) to the CORE GOLDEN.
#[test]
fn g_binding_adjoint_fp_parity_ffi_v3() {
    let mut ev: *mut semiflow_ffi::SmfAdjointFpV3 = std::ptr::null_mut();

    // --- Construct FFI handle ---
    let rc = unsafe { smf_adjoint_fp_new_brownian_1d_v3(A, B, C_COEF, MAX_STEPS, &mut ev) };
    assert_eq!(rc, SemiflowStatus::Ok, "FFI new failed: {rc:?}");
    assert!(!ev.is_null(), "FFI handle must be non-null on Ok");

    // --- Input measure: δ_0 (single Dirac at 0 with weight 1) ---
    let pos_in = [0.0f64];
    let wts_in = [1.0f64];
    let n_in: usize = 1;

    // --- Output buffers ---
    let mut pos_out = [f64::NAN; OUT_CAP];
    let mut wts_out = [f64::NAN; OUT_CAP];
    let mut out_n: usize = 0;

    let rc = unsafe {
        smf_adjoint_fp_step_v3(
            ev,
            TAU,
            1, // n_steps
            pos_in.as_ptr(),
            wts_in.as_ptr(),
            n_in,
            pos_out.as_mut_ptr(),
            wts_out.as_mut_ptr(),
            OUT_CAP,
            &mut out_n,
        )
    };
    assert_eq!(rc, SemiflowStatus::Ok, "FFI step failed: {rc:?}");
    assert_eq!(out_n, 4, "expected 4 output Diracs, got {out_n}");

    // --- Free handle ---
    unsafe { smf_adjoint_fp_free_v3(ev) };

    // --- 0-ULP check ---
    let pos_got = &pos_out[..out_n];
    let wts_got = &wts_out[..out_n];

    let max_ulp_pos = max_ulp_diff(pos_got, &GOLDEN_POS);
    let max_ulp_wts = max_ulp_diff(wts_got, &GOLDEN_WTS);

    println!(
        "G_BINDING_ADJOINT_FP_PARITY sub-test 2 (FFI v3):\n\
         How called: smf_adjoint_fp_new_brownian_1d_v3 + smf_adjoint_fp_step_v3\n\
         positions max ULP vs golden = {max_ulp_pos}  (expected 0)\n\
         weights   max ULP vs golden = {max_ulp_wts}  (expected 0)\n\
         pos_out = {pos_got:?}\n\
         wts_out = {wts_got:?}"
    );

    assert_eq!(
        max_ulp_pos, 0,
        "FFI positions NOT byte-identical to core golden (max ULP = {max_ulp_pos})"
    );
    assert_eq!(
        max_ulp_wts, 0,
        "FFI weights NOT byte-identical to core golden (max ULP = {max_ulp_wts})"
    );
}

// ---------------------------------------------------------------------------
// Error-path: max_steps=0 returns OutOfDomain
// ---------------------------------------------------------------------------

#[test]
fn ffi_adjoint_fp_max_steps_zero_returns_error() {
    let mut ev: *mut semiflow_ffi::SmfAdjointFpV3 = std::ptr::null_mut();
    let rc = unsafe { smf_adjoint_fp_new_brownian_1d_v3(A, B, C_COEF, 0, &mut ev) };
    assert_ne!(rc, SemiflowStatus::Ok, "max_steps=0 should fail");
}

// ---------------------------------------------------------------------------
// Error-path: n_steps exceeds max_steps returns OutOfDomain
// ---------------------------------------------------------------------------

#[test]
fn ffi_adjoint_fp_n_steps_exceeds_max() {
    let mut ev: *mut semiflow_ffi::SmfAdjointFpV3 = std::ptr::null_mut();
    let rc = unsafe { smf_adjoint_fp_new_brownian_1d_v3(A, B, C_COEF, 1, &mut ev) };
    assert_eq!(rc, SemiflowStatus::Ok);

    let pos_in = [0.0f64];
    let wts_in = [1.0f64];
    let mut pos_out = [0.0f64; 64];
    let mut wts_out = [0.0f64; 64];
    let mut out_n: usize = 0;

    // n_steps=2 > max_steps=1 → OutOfDomain
    let rc = unsafe {
        smf_adjoint_fp_step_v3(
            ev,
            TAU,
            2,
            pos_in.as_ptr(),
            wts_in.as_ptr(),
            1,
            pos_out.as_mut_ptr(),
            wts_out.as_mut_ptr(),
            64,
            &mut out_n,
        )
    };
    unsafe { smf_adjoint_fp_free_v3(ev) };
    assert_eq!(
        rc,
        SemiflowStatus::OutOfDomain,
        "n_steps > max_steps should return OutOfDomain"
    );
}

// ---------------------------------------------------------------------------
// ULP helpers
// ---------------------------------------------------------------------------

fn max_ulp_diff(got: &[f64], want: &[f64]) -> u64 {
    assert_eq!(got.len(), want.len(), "length mismatch in ULP check");
    got.iter()
        .zip(want.iter())
        .map(|(&g, &w)| ulp_dist(g, w))
        .max()
        .unwrap_or(0)
}

fn ulp_dist(a: f64, b: f64) -> u64 {
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    ai.wrapping_sub(bi).unsigned_abs()
}
