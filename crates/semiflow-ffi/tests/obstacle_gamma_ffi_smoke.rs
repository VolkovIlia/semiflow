//! Smoke tests for `SmfObstacleGamma` + `SmfObstacleND2` FFI entry points
//! (C-parity pass, ADR-0028/0171, ADR-0153 TIER-2).
//!
//! ## Cross-check methodology
//!
//! The core `binding_obstacle_gamma_parity` integration test (crate
//! `semiflow-core/tests/binding_obstacle_gamma_parity.rs`) establishes
//! the canonical golden: for a perpetual-put value field on `[0,3]`
//! with `N=64` nodes and obstacle `g = K − S`:
//!   - `count > 0` (some interior nodes defined).
//!   - At least one node on the active set (`S ≤ S*`) has `defined = false`.
//!   - `count == sum(defined)`.
//!
//! The FFI smoke test uses a simpler case that can be cross-checked by hand:
//! - Domain `[0, 1]`, N=8, constant obstacle `level = 0.0`.
//! - Value field: `v[i] = 1.0` for all i (all above obstacle).
//! - Expected: count > 0 (most interior nodes defined, boundary nodes refused).
//! - gamma[i] = (v[i+1] - 2*v[i] + v[i-1]) / dx^2 = 0.0 everywhere (constant v).
//!
//! ND2 smoke: 5×5 grid, constant obstacle level = −1, constant v = 0.
//! After one Chernoff step with tau = 0.01, output is >= level (−1) everywhere
//! and close to 0 (diffusion on constant field is near-identity).

#![allow(unsafe_code)]

use semiflow_ffi::{
    smf_free_buf_f64, smf_free_buf_u8,
    smf_obstacle_gamma_free, smf_obstacle_gamma_inactive_gamma,
    smf_obstacle_gamma_new_array, smf_obstacle_gamma_new_const,
    smf_obstacle_gamma_size,
    smf_obstacle_nd2_apply, smf_obstacle_nd2_free, smf_obstacle_nd2_new,
    smf_obstacle_nd2_shape,
    SemiflowStatus, SmfObstacleGamma, SmfObstacleND2,
};

// ---------------------------------------------------------------------------
// SmfObstacleGamma — null-safety
// ---------------------------------------------------------------------------

#[test]
fn obstacle_gamma_new_const_null_out_returns_null_ptr() {
    let s = unsafe {
        smf_obstacle_gamma_new_const(0.0, 1.0, 8, 0.0, std::ptr::null_mut())
    };
    assert_eq!(s, SemiflowStatus::NullPtr);
}

#[test]
fn obstacle_gamma_new_array_null_obs_returns_null_ptr() {
    let mut ptr: *mut SmfObstacleGamma = std::ptr::null_mut();
    let s = unsafe {
        smf_obstacle_gamma_new_array(0.0, 1.0, 8, std::ptr::null(), 8, &mut ptr)
    };
    assert_eq!(s, SemiflowStatus::NullPtr);
    assert!(ptr.is_null());
}

#[test]
fn obstacle_gamma_free_null_is_noop() {
    unsafe { smf_obstacle_gamma_free(std::ptr::null_mut()) };
}

#[test]
fn obstacle_gamma_size_null_returns_zero() {
    assert_eq!(unsafe { smf_obstacle_gamma_size(std::ptr::null()) }, 0);
}

// ---------------------------------------------------------------------------
// SmfObstacleGamma — constructor + size
// ---------------------------------------------------------------------------

#[test]
fn obstacle_gamma_new_const_success() {
    let mut ptr: *mut SmfObstacleGamma = std::ptr::null_mut();
    let s = unsafe { smf_obstacle_gamma_new_const(0.0, 1.0, 8, 0.0, &mut ptr) };
    assert_eq!(s, SemiflowStatus::Ok);
    assert!(!ptr.is_null());
    assert_eq!(unsafe { smf_obstacle_gamma_size(ptr) }, 8);
    unsafe { smf_obstacle_gamma_free(ptr) };
}

#[test]
fn obstacle_gamma_new_const_n_too_small_returns_out_of_domain() {
    let mut ptr: *mut SmfObstacleGamma = std::ptr::null_mut();
    let s = unsafe { smf_obstacle_gamma_new_const(0.0, 1.0, 3, 0.0, &mut ptr) };
    assert_ne!(s, SemiflowStatus::Ok, "n=3 must be rejected");
    assert!(ptr.is_null());
}

#[test]
fn obstacle_gamma_new_const_nan_level_returns_error() {
    let mut ptr: *mut SmfObstacleGamma = std::ptr::null_mut();
    let s = unsafe { smf_obstacle_gamma_new_const(0.0, 1.0, 8, f64::NAN, &mut ptr) };
    assert_ne!(s, SemiflowStatus::Ok, "NaN level must be rejected");
    assert!(ptr.is_null());
}

#[test]
fn obstacle_gamma_new_array_success() {
    let obs: Vec<f64> = vec![0.0; 8];
    let mut ptr: *mut SmfObstacleGamma = std::ptr::null_mut();
    let s = unsafe {
        smf_obstacle_gamma_new_array(0.0, 1.0, 8, obs.as_ptr(), 8, &mut ptr)
    };
    assert_eq!(s, SemiflowStatus::Ok);
    assert!(!ptr.is_null());
    assert_eq!(unsafe { smf_obstacle_gamma_size(ptr) }, 8);
    unsafe { smf_obstacle_gamma_free(ptr) };
}

// ---------------------------------------------------------------------------
// SmfObstacleGamma — inactive_gamma cross-check
//
// Domain [0,1], N=8, level=0.0, v[i]=1.0 (strictly above obstacle everywhere).
//
// Expected (by hand):
//   - All interior nodes (1..6) have v[i] = 1 > 0 = g(x), so inactive=true.
//   - Boundary nodes 0 and 7 are always defined=false (no centred stencil).
//   - With 1-node guard band: nodes at boundary of inactive region are excluded.
//     But here ALL nodes are inactive, so all interior nodes with a complete
//     3-point stencil are defined.
//   - gamma[i] = (1 - 2 + 1) / dx^2 = 0 / dx^2 = 0 for all i (constant v).
//   - count == 6 (nodes 1..6, all interior with full stencil).
// ---------------------------------------------------------------------------

unsafe fn make_gamma_const(n: usize, level: f64) -> *mut SmfObstacleGamma {
    let mut ptr: *mut SmfObstacleGamma = std::ptr::null_mut();
    let s = unsafe { smf_obstacle_gamma_new_const(0.0, 1.0, n, level, &mut ptr) };
    assert_eq!(s, SemiflowStatus::Ok);
    assert!(!ptr.is_null());
    ptr
}

#[test]
fn obstacle_gamma_inactive_gamma_constant_v() {
    let n = 8_usize;
    let ptr = unsafe { make_gamma_const(n, 0.0) };

    // v = [1, 1, ..., 1] — all strictly above obstacle 0.
    let v: Vec<f64> = vec![1.0; n];
    let mut gamma_ptr: *mut f64 = std::ptr::null_mut();
    let mut defined_ptr: *mut u8 = std::ptr::null_mut();
    let mut count: usize = 0;

    let s = unsafe {
        smf_obstacle_gamma_inactive_gamma(
            ptr,
            v.as_ptr(),
            n,
            &mut gamma_ptr,
            &mut defined_ptr,
            &mut count,
        )
    };
    assert_eq!(s, SemiflowStatus::Ok, "inactive_gamma should succeed");
    assert!(!gamma_ptr.is_null(), "gamma buffer must be non-null");
    assert!(!defined_ptr.is_null(), "defined buffer must be non-null");

    // count > 0: at least some nodes are defined.
    assert!(count > 0, "count must be > 0 (some interior nodes active)");

    // Verify: boundary nodes (0 and n-1) are always refused.
    let defined_slice = unsafe { std::slice::from_raw_parts(defined_ptr, n) };
    assert_eq!(defined_slice[0], 0, "boundary node 0 must be refused");
    assert_eq!(defined_slice[n - 1], 0, "boundary node n-1 must be refused");

    // Verify: interior nodes are defined (v=1 > 0=g everywhere).
    for i in 1..(n - 1) {
        assert_eq!(defined_slice[i], 1, "interior node {i} must be defined");
    }
    // count == n-2 (all interior defined, none excluded by guard band since all inactive).
    assert_eq!(count, n - 2, "count must equal n-2 for constant v above obstacle");

    // Verify gamma = 0 at all defined nodes (constant v → zero 2nd derivative).
    let gamma_slice = unsafe { std::slice::from_raw_parts(gamma_ptr, n) };
    for i in 0..n {
        if defined_slice[i] == 1 {
            assert!(
                gamma_slice[i].abs() < 1e-10,
                "gamma[{i}] = {} must be ~0 for constant v",
                gamma_slice[i]
            );
        }
    }

    // Free buffers.
    unsafe { smf_free_buf_f64(gamma_ptr, n) };
    unsafe { smf_free_buf_u8(defined_ptr, n) };
    unsafe { smf_obstacle_gamma_free(ptr) };
}

#[test]
fn obstacle_gamma_inactive_gamma_null_returns_null_ptr() {
    let v = vec![1.0_f64; 8];
    let mut g: *mut f64 = std::ptr::null_mut();
    let mut d: *mut u8 = std::ptr::null_mut();
    let mut c: usize = 0;
    let s = unsafe {
        smf_obstacle_gamma_inactive_gamma(std::ptr::null(), v.as_ptr(), 8, &mut g, &mut d, &mut c)
    };
    assert_eq!(s, SemiflowStatus::NullPtr);
}

#[test]
fn obstacle_gamma_inactive_gamma_length_mismatch_returns_grid_mismatch() {
    let ptr = unsafe { make_gamma_const(8, 0.0) };
    let v = vec![1.0_f64; 5]; // wrong length
    let mut g: *mut f64 = std::ptr::null_mut();
    let mut d: *mut u8 = std::ptr::null_mut();
    let mut c: usize = 0;
    let s = unsafe {
        smf_obstacle_gamma_inactive_gamma(ptr, v.as_ptr(), 5, &mut g, &mut d, &mut c)
    };
    assert_eq!(s, SemiflowStatus::GridMismatch);
    unsafe { smf_obstacle_gamma_free(ptr) };
}

// ---------------------------------------------------------------------------
// SmfObstacleND2 — null-safety
// ---------------------------------------------------------------------------

#[test]
fn obstacle_nd2_new_null_out_returns_null_ptr() {
    let s = unsafe {
        smf_obstacle_nd2_new(0.0, 1.0, 5, 0.0, 1.0, 5, 0.0, std::ptr::null_mut())
    };
    assert_eq!(s, SemiflowStatus::NullPtr);
}

#[test]
fn obstacle_nd2_free_null_is_noop() {
    unsafe { smf_obstacle_nd2_free(std::ptr::null_mut()) };
}

// ---------------------------------------------------------------------------
// SmfObstacleND2 — shape + apply
// ---------------------------------------------------------------------------

unsafe fn make_nd2(nx: usize, ny: usize, level: f64) -> *mut SmfObstacleND2 {
    let mut ptr: *mut SmfObstacleND2 = std::ptr::null_mut();
    let s = unsafe {
        smf_obstacle_nd2_new(0.0, 1.0, nx, 0.0, 1.0, ny, level, &mut ptr)
    };
    assert_eq!(s, SemiflowStatus::Ok, "smf_obstacle_nd2_new({nx},{ny})");
    assert!(!ptr.is_null());
    ptr
}

#[test]
fn obstacle_nd2_shape() {
    let ptr = unsafe { make_nd2(5, 7, -1.0) };
    let mut nx: usize = 0;
    let mut ny: usize = 0;
    let s = unsafe { smf_obstacle_nd2_shape(ptr, &mut nx, &mut ny) };
    assert_eq!(s, SemiflowStatus::Ok);
    assert_eq!(nx, 5);
    assert_eq!(ny, 7);
    unsafe { smf_obstacle_nd2_free(ptr) };
}

#[test]
fn obstacle_nd2_shape_null_returns_null_ptr() {
    let mut nx: usize = 0;
    let mut ny: usize = 0;
    let s = unsafe { smf_obstacle_nd2_shape(std::ptr::null(), &mut nx, &mut ny) };
    assert_eq!(s, SemiflowStatus::NullPtr);
}

/// Cross-check vs PyO3: constant v=0 above obstacle level=−1.
///
/// After one Chernoff step of an isotropic heat semigroup, a constant initial
/// field u₀ = 0 remains exactly 0 (constant is in the null space of Δ).
/// The obstacle projection max(u, g) leaves it unchanged since 0 > −1.
/// Expected: out[i] ≈ 0 for all i, >= −1 (obstacle not violated).
#[test]
fn obstacle_nd2_apply_constant_field() {
    let nx = 5_usize;
    let ny = 5_usize;
    let n = nx * ny;
    let ptr = unsafe { make_nd2(nx, ny, -1.0) };

    let v: Vec<f64> = vec![0.0; n];
    let mut out: Vec<f64> = vec![f64::NAN; n];
    let s = unsafe {
        smf_obstacle_nd2_apply(ptr, 0.01, v.as_ptr(), n, out.as_mut_ptr(), n)
    };
    assert_eq!(s, SemiflowStatus::Ok, "apply should succeed");

    for (i, &val) in out.iter().enumerate() {
        assert!(
            val.is_finite(),
            "output[{i}] is not finite: {val}"
        );
        assert!(
            val >= -1.0 - 1e-12,
            "obstacle violated: out[{i}] = {val} < level = -1.0"
        );
        assert!(
            val.abs() < 1e-10,
            "constant v=0 should stay ~0 after one step; got out[{i}] = {val}"
        );
    }

    unsafe { smf_obstacle_nd2_free(ptr) };
}

#[test]
fn obstacle_nd2_apply_tau_invalid() {
    let ptr = unsafe { make_nd2(5, 5, 0.0) };
    let v: Vec<f64> = vec![1.0; 25];
    let mut out: Vec<f64> = vec![0.0; 25];
    let s = unsafe {
        smf_obstacle_nd2_apply(ptr, 0.0, v.as_ptr(), 25, out.as_mut_ptr(), 25)
    };
    assert_eq!(s, SemiflowStatus::OutOfDomain, "tau=0 must be rejected");
    unsafe { smf_obstacle_nd2_free(ptr) };
}

#[test]
fn obstacle_nd2_apply_length_mismatch() {
    let ptr = unsafe { make_nd2(5, 5, 0.0) };
    let v: Vec<f64> = vec![1.0; 10]; // wrong: should be 25
    let mut out: Vec<f64> = vec![0.0; 10];
    let s = unsafe {
        smf_obstacle_nd2_apply(ptr, 0.01, v.as_ptr(), 10, out.as_mut_ptr(), 10)
    };
    assert_eq!(s, SemiflowStatus::GridMismatch);
    unsafe { smf_obstacle_nd2_free(ptr) };
}
