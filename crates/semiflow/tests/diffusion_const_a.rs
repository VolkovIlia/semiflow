//! Tests for `DiffusionChernoff::new_const_a` fast path (v0.13.0, Wave D, D1).
//!
//! Gate: `CONST_A_BIT_EQUAL` — output of `new_const_a` must be bit-identical to
//! `new` with `|_| 0.0` derivative args for the same `a_value`, grid, and `tau`.

use semiflow_core::{DiffusionChernoff, Grid1D, GridFn1D};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Small non-trivial grid for bit-equality tests.
fn grid64() -> Grid1D<f64> {
    Grid1D::new(-4.0, 4.0, 64).unwrap()
}

/// Gaussian initial condition.
fn gauss(grid: Grid1D<f64>) -> GridFn1D<f64> {
    GridFn1D::from_fn(grid, |x| (-x * x).exp())
}

/// Constant-coefficient `DiffusionChernoff` via `new` (fn-ptr, zero derivatives).
fn dc_fnptr(a: f64) -> DiffusionChernoff<f64> {
    // Cannot capture `a` in fn-pointer; use closure path for parametric a.
    DiffusionChernoff::with_closure(
        move |_: f64| a,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        a,
        grid64(),
    )
}

/// Same constant coefficient via `new_const_a` (`ConstA` fast path, D1).
fn dc_const_a(a: f64) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new_const_a(a, a, grid64())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `new_const_a` sanity check: shape and positivity preserved.
#[test]
fn test_const_a_shape_and_sign() {
    let dc = dc_const_a(1.0);
    let u0 = gauss(grid64());
    let u1 = dc.apply_chernoff(0.01, &u0).unwrap();
    assert_eq!(u1.values.len(), 64);
    assert!(u1.values.iter().all(|&v| v > 0.0));
}

/// `is_const_a` returns `true` for `new_const_a` and `false` for `new`.
#[test]
fn test_is_const_a_flag() {
    let dc_fast = dc_const_a(1.0);
    let dc_slow = dc_fnptr(1.0);
    assert!(dc_fast.is_const_a(), "new_const_a should set is_const_a");
    assert!(!dc_slow.is_const_a(), "new should not set is_const_a");
}

/// `CONST_A_BIT_EQUAL` gate: bit-identical output between `new_const_a`
/// and `new` with zero derivative closures for `a=1.0`.
///
/// This is the regression gate (D1, v0.13.0): the `ConstA` fast path must
/// produce numerically identical results to the general path with `a'=0, a''=0`.
#[test]
fn test_const_a_bit_equal_to_fnptr() {
    let a_val = 1.0_f64;
    let tau = 0.01_f64;
    let dc_fast = dc_const_a(a_val);
    let dc_slow = dc_fnptr(a_val);
    let u0 = gauss(grid64());

    let out_fast = dc_fast.apply_chernoff(tau, &u0).unwrap();
    let out_slow = dc_slow.apply_chernoff(tau, &u0).unwrap();

    assert_eq!(out_fast.values.len(), out_slow.values.len());
    for (i, (a, b)) in out_fast
        .values
        .iter()
        .zip(out_slow.values.iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "CONST_A_BIT_EQUAL FAIL at node {i}: fast={a:.18e} slow={b:.18e}"
        );
    }
}

/// `CONST_A_BIT_EQUAL` for `a=2.5` (non-unit constant, wider variance).
#[test]
fn test_const_a_bit_equal_a25() {
    let a_val = 2.5_f64;
    let tau = 0.005_f64;
    let dc_fast = dc_const_a(a_val);
    let dc_slow = dc_fnptr(a_val);
    let u0 = gauss(grid64());

    let out_fast = dc_fast.apply_chernoff(tau, &u0).unwrap();
    let out_slow = dc_slow.apply_chernoff(tau, &u0).unwrap();

    for (i, (a, b)) in out_fast
        .values
        .iter()
        .zip(out_slow.values.iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "CONST_A_BIT_EQUAL (a=2.5) FAIL at node {i}: fast={a:.18e} slow={b:.18e}"
        );
    }
}

/// `Clone` of `ConstA` preserves path: cloned instance still uses fast path.
#[test]
fn test_const_a_clone_preserves_variant() {
    let dc = dc_const_a(1.0);
    let dc2 = dc.clone();
    assert!(dc2.is_const_a(), "cloned ConstA should still be ConstA");
    let u0 = gauss(grid64());
    let out = dc2.apply_chernoff(0.01, &u0).unwrap();
    assert_eq!(out.values.len(), 64);
}
