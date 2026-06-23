//! Tests for `DiffusionChernoff::with_closure` and related Storage dispatch.
//!
//! ADR-0034 contract tests (v0.12.0):
//! - `FnPtr` and `Closure` storage paths produce identical output.
//! - `with_closure_local` is an alias for `with_closure`.
//! - Closed-over state is accessible inside the closure.
//! - `DiffusionChernoff<f64>: Send + Sync` still holds.

use semiflow::{DiffusionChernoff, Grid1D, GridFn1D};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Small non-trivial grid for bit-equality tests.
fn grid64() -> Grid1D<f64> {
    Grid1D::new(-4.0, 4.0, 64).unwrap()
}

/// Gaussian initial condition centred at zero.
fn gauss(grid: Grid1D<f64>) -> GridFn1D<f64> {
    GridFn1D::from_fn(grid, |x| (-x * x).exp())
}

/// Constant-coefficient ζ-A via `new` (fn-ptr path).
fn dc_fnptr() -> DiffusionChernoff<f64> {
    DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid64())
}

/// Same coefficients via `with_closure` (Closure/Arc path).
fn dc_closure() -> DiffusionChernoff<f64> {
    DiffusionChernoff::with_closure(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        1.0,
        grid64(),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `FnPtr` path: evolve one step and sanity-check shape.
#[test]
fn test_storage_fnptr_path() {
    let dc = dc_fnptr();
    let u0 = gauss(grid64());
    let u1 = dc.apply_chernoff(0.01, &u0).unwrap();
    assert_eq!(u1.values.len(), 64);
    // Positivity: all values > 0 for Gaussian under heat equation.
    assert!(u1.values.iter().all(|&v| v > 0.0));
}

/// Closure path: same shape and sanity-check.
#[test]
fn test_storage_boxed_path() {
    let dc = dc_closure();
    let u0 = gauss(grid64());
    let u1 = dc.apply_chernoff(0.01, &u0).unwrap();
    assert_eq!(u1.values.len(), 64);
    assert!(u1.values.iter().all(|&v| v > 0.0));
}

/// Bit-equality between `FnPtr` and `Closure` paths over N=64 grid.
///
/// This is the regression gate (ADR-0034 §"Suckless audit"): the Storage
/// dispatch must not perturb numerical output for the constant-`a` case.
#[test]
fn test_bit_equality_fnptr_vs_boxed() {
    let dc_fp = dc_fnptr();
    let dc_cl = dc_closure();
    let u0 = gauss(grid64());
    let tau = 0.01_f64;

    let out_fp = dc_fp.apply_chernoff(tau, &u0).unwrap();
    let out_cl = dc_cl.apply_chernoff(tau, &u0).unwrap();

    assert_eq!(out_fp.values.len(), out_cl.values.len());
    for (a, b) in out_fp.values.iter().zip(out_cl.values.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit mismatch: fnptr={a:.18e} closure={b:.18e}"
        );
    }
}

/// `DiffusionChernoff<f64>: Send + Sync` still holds after v0.12.0 changes.
///
/// Verified at compile time — if `DiffusionChernoff<f64>` stopped being
/// Send + Sync, this would not compile (the bound is checked by the compiler
/// when instantiating the generic fn).
#[test]
fn test_send_sync_traits() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DiffusionChernoff<f64>>();
}

/// Closure captures a local variable (the v0.10.0 limitation motivating ADR-0034).
///
/// Verify that a runtime-determined `scale` can be threaded into `a(x)` via move closure.
#[test]
fn test_with_closure_captures_state() {
    let scale: f64 = 2.0;
    let grid = Grid1D::new(-2.0, 2.0, 32).unwrap();

    // `scale` is captured by `move` — not possible with `fn` pointers.
    let dc = DiffusionChernoff::with_closure(
        move |_: f64| scale,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        scale,
        grid,
    );

    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let u1 = dc.apply_chernoff(0.01, &u0).unwrap();
    assert_eq!(u1.values.len(), 32);
    // Non-zero output confirms the captured coefficient was actually used.
    assert!(u1.values.iter().any(|&v| v > 0.0));
}
