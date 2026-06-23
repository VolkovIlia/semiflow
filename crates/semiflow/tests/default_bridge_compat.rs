//! AC-11: `ChernoffFunction::apply_into` and `ApplyChernoffExt::apply_chernoff` (v3.0).
//!
//! In v3.0 the `ChernoffFunction` trait requires `apply_into` directly.
//! The `apply_chernoff` extension method from `ApplyChernoffExt` provides the
//! allocating interface `(tau, &src) → Result<S, E>` for callers with `S: Clone`.
//!
//! This test verifies:
//!   1. A minimal `ChernoffFunction` impl (only `apply_into`, `order`, `growth`)
//!      compiles and runs correctly.
//!   2. `apply_chernoff` produces the same result as manual `apply_into` + clone.
//!
//! Reference: ADR-0043, ADR-0074.

use semiflow_core::{
    chernoff::{ApplyChernoffExt, ChernoffFunction, Growth},
    error::SemiflowError,
    scratch::ScratchPool,
    Grid1D, GridFn1D,
};

// ---------------------------------------------------------------------------
// Mock: identity-scale operator (apply_into = scale all values by constant)
// ---------------------------------------------------------------------------

/// `IdentityScaleChernoff(k)`: `apply_into` scales each value by `k`.
/// Minimal v3.0 impl: only `apply_into`, `order`, `growth`.
struct IdentityScaleChernoff {
    k: f64,
    #[allow(dead_code)]
    grid: Grid1D<f64>,
}

impl ChernoffFunction<f64> for IdentityScaleChernoff {
    type S = GridFn1D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "IdentityScaleChernoff: tau < 0",
                value: tau,
            });
        }
        let k = self.k;
        for (o, &v) in dst.values.iter_mut().zip(src.values.iter()) {
            *o = v * k;
        }
        Ok(())
    }

    fn order(&self) -> u32 {
        1
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn apply_chernoff_matches_apply_into() {
    let grid = Grid1D::new(-1.0, 1.0, 16).unwrap();
    let op = IdentityScaleChernoff { k: 2.5, grid };
    let src = GridFn1D::from_fn(grid, |x| x * x);
    let tau = 0.1_f64;

    // Reference: via `apply_into` with pre-allocated dst.
    let mut expected = GridFn1D::from_fn(grid, |_| 0.0);
    let mut scratch = ScratchPool::new();
    op.apply_into(tau, &src, &mut expected, &mut scratch)
        .expect("apply_into failed");

    // Test: via `apply_chernoff` extension method.
    let result = op.apply_chernoff(tau, &src).expect("apply_chernoff failed");

    // Must be bit-identical (same arithmetic path).
    assert_eq!(
        expected.values, result.values,
        "apply_chernoff must produce bit-identical output to apply_into"
    );
}

#[test]
fn apply_into_preserves_dst_len() {
    let grid = Grid1D::new(-1.0, 1.0, 8).unwrap();
    let op = IdentityScaleChernoff { k: 1.0, grid };
    let src = GridFn1D::from_fn(grid, |x| x);
    let mut dst = GridFn1D::from_fn(grid, |_| 999.0);
    let mut scratch = ScratchPool::new();
    op.apply_into(0.01, &src, &mut dst, &mut scratch)
        .expect("apply_into failed");
    assert_eq!(dst.values.len(), 8, "dst len must be preserved");
}
