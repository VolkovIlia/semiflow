//! Proptest — 4 consistency properties for `NonSeparable2DChernoff` (ADR-0016).
//!
//! **P1 — zero-c bit-equality with `Strang2D`** (200 cases):
//!   `NonSeparable2DChernoff` with `c ≡ 0` must equal `Strang2D` output
//!   exactly (bit-for-bit) on random inputs.
//!
//! **P2 — shape preservation** (200 cases):
//!   `apply` must return a `GridFn2D` with the same `nx*ny` length.
//!
//! **P3 — linearity** (100 cases):
//!   `apply(τ, α·f + β·g) = α·apply(τ, f) + β·apply(τ, g)` within 1e-10.
//!
//! **P4 — CFL boundary** (50 cases):
//!   For `τ` just above the CFL threshold, `apply` returns `CflViolated`.
//!
//! Reference: `contracts/semiflow-core.properties.yaml` Block C properties.

use proptest::prelude::*;
use semiflow_core::{
    chernoff::ApplyChernoffExt, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    NonSeparable2DChernoff, SemiflowError, Strang2D,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_inner_grids(n: usize) -> (Grid1D, Grid1D) {
    let g = Grid1D::new(-1.0, 1.0, n).unwrap();
    (g, g)
}

// ix/iy and ix2/iy2 are parallel operator pairs for the two arms of the split.
#[allow(clippy::similar_names)]
fn make_ops(
    n: usize,
) -> (
    NonSeparable2DChernoff<DiffusionChernoff, DiffusionChernoff>,
    Strang2D<DiffusionChernoff, DiffusionChernoff>,
) {
    let (gx, gy) = make_inner_grids(n);
    let ix = DiffusionChernoff::new(|_| 0.4, |_| 0.0, |_| 0.0, 0.4, gx);
    let iy = DiffusionChernoff::new(|_| 0.3, |_| 0.0, |_| 0.0, 0.3, gy);
    let ix2 = DiffusionChernoff::new(|_| 0.4, |_| 0.0, |_| 0.0, 0.4, gx);
    let iy2 = DiffusionChernoff::new(|_| 0.3, |_| 0.0, |_| 0.0, 0.3, gy);
    let grid = Grid2D::new(
        Grid1D::new(-1.0, 1.0, n).unwrap(),
        Grid1D::new(-1.0, 1.0, n).unwrap(),
    );
    let ns = NonSeparable2DChernoff::new(ix, iy, |_, _| 0.0, 0.0, grid).unwrap();
    let s3 = Strang2D::new(ix2, iy2);
    (ns, s3)
}

fn grid2d(n: usize) -> Grid2D {
    Grid2D::new(
        Grid1D::new(-1.0, 1.0, n).unwrap(),
        Grid1D::new(-1.0, 1.0, n).unwrap(),
    )
}

// ---------------------------------------------------------------------------
// P1 — zero-c bit-equality with Strang2D
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// P1: `NonSeparable2DChernoff` with `c≡0` must match `Strang2D` bit-for-bit.
    #[test]
    fn zero_c_equals_strang2d(
        amp in 0.5f64..=2.0f64,
        tau in 1e-4f64..=5e-3f64,
    ) {
        let n = 12_usize;
        let (ns, s3) = make_ops(n);
        let f = GridFn2D::from_fn(grid2d(n), |x, y| amp * (-(x*x + y*y)).exp());
        let out_ns = ns.apply_chernoff(tau, &f).unwrap();
        let out_s3 = s3.apply_chernoff(tau, &f).unwrap();
        for k in 0..n*n {
            prop_assert_eq!(
                out_ns.values[k].to_bits(),
                out_s3.values[k].to_bits(),
                "bit mismatch at k={}: ns={}, s3={}",
                k, out_ns.values[k], out_s3.values[k]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// P2 — shape preservation
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// P2: output length == nx * ny.
    #[test]
    fn shape_preserved(
        amp in 0.5f64..=2.0f64,
        tau in 1e-4f64..=5e-3f64,
    ) {
        let n = 10_usize;
        let (gx, gy) = make_inner_grids(n);
        let ix = DiffusionChernoff::new(|_| 0.4, |_| 0.0, |_| 0.0, 0.4, gx);
        let iy = DiffusionChernoff::new(|_| 0.3, |_| 0.0, |_| 0.0, 0.3, gy);
        let grid = grid2d(n);
        let op = NonSeparable2DChernoff::new(ix, iy, |_, _| 0.01, 0.01, grid).unwrap();
        let f = GridFn2D::from_fn(grid, |x, y| amp * (x + y).sin());
        let out = op.apply_chernoff(tau, &f).unwrap();
        prop_assert_eq!(out.values.len(), n * n);
    }
}

// ---------------------------------------------------------------------------
// P3 — linearity
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    /// P3: `apply(τ, α·f + β·g) ≈ α·apply(τ,f) + β·apply(τ,g)` within 1e-10.
    #[test]
    fn linearity(
        alpha in -2.0f64..=2.0f64,
        beta  in -2.0f64..=2.0f64,
        tau   in 1e-4f64..=5e-3f64,
    ) {
        let n = 8_usize;
        let (gx, gy) = make_inner_grids(n);
        let ix = DiffusionChernoff::new(|_| 0.4, |_| 0.0, |_| 0.0, 0.4, gx);
        let iy = DiffusionChernoff::new(|_| 0.3, |_| 0.0, |_| 0.0, 0.3, gy);
        let ix2 = DiffusionChernoff::new(|_| 0.4, |_| 0.0, |_| 0.0, 0.4, gx);
        let iy2 = DiffusionChernoff::new(|_| 0.3, |_| 0.0, |_| 0.0, 0.3, gy);
        let grid1 = grid2d(n);
        let grid2 = grid2d(n);
        let op1 = NonSeparable2DChernoff::new(ix, iy, |_, _| 0.01, 0.01, grid1).unwrap();
        let op2 = NonSeparable2DChernoff::new(ix2, iy2, |_, _| 0.01, 0.01, grid2).unwrap();
        let f = GridFn2D::from_fn(grid2d(n), |x, y| (x - y).sin());
        let g = GridFn2D::from_fn(grid2d(n), |x, y| (x * y).cos());
        // α·f + β·g
        let mut fg = f.clone();
        fg.axpy(alpha, &f);   // fg = f (axpy adds alpha*f to self; reset first)
        // Actually construct alpha*f + beta*g directly
        let mut combo = GridFn2D::from_fn(grid2d(n), |_, _| 0.0);
        for k in 0..n*n { combo.values[k] = alpha * f.values[k] + beta * g.values[k]; }
        let out_combo = op1.apply_chernoff(tau, &combo).unwrap();
        let out_f = op2.apply_chernoff(tau, &f).unwrap();
        let out_g = op1.apply_chernoff(tau, &g).unwrap();
        for k in 0..n*n {
            let lhs = out_combo.values[k];
            let rhs = alpha * out_f.values[k] + beta * out_g.values[k];
            prop_assert!(
                (lhs - rhs).abs() < 1e-10,
                "linearity at k={k}: lhs={lhs}, rhs={rhs}, diff={}",
                (lhs - rhs).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// P4 — CFL boundary
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 50, ..ProptestConfig::default() })]

    /// P4: For τ strictly above the CFL threshold, `apply` returns `CflViolated`.
    #[test]
    fn cfl_boundary_returned(
        excess in 1.01f64..=2.0f64,
    ) {
        let n = 8_usize;
        let (gx, gy) = make_inner_grids(n);
        let ix = DiffusionChernoff::new(|_| 0.4, |_| 0.0, |_| 0.0, 0.4, gx);
        let iy = DiffusionChernoff::new(|_| 0.3, |_| 0.0, |_| 0.0, 0.3, gy);
        let grid = grid2d(n);
        let c_norm = 1.0_f64;
        let op = NonSeparable2DChernoff::new(ix, iy, |_, _| 1.0, c_norm, grid).unwrap();
        let dx = Grid1D::new(-1.0, 1.0, n).unwrap().dx();
        let dx_dy = dx * dx; // square grid so dx == dy
        // τ at which 4 * τ * c_norm == dx*dy → threshold
        let tau_threshold = dx_dy / (4.0 * c_norm);
        let tau_violated = tau_threshold * excess;
        let f = GridFn2D::from_fn(grid, |_, _| 1.0);
        let res = op.apply_chernoff(tau_violated, &f);
        prop_assert!(
            matches!(res, Err(SemiflowError::CflViolated { .. })),
            "expected CflViolated for tau={tau_violated}, got {:?}",
            res
        );
    }
}
