//! Proptest — Reduction lemma + `AxisLift` 1D consistency (v0.5.0, ADR-0012).
//!
//! Two proptest blocks:
//!
//! **Block 1 — Y-reduction lemma (100 cases)**
//! For `f(x,y) = g(x)` (Y-independent), `AxisLift<_, Axis::X>::apply(τ, f)`
//! must row-equal `DiffusionChernoff.apply(τ, row_j(f))` for every j.
//! `AxisLift<_, Axis::Y>::apply(τ, f)` must be identity within 16 ULP/node.
//! Invariant I-T4 (tensor.yaml) and Lemma 10.2 (math.md §10.4).
//!
//! **Block 2 — `AxisLift` 1D consistency (200 cases)**
//! For random 2D input `f(x,y) = g(x)·h(y)`, `AxisLift<_, Axis::X>::apply`
//! row-equals per-row 1D `apply` within ≤16 ULP/node; symmetrically for Y.
//! Direct realisation of `axis_lift_1d_consistency` (properties.yaml, 200 cases).
//!
//! Reference: `contracts/semiflow-core.tensor.yaml` §3 I-T4,
//! `contracts/semiflow-core.properties.yaml` `axis_lift_1d_consistency`,
//! `contracts/semiflow-core.math.md` §10.4 Lemma 10.2.

use proptest::prelude::*;
use semiflow::{
    chernoff::ApplyChernoffExt, Axis, AxisLift, DiffusionChernoff, Grid1D, Grid2D, GridFn1D,
    GridFn2D, State,
};

// ---------------------------------------------------------------------------
// Block 1 — Y-reduction lemma  (100 cases)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    /// Y-reduction lemma (Lemma 10.2, math.md §10.4, invariant I-T4):
    ///
    /// For Y-independent input `f(x, y) = g(x)`:
    ///   - X-lift row-equals 1D `DiffusionChernoff.apply(τ, row_j)` for every j.
    ///   - Y-lift is identity within 16 ULP/node.
    #[test]
    fn axis_lift_y_reduction_lemma(
        g_amp in 0.5f64..=2.0f64,
        g_mu  in -2.0f64..=2.0f64,
        g_sig in 0.3f64..=1.5f64,
        tau   in 0.001f64..=0.1f64,
        nx    in 16usize..=32usize,
        ny    in 16usize..=32usize,
    ) {
        let gx = Grid1D::new(-10.0, 10.0, nx).unwrap();
        let gy = Grid1D::new(-10.0, 10.0, ny).unwrap();
        let grid2d = Grid2D::new(gx, gy);

        // Y-independent input: f(x, y) = g(x).
        let g = {
            let sig2 = g_sig * g_sig;
            move |x: f64| g_amp * (-((x - g_mu).powi(2)) / (2.0 * sig2)).exp()
        };
        let f2d = GridFn2D::from_separable(grid2d, g, |_| 1.0_f64);

        // 1D reference: g on grid.x.
        let f1d = GridFn1D::from_fn(gx, g);

        // Constant-a ζ-A Chernoff: a=0.5, a'=0, a''=0.
        let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
        let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

        // --- X-lift: must row-equal 1D apply ---
        let lift_x = AxisLift::new(cx.clone(), Axis::X);
        let evolved_2d_x = lift_x.apply_chernoff(tau, &f2d).unwrap();
        let evolved_1d = cx.apply_chernoff(tau, &f1d).unwrap();

        let tol = 16.0 * f64::EPSILON * (1.0 + evolved_1d.norm_sup());
        for j in 0..ny {
            let row_j = evolved_2d_x.row(j);
            for i in 0..nx {
                let diff = (row_j.values[i] - evolved_1d.values[i]).abs();
                prop_assert!(
                    diff <= tol,
                    "X-lift row inconsistency (I-T4): (i={}, j={}): diff={:.4e}, tol={:.4e}",
                    i, j, diff, tol
                );
            }
        }

        // --- Y-lift on Y-independent input: must be identity ---
        let lift_y = AxisLift::new(cy, Axis::Y);
        let evolved_2d_y = lift_y.apply_chernoff(tau, &f2d).unwrap();
        let mut diff_y = evolved_2d_y.clone();
        diff_y.axpy(-1.0, &f2d);
        let tol_y = 16.0 * f64::EPSILON * (1.0 + f2d.norm_sup());
        prop_assert!(
            diff_y.norm_sup() <= tol_y,
            "Y-lift on Y-independent input is not identity (Lemma 10.2): \
             diff={:.4e}, tol={:.4e}",
            diff_y.norm_sup(), tol_y
        );
    }
}

// ---------------------------------------------------------------------------
// Block 2 — AxisLift 1D consistency  (200 cases)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// AxisLift 1D consistency (properties.yaml `axis_lift_1d_consistency`, 200 cases):
    ///
    /// For tensor input `f(x, y) = g(x) · h(y)`:
    ///   - `AxisLift<_, X>::apply(τ, f)` row j equals `cx.apply(τ, row_j(f))`
    ///     within ≤16 ULP/node.
    ///   - `AxisLift<_, Y>::apply(τ, f)` col i equals `cy.apply(τ, col_i(f))`
    ///     within ≤16 ULP/node.
    #[test]
    fn axis_lift_1d_consistency(
        g_amp in 0.5f64..=2.0f64,
        g_mu  in -2.0f64..=2.0f64,
        h_amp in 0.5f64..=2.0f64,
        h_mu  in -2.0f64..=2.0f64,
        tau   in 0.001f64..=0.1f64,
        nx    in 16usize..=32usize,
        ny    in 16usize..=32usize,
    ) {
        let gx = Grid1D::new(-10.0, 10.0, nx).unwrap();
        let gy = Grid1D::new(-10.0, 10.0, ny).unwrap();
        let grid2d = Grid2D::new(gx, gy);

        let g = move |x: f64| g_amp * (-(x - g_mu).powi(2)).exp();
        let h = move |y: f64| h_amp * (-(y - h_mu).powi(2)).exp();
        let f2d = GridFn2D::from_separable(grid2d, g, h);

        // Constant-a ζ-A Chernoff per axis.
        let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
        let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

        // --- X-axis consistency ---
        let lift_x = AxisLift::new(cx.clone(), Axis::X);
        let out_x = lift_x.apply_chernoff(tau, &f2d).unwrap();

        for j in 0..ny {
            let row_in = f2d.row(j);
            let row_evolved = cx.apply_chernoff(tau, &row_in).unwrap();
            let row_out = out_x.row(j);
            let tol = 16.0 * f64::EPSILON * (1.0 + row_evolved.norm_sup());
            for i in 0..nx {
                let diff = (row_out.values[i] - row_evolved.values[i]).abs();
                prop_assert!(
                    diff <= tol,
                    "X-axis consistency violated at (i={}, j={}): diff={:.4e}, tol={:.4e}",
                    i, j, diff, tol
                );
            }
        }

        // --- Y-axis consistency ---
        let lift_y = AxisLift::new(cy.clone(), Axis::Y);
        let out_y = lift_y.apply_chernoff(tau, &f2d).unwrap();

        for i in 0..nx {
            let col_in = f2d.col(i);
            let col_evolved = cy.apply_chernoff(tau, &col_in).unwrap();
            let col_out = out_y.col(i);
            let tol = 16.0 * f64::EPSILON * (1.0 + col_evolved.norm_sup());
            for j in 0..ny {
                let diff = (col_out.values[j] - col_evolved.values[j]).abs();
                prop_assert!(
                    diff <= tol,
                    "Y-axis consistency violated at (i={}, j={}): diff={:.4e}, tol={:.4e}",
                    i, j, diff, tol
                );
            }
        }
    }
}
