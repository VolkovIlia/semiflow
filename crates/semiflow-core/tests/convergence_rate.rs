//! G3 — Convergence-rate test (empirical log-log slope >= 0.95).
//!
//! Runs the Gaussian heat-kernel test for n in {25, 50, 100, 200, 400},
//! records err(n) = sup-norm error vs oracle, fits a log-log least-squares
//! line on (log n, log err), and asserts slope <= -0.95 (order >= 0.95).
//!
//! Reference: inequality (9) of Theorem 6 (Remizov 2025) predicts O(1/n)
//! convergence, which gives log-log slope -1. The gate is relaxed to -0.95
//! to tolerate grid discretisation and boundary artifacts.

use semiflow_core::{ChernoffSemigroup, Grid1D, GridFn1D, ShiftChernoff1D};

const N_VALUES: [usize; 5] = [25, 50, 100, 200, 400];

/// Compute sup-norm error of Chernoff(n) vs. oracle at t=1.
fn error_at_n(n_steps: usize) -> f64 {
    let grid = Grid1D::new(-10.0, 10.0, 1000).unwrap();
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let chernoff = ShiftChernoff1D::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.0, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps).unwrap();
    let u_t = semigroup.evolve(1.0, &f0).unwrap();

    let inv_sqrt3 = (3.0_f64).sqrt().recip();
    let mut max_err: f64 = 0.0;
    for i in 0..u_t.values.len() {
        let x = grid.x_at(i);
        let oracle = inv_sqrt3 * (-(x * x) / 3.0).exp();
        let err = (u_t.values[i] - oracle).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

/// Least-squares slope of points (log n, log err) using simple formula.
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
// sum_x/sum_y/sum_xy: OLS variable names by convention; n/ns are grid counts ≤ 2^16
fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();

    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|&x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();

    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

#[test]
fn g3_convergence_slope() {
    let errs: Vec<f64> = N_VALUES.iter().map(|&n| error_at_n(n)).collect();

    for (n, e) in N_VALUES.iter().zip(errs.iter()) {
        println!("n={n:4}, err={e:.4e}");
    }

    let slope = log_log_slope(&N_VALUES, &errs);
    println!("log-log slope = {slope:.4}");

    assert!(
        slope <= -0.95,
        "G3 FAIL: convergence slope {slope:.4} > -0.95 (order < 0.95)"
    );
}
