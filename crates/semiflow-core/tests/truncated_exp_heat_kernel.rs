//! G3-truncated-exp — heat-equation diagonal convergence with `TruncatedExpDiffusionChernoff` (v0.4.0).
//!
//! Tests global O(τ²) order of `TruncatedExpDiffusionChernoff` by co-refining spatial
//! and temporal grids together (diagonal refinement), which is required because
//! the truncated-exp CFL constraint `2τ·a < dx²` couples the two grids.
//!
//! ## Setup
//!
//! PDE:  `∂_t u = a ∂_xx u`, `a = 0.5`, `u(x,0) = cos(π x / L_half)`.
//! Domain: `[-1, 1]` (L=2).  `L_half = 1`.
//! Exact oracle: `u(x, T) = exp(-π²·a·T) cos(π x)`.
//!
//! Diagonal refinement: for each n, use `N = n` spatial nodes on `[-1, 1]`.
//! Then `dx = 2/(n-1)`, `τ = T/n`, and:
//!   `CFL_factor = 2τ·a/dx² = 2·(T/n)·0.5 / (2/(n-1))² ≈ T·(n-1)²/(4n)`
//!   ≈ T·n/4 for large n. With T=0.02 and n ≤ 200: `CFL_factor` ≤ 0.02·200/4 = 1.
//!
//! T=0.02 is chosen so CFL is satisfied for n ∈ {32, 64, 128} (factor ≤ 0.64).
//! n=256 gives `CFL_factor` ≈ 0.98 (borderline — excluded from `N_VALUES`).
//!
//! ## Gate: G3-truncated-exp
//!
//! Log-log slope ≤ −1.95 over n ∈ {32, 64, 128} (diagonal refinement).
//!
//! Reference: contracts/semiflow-core.math.md §9.2.3.C, ADR-0011.

use core::f64::consts::PI;

use semiflow_core::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExpDiffusionChernoff};

// ---------------------------------------------------------------------------
// Gate constants
// ---------------------------------------------------------------------------

const SLOPE_GATE: f64 = -1.95;
/// Diagonal refinement: `N_VALUES` is both n (steps) and N (nodes).
const N_VALUES: [usize; 3] = [32, 64, 128];
const T_FINAL: f64 = 0.02;
const A_CONST: f64 = 0.5;
const X_MIN: f64 = -1.0;
const X_MAX: f64 = 1.0;

// ---------------------------------------------------------------------------
// Oracle: u(x, T) = exp(-π²·a·T) · cos(πx)
// ---------------------------------------------------------------------------

fn oracle(x: f64, t: f64) -> f64 {
    (-(PI * PI) * A_CONST * t).exp() * (PI * x).cos()
}

// ---------------------------------------------------------------------------
// Sup-norm error at n steps (diagonal: N = n)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)] // n ≤ 256, step count ≤ N_VALUES max; well within f64 mantissa
fn error_at_n(n: usize) -> f64 {
    let tau = T_FINAL / n as f64;
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid valid");
    let dx = grid.dx();
    let dx2 = dx * dx;

    // Verify CFL (should always hold by construction; panic if not).
    let cfl = 2.0 * tau * A_CONST / dx2;
    assert!(
        cfl < 1.0,
        "CFL violated for n={n}: 2τa/dx²={cfl:.4} >= 1 (T={T_FINAL}, a={A_CONST})"
    );
    eprintln!("  n={n:4}: tau={tau:.4e}, dx={dx:.4e}, CFL={cfl:.4}");

    let trunc_exp =
        TruncatedExpDiffusionChernoff::new(|_| A_CONST, |_| 0.0_f64, |_| 0.0_f64, A_CONST, grid);

    let mut state = GridFn1D::from_fn(grid, |x| (PI * x).cos());
    for _ in 0..n {
        state = trunc_exp
            .apply_chernoff(tau, &state)
            .expect("trunc_exp apply ok");
    }

    let mut max_err: f64 = 0.0;
    for i in 0..n {
        let x = grid.x_at(i);
        let err = (state.values[i] - oracle(x, T_FINAL)).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Log-log slope (OLS, verbatim from convergence_rate_strang.rs)
// ---------------------------------------------------------------------------

#[allow(clippy::similar_names, clippy::cast_precision_loss)]
// sum_x/sum_y/sum_xy: OLS variable names by convention; ns ≤ N_VALUES max
fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|&x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// G3-truncated-exp
// ---------------------------------------------------------------------------

/// G3-truncated-exp: `TruncatedExpDiffusionChernoff` heat-kernel slope ≤ −1.95.
///
/// Diagonal refinement (N = n): CFL satisfied, O(dx²) = O(τ²) → slope −2.
#[test]
#[allow(clippy::cast_precision_loss)] // n ≤ 256 in N_VALUES; well within f64 mantissa
fn g3_truncated_exp_heat_kernel_slope() {
    eprintln!("G3-truncated-exp: diagonal refinement heat-kernel test (T={T_FINAL}, a={A_CONST})");
    let errs: Vec<f64> = N_VALUES.iter().map(|&n| error_at_n(n)).collect();

    eprintln!("{:>6}  {:>12}  {:>12}", "n", "tau", "sup_err");
    for (&n, &e) in N_VALUES.iter().zip(errs.iter()) {
        eprintln!("{n:>6}  {:.6e}  {e:.6e}", T_FINAL / n as f64);
    }

    let slope = log_log_slope(&N_VALUES, &errs);
    eprintln!("G3-truncated-exp: log-log slope = {slope:.4}  (gate: <= {SLOPE_GATE})");

    assert!(
        slope <= SLOPE_GATE,
        "G3-truncated-exp FAIL: slope {slope:.4} > {SLOPE_GATE} (order < 1.95) — Gate FAILED"
    );
}
