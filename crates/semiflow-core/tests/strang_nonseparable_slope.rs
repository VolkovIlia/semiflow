//! `G3_NS2D` + `G3_NS2D_var` — empirical convergence-rate tests (slope ≤ −1.95).
//!
//! Gated: `#[cfg(feature = "slow-tests")]`
//!
//! Validates order-2 (spatial) convergence of `NonSeparable2DChernoff` via
//! log-log OLS slope with fixed `N=1000` Chernoff steps (matching the G3-2D
//! gate methodology from `strang_advdiff_2d.rs`).
//!
//! **`G3_NS2D`** — constant coupling `c = 0.05`, `axx = ayy = 0.1`.
//!   Uses oracle error vs. rotated-Gaussian closed form.
//!   `n_spatial` ∈ {32, 64, 128, 256}.
//!
//! **`G3_NS2D_var`** — variable coupling `c(x,y) = 0.05·tanh(x+y)`.
//!   Uses spatial self-convergence (coarse n vs fine 2n−1, same c) to avoid
//!   oracle mismatch from the variable coupling.
//!   `n_spatial` ∈ {64, 128, 256} — n=32 is pre-asymptotic for the tanh coupling
//!   (the cross-stencil error at n=32 is dominated by boundary effects from the
//!   non-uniform coupling gradient, not the underlying O(dx²) rate).
//!
//! # Design
//! Same methodology as the existing G3-2D gate for `Strang2D`:
//! - Fix N large (N=1000) so temporal error is negligible.
//! - Vary `n_spatial` to probe spatial convergence.
//! - For const-c: measure oracle error (rotated Gaussian).
//! - For var-c: measure spatial self-convergence (no oracle — avoids model mismatch).
//! - Slope in `log(n_spatial)` vs log(sup-err) must be ≤ −1.95.
//!
//! The 4-point cross-stencil is O(dx^2 + dy^2), so spatial convergence ≈ -2. ✓
//!
//! # CFL
//! Worst case: n=64 (var-c), domain [-6,6], dx=12/63≈0.190, dx*dy≈0.036.
//! N=1000 steps, T=0.5: tau=5e-4. CFL: 4*tau*c=4*5e-4*0.05=1e-4 << 0.036. ✓ (All safe.)
//!
//! # Gate
//! `slope ≤ −1.95` (NON-NEGOTIABLE, math.md §10.7-bis, ADR-0016).

#![cfg(feature = "slow-tests")]

use semiflow_core::{
    chernoff::ApplyChernoffExt, BoundaryPolicy, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    NonSeparable2DChernoff,
};

// ---------------------------------------------------------------------------
// Gate constants
// ---------------------------------------------------------------------------

const SLOPE_GATE: f64 = -1.95;

/// `n_spatial` values for const-c oracle test (full 4-point range).
const N_SPATIAL: [usize; 4] = [32, 64, 128, 256];
/// `n_spatial` values for var-c self-convergence test.
/// Starts at 64 to skip the pre-asymptotic n=32 regime where the tanh
/// coupling gradient introduces non-negligible boundary effects.
const N_SPATIAL_VAR: [usize; 3] = [64, 128, 256];
/// Fixed Chernoff steps (large → temporal error negligible).
const N_STEPS: usize = 1000;
const T_FINAL: f64 = 0.5;
const AXX: f64 = 0.1;
const AYY: f64 = 0.1;
const X_MIN: f64 = -6.0;
const X_MAX: f64 = 6.0;

// ---------------------------------------------------------------------------
// Oracle (rotated Gaussian, 2x2 covariance propagation)
// ---------------------------------------------------------------------------

fn sigma_t_ns(axx: f64, ayy: f64, c: f64, t: f64) -> [f64; 3] {
    [0.5 + 2.0 * t * axx, t * c, 0.5 + 2.0 * t * ayy]
}

fn det2_ns(s: &[f64; 3]) -> f64 {
    s[0] * s[2] - s[1] * s[1]
}

// c, t, x, y, q are standard mathematical notation for the CEV/heat oracle.
#[allow(clippy::many_single_char_names)]
fn oracle_ns(axx: f64, ayy: f64, c: f64, t: f64, x: f64, y: f64) -> f64 {
    if t == 0.0 {
        return (-x * x - y * y).exp();
    }
    let sig = sigma_t_ns(axx, ayy, c, t);
    let det_t = det2_ns(&sig);
    let det_0 = 0.25_f64;
    let norm = (det_0 / det_t).sqrt();
    let q = sig[2] * x * x - 2.0 * sig[1] * x * y + sig[0] * y * y;
    norm * (-0.5 * q / det_t).exp()
}

// ---------------------------------------------------------------------------
// Function pointers
// ---------------------------------------------------------------------------

fn axx_fn(_: f64) -> f64 {
    AXX
}
fn ayy_fn(_: f64) -> f64 {
    AYY
}
fn c_fn_const(_: f64, _: f64) -> f64 {
    0.05
}
fn c_fn_var(x: f64, y: f64) -> f64 {
    0.05 * (x + y).tanh()
}

// ---------------------------------------------------------------------------
// Runner: evolve n_spatial-node grid for N_STEPS, return sup-error
// ---------------------------------------------------------------------------

/// Run the operator on an n_spatial-node grid for `N_STEPS`, return values + grid.
// N_STEPS = 1000 — well within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn run_values(n_spatial: usize, c_fn: fn(f64, f64) -> f64, c_norm: f64) -> (Vec<f64>, Grid1D) {
    let tau = T_FINAL / N_STEPS as f64;
    let gx = Grid1D::new(X_MIN, X_MAX, n_spatial)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic);
    let gy = gx;
    let grid = Grid2D::new(gx, gy);
    let ix = DiffusionChernoff::new(axx_fn, |_| 0.0, |_| 0.0, AXX, gx);
    let iy = DiffusionChernoff::new(ayy_fn, |_| 0.0, |_| 0.0, AYY, gy);
    let op = NonSeparable2DChernoff::new(ix, iy, c_fn, c_norm, grid).unwrap();
    let mut u = GridFn2D::from_fn(grid, |x, y| (-x * x - y * y).exp());
    for _ in 0..N_STEPS {
        u = op.apply_chernoff(tau, &u).unwrap();
    }
    (u.values, gx)
}

/// Run and compare against closed-form oracle (for constant c).
fn run_error_oracle(
    n_spatial: usize,
    c_fn: fn(f64, f64) -> f64,
    c_norm: f64,
    c_oracle: f64,
) -> f64 {
    let (vals, gx) = run_values(n_spatial, c_fn, c_norm);
    let mut sup = 0.0_f64;
    for j in 0..n_spatial {
        let yj = gx.x_at(j);
        for i in 0..n_spatial {
            let xi = gx.x_at(i);
            let k = j * n_spatial + i;
            let exact = oracle_ns(AXX, AYY, c_oracle, T_FINAL, xi, yj);
            let err = (vals[k] - exact).abs();
            if err > sup {
                sup = err;
            }
        }
    }
    sup
}

/// Spatial self-convergence: compare `n_spatial` solution to 2*`n_spatial` solution
/// subsampled at matching nodes. No oracle needed — cancels model mismatch.
fn run_error_self(n_spatial: usize, c_fn: fn(f64, f64) -> f64, c_norm: f64) -> f64 {
    let (u_coarse, _) = run_values(n_spatial, c_fn, c_norm);
    let (u_fine, _) = run_values(n_spatial * 2 - 1, c_fn, c_norm);
    // Fine grid has nodes at x_i = X_MIN + i*dx_fine, i=0..2n-2.
    // Coarse node i corresponds to fine node 2*i (every other node).
    let mut sup = 0.0_f64;
    let n2 = n_spatial * 2 - 1;
    for j in 0..n_spatial {
        for i in 0..n_spatial {
            let k_c = j * n_spatial + i;
            let k_f = (j * 2) * n2 + (i * 2);
            let err = (u_coarse[k_c] - u_fine[k_f]).abs();
            if err > sup {
                sup = err;
            }
        }
    }
    sup
}

// ---------------------------------------------------------------------------
// OLS log-log slope
// ---------------------------------------------------------------------------

// test grids ≤ 2¹⁶ nodes — all values within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn loglog_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let log_n: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let log_e: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
    let mean_x = log_n.iter().sum::<f64>() / m;
    let mean_y = log_e.iter().sum::<f64>() / m;
    let num: f64 = log_n
        .iter()
        .zip(log_e.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_n.iter().map(|x| (x - mean_x).powi(2)).sum();
    num / den
}

// ---------------------------------------------------------------------------
// G3_NS2D — constant coupling
// ---------------------------------------------------------------------------

/// `G3_NS2D`: spatial-order slope ≤ -1.95 for `NonSeparable2D` with const c=0.05.
/// Methodology follows G3-2D (`strang_advdiff_2d.rs`): fixed N=1000, vary `n_spatial`.
// n_spatial ≤ 256 — within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
#[test]
fn g3_ns2d_constant_c_slope() {
    let mut errs = Vec::with_capacity(N_SPATIAL.len());
    for &n in &N_SPATIAL {
        let e = run_error_oracle(n, c_fn_const, 0.05, 0.05);
        println!(
            "G3_NS2D const: n_spatial={n:4}, dx={:.4}, err={e:.4e}",
            (X_MAX - X_MIN) / (n - 1) as f64
        );
        errs.push(e);
    }
    let slope = loglog_slope(&N_SPATIAL, &errs);
    println!("G3_NS2D (const c): slope = {slope:.4}  (gate <= {SLOPE_GATE})");
    assert!(
        slope <= SLOPE_GATE,
        "G3_NS2D slope {slope:.4} > gate {SLOPE_GATE}"
    );
}

// ---------------------------------------------------------------------------
// G3_NS2D_var — variable coupling
// ---------------------------------------------------------------------------

/// `G3_NS2D_var`: spatial-order slope ≤ -1.95 for c(x,y)=0.05*tanh(x+y).
/// Uses spatial self-convergence (coarse n vs fine 2n−1) to avoid oracle mismatch.
/// Starts at n=64 to skip the pre-asymptotic regime at n=32.
// n_spatial ≤ 256 — within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
#[test]
fn g3_ns2d_variable_c_slope() {
    let mut errs = Vec::with_capacity(N_SPATIAL_VAR.len());
    for &n in &N_SPATIAL_VAR {
        let e = run_error_self(n, c_fn_var, 0.05);
        println!(
            "G3_NS2D_var: n_spatial={n:4}, dx={:.4}, self_err={e:.4e}",
            (X_MAX - X_MIN) / (n - 1) as f64
        );
        errs.push(e);
    }
    let slope = loglog_slope(&N_SPATIAL_VAR, &errs);
    println!("G3_NS2D_var: slope = {slope:.4}  (gate <= {SLOPE_GATE})");
    assert!(
        slope <= SLOPE_GATE,
        "G3_NS2D_var slope {slope:.4} > gate {SLOPE_GATE}"
    );
}
