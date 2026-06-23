//! `G4_NS2D_aniso` — empirical convergence-rate test (slope ≤ -1.95).
//!
//! math.md §10.7-ter.7; ADR-0023; gate `G4_NS2D_aniso`.
//!
//! Gated: `#[cfg(feature = "slow-tests")]`
//!
//! Validates order-2 (spatial) convergence of `NonSeparable2DAnisotropicChernoff`
//! via log-log OLS slope using **spatial self-convergence** (coarse n vs fine 2n−1).
//!
//! # Design
//! Same methodology as `G3_NS2D_var` (`strang_nonseparable_slope.rs`):
//! - Domain `[-5, 5]²`, `T = 0.5`, `N_STEPS = 500` time steps (fixed temporal;
//!   large step count keeps temporal error negligible and satisfies CFL at all grids).
//! - β(x,y) = 0.05 · exp(-(x²+y²)/4) — non-constant; exercises `[A,[B,M_β]]`.
//! - Initial datum: `u_0(x,y) = exp(-(x²+y²))`.
//! - Per-axis legs: `DiffusionChernoff` (ζ-A) with `a = 0.1`.
//! - `n_spatial ∈ {32, 64, 128, 256}` — probe set (n=32 diagnostic only).
//! - For each probe n, compare against `run_to_T(2n−1)` subsampled at coarse nodes:
//!   `error(n) = sup_{i,j} |u_n[i,j] − u_{2n−1}[2i, 2j]|`.
//! - Slope computed on `{64, 128, 256}` (skip n=32, pre-asymptotic per `G3_NS2D_var`).
//! - Variable β rules out a closed-form oracle; self-convergence cancels model mismatch.
//!
//! # Why self-convergence instead of a high-resolution reference
//! A single `N_ref=1024` reference has residual error ~1e-3. At n=256 the probe
//! error reaches that floor, reversing the apparent convergence (slope → 0 or positive).
//! Self-convergence (each comparison local: coarse vs next-finer) eliminates the
//! reference floor entirely, so the spatial O(dx²) rate is measured cleanly.
//!
//! # CFL safety
//! Worst case: finest comparison grid is `2·256−1 = 511` nodes.
//! dx = 10 / (511−1) ≈ 0.01961, dx² ≈ 3.845e-4.
//! τ = 0.5 / 500 = 1.0e-3. CFL: `4·τ·β_norm` = 4·1e-3·0.05 = 2.0e-4 < 3.845e-4. ✓
//!
//! # Gate
//! `slope ≤ −1.95` (NON-NEGOTIABLE, math.md §10.7-ter.7, ADR-0023).

#![cfg(feature = "slow-tests")]

use semiflow::{
    chernoff::ApplyChernoffExt, BoundaryPolicy, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    NonSeparable2DAnisotropicChernoff,
};

// ---------------------------------------------------------------------------
// Gate constants
// ---------------------------------------------------------------------------

const SLOPE_GATE: f64 = -1.95;
/// Probe `n_spatial` values; n=32 included as diagnostic, slope on {64,128,256}.
const N_SPATIAL: [usize; 4] = [32, 64, 128, 256];
/// Fixed Chernoff steps — large enough to make temporal error negligible and
/// satisfy CFL at the finest self-convergence grid (2·256−1 = 511).
const N_STEPS: usize = 500;
const T_FINAL: f64 = 0.5;
const AXX: f64 = 0.1;
const AYY: f64 = 0.1;
const X_MIN: f64 = -5.0;
const X_MAX: f64 = 5.0;
/// Supremum of β(x,y) = 0.05·exp(-(x²+y²)/4) over all (x,y).
const BETA_NORM: f64 = 0.05;

// ---------------------------------------------------------------------------
// Function pointers
// ---------------------------------------------------------------------------

fn axx_fn(_: f64) -> f64 {
    AXX
}
fn ayy_fn(_: f64) -> f64 {
    AYY
}
/// β(x,y) = 0.05 · exp(-(x²+y²)/4) — smooth, bounded by `BETA_NORM`, non-constant.
/// Exercises the `[A,[B,M_β]]` commutator (math.md §10.7-ter.7, ADR-0023).
fn beta_fn(x: f64, y: f64) -> f64 {
    0.05 * (-(x * x + y * y) / 4.0).exp()
}

// ---------------------------------------------------------------------------
// Runner: evolve n-node grid for N_STEPS steps, return final GridFn2D.
// ---------------------------------------------------------------------------

/// Evolve `NonSeparable2DAnisotropicChernoff` on an `n×n` grid for `N_STEPS`
/// Chernoff steps, starting from `u_0(x,y) = exp(-(x²+y²))`. Returns the
/// final state as a `GridFn2D` (values in row-major order: index = j*n + i).
// n ≤ 511 — well within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn run_to_t(n: usize) -> GridFn2D {
    let tau = T_FINAL / N_STEPS as f64;
    let gx = Grid1D::new(X_MIN, X_MAX, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic);
    let gy = gx;
    let grid = Grid2D::new(gx, gy);
    let ix = DiffusionChernoff::new(axx_fn, |_| 0.0, |_| 0.0, AXX, gx);
    let iy = DiffusionChernoff::new(ayy_fn, |_| 0.0, |_| 0.0, AYY, gy);
    let op = NonSeparable2DAnisotropicChernoff::new(ix, iy, beta_fn, BETA_NORM, grid).unwrap();
    let mut u = GridFn2D::from_fn(grid, |x, y| (-x * x - y * y).exp());
    for _ in 0..N_STEPS {
        u = op.apply_chernoff(tau, &u).unwrap();
    }
    u
}

// ---------------------------------------------------------------------------
// Self-convergence error
// ---------------------------------------------------------------------------

/// Spatial self-convergence error for probe size `n`.
///
/// Runs the operator at resolution `n` and at the finer resolution `2n−1`.
/// The fine grid has `n2 = 2n−1` nodes; coarse node `(i, j)` corresponds to
/// fine node `(2i, 2j)` (every other node, zero-offset). Returns
/// `sup_{i,j} |u_n[i,j] − u_{2n−1}[2i, 2j]|`.
fn self_conv_error(n: usize) -> f64 {
    let u_coarse = run_to_t(n);
    let n2 = 2 * n - 1;
    let u_fine = run_to_t(n2);
    let mut sup = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            let k_c = j * n + i;
            let k_f = (j * 2) * n2 + (i * 2);
            let err = (u_coarse.values[k_c] - u_fine.values[k_f]).abs();
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

// test grids ≤ 511 nodes — within f64 52-bit mantissa range.
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
// G4_NS2D_aniso — variable anisotropic β, spatial self-convergence
// ---------------------------------------------------------------------------

/// `G4_NS2D_aniso`: spatial slope ≤ -1.95 for `NonSeparable2DAnisotropicChernoff`
/// with β(x,y) = 0.05·exp(-(x²+y²)/4). Uses self-convergence (coarse n vs
/// fine 2n−1) — no oracle needed; eliminates the high-resolution reference floor.
/// Slope computed on {64, 128, 256}; n=32 included as diagnostic only.
// n_spatial ≤ 256; fine grids ≤ 511 — within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
#[test]
fn g4_ns2d_aniso_slope() {
    // CFL sanity: finest self-convergence grid is 2*256-1 = 511 nodes.
    // dx_finest = (X_MAX - X_MIN) / (511 - 1), tau = T_FINAL / N_STEPS.
    let n_finest: usize = 2 * *N_SPATIAL.last().unwrap() - 1;
    let tau = T_FINAL / N_STEPS as f64;
    let dx_finest = (X_MAX - X_MIN) / (n_finest - 1) as f64;
    let tau_cfl_max = dx_finest * dx_finest / (4.0 * BETA_NORM);
    assert!(
        tau < tau_cfl_max,
        "tau={tau:.3e} violates CFL at finest grid n={n_finest} (tau_max={tau_cfl_max:.3e})"
    );

    // Compute self-convergence errors for all probe sizes.
    let errs: Vec<f64> = N_SPATIAL.iter().map(|&n| self_conv_error(n)).collect();

    // Print diagnostics for all probe sizes.
    for (&n, &e) in N_SPATIAL.iter().zip(&errs) {
        let dx = (X_MAX - X_MIN) / (n - 1) as f64;
        println!("G4_NS2D_aniso: n_spatial={n:4}, dx={dx:.4}, self_err={e:.4e}");
    }

    // Slope on {64, 128, 256} — skip n=32 (pre-asymptotic, per G3_NS2D_var convention).
    let slope = loglog_slope(&N_SPATIAL[1..], &errs[1..]);
    println!("G4_NS2D_aniso: slope = {slope:.4}  (gate <= {SLOPE_GATE})");

    assert!(
        slope <= SLOPE_GATE,
        "G4_NS2D_aniso slope {slope:.4} > gate {SLOPE_GATE}",
    );
}
