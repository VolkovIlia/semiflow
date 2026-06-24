//! `G_KAHLER_CURV` — CP¹ Fubini-Study `ManifoldChernoff` curvature-correction slope gate.
//!
//! Gate (ADR-0129, v7.0.0 `G_KAHLER_CURV)`:
//!   `FubiniStudyCp1` backend; R = 2 (constant); `ManifoldChernoff` with R/12 correction.
//!   Self-convergence slope ≤ −1.95 (order-2 spatial, mirror `G4_NS2D_aniso` pattern).
//!
//! # Methodology — self-convergence in equatorial belt
//!
//! The FS stereographic chart (u,v) has conformal factor σ = 2/(1+r²) that varies
//! enormously across the chart. Near r = 0 (south pole), σ ≈ 2 and the GH-5
//! support (√τ·(1+r²) ≈ √τ) is small relative to the grid spacing at standard n,
//! putting the scheme in the pre-asymptotic regime there. Near r = 2 the FS scale
//! matches the S² equatorial scale (2√τ), and the scheme enters the O(h²) regime.
//!
//! **Initial datum**: `exp(−8·(r−2)²)` — an annular Gaussian ring at r ≈ 2,
//! concentrated where the GH support (√τ·5 ≈ 0.177) spans multiple cells even at
//! coarse n=32 (h ≈ 0.19 on L=3, GH/h ≈ 1.9). This ensures the O(h²) spatial
//! convergence regime is active over most of the function support.
//!
//! **Self-convergence**: for each n, compare to the 2n−1-grid result at matching
//! nodes in the annular evaluation region 1 ≤ r ≤ 2. This eliminates the oracle
//! dependency and the temporal floor (both coarse and fine grids share the same
//! temporal error, which cancels in the difference).
//!
//! **OLS slope** on n ∈ {32, 64, 128, 256}: gate ≤ −1.95.
//!   n=32 is included even though it is pre-asymptotic near r<1 (the evaluation
//!   region r ∈ [1,2] excludes that area); the OLS averages over the transition.
//!
//! # GH coverage analysis (τ=0.00125, L=3)
//! At r=2: scale = √τ·(1+4) = 0.035·5 = 0.177.
//!   n=32  (h=0.194): GH/h = 1.84  ✓ (asymptotic)
//!   n=64  (h=0.095): GH/h = 3.74  ✓
//!   n=128 (h=0.047): GH/h = 7.53  ✓
//!   n=256 (h=0.024): GH/h = 14.8  ✓
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS; n ≤ 256 ≤ 2^52
#![allow(clippy::similar_names)] // nx_c/nx_f/ny_c are math index names
#![allow(clippy::manual_range_contains)] // `r2 < MIN || r2 > MAX` exclusion is clearer

use semiflow::{
    ChernoffFunction, FubiniStudyCp1, Grid1D, Grid2D, GridFn2D, ManifoldChernoff, ScratchPool,
};

// ─── Gate constants ───────────────────────────────────────────────────────────

const T_HORIZON: f64 = 0.05;
const N_CHERNOFF: usize = 40; // τ = T/N = 0.00125
const SLOPE_GATE: f64 = -1.95;
const N_CHART_SWEEP: [usize; 4] = [32, 64, 128, 256];
const L: f64 = 3.0;
/// Annular Gaussian: exp(−8·(r−R0)²) concentrated at r ≈ R0.
const R0: f64 = 2.0;
const KAPPA: f64 = 8.0;
/// Annular evaluation band: r² ∈ [`R_EVAL_MIN_SQ`, `R_EVAL_MAX_SQ`].
/// At r=R0=2: GH max displacement (0.357) < L-R0 = 1 → boundary-safe.
/// Inner limit 1.0 avoids the near-origin pre-asymptotic regime.
const R_EVAL_MIN_SQ: f64 = 1.0;
const R_EVAL_MAX_SQ: f64 = 4.0;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Annular Gaussian concentrated at r ≈ R0. The function is essentially zero
/// at r = 0 (exp(−32) ≈ 2e-14) and at r = L = 3 (exp(−8) ≈ 3e-4).
fn initial_datum(u: f64, v: f64) -> f64 {
    let r = (u * u + v * v).sqrt();
    let dr = r - R0;
    (-KAPPA * dr * dr).exp()
}

fn build_grid(n: usize) -> Grid2D<f64> {
    let axis = Grid1D::new(-L, L, n).unwrap();
    Grid2D::new(axis, axis)
}

/// Run `N_CHERNOFF` Chernoff steps; return final `GridFn2D`.
fn run_to_t(n: usize) -> GridFn2D<f64> {
    let tau = T_HORIZON / N_CHERNOFF as f64;
    let grid = build_grid(n);
    let backend = FubiniStudyCp1::new();
    let chernoff = ManifoldChernoff::new(backend, true);
    let mut u = GridFn2D::from_fn(grid, initial_datum);
    let mut dst = u.clone();
    let mut scratch = ScratchPool::new();
    for _ in 0..N_CHERNOFF {
        chernoff
            .apply_into(tau, &u, &mut dst, &mut scratch)
            .unwrap();
        core::mem::swap(&mut u, &mut dst);
    }
    u
}

/// Self-convergence error: sup_{annular band} |`u_n` − u_{2n−1}| at matching nodes.
///
/// Co-aligned grids: `Grid1D::new(−L, L, n)[i]` = −L + i·2L/(n−1).
///   Fine (2n−1): −L + 2i·2L/(2(n−1)) = −L + i·2L/(n−1) = Coarse[i]. ✓
#[allow(clippy::cast_precision_loss)]
fn self_conv_error(n: usize) -> f64 {
    let n_fine = 2 * n - 1;
    let u_coarse = run_to_t(n);
    let u_fine = run_to_t(n_fine);
    let nx_c = u_coarse.grid.nx();
    let ny_c = u_coarse.grid.ny();
    let nx_f = u_fine.grid.nx();
    let mut max_err = 0.0f64;
    for j in 1..ny_c - 1 {
        for i in 1..nx_c - 1 {
            let uu = u_coarse.grid.x.x_at(i);
            let vv = u_coarse.grid.y.x_at(j);
            let r2 = uu * uu + vv * vv;
            if r2 < R_EVAL_MIN_SQ || r2 > R_EVAL_MAX_SQ {
                continue;
            }
            let val_c = u_coarse.values[j * nx_c + i];
            let val_f = u_fine.values[(2 * j) * nx_f + (2 * i)];
            let err = (val_c - val_f).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// ─── OLS slope ────────────────────────────────────────────────────────────────

#[allow(clippy::cast_precision_loss)]
fn ols_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let log_x: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let log_y: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let mean_x = log_x.iter().sum::<f64>() / m;
    let mean_y = log_y.iter().sum::<f64>() / m;
    let num: f64 = log_x
        .iter()
        .zip(log_y.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_x.iter().map(|x| (x - mean_x).powi(2)).sum();
    num / den
}

// ─── G_KAHLER_CURV ────────────────────────────────────────────────────────────

/// `G_KAHLER_CURV` — curvature-corrected `ManifoldChernoff`<`FubiniStudyCp1`> slope ≤ −1.95.
///
/// Self-convergence in annular belt r ∈ [1, 2] on n ∈ {32, 64, 128, 256}.
/// OLS slope in log(err) vs log(n). Gate: slope ≤ −1.95.
///
/// A slope steeper than ≈ −2.3 signals measurement artefact; investigate.
#[test]
#[ignore = "slow flagship gate; run with: cargo run -p xtask -- test-flagship"]
fn g_kahler_curv() {
    let mut errs = Vec::with_capacity(N_CHART_SWEEP.len());
    for &n in &N_CHART_SWEEP {
        let err = self_conv_error(n);
        println!("G_KAHLER_CURV: n={n:4} → self_conv_err={err:.4e}");
        errs.push(err);
    }
    let slope = ols_slope(&N_CHART_SWEEP, &errs);
    println!("G_KAHLER_CURV: slope={slope:.4}  (gate ≤ {SLOPE_GATE})");
    assert!(
        slope <= SLOPE_GATE,
        "G_KAHLER_CURV FAIL: slope={slope:.4} > {SLOPE_GATE}. errs={errs:?}",
    );
}
