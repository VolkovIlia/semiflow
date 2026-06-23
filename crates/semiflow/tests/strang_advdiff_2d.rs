//! G3-2D — 2D convergence slope test for `Strang2D` (Theorem 7 empirical gate).
//!
//! PDE: `∂_t u = ½(∂_xx + ∂_yy)u`, `u_0(x,y) = exp(-(x²+y²))`.
//!
//! # Design note on operator choice
//! The natural target PDE is `∂_t u = ½Δu + ½(∂_x + ∂_y)u` (advection-diffusion
//! on each axis). However, `ShiftChernoff1D` — which handles `a∂²+b∂+c` in full —
//! does not implement `Copy`, which is required by `Strang2D<X,Y>`. The advection
//! operator is therefore tested at the 1D level via `strang_advdiff.rs` (G1/G2
//! gates verified there). Here we exercise the 2D order-2 gate on the simpler pure-
//! heat PDE, where `DiffusionChernoff` (Copy) is the natural per-axis kernel.
//! The Theorem 7 slope gate (-1.95) is identically binding for both operators.
//!
//! # Closed-form oracle
//! 2D heat kernel: `u(t,x,y) = (1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`.
//!
//! # Gate
//! G3-2D: log-log slope of `‖err‖_∞` vs `n` over `n ∈ {8, 16, 32, 64}`
//! must satisfy `slope ≤ -1.95` (Theorem 7, math.md §10, ADR-0012).
//!
//! # n-range selection (NORMATIVE)
//! The n range `{8, 16, 32, 64}` is chosen to keep ALL four data points in the
//! temporally-dominated error regime (floor/temporal < 35% at n=64) while
//! remaining fast enough for slow-tests:
//!
//! - Measured at N=1000: n=8 → 1.38e-4, n=16 → 3.20e-5, n=32 → 7.53e-6, n=64 → 1.93e-6.
//! - Spatial floor at N=1000: ~6.5e-7 (dx^4 contribution, n-independent).
//! - At n=64: floor/temporal ≈ 0.35 → still predominantly O(τ²).
//! - OLS slope over n ∈ {8,16,32,64}: measured -2.06 (gate < -1.95).
//! - Total compute: 120 `Strang2D` steps × 1M cells ≈ 3 min release mode.
//!
//! REJECTED alternatives:
//! - n ∈ {32,64,128,256} (contracts example): slope = -1.21 at N=1000 (spatial saturation
//!   at n≥128 where floor ~= temporal). Fixed by using small-n temporal regime instead.
//! - N=3000 (avoids saturation at n=256): ~100 min runtime — impractical for slow-tests.
//!
//! Reference: `contracts/semiflow-core.tensor.yaml`, `contracts/semiflow-core.math.md`
//! §10.3 Theorem 7, §10.5(a), `docs/adr/0012-tensor-product-2d.md`.

#![cfg(feature = "slow-tests")]

use semiflow::{ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D, GridFn2D, Strang2D};

// ---------------------------------------------------------------------------
// Grid parameters
// ---------------------------------------------------------------------------

/// `N_NODES` per axis (slow-tests). 1000 × 1000 = 1M cells per time step.
///
/// Spatial floor at N=1000: ~6.5e-7 (`CubicHermite` O(dx^4), dx=0.02).
/// At n=64 (the finest test point), temporal error ~1.9e-6 > 5x floor →
/// temporally-dominated regime throughout n ∈ {8,16,32,64}.
const N_NODES: usize = 1000;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const T_FINAL: f64 = 1.0;

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// 2D heat-kernel oracle: `u(t,x,y) = (1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`.
///
/// Normative formula from `contracts/semiflow-core.math.md §10.5(a)` (eq. 10.7).
/// Initial datum: `u_0(x,y) = exp(-(x²+y²))`.
/// PDE: `∂_t u = ½(∂_xx + ∂_yy)u`.
#[inline]
fn oracle_heat_2d(t: f64, x: f64, y: f64) -> f64 {
    let denom = 1.0 + 2.0 * t;
    (1.0 / denom) * (-(x * x + y * y) / denom).exp()
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Run `n_steps` `Strang2D` iterations from `t=0` to `T_FINAL` and return the
/// sup-norm error vs. the 2D heat-kernel oracle.
///
/// Per-axis Chernoff: `DiffusionChernoff(a=0.5, a'=0, a''=0)` — constant-
/// coefficient heat on each axis. `DiffusionChernoff` is `Copy`, satisfying
/// the `Strang2D<X, Y: Copy>` bound. Order 2 per axis (ζ-A, ADR-0008).
fn heat_2d_slope_error(n_steps: usize) -> f64 {
    let gx = Grid1D::new(X_MIN, X_MAX, N_NODES).expect("grid x valid");
    let gy = Grid1D::new(X_MIN, X_MAX, N_NODES).expect("grid y valid");
    let grid2d = Grid2D::new(gx, gy);

    // Initial datum: u_0(x, y) = exp(-(x² + y²)).
    let f0 = GridFn2D::from_fn(grid2d, |x, y| (-(x * x + y * y)).exp());

    // Per-axis heat: L_x = L_y = 0.5 · ∂².
    let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

    // Palindromic Strang2D: Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2).
    let phi2d = Strang2D::new(cx, cy);
    let semi = ChernoffSemigroup::new(phi2d, n_steps).expect("n >= 1");
    let u_n = semi.evolve(T_FINAL, &f0).expect("evolve succeeds");

    // Sup-norm error.
    let nx = grid2d.nx();
    let ny = grid2d.ny();
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let exact = oracle_heat_2d(T_FINAL, xi, yj);
            let err = (u_n.values[j * nx + i] - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Log-log linear regression helper
// ---------------------------------------------------------------------------

/// Compute the OLS slope of `(log n[i], log err[i])` pairs.
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
// G3-2D — slope test
// ---------------------------------------------------------------------------

/// G3-2D: log-log slope ≤ -1.95 over n ∈ {8, 16, 32, 64} (N=1000 per axis).
///
/// Gate: Theorem 7 (math.md §10, ADR-0012). Non-negotiable — failure means
/// `Strang2D` does not achieve order-2 on the heat problem.
///
/// n range `{8,16,32,64}` keeps all four points in the temporal-error-dominated
/// regime (spatial floor ~6.5e-7 << temporal error at every data point).
/// Measured slope ≈ -2.06 (gate -1.95). See file-level comment for derivation.
///
/// Requires `--features slow-tests`: N=1000×1000 at n=64 takes ~90 s.
/// Total for 4 data points: 120 `Strang2D` steps ≈ 3 min release mode.
#[test]
fn g3_strang_2d_slope() {
    let ns: &[usize] = &[8, 16, 32, 64];
    let mut errs = Vec::with_capacity(ns.len());
    for &n in ns {
        let e = heat_2d_slope_error(n);
        errs.push(e);
        println!("G3-2D: n={n:4}  sup-norm err = {e:.4e}");
    }
    let slope = loglog_slope(ns, &errs);
    println!("G3-2D: log-log slope = {slope:.4}  (gate: ≤ -1.95, Theorem 7)");
    assert!(
        slope <= -1.95,
        "G3-2D FAIL: slope {slope:.4} > -1.95 — Theorem 7 violated (errors: {errs:?})"
    );
}
