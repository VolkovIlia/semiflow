//! Liouville oracle tests for `Diffusion4thChernoff` (v0.6.0, ADR-0013).
//!
//! Three sub-tests:
//!
//! ## 1. `spatial_convergence_variable_a_liouville`
//!
//! HEADLINE test: dx-sweep for variable-coefficient diffusion.
//! Fixed n=4000 (τ≈1.25e-4, τ²≈1.56e-8 — temporal error negligible at all N tested).
//! N ∈ {400, 800, 1600} — asymptotic 4th-order regime (justified below).
//!
//! PDE: `u_t = ∂_x((1+γx)²·∂_x u)`, γ=0.05. Exact solution via Liouville transform.
//!
//! Both `Diffusion4thChernoff` (ζ⁴) and `DiffusionChernoff` (ζ-A, v0.5.0) are
//! measured. Gates:
//! - `Diffusion4thChernoff`: slope ≤ −3.85.
//! - `DiffusionChernoff`:    slope ≤ −1.85.
//! - At each N: D4th error ≤ D2 error × 1.05 (no regression).
//!
//! **Implementation note (honest documentation)**:
//! For γ=0.05, the τ²-scaled ζ correction (magnitude ~ τ²·a·a'·f''' ~ 10⁻⁸)
//! is negligible vs. the K-kernel's O(dx⁴) spatial error at all N tested.
//! Consequently D4th and D2 converge at the SAME rate (≈ -4); the FD stencil
//! upgrade from O(Δ²) to O(Δ⁶) is not distinguishable at γ=0.05.
//! Both slopes satisfy their gates: D4th ≤ -3.85 and D2 ≤ -1.85 (a -4 slope
//! satisfies both). The per-N ordering (D4th ≤ D2 × 1.05) is verified.
//!
//! Selected N range: {400, 800, 1600}.
//! Pre-test calibration (run at n=4000, T=0.5):
//!   N=200:  err≈1.95e-3 (order ~2.4, pre-asymptotic — excluded)
//!   N=400:  err≈3.78e-4 (entering asymptotic, ratio 5.2×)
//!   N=800:  err≈4.14e-5 (asymptotic, ratio 9.1×)
//!   N=1600: err≈1.32e-6 (asymptotic, ratio 31×)
//!   N=3200: err≈2.25e-7 (temporal floor, ratio 5.9× — excluded: tau² dominates)
//! Reasoning: N={400,800,1600} are all in the dx-monotone-decreasing regime;
//! slope over this range is ≈ -4.08, confirming K-kernel 4th-order behaviour.
//!
//! ## 2. `temporal_consistency_variable_a`
//!
//! Sanity check: absolute error < 5e-2 at single (N=200, n=128) point.
//! NOT the headline gate — see `spatial_convergence_variable_a_liouville`.
//!
//! ## 3. `oracle_at_small_t_matches_ic`
//!
//! Self-consistency: oracle at T=1e-4 ≈ IC.
//!
//! ## Liouville Oracle
//!
//! Change of variable y = ln(1+γx)/γ maps `u_t = ∂_x((1+γx)²·∂_x u)` to
//! the standard heat equation `v_t = v_yy` on y-space, enabling an exact
//! Gaussian solution via heat-kernel convolution.

use core::cell::Cell;
use std::f64::consts::PI;

use semiflow::{
    boundary::InterpKind, chernoff::ApplyChernoffExt, Diffusion4thChernoff, DiffusionChernoff,
    Grid1D, GridFn1D,
};

const GAMMA: f64 = 0.05;
const SIGMA: f64 = 1.0;
const T: f64 = 0.5;
const X_MIN: f64 = -15.0;
const X_MAX: f64 = 15.0;
const N_QUAD: usize = 16000;
const Y_MAX: f64 = 14.0;

thread_local! {
    static G_CELL: Cell<f64> = const { Cell::new(0.05) };
}

fn a_fn(x: f64) -> f64 {
    let g = G_CELL.with(Cell::get);
    let v = 1.0 + g * x;
    v * v
}
fn ap_fn(x: f64) -> f64 {
    let g = G_CELL.with(Cell::get);
    2.0 * g * (1.0 + g * x)
}
fn app_fn(x: f64) -> f64 {
    let g = G_CELL.with(Cell::get);
    let _ = x;
    2.0 * g * g
}

/// Liouville transform oracle for `u_t = ∂_x((1+γx)² ∂_x u)`.
///
/// Change of variable y = ln(1+γx)/γ maps the PDE to `v_t = v_yy`.
/// Heat-kernel convolution in y-space gives the exact solution.
#[allow(clippy::cast_precision_loss)] // N_QUAD = 16000 ≤ 2^16; k ≤ N_QUAD; well within f64 mantissa
fn liouville_oracle(gamma: f64, t: f64, x: f64) -> f64 {
    let y_query = (1.0 + gamma * x).ln() / gamma;
    let dy = 2.0 * Y_MAX / (N_QUAD - 1) as f64;
    let inv = 1.0 / (4.0 * PI * t).sqrt();
    let mut integral = 0.0_f64;

    for k in 0..N_QUAD {
        let yp = -Y_MAX + k as f64 * dy;
        let xp = ((gamma * yp).exp() - 1.0) / gamma;
        let w0 = (0.5 * gamma * yp).exp() * (-(xp * xp) / (2.0 * SIGMA * SIGMA)).exp();
        let dz = y_query - yp;
        let kern = (-(dz * dz) / (4.0 * t)).exp() * inv;
        let weight = if k == 0 || k == N_QUAD - 1 { 0.5 } else { 1.0 };
        integral += weight * kern * w0 * dy;
    }

    let a_factor = (1.0 + gamma * x).powf(-0.5);
    let decay = (-gamma * gamma * t / 4.0).exp();
    a_factor * decay * integral
}

/// OLS slope: log(err) ~ slope·log(N) + const.
/// Negative slope → err decreases as N increases (convergent).
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
// sum_x/sum_y/sum_xy: OLS variable names by convention; ns.len() ≤ N_SWEEP max
fn log_log_slope_n(ns: &[f64], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let lx: Vec<f64> = ns.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = lx.iter().sum();
    let sum_y: f64 = ly.iter().sum();
    let sum_xx: f64 = lx.iter().map(|&v| v * v).sum();
    let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

/// Run one spatial grid and return (`err_d4`, `err_d2`) vs. Liouville oracle.
///
/// Helper for `spatial_convergence_variable_a_liouville` to stay ≤50 lines.
#[allow(clippy::cast_precision_loss)] // n_spatial ≤ 1600, N_FIXED_STEPS = 4000; well within f64 mantissa
fn run_one_liouville_grid(
    n_spatial: usize,
    tau: f64,
    a_norm: f64,
    n_fixed_steps: usize,
) -> (f64, f64) {
    // Pin to CubicHermite: the slope gate -3.85 was calibrated with CubicHermite
    // K-kernel sampling (the pre-v6.0 default). v6.0 changed Grid1D::new default
    // to SepticHermite (ADR-0109); pin here so the gate remains valid.
    let grid = Grid1D::new(X_MIN, X_MAX, n_spatial)
        .expect("grid valid")
        .with_interp(InterpKind::CubicHermite);
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-(x * x) / (2.0 * SIGMA * SIGMA)));
    G_CELL.with(|c| c.set(GAMMA));
    let d4 = Diffusion4thChernoff::new(a_fn, ap_fn, app_fn, a_norm, grid);
    let d2 = DiffusionChernoff::new(a_fn, ap_fn, app_fn, a_norm, grid);
    let mut s4 = f0.clone();
    let mut s2 = f0.clone();
    for _ in 0..n_fixed_steps {
        G_CELL.with(|c| c.set(GAMMA));
        s4 = d4.apply_chernoff(tau, &s4).expect("d4 apply");
        s2 = d2.apply_chernoff(tau, &s2).expect("d2 apply");
    }
    let mut err_d4 = 0.0_f64;
    let mut err_d2 = 0.0_f64;
    for i in 0..s4.values.len() {
        let x = grid.x_at(i);
        let exact = liouville_oracle(GAMMA, T, x);
        err_d4 = err_d4.max((s4.values[i] - exact).abs());
        err_d2 = err_d2.max((s2.values[i] - exact).abs());
    }
    (err_d4, err_d2)
}

/// Spatial dx-sweep, both D4th (ζ⁴) and D2 (ζ-A), variable-a Liouville.
///
/// Fixed n=4000 (τ=T/n=1.25e-4, τ²≈1.56e-8 — temporal error negligible).
/// N ∈ {400, 800, 1600} — asymptotic 4th-order regime (calibrated; see module docstring).
///
/// Gates:
/// - D4th slope ≤ −3.85 (variable-a; measured ≈ −4.08, K-kernel dominated).
/// - D2  slope ≤ −1.85 (regression guard; measured ≈ −4.08, both schemes converge).
/// - Per-N: D4th error ≤ D2 error × 1.05 (no regression vs `DiffusionChernoff`).
///
/// Domain [−15, 15] keeps ±3·dx FD stencil inside grid for all N. T=0.5.
#[test]
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
// n_spatial ≤ 1600; err_d4/err_d2 vs errs_d4/errs_d2: conventional D4/D2 naming suffix
fn spatial_convergence_variable_a_liouville() {
    // N range: {400, 800, 1600} — asymptotic 4th-order regime only.
    // N=200 excluded: pre-asymptotic (order ~2.4). N=3200 excluded: temporal floor.
    const N_FIXED_STEPS: usize = 4000;
    const N_SWEEP: [usize; 3] = [400, 800, 1600];
    const SLOPE_GATE_D4: f64 = -3.85;
    const SLOPE_GATE_D2: f64 = -1.85;

    G_CELL.with(|c| c.set(GAMMA));
    let a_norm = (1.0 + GAMMA * X_MAX.abs()).powi(2);
    let tau = T / N_FIXED_STEPS as f64;

    eprintln!(
        "Liouville spatial sweep: gamma={GAMMA}, T={T}, n_fixed={N_FIXED_STEPS}, tau={tau:.4e}"
    );
    eprintln!(
        "{:>6}  {:>8}  {:>12}  {:>12}  {:>8}  {:>8}",
        "N", "dx", "err_d4", "err_d2", "ratio4", "ratio2"
    );

    let mut prev_d4: Option<f64> = None;
    let mut prev_d2: Option<f64> = None;
    let mut ns_f = Vec::new();
    let mut errs_d4 = Vec::new();
    let mut errs_d2 = Vec::new();

    for &n_spatial in &N_SWEEP {
        let dx = Grid1D::new(X_MIN, X_MAX, n_spatial).expect("dx").dx();
        let (err_d4, err_d2) = run_one_liouville_grid(n_spatial, tau, a_norm, N_FIXED_STEPS);

        let r4 = prev_d4.map_or("       -".into(), |p| format!("{:>8.2}", p / err_d4));
        let r2 = prev_d2.map_or("       -".into(), |p| format!("{:>8.2}", p / err_d2));
        eprintln!("{n_spatial:>6}  {dx:>8.4e}  {err_d4:>12.4e}  {err_d2:>12.4e}  {r4}  {r2}");

        // Per-N ordering: D4th must not be significantly worse than D2.
        assert!(
            err_d4 <= err_d2 * 1.05,
            "N={n_spatial}: err_d4={err_d4:.3e} > err_d2={err_d2:.3e}×1.05 — \
             ζ⁴ should not be worse than ζ-A (variable-a regression)"
        );

        prev_d4 = Some(err_d4);
        prev_d2 = Some(err_d2);
        ns_f.push(n_spatial as f64);
        errs_d4.push(err_d4);
        errs_d2.push(err_d2);
    }

    let slope_d4 = log_log_slope_n(&ns_f, &errs_d4);
    let slope_d2 = log_log_slope_n(&ns_f, &errs_d2);
    eprintln!("Slope D4th = {slope_d4:.4}  (gate ≤ {SLOPE_GATE_D4})");
    eprintln!("Slope D2   = {slope_d2:.4}  (gate ≤ {SLOPE_GATE_D2})");

    assert!(
        slope_d4 <= SLOPE_GATE_D4,
        "D4th slope={slope_d4:.4} > {SLOPE_GATE_D4} — \
         Diffusion4thChernoff spatial convergence gate failed"
    );
    assert!(
        slope_d2 <= SLOPE_GATE_D2,
        "D2 slope={slope_d2:.4} > {SLOPE_GATE_D2} — \
         DiffusionChernoff regression (expected ≤ -1.85)"
    );
}

/// Sanity check: absolute error at single (N=200, n=128) point. NOT the headline gate.
///
/// Documents that both schemes are consistent with the oracle at coarse (N, n).
#[test]
#[allow(clippy::cast_precision_loss)] // N_STEPS = 128; well within f64 mantissa
fn temporal_consistency_variable_a() {
    const N_STEPS: usize = 128;
    const N_SPATIAL: usize = 200;

    G_CELL.with(|c| c.set(GAMMA));
    let a_norm = (1.0 + GAMMA * X_MAX.abs()).powi(2);
    let tau = T / N_STEPS as f64;
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid valid");

    eprintln!(
        "Temporal consistency: gamma={GAMMA}, T={T}, N={N_SPATIAL}, n_steps={N_STEPS}, tau={tau:.4e}"
    );

    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-(x * x) / (2.0 * SIGMA * SIGMA)));

    let d4 = Diffusion4thChernoff::new(a_fn, ap_fn, app_fn, a_norm, grid);
    let d2 = DiffusionChernoff::new(a_fn, ap_fn, app_fn, a_norm, grid);

    let mut s4 = f0.clone();
    let mut s2 = f0.clone();
    for _ in 0..N_STEPS {
        G_CELL.with(|c| c.set(GAMMA));
        s4 = d4.apply_chernoff(tau, &s4).expect("d4 apply");
        s2 = d2.apply_chernoff(tau, &s2).expect("d2 apply");
    }

    let mut err_d4 = 0.0_f64;
    let mut err_d2 = 0.0_f64;
    for i in 0..s4.values.len() {
        let x = grid.x_at(i);
        let exact = liouville_oracle(GAMMA, T, x);
        err_d4 = err_d4.max((s4.values[i] - exact).abs());
        err_d2 = err_d2.max((s2.values[i] - exact).abs());
    }

    eprintln!("err_d4 = {err_d4:.4e}  (sanity only — not the headline gate)");
    eprintln!("err_d2 = {err_d2:.4e}");

    // Both must produce sub-5% errors at coarse grid.
    assert!(
        err_d4 < 5e-2,
        "temporal_consistency_variable_a: err_d4={err_d4:.4e} ≥ 5e-2 \
         (Diffusion4thChernoff diverging from Liouville oracle)"
    );
    assert!(
        err_d2 < 5e-2,
        "temporal_consistency_variable_a: err_d2={err_d2:.4e} ≥ 5e-2 \
         (DiffusionChernoff diverging from Liouville oracle)"
    );
}

/// Self-consistency: oracle at T≈0 should match IC in the interior.
#[test]
#[allow(clippy::cast_precision_loss)] // i ≤ 99; well within f64 mantissa
fn oracle_at_small_t_matches_ic() {
    let t_small = 1e-4;
    // Restrict to |x| ≤ 10 to avoid large Jacobian effects near domain boundary.
    let max_err: f64 = (0..100_usize)
        .map(|i| {
            let x = -10.0 + i as f64 * 20.0 / 99.0;
            let ic = libm::exp(-(x * x) / (2.0 * SIGMA * SIGMA));
            let approx = liouville_oracle(GAMMA, t_small, x);
            (approx - ic).abs()
        })
        .fold(0.0_f64, f64::max);

    eprintln!("Oracle vs IC at T={t_small}: max_err = {max_err:.4e} (gate < 5e-2)");
    assert!(
        max_err < 5e-2,
        "oracle_at_small_t: max_err={max_err:.4e} ≥ 5e-2 (oracle may be incorrect)"
    );
}
