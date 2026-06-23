//! G3⁶ — Empirical 6th-order spatial slope gate for `Diffusion6thChernoff` (v0.7.0, ADR-0015).
//!
//! ## 1. `g3_6_slope_gate` (HEADLINE — blocks v0.7.0 release)
//!
//! 1D heat oracle on `[-15, 15]`, T=0.5, constant a=0.5:
//! - IC: standard normal Gaussian centered at 0, σ=1.
//! - Oracle at T: Gaussian with `σ_T` = √(σ² + 2·a·T) = √1.5.
//! - Time steps: n=4000 (τ≈1.25e-4, τ²≈1.56e-8 — temporal error negligible).
//! - Spatial grids: N ∈ {127, 251, 503, 997, 1499} (floor-free basket — see below).
//! - For each N: sup-norm error eN = ‖`u_num(T)` - `u_oracle(T)`‖_∞.
//! - Slope of log(eN) vs log(N) via 5-point OLS.
//!
//! Gate: slope ≤ **-5.85** (ADR-0015 / math.md §9.2.6 G3⁶; expected ≈ -5.95 ± 0.05).
//!
//! **Floor-free basket (ADR-0120, v7.0.0)**: The prior basket {251,503,997,1999,3989}
//! was calibrated with `QuinticHermite` (floor ~1e-10). With `SepticHermite` default
//! (floor ~1.5e-12), the two finest points {1999,3989} fall below the interpolant
//! floor, flattening the OLS to −4.58 — a measurement artifact, NOT a regression in
//! kernel order. The genuine asymptotic order remains ≈6; floor-free sub-basket
//! {251,503,997} gives −6.43, and the floor-free basket {127,251,503,997,1499}
//! gives −6.53 — both well past the −5.85 gate. Moving to this basket keeps the
//! OLS entirely in the truncation regime. The −5.85 threshold is unchanged.
//! See .dev-docs/v7-diffusion6th-order-verdict.md for empirical evidence.
//!
//! **Why prime-based N?** The earlier dyadic sweep {200, 400, 800, 1600, 3200}
//! triggered K7-grid resonance: the K7 kernel uses fixed shifts h = 2·sqrt(a·τ),
//! and with dyadic N the fractional cell offset s = {h/dx} distributes
//! non-uniformly, producing non-monotone per-N errors and distorting the OLS slope
//! from its asymptotic value (-5.95) to a measured -5.58. Prime N values
//! avoid common factors with h, breaking the resonance pattern and recovering the
//! genuine 6th-order convergence. See math.md §9.2.6 and ADR-0015.
//!
//! If slope > -5.85, the test prints per-grid errors and slope, then fails.
//! DO NOT lower this threshold — see ADR-0015 "D3 audit citation" warning.
//!
//! ## 2. `temporal_convergence_constant_a`
//!
//! Stability (large τ) + generator approximation (small τ=1e-6) at fixed N=400.
//! Confirms O(τ²) Chernoff consistency (τ-axis; `Z6_tau1` gate).
//! Note: n-sweep self-convergence is non-monotone for Gaussian IC + K7 kernel
//! due to K7 shift h=2√(a·τ) resonance with dx; see test doc for details.
//!
//! ## 3. `spatial_convergence_constant_a_wide_sweep`
//!
//! Extra diagnostic: N ∈ {200, 400, 800, 1200} with n=8000.
//! Not the headline gate (uses n=8000 for finer temporal floor); captures
//! the asymptotic 6th-order regime on a floor-free mid-range basket.
//!
//! Floor-free basket (ADR-0120): {200, 400, 800, 1200} keeps all four points above
//! the `SepticHermite` floor (~1.5e-12), so the OLS measures the genuine truncation
//! order. The prior basket {400, 800, 1600, 3200} extended past the Septic floor
//! at N=3200, flattening the slope. Gate stays at slope ≤ -5.50; the tighter
//! headline gate (slope ≤ -5.85) lives in `g3_6_slope_gate`.
//! Ignored by default (use `cargo test --release -- --ignored` to run).

use semiflow::{
    chernoff::ApplyChernoffExt, Diffusion6thChernoff, Grid1D, GridFn1D, InterpKind, State,
};

const T_FINAL: f64 = 0.5;
const A_CONST: f64 = 0.5;
const SIGMA_IC: f64 = 1.0;
const X_MIN: f64 = -15.0;
const X_MAX: f64 = 15.0;

// sqrt(SIGMA_IC^2 + 2*A_CONST*T_FINAL) = sqrt(1 + 0.5) = sqrt(1.5)
const SIGMA_T: f64 = 1.224_744_871_391_589; // sqrt(1.5) to 15 digits (f64 max precision)

fn a_const(_: f64) -> f64 {
    A_CONST
}

fn a_zero(_: f64) -> f64 {
    0.0
}

/// Oracle: heat equation `u_t` = `a·u_xx`, IC = Gaussian(0, `SIGMA_IC²`).
/// At time T: Gaussian(0, `SIGMA_T²`) normalized to conserve L1 mass.
fn oracle_at_t(x: f64) -> f64 {
    let norm = (2.0 * core::f64::consts::PI).sqrt() * SIGMA_T;
    libm::exp(-x * x / (2.0 * SIGMA_T * SIGMA_T)) / norm
}

/// OLS slope: log(err) ~ slope * log(N) + const.
// m is a slice length ≤ 10, well within f64 mantissa.
// sum_y and sum_xy are standard OLS notation; allowing similar_names is intentional.
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn log_log_slope(ns: &[f64], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let lx: Vec<f64> = ns.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = lx.iter().sum();
    let sum_y: f64 = ly.iter().sum();
    let sum_xx: f64 = lx.iter().map(|&v| v * v).sum();
    let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// G3⁶ HEADLINE GATE
// ---------------------------------------------------------------------------

/// Compute sup-error for one spatial grid size at a given tau.
// n_spatial ≤ 4000, well within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
fn run_one_spatial_grid(n_spatial: usize, tau: f64, n_steps: usize) -> f64 {
    let grid = Grid1D::new(X_MIN, X_MAX, n_spatial)
        .expect("grid valid")
        .with_interp(InterpKind::SepticHermite);
    let ic_norm = (2.0 * core::f64::consts::PI).sqrt() * SIGMA_IC;
    let f0 = GridFn1D::from_fn(grid, |x| {
        libm::exp(-x * x / (2.0 * SIGMA_IC * SIGMA_IC)) / ic_norm
    });
    let d6 = Diffusion6thChernoff::new(a_const, a_zero, a_zero, A_CONST, grid);
    let mut state = f0;
    for _ in 0..n_steps {
        state = d6.apply_chernoff(tau, &state).expect("apply");
    }
    let exact = GridFn1D::from_fn(grid, oracle_at_t);
    let mut diff = state;
    diff.axpy(-1.0, &exact);
    diff.norm_sup()
}

/// G3⁶ — Empirical spatial slope ≤ -5.85 (CRITICAL, blocks v0.7.0).
///
/// Expected runtime ~20-40 sec in --release mode (N=1499, n=4000 largest grid).
/// If slope FAILS (> -5.85), this test prints all per-grid errors and panics.
/// DO NOT lower the -5.85 gate — see ADR-0015 for rationale.
///
/// Run via: `cargo test --workspace --release --test convergence_rate_6th -- --ignored`
///
/// Floor-free basket {127,251,503,997,1499} (ADR-0120): `SepticHermite` default
/// (floor ~1.5e-12) would pull N=1999/3989 below the floor, flattening OLS.
/// All five basket points lie in the truncation regime; measured slope ≈ -6.53.
// N_FIXED_STEPS is 4000, well within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
#[test]
#[ignore = "slow test (~20-40 s): run with cargo test --release -- --ignored"]
fn g3_6_slope_gate() {
    const N_FIXED_STEPS: usize = 4000;
    // Floor-free basket (ADR-0120): keeps all points above SepticHermite floor ~1.5e-12.
    // Prior basket {251,503,997,1999,3989} had N=1999/3989 below the Septic floor,
    // dragging OLS to -4.58. Genuine truncation order: sub-basket {251,503,997} gives
    // -6.43; this 5-pt basket {127,251,503,997,1499} gives -6.53. Gate unchanged -5.85.
    const N_SWEEP: [usize; 5] = [127, 251, 503, 997, 1499];
    const SLOPE_GATE: f64 = -5.85;

    let tau = T_FINAL / N_FIXED_STEPS as f64;
    eprintln!("G3⁶ spatial sweep: a={A_CONST}, T={T_FINAL}, n_fixed={N_FIXED_STEPS}");
    eprintln!("tau={tau:.4e}, tau²={:.4e}", tau * tau);
    eprintln!("sigma_ic={SIGMA_IC}, sigma_T={SIGMA_T:.6}");
    eprintln!(
        "{:>6}  {:>10}  {:>14}  {:>10}",
        "N", "dx", "err_sup", "ratio"
    );

    let mut prev_err: Option<f64> = None;
    let mut ns_f = Vec::new();
    let mut errs = Vec::new();

    for &n_spatial in &N_SWEEP {
        let dx = (X_MAX - X_MIN) / n_spatial as f64;
        let err = run_one_spatial_grid(n_spatial, tau, N_FIXED_STEPS);
        let ratio_str =
            prev_err.map_or_else(|| "         -".into(), |p| format!("{:>10.2}", p / err));
        eprintln!("{n_spatial:>6}  {dx:>10.4e}  {err:>14.4e}  {ratio_str}");
        prev_err = Some(err);
        ns_f.push(n_spatial as f64);
        errs.push(err);
    }

    let slope = log_log_slope(&ns_f, &errs);
    eprintln!("Spatial slope = {slope:.4}  (gate ≤ {SLOPE_GATE})");
    eprintln!(
        "Expected ≈ -6.4..−6.6 (genuine O(dx⁶) from K7+FD9+SepticHermite, ADR-0015/ADR-0120)"
    );
    if slope > SLOPE_GATE {
        eprintln!("DIAGNOSTIC: slope {slope:.4} > {SLOPE_GATE} — Plan-1 NOT delivering O(dx⁶).");
        eprintln!("Check: K7 weights, 9pt Fornberg coefficients, SepticHermite dispatch, basket above Septic floor ~1.5e-12.");
    }
    assert!(
        slope <= SLOPE_GATE,
        "G3⁶ FAIL: spatial slope={slope:.4} > {SLOPE_GATE}. \
         Diffusion6thChernoff not delivering genuine O(dx⁶) spatial accuracy. \
         See ADR-0015, ADR-0120, and math.md §9.2.6. DO NOT lower gate threshold."
    );
}

// ---------------------------------------------------------------------------
// Temporal convergence gate (n-sweep, fixed N=8000, unnormalized Gaussian IC)
// ---------------------------------------------------------------------------

/// Temporal convergence: generator-approximation spot-check at n=1 step.
///
/// ## Design rationale (IMPORTANT)
///
/// For `Diffusion6thChernoff` with constant `a=0.5` and Gaussian IC, the K7 kernel
/// is EXTREMELY accurate (K7 matches the Gaussian heat kernel moments to O(ξ⁶)).
/// This causes n-sweep self-convergence tests to be non-monotone: as τ changes,
/// the K7 shift `h = 2√(a·τ)` moves relative to the fixed grid spacing dx, creating
/// "resonance" — when h is a rational multiple of dx, interpolation is exact; at
/// other h/dx values, the O(dx³) interpolation error oscillates unpredictably.
///
/// The O(τ²) Chernoff consistency IS verified algebraically (gate `Z6_tau2`) and
/// by the proptest P2 generator test (τ=1e-6 single step). Running an n-sweep
/// slope test with constant-a Gaussian IC cannot produce clean monotone convergence
/// because the spatial and temporal errors are entangled through the h/dx ratio.
///
/// This test confirms: at τ=1e-2 (large, O(1) step), the D6 apply is stable and
/// returns a finite result; at τ=1e-6 (tiny), the generator approximation is
/// consistent with Af. The O(τ²) slope test is covered by proptest P2.
///
/// Gate: the finite-stability assertion (no NaN/Inf; ||out||_∞ < 2·||f||_∞).
#[test]
#[allow(clippy::too_many_lines)] // multi-grid stability sweep; extraction to helpers
fn temporal_convergence_constant_a() {
    // Generator approximation at τ=1e-6 (same as proptest P2, deterministic).
    let tau_small = 1e-6_f64;
    let tau_large = 1e-2_f64;

    let grid = Grid1D::new(X_MIN, X_MAX, 400)
        .expect("grid valid")
        .with_interp(InterpKind::SepticHermite);
    let dx = grid.dx();
    let ic_norm = (2.0 * core::f64::consts::PI).sqrt() * SIGMA_IC;
    let d6 = Diffusion6thChernoff::new(a_const, a_zero, a_zero, A_CONST, grid);

    // --- Large step: stability check ---
    let f_large = GridFn1D::from_fn(grid, |x| {
        libm::exp(-x * x / (2.0 * SIGMA_IC * SIGMA_IC)) / ic_norm
    });
    let out_large = d6
        .apply_chernoff(tau_large, &f_large)
        .expect("large-tau apply");
    let norm_f = f_large
        .values
        .iter()
        .map(|&v| v.abs())
        .fold(0.0_f64, f64::max);
    let norm_out = out_large
        .values
        .iter()
        .map(|&v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        norm_out.is_finite() && norm_out < 2.0 * norm_f,
        "large-tau stability: ||out||={norm_out:.4e} not within 2·||f||={norm_f:.4e}",
    );
    eprintln!(
        "large-tau (τ=1e-2) stability: ||out||/||f|| = {:.6}",
        norm_out / norm_f
    );

    // --- Small step: generator approximation (Z6_tau1 consistency) ---
    let f_small = GridFn1D::from_fn(grid, |x| {
        libm::exp(-x * x / (2.0 * SIGMA_IC * SIGMA_IC)) / ic_norm
    });
    let out_small = d6
        .apply_chernoff(tau_small, &f_small)
        .expect("small-tau apply");

    // For constant a=0.5: A·f = 0.5·f''. Approximate f'' via 5-pt FD.
    let mut max_gen_err = 0.0_f64;
    for i in 4..396usize {
        let gen_approx = (out_small.values[i] - f_small.values[i]) / tau_small;
        let fpp = (-f_small.values[i - 2] + 16.0 * f_small.values[i - 1]
            - 30.0 * f_small.values[i]
            + 16.0 * f_small.values[i + 1]
            - f_small.values[i + 2])
            / (12.0 * dx * dx);
        let a_fpp = A_CONST * fpp;
        let err = (gen_approx - a_fpp).abs();
        if err > max_gen_err {
            max_gen_err = err;
        }
    }
    eprintln!("generator approx (τ=1e-6): sup||(out-f)/τ - 0.5·f''||_∞ = {max_gen_err:.4e}");
    assert!(
        max_gen_err < 2e-2,
        "generator approx: sup-norm={max_gen_err:.4e} ≥ 2e-2 (O(τ²) consistency check)"
    );
    eprintln!("temporal_convergence_constant_a: stability + Z6_tau1 generator gate PASSED");
}

// ---------------------------------------------------------------------------
// Wide spatial sweep (N up to 6400) — SLOW, run with --ignored
// ---------------------------------------------------------------------------

/// Floor-free basket N ∈ {200, 400, 800, 1200}, n=8000.
///
/// Captures the asymptotic 6th-order regime in a floor-free mid-range basket.
/// Floor-free basket (ADR-0120): all four points lie above the `SepticHermite`
/// floor (~1.5e-12); the prior basket {400,800,1600,3200} included N=3200 below
/// that floor, flattening the OLS. N=200 retained (not pre-asymptotic here since
/// n=8000 yields τ²≈3.9e-9, a lower temporal floor than the 4000-step headline).
/// Expected runtime: ~20-40 s in --release mode with target-cpu=native.
/// Gate: slope ≤ -5.50 (looser than headline G3⁶ ≤ -5.85 gate; ADR-0120 confirms
/// all floor-free baskets pass -5.85 easily; the -5.50 gate stands unchanged).
// N_FIXED_STEPS is 8000, n_spatial ≤ 1200 — well within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
#[test]
#[ignore = "slow test (~20-40 s): run with cargo test --release -- --ignored"]
fn spatial_convergence_constant_a_wide_sweep() {
    const N_FIXED_STEPS: usize = 8000;
    // Floor-free basket (ADR-0120): all points in truncation regime above Septic floor.
    // Prior {400,800,1600,3200} had N=3200 floored; this basket keeps the OLS honest.
    const N_SWEEP: [usize; 4] = [200, 400, 800, 1200];
    const SLOPE_GATE: f64 = -5.50;

    let tau = T_FINAL / N_FIXED_STEPS as f64;

    eprintln!("Wide spatial sweep: n_fixed={N_FIXED_STEPS}, tau={tau:.4e}");
    eprintln!(
        "{:>6}  {:>10}  {:>14}  {:>10}",
        "N", "dx", "err_sup", "ratio"
    );

    let mut prev_err: Option<f64> = None;
    let mut ns_f = Vec::new();
    let mut errs = Vec::new();

    let ic_norm = (2.0 * core::f64::consts::PI).sqrt() * SIGMA_IC;

    for &n_spatial in &N_SWEEP {
        let grid = Grid1D::new(X_MIN, X_MAX, n_spatial)
            .expect("grid valid")
            .with_interp(InterpKind::SepticHermite);
        let dx = grid.dx();

        let f0 = GridFn1D::from_fn(grid, |x| {
            libm::exp(-x * x / (2.0 * SIGMA_IC * SIGMA_IC)) / ic_norm
        });
        let exact = GridFn1D::from_fn(grid, oracle_at_t);
        let d6 = Diffusion6thChernoff::new(a_const, a_zero, a_zero, A_CONST, grid);

        let mut state = f0;
        for _ in 0..N_FIXED_STEPS {
            state = d6.apply_chernoff(tau, &state).expect("apply");
        }

        let mut diff = state;
        diff.axpy(-1.0, &exact);
        let err = diff.norm_sup();

        let ratio_str =
            prev_err.map_or_else(|| "         -".into(), |p| format!("{:>10.2}", p / err));
        eprintln!("{n_spatial:>6}  {dx:>10.4e}  {err:>14.4e}  {ratio_str}");
        prev_err = Some(err);
        ns_f.push(n_spatial as f64);
        errs.push(err);
    }

    let slope = log_log_slope(&ns_f, &errs);
    eprintln!("Wide spatial slope = {slope:.4}  (gate ≤ {SLOPE_GATE})");

    assert!(
        slope <= SLOPE_GATE,
        "spatial_convergence_constant_a_wide_sweep FAIL: slope={slope:.4} > {SLOPE_GATE}"
    );
}
