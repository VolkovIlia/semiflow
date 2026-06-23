//! Convergence tests for `TruncatedExp4thDiffusionChernoff` (v0.6.0, ADR-0013).
//!
//! CFL constraint: `τ < 3·dx² / (8·a_norm)` — 25% tighter than v0.4.0.
//!
//! ## `spatial_convergence_constant_a`
//!
//! DX-SWEEP at fixed large `n=40_000` (τ=2.5e-5). Sweeps N (number of spatial nodes)
//! to measure O(dx⁴) spatial floor.
//! Gate: log-log slope (log err vs log N) ≤ −3.85.
//!
//! CFL check with a=0.5: need `dx > sqrt(tau * 8 * 0.5 / 3) = sqrt(tau * 4/3)`.
//! At tau=2.5e-5: dx > sqrt(2.5e-5 * 4/3) ≈ 0.00577. Domain [-15,15]: N < 5196.
//! Safe range: N ∈ {200, 400, 800, 1600, 3200}.
//! At N=3200: dx=30/3200=0.009375, `CFL_max`≈6.6e-5 >> tau=2.5e-5 ✓.
//!
//! ## `temporal_convergence_constant_a`
//!
//! n-SWEEP at fixed N=400 (dx=0.075, `CFL_max`≈0.00338). Uses large n values to
//! stay in the convergent regime. Gate: slope ≤ −1.85.
//!
//! At N=400 spatial nodes: `CFL_max` = 3*(0.075)²/(8*0.5) ≈ 0.003375.
//! Use n ∈ {400, 800, 1600, 3200} → τ ∈ {0.0025, 0.00125, 6.25e-4, 3.125e-4} — all < `CFL_max`.
//!
//! Design note (Phase 1 lesson): n-sweep measures Chernoff-temporal convergence
//! (order 2) — NOT spatial order. The spatial gate requires the DX-SWEEP above.

use semiflow_core::{ChernoffSemigroup, Grid1D, GridFn1D, State, TruncatedExp4thDiffusionChernoff};

const T_FINAL: f64 = 1.0;
const X_MIN: f64 = -15.0;
const X_MAX: f64 = 15.0;
const A_CONST: f64 = 0.5;

fn a_half(_: f64) -> f64 {
    A_CONST
}
fn a_zero(_: f64) -> f64 {
    0.0
}

/// Oracle: u(1,x) = 3^{−1/2}·exp(−x²/3) for a=0.5, IC=exp(−x²), T=1.
fn oracle(x: f64) -> f64 {
    (3.0_f64).sqrt().recip() * libm::exp(-x * x / 3.0)
}

/// OLS log-log slope: log(err) ~ slope * log(x) + const.
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
// sum_x/sum_y/sum_xy: OLS variable names by convention; xs.len() ≤ 2^16; well within f64 mantissa
fn log_log_slope(xs: &[f64], errs: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let lx: Vec<f64> = xs.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = lx.iter().sum();
    let sum_y: f64 = ly.iter().sum();
    let sum_xx: f64 = lx.iter().map(|&v| v * v).sum();
    let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

/// Spatial convergence: N-sweep at fixed `n=40_000` (τ=2.5e-5). Gate ≤ −3.85.
///
/// HEADLINE M⁴ test: confirms O(dx⁴) spatial floor for `TruncatedExp4thDiffusionChernoff`
/// at constant a (standard 4th-order Laplacian collapse, gate M⁴_const-a-fast).
///
/// CFL at N=3200: dx=0.009375, CFL_max=6.6e-5 >> tau=2.5e-5 ✓.
/// Expected ratio per N-doubling: ≈ 16× (2⁴ = 16) confirming 4th order.
#[test]
#[allow(clippy::cast_precision_loss)]
// N_FIXED=40_000, n_spatial ≤ 3200; well within f64 52-bit mantissa
fn spatial_convergence_constant_a() {
    const N_FIXED: usize = 40_000;
    const N_SWEEP: [usize; 5] = [200, 400, 800, 1600, 3200];
    const SLOPE_GATE: f64 = -3.85;
    let tau = T_FINAL / N_FIXED as f64;

    eprintln!(
        "Spatial sweep: TruncatedExp4thDiffusionChernoff a=0.5, n_fixed={N_FIXED}, T={T_FINAL}"
    );
    eprintln!("tau={tau:.3e}, Domain [{X_MIN}, {X_MAX}]; BoundaryPolicy::Reflect (default)");
    eprintln!(
        "{:>6}  {:>8}  {:>10}  {:>12}  {:>8}",
        "N", "dx", "CFL_max", "err_sup", "ratio"
    );

    let mut prev_err: Option<f64> = None;
    let mut ns_spatial = Vec::new();
    let mut errs = Vec::new();

    for &n_spatial in &N_SWEEP {
        let grid = Grid1D::new(X_MIN, X_MAX, n_spatial).expect("grid valid");
        let dx = grid.dx();
        let cfl_max = 3.0 * dx * dx / (8.0 * A_CONST);
        assert!(
            tau < cfl_max,
            "CFL violated at N={n_spatial}: tau={tau:.3e} >= cfl_max={cfl_max:.3e}"
        );

        let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
        let exact = GridFn1D::from_fn(grid, oracle);

        let m4 = TruncatedExp4thDiffusionChernoff::new(a_half, a_zero, a_zero, A_CONST, grid);
        let sem = ChernoffSemigroup::new(m4, N_FIXED).expect("semigroup valid");
        let result = sem.evolve(T_FINAL, &f0).expect("evolve ok");

        let mut diff = result;
        diff.axpy(-1.0, &exact);
        let err = diff.norm_sup();

        let ratio_str = prev_err.map_or_else(|| "       -".into(), |p| format!("{:>8.2}", p / err));
        eprintln!("{n_spatial:>6}  {dx:>8.4e}  {cfl_max:>10.4e}  {err:>12.4e}  {ratio_str}");
        prev_err = Some(err);
        ns_spatial.push(n_spatial as f64);
        errs.push(err);
    }

    let slope = log_log_slope(&ns_spatial, &errs);
    eprintln!("Spatial slope = {slope:.4}  (gate ≤ {SLOPE_GATE})");
    eprintln!("Expected ≈ −4.0 (O(dx⁴) 4th-order TruncatedExp stencil, constant-a)");

    assert!(
        slope <= SLOPE_GATE,
        "spatial_convergence_constant_a FAIL: slope={slope:.4} > {SLOPE_GATE} — \
         TruncatedExp4thDiffusionChernoff O(dx⁴) spatial convergence not confirmed at fixed n={N_FIXED}"
    );
}

/// Temporal convergence: DIAGONAL refinement (N=n). Gate ≤ −1.85.
///
/// Diagonal refinement is the standard approach for TruncatedExp-type CFL-constrained
/// integrators (see `truncated_exp_heat_kernel.rs`). N=n ensures CFL is satisfied by
/// construction: τ·‖G⁴‖ < 2 iff 8·(T/n)·a·n² / (3·(domain/n)²) < 2,
/// which for this setup reduces to a fixed CFL factor independent of n.
///
/// Setup: `∂_t u = a·∂_xx u`, a=0.5, IC=exp(−x²), oracle u(T,x).
/// Domain [-15, 15] (L=30). Diagonal: N=n spatial nodes, n temporal steps.
/// dx = 30/n, τ = T/n. CFL factor = 8·a·τ/(3·dx²) = 8·0.5·(1/n)/(3·(30/n)²)
///   = 4·n / (3·900) = n/675.
/// For n ∈ {64, 128, 256, 512}: CFL factor = {0.095, 0.190, 0.379, 0.759} < 1 ✓.
///
/// At each n, BOTH spatial AND temporal error shrink together. For Chernoff
/// (temporal order p=2), total error ∝ `C_t·τ²` + `C_x·dx⁴` → dominated by τ²
/// when τ >> dx² (coarse n) and by dx⁴ when τ << dx². In the diagonal sweep,
/// τ ∝ 1/n and dx ∝ 1/n, so temporal error ∝ n⁻² dominates for all n.
#[test]
#[allow(clippy::cast_precision_loss)]
// n ≤ 512; well within f64 52-bit mantissa
fn temporal_convergence_constant_a() {
    const N_SWEEP: [usize; 4] = [64, 128, 256, 512];
    const SLOPE_GATE: f64 = -1.85;

    eprintln!("Temporal sweep (diagonal N=n): TruncatedExp4thDiffusionChernoff a=0.5, T={T_FINAL}");
    eprintln!(
        "{:>6}  {:>10}  {:>10}  {:>12}  {:>8}",
        "n=N", "tau", "CFL", "err_sup", "ratio"
    );

    let mut prev_err: Option<f64> = None;
    let mut ns_f = Vec::new();
    let mut errs = Vec::new();

    for &n in &N_SWEEP {
        let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid valid");
        let dx = grid.dx();
        let tau = T_FINAL / n as f64;
        let cfl_factor = 8.0 * A_CONST * tau / (3.0 * dx * dx);
        assert!(
            cfl_factor < 1.0,
            "CFL violated at n={n}: factor={cfl_factor:.4} >= 1"
        );

        let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
        let exact = GridFn1D::from_fn(grid, oracle);

        let m4 = TruncatedExp4thDiffusionChernoff::new(a_half, a_zero, a_zero, A_CONST, grid);
        let sem = ChernoffSemigroup::new(m4, n).expect("semigroup valid");
        let result = sem.evolve(T_FINAL, &f0).expect("evolve ok");

        let mut diff = result;
        diff.axpy(-1.0, &exact);
        let err = diff.norm_sup();

        let ratio_str = prev_err.map_or_else(|| "       -".into(), |p| format!("{:>8.2}", p / err));
        eprintln!("{n:>6}  {tau:>10.4e}  {cfl_factor:>10.4}  {err:>12.4e}  {ratio_str}");
        prev_err = Some(err);
        ns_f.push(n as f64);
        errs.push(err);
    }

    let slope = log_log_slope(&ns_f, &errs);
    eprintln!("Diagonal slope = {slope:.4}  (gate ≤ {SLOPE_GATE})");
    eprintln!("Expected ≈ −2.0 (O(τ²) temporal dominates in diagonal sweep)");

    assert!(
        slope <= SLOPE_GATE,
        "temporal_convergence_constant_a FAIL: slope={slope:.4} > {SLOPE_GATE} — \
         TruncatedExp4thDiffusionChernoff O(τ²) temporal convergence not confirmed (diagonal N=n)"
    );
}
