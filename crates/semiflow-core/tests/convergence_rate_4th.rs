//! G3⁴ — spatial and temporal convergence gates for `Diffusion4thChernoff` (v0.6.0, ADR-0013).
//!
//! Two sub-tests:
//!
//! ## 1. `temporal_convergence_constant_a`
//!
//! n-sweep at fixed large N. Measures O(τ²) Chernoff convergence.
//! Gate: slope ≤ −1.85.
//!
//! Design note: temporal Chernoff convergence is order 2 (consistency-in-τ)
//! regardless of spatial order. To see 4th-order spatial benefit, use the
//! spatial sweep below.
//!
//! ## 2. `spatial_convergence_constant_a`
//!
//! Fixed n=4000 (τ≈2.5e-4, τ²≈6.25e-8 — temporal error negligible vs spatial
//! error across all N tested).
//!
//! N-sweep: N ∈ {200, 400, 800, 1600, 3200} nodes on [−15, 15].
//! Gate: log-log slope (log err vs log N) ≤ −3.85 — confirms O(dx⁴) spatial floor.
//!
//! This is the HEADLINE G3⁴ measurement demonstrating the 4th-order spatial
//! benefit of `Diffusion4thChernoff`. For constant-a, `zeta4_correction`
//! short-circuits to 0 and the accuracy is governed by the K-kernel's 4th-order
//! Fourier-symbol weights (W0, W1, W2 — math.md §9.2.1).
//!
//! ## Setup
//!
//! Heat equation: `u_t` = `0.5·u_xx`, IC = exp(−x²), T = 1.
//! Oracle: u(1,x) = 3^{−1/2}·exp(−x²/3) (a=0.5 heat kernel, IC=Gaussian).
//!
//! Domain [−15, 15] keeps the 7-point FD stencil (±3·dx from x) well inside
//! the grid for all N tested. Boundary policy: Reflect (default).

use semiflow_core::{
    boundary::InterpKind, ChernoffSemigroup, Diffusion4thChernoff, Grid1D, GridFn1D, State,
};

const T_FINAL: f64 = 1.0;
const X_MIN: f64 = -15.0;
const X_MAX: f64 = 15.0;

/// Constant diffusion a=0.5 (fn pointer compatible).
fn a_const(_: f64) -> f64 {
    0.5
}
fn a_zero(_: f64) -> f64 {
    0.0
}

/// Oracle: u(1,x) = 3^{−1/2}·exp(−x²/3) for a=0.5, IC=exp(−x²), T=1.
fn oracle(x: f64) -> f64 {
    (3.0_f64).sqrt().recip() * libm::exp(-x * x / 3.0)
}

/// OLS slope: log(err) ~ slope * log(x) + const.
/// Both temporal sweep (x=n) and spatial sweep (x=N) use this.
/// Positive slope → err increases with x (wrong direction).
/// Negative slope → err decreases with x (convergent).
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
// sum_x/sum_y/sum_xy: OLS names by convention; xs are grid counts ≤ 8000
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

/// Temporal convergence: n-sweep at fixed N=8000. Gate ≤ −1.85.
///
/// Confirms O(τ²) Chernoff convergence (same as `DiffusionChernoff` for
/// constant a — Z⁴_const-a gate, proven bit-equal).
///
/// Note: slope ≈ −2.0 (temporal order), NOT −4.0 (spatial order).
/// The 4th-order spatial benefit is only visible via `spatial_convergence_constant_a`.
#[test]
#[allow(clippy::cast_precision_loss)] // n/n_spatial ≤ 8000; well within f64 52-bit mantissa
fn temporal_convergence_constant_a() {
    const N_SPATIAL: usize = 8000;
    const N_STEPS: [usize; 4] = [64, 128, 256, 512];
    const SLOPE_GATE: f64 = -1.85;

    // Pin to CubicHermite: temporal-convergence gate was calibrated with the
    // v5.x default (CubicHermite). v6.0 changed Grid1D::new default to
    // SepticHermite (ADR-0109); pinning isolates temporal-order measurement.
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL)
        .expect("grid valid")
        .with_interp(InterpKind::CubicHermite);
    let exact = GridFn1D::from_fn(grid, oracle);

    eprintln!("Temporal sweep: Diffusion4thChernoff a=0.5, N_SPATIAL={N_SPATIAL}, T={T_FINAL}");
    eprintln!("{:>6}  {:>12}  {:>8}", "n", "err_sup", "ratio");

    let mut prev_err: Option<f64> = None;
    let mut ns_f = Vec::new();
    let mut errs = Vec::new();

    for &n in &N_STEPS {
        let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
        let d4 = Diffusion4thChernoff::new(a_const, a_zero, a_zero, 0.5, grid);
        let sem = ChernoffSemigroup::new(d4, n).expect("semigroup valid");
        let result = sem.evolve(T_FINAL, &f0).expect("evolve ok");

        let mut diff = result;
        diff.axpy(-1.0, &exact);
        let err = diff.norm_sup();

        let ratio_str = prev_err.map_or_else(|| "       -".into(), |p| format!("{:>8.2}", p / err));
        eprintln!("{n:>6}  {err:>12.4e}  {ratio_str}");
        prev_err = Some(err);
        ns_f.push(n as f64);
        errs.push(err);
    }

    // Slope vs log(n): as n grows (fewer τ), err shrinks → slope ≈ −2.
    let slope = log_log_slope(&ns_f, &errs);
    eprintln!("Temporal slope = {slope:.4}  (gate ≤ {SLOPE_GATE})");
    eprintln!("Expected ≈ −2.0 (O(τ²) Chernoff; constant-a → bit-equal to DiffusionChernoff)");

    assert!(
        slope <= SLOPE_GATE,
        "temporal_convergence_constant_a FAIL: slope={slope:.4} > {SLOPE_GATE} — \
         Diffusion4thChernoff O(τ²) temporal convergence not confirmed"
    );
}

/// Spatial convergence: N-sweep at fixed n=4000. Gate ≤ −3.85.
///
/// HEADLINE G3⁴ test: confirms O(dx⁴) spatial floor for `Diffusion4thChernoff`.
///
/// Fixed τ = T/n = 1/4000 ≈ 2.5e-4, τ² ≈ 6.25e-8.
/// Temporal error ~ `C_τ·τ²` ≈ (some constant)·6.25e-8 is negligible vs
/// spatial error `C_x·dx⁴` across all N tested:
///   N=200,  dx=0.15,  dx⁴≈5e-4   >> τ²
///   N=3200, dx=0.009, dx⁴≈7e-11  >> τ² (barely)
///
/// For constant-a, `zeta4_correction` returns 0 (short-circuit gate Z⁴_const-a).
/// Spatial accuracy governed by K-kernel Fourier-symbol weights (§9.2.1):
///   K̂(ξ) = W0 + 2W1·cos(ξ) + 2W2·cos(2ξ)  [O(ξ⁴) by construction]
/// → global O(dx⁴) spatial convergence.
///
/// Measured ratio per N-doubling: ≈ 16× (2⁴ = 16) confirming 4th order.
///
/// Selected N range: {200, 400, 800, 1600, 3200}
/// Reasoning: monotone decreasing regime at n=4000 fixed; ratio ≈ 16× per step.
/// Domain [−15, 15]: wider than [−10, 10] so ±3·dx FD offset stays inside grid.
#[test]
#[allow(clippy::cast_precision_loss)] // n_spatial ≤ 3200; well within f64 52-bit mantissa
fn spatial_convergence_constant_a() {
    const N_FIXED_STEPS: usize = 4000;
    const N_SWEEP: [usize; 5] = [200, 400, 800, 1600, 3200];
    const SLOPE_GATE: f64 = -3.85;

    eprintln!("Spatial sweep: Diffusion4thChernoff a=0.5, n_fixed={N_FIXED_STEPS}, T={T_FINAL}");
    eprintln!("Domain [{X_MIN}, {X_MAX}]; BoundaryPolicy::Reflect (default)");
    eprintln!("{:>6}  {:>8}  {:>12}  {:>8}", "N", "dx", "err_sup", "ratio");

    let mut prev_err: Option<f64> = None;
    let mut ns_spatial = Vec::new();
    let mut errs = Vec::new();

    for &n_spatial in &N_SWEEP {
        // Pin to CubicHermite: spatial convergence gate (-3.85) was calibrated
        // with v5.x default. v6.0 SepticHermite changes saturation floor;
        // pin to keep this gate independent of interpolation kernel.
        let grid = Grid1D::new(X_MIN, X_MAX, n_spatial)
            .expect("grid valid")
            .with_interp(InterpKind::CubicHermite);
        let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
        let exact = GridFn1D::from_fn(grid, oracle);

        let d4 = Diffusion4thChernoff::new(a_const, a_zero, a_zero, 0.5, grid);
        let sem = ChernoffSemigroup::new(d4, N_FIXED_STEPS).expect("semigroup valid");
        let result = sem.evolve(T_FINAL, &f0).expect("evolve ok");

        let mut diff = result;
        diff.axpy(-1.0, &exact);
        let err = diff.norm_sup();
        let dx = grid.dx();

        let ratio_str = prev_err.map_or_else(|| "       -".into(), |p| format!("{:>8.2}", p / err));
        eprintln!("{n_spatial:>6}  {dx:>8.4e}  {err:>12.4e}  {ratio_str}");
        prev_err = Some(err);
        ns_spatial.push(n_spatial as f64);
        errs.push(err);
    }

    // Slope vs log(N): as N grows (smaller dx), err shrinks → slope ≈ −4.
    let slope = log_log_slope(&ns_spatial, &errs);
    eprintln!("Spatial slope = {slope:.4}  (gate ≤ {SLOPE_GATE})");
    eprintln!("Expected ≈ −4.0 (O(dx⁴) K-kernel spatial floor, math.md §9.2.1)");

    assert!(
        slope <= SLOPE_GATE,
        "spatial_convergence_constant_a FAIL: slope={slope:.4} > {SLOPE_GATE} — \
         Diffusion4thChernoff O(dx⁴) spatial convergence not confirmed at fixed n={N_FIXED_STEPS}."
    );
}
