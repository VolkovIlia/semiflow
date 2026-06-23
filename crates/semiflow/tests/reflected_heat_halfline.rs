//! G27 — Reflected heat on half-line [0, ∞) (`RELEASE_BLOCKING`).
//!
//! Gate (properties.yaml G27, `RELEASE_BLOCKING`, 2 sub-tests):
//!
//!   Operator: L = ∂_xx (`DiffusionChernoff` with a(x)≡1, a'≡0, a''≡0).
//!   Region:   R = [0, ∞) — `HalfSpaceRegion` with origin=[0.0], normal=[1.0].
//!   IC:       g(x) = exp(-x²) (Gaussian, supported on [0,∞)).
//!   `T_final`:  T = 0.1.
//!
//!   Oracle (closed form, math §25.5):
//!     `K_N(x`, y; t) = (4πt)^{-1/2} · [exp(-(x-y)²/(4t)) + exp(-(x+y)²/(4t))]
//!     u^N(T, x) = ∫_0^∞ `K_N(x`, y; T) · g(y) dy
//!              = exp(-x²/(1 + 4T)) / √(1 + 4T)
//!     (closed-form Gaussian integral — verified by sympy in T22N setup.)
//!
//!   Sub-test 1 (RESIDUAL at fixed N):
//!     Grid N=1024 on [0, 10], `n_Chernoff=100`, τ=T/100=0.001.
//!     MUST satisfy ‖F^100(g) - u^N(T, ·)‖_∞ ≤ 1e-6.
//!
//!   Sub-test 2 (SLOPE / ORDER PRESERVATION):
//!     Sweep n ∈ {16, 32, 64, 128} at fixed T=0.1 (τ = T/n).
//!     OLS slope of log(err) vs log(n) MUST be ≤ -0.95.
//!     This verifies Proposition 25.1: wrapper preserves inner order.
//!
//! Feature gate: `slow-tests`. Both sub-tests MUST pass.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // n, n_steps ≤ 1024, well within 2^52
#![allow(clippy::doc_markdown)]        // math notation in doc comments (K_N, n_Chernoff, etc.)

use semiflow::{
    reflection::{HalfSpaceRegion, ReflectedHeatChernoff},
    BoundaryPolicy, ChernoffFunction, DiffusionChernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Gate constants (normative — do NOT relax without ADR + properties.yaml bump)
// ---------------------------------------------------------------------------

/// Sub-test 1: sup-norm residual must be ≤ this at N=1024, n_Chernoff=100.
const RESIDUAL_GATE: f64 = 1e-6;

/// Sub-test 2: OLS log-log slope of err vs log(n) must be ≤ this.
const SLOPE_GATE: f64 = -0.95;

/// Fixed final time.
const T_FINAL: f64 = 0.1;

/// Grid extent (half-line truncated to [0, GRID_MAX]).
const GRID_MAX: f64 = 10.0;

/// n_Chernoff sweep for sub-test 2.
const N_SWEEP: [usize; 4] = [16, 32, 64, 128];

// ---------------------------------------------------------------------------
// Oracle: u^N(T, x) = exp(-x² / (1 + 4T)) / √(1 + 4T)
// ---------------------------------------------------------------------------

/// Closed-form reference solution for reflected heat on [0,∞) with IC g(x)=exp(-x²).
///
/// Derived as ∫_0^∞ K_N(x, y; T) · exp(-y²) dy where K_N is the image-method
/// kernel K(x,y;T) + K(x,-y;T); both Gaussian integrals combine to this form.
/// Verified symbolically by T22N check.
fn oracle(t: f64, x: f64) -> f64 {
    let denom = 1.0 + 4.0 * t;
    (-x * x / denom).exp() / denom.sqrt()
}

// ---------------------------------------------------------------------------
// Operator and region setup
// ---------------------------------------------------------------------------

/// Build the `ReflectedHeatChernoff` for the G27 test.
///
/// Inner: `DiffusionChernoff(a=1, a'=0, a''=0)` — pure heat `∂_t u = ∂_xx u`.
/// Region: `HalfSpaceRegion<f64, 1>` with origin=0, outward normal=+1 (half-line [0,∞)).
/// Grid boundary: `ZeroExtend` (nodes outside [0, GRID_MAX] are treated as 0).
fn build_wrapper(
    grid: Grid1D<f64>,
) -> ReflectedHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRegion<f64, 1>> {
    let inner = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    let region = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).expect("unit normal");
    ReflectedHeatChernoff::new(inner, region).expect("wrapper construction")
}

// ---------------------------------------------------------------------------
// Core evolution function
// ---------------------------------------------------------------------------

/// Evolve the reflected-heat wrapper for `n_steps` steps and return sup-norm
/// error vs the analytical oracle on the interior nodes.
fn sup_error_at(n_spatial: usize, n_steps: usize) -> f64 {
    let tau = T_FINAL / n_steps as f64;

    let grid = Grid1D::new(0.0_f64, GRID_MAX, n_spatial)
        .expect("grid construction")
        .with_boundary(BoundaryPolicy::ZeroExtend);

    let wrapper = build_wrapper(grid);
    let mut u = GridFn1D::from_fn(grid, |x| (-x * x).exp()); // IC: g(x) = exp(-x²)
    let mut scratch = ScratchPool::new();

    for _ in 0..n_steps {
        let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
        wrapper
            .apply_into(tau, &u, &mut dst, &mut scratch)
            .expect("apply_into");
        u = dst;
    }

    // Compute sup-norm error on interior nodes (skip boundary nodes).
    (1..n_spatial - 1)
        .map(|i| (u.values[i] - oracle(T_FINAL, grid.x_at(i))).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// OLS log-log slope: log(err) vs log(n)
// ---------------------------------------------------------------------------

fn ols_slope(n_values: &[usize], errs: &[f64]) -> f64 {
    let m = n_values.len() as f64;
    // Use log(n) as x-axis: as n grows, err shrinks, giving negative slope.
    let log_x: Vec<f64> = n_values.iter().map(|&n| (n as f64).ln()).collect();
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

// ---------------------------------------------------------------------------
// G27 sub-test 1: Residual at N=1024, n_Chernoff=100
// ---------------------------------------------------------------------------

/// G27(1) — Reflected heat residual ≤ 1e-6 at N=1024, n_Chernoff=100, T=0.1.
///
/// Verifies that `ReflectedHeatChernoff<DiffusionChernoff, HalfSpaceRegion<f64,1>>`
/// approximates the Neumann reflected heat with the correct kernel at a fixed resolution.
/// N=1024 required: Catmull-Rom boundary stencil is 2nd-order at x=0 (Reflect symmetry),
/// so spatial error floor at N=64 (~4.4e-3) exceeds the 1e-6 gate; N=1024 gives ~1.5e-7.
#[test]
fn g27_reflected_halfline_residual() {
    const N_SPATIAL: usize = 1024;
    const N_STEPS: usize = 100;

    let tau = T_FINAL / N_STEPS as f64;
    let err = sup_error_at(N_SPATIAL, N_STEPS);

    println!(
        "G27(1): N={N_SPATIAL}, n_steps={N_STEPS}, tau={tau:.4e}, err={err:.4e} (gate ≤ {RESIDUAL_GATE:.0e})"
    );

    assert!(
        err <= RESIDUAL_GATE,
        "G27(1) FAIL: residual {err:.4e} > {RESIDUAL_GATE:.0e}. \
         ReflectedHeatChernoff<DiffusionChernoff, HalfSpaceRegion> \
         does not match oracle at N={N_SPATIAL}, n_steps={N_STEPS}, T={T_FINAL}."
    );

    println!("G27(1) PASS");
}

// ---------------------------------------------------------------------------
// G27 sub-test 2: Slope / order preservation
// ---------------------------------------------------------------------------

/// G27(2) — OLS log-log slope ≤ -0.95 on n ∈ {16, 32, 64, 128}.
///
/// Verifies Proposition 25.1: the reflecting wrapper preserves the order of the
/// inner Chernoff function. `DiffusionChernoff` convergences at O(1/n) per step
/// (global order 1 for Chernoff iteration); the wrapper must not cap this lower.
/// The gate -0.95 gives 5% margin vs the theoretical -1.0 slope.
#[test]
#[allow(clippy::cast_precision_loss)]
fn g27_reflected_halfline_slope() {
    let mut errs = Vec::with_capacity(N_SWEEP.len());

    for &n in &N_SWEEP {
        let tau = T_FINAL / n as f64;
        let err = sup_error_at(n, n);
        println!("G27(2): n={n:4}, tau={tau:.4e}, err={err:.4e}");
        errs.push(err);
    }

    let slope = ols_slope(&N_SWEEP, &errs);
    println!("G27(2): slope={slope:.4} (gate ≤ {SLOPE_GATE})");

    assert!(
        slope <= SLOPE_GATE,
        "G27(2) FAIL: slope {slope:.4} > {SLOPE_GATE}. \
         Reflection wrapper does not preserve inner Chernoff order. \
         errs={errs:?}, n_sweep={N_SWEEP:?}."
    );

    println!("G27(2) PASS");
}
