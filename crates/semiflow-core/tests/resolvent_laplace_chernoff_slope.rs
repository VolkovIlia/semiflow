// Integration test: allows for continuation-after-list-item in doc comments.
#![allow(clippy::doc_lazy_continuation)]
//! G24 — Laplace-Chernoff Resolvent residual + slope (math.md §22.5, ADR-0069).
//!
//! Gate (properties.yaml v0.9.0 G24, `RELEASE_BLOCKING)`:
//!   (1) Residual sub-test: ||(λI − `A_cont`) `R̃_n` g − g||_∞ ≤ 1e-3
//!       at n=64, λ=1.0, Gaussian IC on [-5,5] N=512 (Reflect BC).
//!   (2) Slope sub-test: OLS of `log(||R̃_n` g − R̃_{`n_ref`} g||) vs log(n) ≥ 1.0.
//!       Self-convergence with `n_ref` = `N_REF_SLOPE` = 4096 as proxy for n→∞.
//!       Sweep n ∈ {16, 32, 64, 128}; variable a(x) to expose Chernoff errors.
//!       `DiffusionChernoff` is order-2 (C(τ) = e^{τA} + O(τ³)) → OLS slope ≈ −3.
//!       Gate ≥ 1.0 provides >2 decades of safety margin.
//!   (3) Stress sub-test: `TrapezoidWithTail` at λ=0.01, no panic, Ok result.
//!
//! All three sub-tests must pass. Feature gate: `slow-tests`.
//!
//! ## G24(1) — Residual BC choice (Reflect)
//!
//! Reflect BC: Gaussian shift queries reflected at boundaries. For small-τ steps
//! (τ = `s_k/(n·λ)` with n=64 and GL32 nodes `s_k` ≤ 111), the domain [-5,5] with
//! Reflect BC gives residual < 1e-3 at n=64 (verified numerically).
//!
//! ## G24(2) — Slope test design
//!
//! For constant a=1 and Gaussian IC, `DiffusionChernoff` is essentially exact:
//! the Gaussian is in the eigenspace of the constant-coefficient heat kernel,
//! causing higher-order cancellations that push the Chernoff error well below
//! the O(τ³) bound. Self-convergence errors reach machine precision at n=64.
//!
//! To expose the genuine O(1/n²) Chernoff convergence, the slope test uses:
//!   - Variable coefficient a(x) = 1 + 0.5·x·exp(-x²) (bounded in [0.74, 1.22])
//!   - `a_prime(x)` = 0.5·(1 - 2x²)·exp(-x²)
//!   - `a_double_prime(x)` = 0.5·(-6x + 4x³)·exp(-x²)
//! For this a(x), the ζ-A τ²-correction is non-trivial, ensuring the leading
//! Chernoff error is exactly O(τ³). Self-convergence then gives:
//!   err(n) = ||`R̃_n` g − R̃_{`n_ref`} g||_∞ ~ C/n²
//! Numerically (`n_ref=4096)`: errs ≈ {3.20e-2, 6.15e-3, 3.05e-4, 1.04e-5}
//! for n ∈ {16, 32, 64, 128} → OLS slope ≈ −3.9 (rate ≈ 3.9 >> 1.0).
//!
//! See Also: G24(1) uses constant a=1 for the residual gate (n=64 is sufficient
//! for the 1e-3 residual bound; the slope gate is tested separately above).

#![cfg(feature = "slow-tests")]

use semiflow_core::{
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
    DiffusionChernoff, Grid1D, GridFn1D,
};

// ---------------------------------------------------------------------------
// Gate constants — do NOT relax without ADR + properties.yaml bump.
// ---------------------------------------------------------------------------

const RESIDUAL_GATE: f64 = 1e-3;
const SLOPE_GATE: f64 = 1.0;
const LAMBDA_MAIN: f64 = 1.0;
const LAMBDA_STRESS: f64 = 0.01;
const N_RESIDUAL: usize = 64;
const N_SPATIAL: usize = 512;
const N_SPATIAL_STRESS: usize = 256;
/// Chernoff truncation sweep for the slope test (per gate spec).
///
/// n ∈ {16, 32, 64, 128}: in the asymptotic Chernoff regime for variable a.
/// With variable a(x), self-conv errors are {3.2e-2, 6.2e-3, 3.1e-4, 1.0e-5}
/// giving OLS slope ≈ −3.9 >> gate −1.0.
const N_SWEEP: [usize; 4] = [16, 32, 64, 128];
/// Reference n for self-convergence: n_ref contribution ~ C/4096² ≈ negligible.
const N_REF_SLOPE: usize = 4096;
/// Domain for all tests: [-L, L] with L=5.
const DOMAIN_L: f64 = 5.0;
/// a_norm_bound for the variable-coefficient slope test: max|a(x)| ≤ 1.5.
const A_NORM_BOUND: f64 = 1.5;

// ---------------------------------------------------------------------------
// Variable coefficient for G24(2)
//
// a(x) = 1 + 0.5 · x · exp(-x²)
//   → range: [1 + 0.5·min(x·exp(-x²)), 1 + 0.5·max(x·exp(-x²))]
//           = [1 − 0.5/(2√e), 1 + 0.5/(2√e)] ≈ [0.848, 1.152] ⊂ (0, ∞)
//
// a'(x) = 0.5·(1 - 2x²)·exp(-x²)
// a''(x) = 0.5·(-6x + 4x³)·exp(-x²)
//
// These are C∞, the ζ-A τ²-correction is non-trivial (a' ≠ 0),
// so the Chernoff formula gives a genuine O(τ³) leading error.
// ---------------------------------------------------------------------------

fn a_slope(x: f64) -> f64 {
    1.0 + 0.5 * x * (-x * x).exp()
}

fn a_prime_slope(x: f64) -> f64 {
    0.5 * (1.0 - 2.0 * x * x) * (-x * x).exp()
}

fn a_double_prime_slope(x: f64) -> f64 {
    0.5 * (-6.0 * x + 4.0 * x * x * x) * (-x * x).exp()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn gaussian_ic(x: f64) -> f64 {
    (-x * x).exp()
}

/// OLS slope of log(y) vs log(x). Both slices must be non-empty and equal length.
#[allow(clippy::cast_precision_loss)]
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let log_x: Vec<f64> = xs.iter().map(|&v| v.ln()).collect();
    let log_y: Vec<f64> = ys.iter().map(|&v| v.ln()).collect();
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
// G24 sub-test (1): residual ≤ 1e-3 (Reflect BC)
// ---------------------------------------------------------------------------

/// G24(1) — resolvent identity residual ≤ 1e-3 at n=64, λ=1.0 (Reflect BC).
///
/// Checks: ||(λI − A_cont) R̃_n(λ) g − g||_∞ ≤ 1e-3 on interior nodes.
/// A_cont = ∂_xx approximated by central FD (O(dx²) ~ 4e-4 for N=512).
/// With Reflect BC and constant a=1, the resolvent identity residual < 1e-3
/// at n=64 (verified numerically: residual = 1.57e-4).
#[test]
fn g24_resolvent_residual_at_n_64() {
    // Reflect BC (default): gives smaller resolvent identity residual than ZeroExtend.
    let grid = Grid1D::new(-DOMAIN_L, DOMAIN_L, N_SPATIAL).unwrap();
    let dx = grid.dx();
    let diff = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let resolvent =
        LaplaceChernoffResolvent::new(diff, N_RESIDUAL, LaplaceQuadrature::GaussLaguerre32)
            .unwrap();

    let g = GridFn1D::from_fn(grid, gaussian_ic);
    let r_g = resolvent.eval(LAMBDA_MAIN, &g).unwrap();

    let r_vals = &r_g.values;
    let g_vals = &g.values;
    let n = r_vals.len();

    let max_err: f64 = (1..n - 1)
        .map(|i| {
            let lap = (r_vals[i + 1] - 2.0 * r_vals[i] + r_vals[i - 1]) / (dx * dx);
            let lhs = LAMBDA_MAIN * r_vals[i] - lap;
            (lhs - g_vals[i]).abs()
        })
        .fold(0.0, f64::max);

    println!(
        "G24(1) residual = {:.6e}  (gate ≤ {:.0e})",
        max_err, RESIDUAL_GATE
    );
    assert!(
        max_err <= RESIDUAL_GATE,
        "G24(1) FAIL: residual {max_err:.6e} > {RESIDUAL_GATE:.0e}"
    );
}

// ---------------------------------------------------------------------------
// G24 sub-test (2): slope ≥ 1.0 (self-convergence, variable a, n ∈ {16..128})
// ---------------------------------------------------------------------------

/// G24(2) — slope ≥ 1.0: OLS of log(||R̃_n g − R̃_{n_ref} g||) vs log(n).
///
/// Uses variable a(x) = 1 + 0.5·x·exp(-x²) so the ζ-A τ²-correction is
/// non-trivial and the genuine O(τ³) Chernoff error is exposed.
/// For constant a=1 and Gaussian IC, higher-order cancellations make the
/// Chernoff error negligibly small at n=64 (machine precision), masking slope.
///
/// Self-convergence with n_ref = N_REF_SLOPE = 4096 eliminates the O(dx²)
/// discrete/continuum gap that would otherwise create a fixed floor.
///
/// Theoretical rate for order-2 Chernoff:
///   ||(C(τ))^n − e^{nτA}|| = O(n·τ³) = O(T³/n²)  for T = s_k/λ.
/// GL32-weighted: ||R̃_n − R̃_{n_ref}|| = O(1/n²).
/// Observed OLS slope ≈ −3.9; gate ≥ −1.0 gives >2-decade safety margin.
#[test]
fn g24_resolvent_slope() {
    let grid = Grid1D::new(-DOMAIN_L, DOMAIN_L, N_SPATIAL).unwrap();
    let g = GridFn1D::from_fn(grid, gaussian_ic);

    // Reference: n_ref = 4096 (contribution ~ C/n_ref² ≈ negligible vs n=128 error).
    let diff_ref = DiffusionChernoff::new(
        a_slope,
        a_prime_slope,
        a_double_prime_slope,
        A_NORM_BOUND,
        grid,
    );
    let resolvent_ref =
        LaplaceChernoffResolvent::new(diff_ref, N_REF_SLOPE, LaplaceQuadrature::GaussLaguerre32)
            .unwrap();
    let r_ref = resolvent_ref.eval(LAMBDA_MAIN, &g).unwrap();

    // Sweep n values ∈ {16, 32, 64, 128} — all in Chernoff-dominated regime.
    let mut errs: Vec<f64> = Vec::with_capacity(N_SWEEP.len());
    for &n in &N_SWEEP {
        let diff = DiffusionChernoff::new(
            a_slope,
            a_prime_slope,
            a_double_prime_slope,
            A_NORM_BOUND,
            grid,
        );
        let resolvent =
            LaplaceChernoffResolvent::new(diff, n, LaplaceQuadrature::GaussLaguerre32).unwrap();
        let r_n = resolvent.eval(LAMBDA_MAIN, &g).unwrap();
        // Sup-norm on interior nodes (skip endpoints: Gaussian ≈ 0 there).
        let err = (1..N_SPATIAL - 1)
            .map(|i| (r_n.values[i] - r_ref.values[i]).abs())
            .fold(0.0_f64, f64::max);
        println!("G24(2) n={n:4}: err = {err:.4e}");
        errs.push(err);
    }

    let ns_f64: Vec<f64> = N_SWEEP.iter().map(|&n| n as f64).collect();
    let slope = ols_slope(&ns_f64, &errs);
    let rate = -slope; // convergence rate: err ~ 1/n^r → rate = r (gate: r ≥ 1.0)

    println!(
        "G24(2) log(err) vs log(n) slope = {slope:.4}, rate = {rate:.4}  (gate ≥ {SLOPE_GATE})"
    );
    assert!(
        rate >= SLOPE_GATE,
        "G24(2) FAIL: convergence rate {rate:.4} < {SLOPE_GATE}. \
         Expected rate ≈ 2.0 from order-2 Chernoff; actually ≈ 3.9 for variable-a. \
         If rate < 1: check n_ref={N_REF_SLOPE} is large enough, \
         or verify a_slope / a_prime_slope / a_double_prime_slope are consistent."
    );
}

// ---------------------------------------------------------------------------
// G24 sub-test (3): TrapezoidWithTail stress at λ=0.01
// ---------------------------------------------------------------------------

/// G24(3) — stress test: TrapezoidWithTail fallback at λ=0.01 (marginal regime).
///
/// ω=0 for DiffusionChernoff(a=1). λ=0.01 is near-marginal.
/// t_max = 5000 → exp(-λ·t_max) = exp(-50) ≪ 1e-20 (tail negligible).
/// Gate: no panic, Ok result.
#[test]
fn g24_trapezoid_stress_small_lambda() {
    let grid = Grid1D::new(-DOMAIN_L, DOMAIN_L, N_SPATIAL_STRESS).unwrap();
    let diff = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let t_max = 5000.0_f64;
    let resolvent = LaplaceChernoffResolvent::new(
        diff,
        N_RESIDUAL,
        LaplaceQuadrature::TrapezoidWithTail { t_max },
    )
    .unwrap();

    let g = GridFn1D::from_fn(grid, gaussian_ic);
    let result = resolvent.eval(LAMBDA_STRESS, &g);
    assert!(
        result.is_ok(),
        "G24(3) FAIL: TrapezoidWithTail at λ=0.01 returned error: {:?}",
        result.err()
    );

    let r_g = result.unwrap();
    let max_val = r_g.values.iter().cloned().fold(0.0_f64, f64::max);
    println!("G24(3) TrapezoidWithTail at λ={LAMBDA_STRESS}: max(R̃g) = {max_val:.4e}  [stress OK]");
}
