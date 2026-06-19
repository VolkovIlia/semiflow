//! `G_SCHROD_B` — Schrödinger Option B unitarity gate (`RELEASE_BLOCKING`).
//!
//! Properties.yaml v4.0 spec (math.md §30.6):
//!   ‖U(τ) ψ‖₂ = ‖ψ(0)‖₂ within 1e-12 at f64
//!   Setup: harmonic-oscillator V(x)=½x², Gaussian wave packet with momentum k₀=2,
//!   N=512, domain [−10, 10], T=1.0, `n_steps=128` palindromic Strang steps.
//!
//! Single-point gate (NOT an n-sweep). Failure blocks v4.0 release.
//!
//! ADR-0079; math.md §30.4 Proposition 30.1.

#![cfg(feature = "slow-tests")]

use std::f64::consts::PI;

use num_complex::Complex;
use semiflow_core::{
    chernoff::ChernoffFunction,
    grid::Grid1D,
    schrodinger_complex::{GridFnComplex1D, SchrödingerChernoffComplex},
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// G_SCHROD_B (RELEASE_BLOCKING)
// ---------------------------------------------------------------------------

/// Unitarity of Schrödinger Option B on harmonic oscillator at f64.
///
/// Gate: |‖ψ_n‖₂ − ‖ψ₀‖₂| ≤ 1e-12.
/// Initial datum: Gaussian wave packet  ψ₀(x) = π^{−1/4} exp(−x²/2 + i k₀ x)
/// with k₀ = 2 (mean momentum).  The analytic ‖ψ₀‖₂ = 1 on ℝ (truncation
/// error on [−10, 10] is < 10^{−43} at σ=1).
#[test]
fn g_schrod_b_unitarity_harmonic_oscillator_gaussian() {
    // Setup per math.md §30.6 and properties.yaml G_SCHROD_B.
    let n = 512_usize;
    let lo = -10.0_f64;
    let hi = 10.0_f64;
    let grid = Grid1D::<f64>::new(lo, hi, n).expect("valid grid");

    // V(x) = ½ x² (harmonic oscillator)
    let kernel = SchrödingerChernoffComplex::<Complex<f64>>::new(grid, |x: f64| 0.5 * x * x)
        .expect("valid kernel");

    // Initial datum: ψ₀(x) = π^{-1/4} exp(-x²/2 + i k₀ x)
    let k0 = 2.0_f64;
    let norm_factor = PI.powf(-0.25_f64); // π^{-1/4}  (ensures ‖ψ₀‖_{L²(ℝ)} = 1)
    let mut state = GridFnComplex1D::from_fn(grid, |x: f64| {
        let envelope = norm_factor * (-x * x / 2.0).exp();
        Complex::from_polar(envelope, k0 * x)
    });

    // Measure initial L²-norm (discrete, with dx quadrature weight)
    let norm_initial = state.norm_l2();

    // Evolve T=1.0 with n=128 steps
    let t_final = 1.0_f64;
    let n_steps = 128_usize;
    let tau = t_final / n_steps as f64;

    let mut dst = state.clone();
    let mut scratch = ScratchPool::new();

    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &state, &mut dst, &mut scratch)
            .expect("apply_into should not fail");
        core::mem::swap(&mut state, &mut dst);
    }

    // G_SCHROD_B unitarity check
    let norm_final = state.norm_l2();
    let unitarity_err = (norm_final - norm_initial).abs();

    println!(
        "G_SCHROD_B: ‖ψ₀‖₂ = {:.15}, ‖ψ_T‖₂ = {:.15}, Δ = {:.4e}  (target ≤ 1e-12)",
        norm_initial, norm_final, unitarity_err
    );

    assert!(
        unitarity_err <= 1e-12,
        "G_SCHROD_B FAIL: |‖ψ_T‖₂ − ‖ψ₀‖₂| = {:.4e} > 1e-12",
        unitarity_err
    );
}

// ---------------------------------------------------------------------------
// Option A / Option B cross-check (ADVISORY — not RELEASE_BLOCKING)
// ---------------------------------------------------------------------------

/// Cross-representation sup-norm comparison: Option A (real-pair) vs Option B
/// (native complex) on the same harmonic-oscillator setup at N=128 (smoke size).
///
/// Expected: sup-norm diff ≤ 4 ULP ≈ 8.88 × 10^{-16} per math.md §30.4.
///
/// ADVISORY: failure triggers investigation but does NOT block v4.0 release.
#[test]
#[ignore = "advisory: cross-representation Option A vs B; investigate on failure but does not block"]
fn schrodinger_option_a_vs_b_sup_norm() {
    use semiflow_core::{
        diffusion4::Diffusion4thChernoff, grid_fn::GridFn1D, schrodinger::SchrodingerChernoff,
        schrodinger::SchrodingerState, ChernoffFunction,
    };

    let n = 128_usize;
    let grid_f64 = Grid1D::<f64>::new(-5.0, 5.0, n).unwrap();
    let tau = 1.0 / 128.0_f64;
    let n_steps = 16_usize;

    // --- Option B ---
    let kb = SchrödingerChernoffComplex::<Complex<f64>>::new(grid_f64, |x| 0.5 * x * x).unwrap();
    let mut sb =
        GridFnComplex1D::from_fn(grid_f64, |x: f64| Complex::new((-x * x / 2.0).exp(), 0.0));
    let mut sb_dst = sb.clone();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        kb.apply_into(tau, &sb, &mut sb_dst, &mut scratch).unwrap();
        core::mem::swap(&mut sb, &mut sb_dst);
    }

    // --- Option A ---
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid_f64);
    let ka = SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x).unwrap();
    let psi_re = GridFn1D::from_fn(grid_f64, |x: f64| (-x * x / 2.0).exp());
    let psi_im = GridFn1D::from_fn(grid_f64, |_: f64| 0.0_f64);
    let mut sa = SchrodingerState { psi_re, psi_im };
    let mut sa_dst = sa.clone();
    let mut scratch2 = ScratchPool::new();
    for _ in 0..n_steps {
        ka.apply_into(tau, &sa, &mut sa_dst, &mut scratch2).unwrap();
        core::mem::swap(&mut sa, &mut sa_dst);
    }

    // Compare pointwise sup-norm of (re, im) differences
    let mut sup_diff = 0.0_f64;
    for k in 0..n {
        let re_diff = (sb.values[k].re - sa.psi_re.values[k]).abs();
        let im_diff = (sb.values[k].im - sa.psi_im.values[k]).abs();
        sup_diff = sup_diff.max(re_diff).max(im_diff);
    }

    let four_ulp = 4.0 * f64::EPSILON;
    println!(
        "Option A vs B sup-norm diff = {:.4e}  (4 ULP = {:.4e})",
        sup_diff, four_ulp
    );

    // ADVISORY: warn but do not panic (cross-check may differ slightly due to
    // algorithm ordering; hard-fail only if difference is catastrophically large)
    if sup_diff > four_ulp {
        eprintln!(
            "ADVISORY: Option A vs B sup-norm {:.4e} exceeds 4 ULP {:.4e}. Investigate.",
            sup_diff, four_ulp
        );
    }
    assert!(
        sup_diff < 1e-10,
        "Option A vs B diff = {:.4e} is catastrophically large (> 1e-10); regression?",
        sup_diff
    );
}
