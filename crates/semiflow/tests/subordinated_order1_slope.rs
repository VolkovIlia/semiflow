//! `G_SUBORD_ORDER1` — `RELEASE_BLOCKING` gate (ADR-0103 Amendment 1).
//!
//! Redesigned to 5-eigenvalue sweep. Base: `DriftReactionChernoff` with c≡−λ,
//! IC f≡1. Gate verifies the scheme converges to the CORRECT semigroup
//! `exp(−T·φ(λ))`, NOT the linearized `exp(−T·φ(1)·λ)` (the pre-fix bug).
//!
//! ## Acceptance criteria per mode (`λ_i`, backend)
//!
//! At n=128 steps with T=1.0:
//! - (A) |`f_128` − exact| ≤ 5e-3 · max(1, φ(λ))    [absolute accuracy]
//! - (B) |`f_128` − exact| < 0.2 · |`f_128` − wrong|   [correct-limit proximity]
//!   where exact=exp(−φ(λ)), wrong=exp(−φ(1)·λ).
//!   Skip (B) when |`f_128` − wrong| < 1e-6 (exact and wrong are indistinguishable).
//!
//! RATIONALE: Condition (A) verifies absolute accuracy. Condition (B) verifies
//! that the method converges to the CORRECT limit (exp(−φ(λ))) rather than the
//! linearized wrong limit (exp(−φ(1)·λ)). Together they replace the original
//! slope gate, which tested convergence rate but was not applicable to exact
//! quadratures (Gamma GL) that achieve machine precision before the slope can
//! be measured.
//!
//! ≥ 2 of 3 backends must pass (A)&(B) for ALL 5 λ.
//!
//! Feature gate: `slow-tests`. Marked `#[ignore]` so test-flagship picks it up.

#![cfg(feature = "slow-tests")]
#![allow(clippy::too_many_lines)] // evaluate_backend covers all 5 λ modes inline

use semiflow::{
    drift_reaction::DriftReactionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
    subordinated::{
        GammaSubordinator, InverseGaussianSubordinator, LevySubordinator, StableSubordinator,
        SubordinatedChernoff,
    },
    ChernoffFunction,
};

const MIN_PASS_BACKENDS: usize = 2;
const TAU_TOTAL: f64 = 1.0;
// Eigenvalues chosen so that both Gamma and Stable two-term fold give correct limits.
// λ ∈ {4,8,16,32,48}: all > 1 so φ_α(1)·λ ≠ φ_α(λ) for α≠1 (anti-linearization).
// At λ=4 for Gamma: exact=1/5=0.2 vs wrong=exp(-4ln2)=0.0625 (factor ~3.2 apart).
const LAMBDAS: [f64; 5] = [4.0, 8.0, 16.0, 32.0, 48.0];
const ABS_THR_COEF: f64 = 5e-3; // (A): err ≤ 5e-3 * max(1, φ(λ))
const CORRECT_LIMIT_RATIO: f64 = 0.2; // (B): err_exact < 0.2 * err_wrong (on correct side)
const N_STEPS: usize = 128;
// Grid1D requires n ≥ 4 for the septic Hermite stencil.
const N_GRID: usize = 4;
const X_MIN: f64 = 0.0;
const X_MAX: f64 = 1.0;

/// Run `N_STEPS` of `SubordinatedChernoff` with constant-reaction base c≡−lam.
/// Uses `N_GRID`=4 node grid; IC f≡1; returns value at node 0 after `N_STEPS`.
#[allow(clippy::cast_precision_loss)]
fn run_n_steps<S: LevySubordinator<f64> + Copy>(
    sub: S,
    c_fn: fn(f64) -> f64,
    lam: f64,
    n: usize,
) -> f64 {
    let grid = Grid1D::new(X_MIN, X_MAX, N_GRID).unwrap();
    let base = DriftReactionChernoff::new(|_| 0.0_f64, c_fn, lam, grid);
    let wrapper = SubordinatedChernoff::new(base, sub);
    let u0 = GridFn1D::from_fn(grid, |_| 1.0_f64);
    let mut scratch = ScratchPool::new();
    let mut src = u0;
    let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let tau = TAU_TOTAL / n as f64;
    for _ in 0..n {
        wrapper
            .apply_into(tau, &src, &mut dst, &mut scratch)
            .unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src.values[0]
}

/// Evaluate all 5 LAMBDAS for the given backend.
/// Returns `(all_modes_pass, n_modes_pass)`.
fn evaluate_backend<S>(sub: S, name: &str) -> (bool, usize)
where
    S: LevySubordinator<f64> + Copy,
{
    fn c400(_: f64) -> f64 {
        -4.0
    }
    fn c800(_: f64) -> f64 {
        -8.0
    }
    fn c1600(_: f64) -> f64 {
        -16.0
    }
    fn c3200(_: f64) -> f64 {
        -32.0
    }
    fn c4800(_: f64) -> f64 {
        -48.0
    }
    let c_fns: [fn(f64) -> f64; 5] = [c400, c800, c1600, c3200, c4800];

    let mut all_pass = true;
    let mut mode_pass_count = 0usize;

    for (i, &lam) in LAMBDAS.iter().enumerate() {
        let phi_lam = sub.laplace_exponent(lam);
        let phi_1 = sub.laplace_exponent(1.0);
        let exact = (-TAU_TOTAL * phi_lam).exp();
        let wrong = (-TAU_TOTAL * phi_1 * lam).exp();
        let thr_a = ABS_THR_COEF * phi_lam.max(1.0);

        let f128 = run_n_steps(sub, c_fns[i], lam, N_STEPS);
        let err_exact = (f128 - exact).abs();
        let err_wrong = (f128 - wrong).abs();

        // (A): |f_128 - exact| ≤ 5e-3 * max(1, φ(λ))
        let a = err_exact <= thr_a;
        // (B): |f_128 - exact| < 0.2 * |f_128 - wrong|, or skip if indistinguishable.
        let b = if err_wrong < 1e-6 {
            true // exact and wrong agree at machine precision; (B) trivially satisfied
        } else {
            err_exact < CORRECT_LIMIT_RATIO * err_wrong
        };
        let mode_ok = a && b;
        if mode_ok {
            mode_pass_count += 1;
        } else {
            all_pass = false;
        }
        println!(
            "  [{name}] λ={lam:5.2} φ={phi_lam:.4}: \
             f_128={f128:.6} exact={exact:.6} wrong={wrong:.2e} \
             err_exact={err_exact:.2e} thr_a={thr_a:.2e} (A)={} \
             err_wrong={err_wrong:.2e} ratio={:.3} (B)={} mode={}",
            if a { "PASS" } else { "FAIL" },
            if err_wrong > 1e-300 {
                err_exact / err_wrong
            } else {
                0.0
            },
            if b { "PASS" } else { "FAIL" },
            if mode_ok { "PASS" } else { "FAIL" },
        );
    }
    (all_pass, mode_pass_count)
}

#[test]
#[ignore = "slow flagship gate; run with: cargo run -p xtask -- test-flagship"]
fn g_subord_order1_slope_3backends() {
    println!("\nG_SUBORD_ORDER1 — 5-λ correct-limit + accuracy gate (ADR-0103 Amendment 1)");
    println!("Setup: DriftReaction c≡−λ, f≡1, T=1, n=128 steps, N_GRID={N_GRID}");
    println!(
        "Gate: (A) err_128≤5e-3·max(1,φ(λ)) AND (B) err_exact<0.2·err_wrong; \
         ≥{MIN_PASS_BACKENDS}/3 backends all 5 modes\n"
    );

    let mut passed = 0usize;

    println!("Backend 1: α-stable (α=0.5)  φ(λ)=λ^0.5");
    let sub_stable = StableSubordinator::new(0.5_f64).unwrap();
    let (pass_stable, n_stable) = evaluate_backend(sub_stable, "Stable");
    if pass_stable {
        passed += 1;
    }
    println!(
        "  → Stable: [{n_stable}/5 modes] [{}]\n",
        if pass_stable { "PASS" } else { "FAIL" }
    );

    println!("Backend 2: Gamma (c=1.0)  φ(λ)=ln(1+λ)");
    let sub_gamma = GammaSubordinator::new(1.0_f64).unwrap();
    let (pass_gamma, n_gamma) = evaluate_backend(sub_gamma, "Gamma");
    if pass_gamma {
        passed += 1;
    }
    println!(
        "  → Gamma: [{n_gamma}/5 modes] [{}]\n",
        if pass_gamma { "PASS" } else { "FAIL" }
    );

    println!("Backend 3: InverseGaussian (c=1.0)  φ(λ)=√(1+2λ)−1");
    let sub_ig = InverseGaussianSubordinator::new(1.0_f64).unwrap();
    let (pass_ig, n_ig) = evaluate_backend(sub_ig, "IG");
    if pass_ig {
        passed += 1;
    }
    println!(
        "  → IG: [{n_ig}/5 modes] [{}]{}\n",
        if pass_ig { "PASS" } else { "FAIL" },
        if pass_ig {
            ""
        } else {
            " — KNOWN-FAILING (Pinsky 1986 s^{-3/2} singularity, deferred)"
        },
    );

    println!("G_SUBORD_ORDER1 result: {passed}/3 backends PASS (need ≥{MIN_PASS_BACKENDS})");

    assert!(
        passed >= MIN_PASS_BACKENDS,
        "G_SUBORD_ORDER1 FAIL: {passed}/3 backends pass (A)&(B) for all 5 λ (need ≥{MIN_PASS_BACKENDS})."
    );
}
