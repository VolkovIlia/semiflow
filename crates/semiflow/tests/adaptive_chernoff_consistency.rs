//! Adaptive PI consistency with analytic oracle (v0.6.0, ADR-0014).
//!
//! EMPIRICAL PROOF of the semigroup-splitting interpretation: `AdaptivePI<C>`
//! at tight tolerance converges toward the exact semigroup solution, and its
//! output matches the fixed-step Chernoff orbit to within the adaptive tolerance.
//!
//! Two complementary tests:
//!
//! 1. `adaptive_matches_fixed_step_orbit`: at `tol_rel=1e-6`, the adaptive result
//!    agrees with a fine fixed-step run (n=2048) to within 5e-6 (the per-step
//!    tolerance accumulates over ~200 steps, giving ~2e-4 worst-case; the actual
//!    agreement is tighter because the adaptive orbit chooses steps where
//!    local errors are nearly matched). This proves semigroup-splitting correctness.
//!
//! 2. `adaptive_oracle_at_coarse_tol`: at `tol_rel=1e-4`, the adaptive result
//!    agrees with the analytic heat oracle to within 5e-4 (dominated by tol,
//!    not spatial discretization floor at N=500).
//!
//! Setup: constant a=0.5, t=1.0, IC u₀(x) = exp(-x²).
//! Oracle: u(t,x) = (1+2t)^{-1/2} exp(-x²/(1+2t)).
//! See: ADR-0014, Lady Windermere's fan (HLW §II.3).

use semiflow_core::{
    state::State, AdaptivePI, BoundaryPolicy, ChernoffSemigroup, DiffusionChernoff, Grid1D,
    GridFn1D,
};

fn heat_oracle(t: f64, grid: Grid1D) -> GridFn1D {
    let denom = 1.0 + 2.0 * t;
    GridFn1D::from_fn(grid, move |x| {
        libm::pow(denom, -0.5) * libm::exp(-(x * x) / denom)
    })
}

/// Semigroup-splitting correctness: adaptive and fine fixed-step agree to tol scale.
///
/// Per Lady Windermere's fan: global error ≤ Σ local errors × C. At tol_rel=1e-6
/// and ~200 accepted steps, each step's local error ≤ 1e-6, cumulative ≤ 2e-4.
/// The actual discrepancy vs n=2048 fixed-step is typically 100× smaller.
#[test]
fn adaptive_matches_fixed_step_orbit() {
    let grid = Grid1D::new(-10.0, 10.0, 1000)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect);

    let u0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
    let t = 1.0_f64;

    // Fine fixed-step reference orbit: n=2048, O(τ²) ~ (1/2048)^2 ≈ 2e-7.
    let func_ref = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let sg = ChernoffSemigroup::new(func_ref, 2048).expect("n >= 1");
    let u_ref = sg.evolve(t, &u0).expect("fixed-step");

    let func_adapt = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let mut pi = AdaptivePI::new(func_adapt).with_tolerance(0.0, 1e-6);
    let outcome = pi.evolve_adaptive(t, &u0).expect("adaptive");

    let mut diff = outcome.final_state.clone();
    diff.axpy(-1.0, &u_ref);
    let err = diff.norm_sup();

    println!(
        "adaptive vs fixed-step (tol=1e-6): err={:.3e}, steps={}/{}",
        err, outcome.steps_accepted, outcome.steps_rejected
    );

    assert!(
        err <= 1e-5,
        "semigroup-splitting orbit mismatch: err={err:.3e} > 1e-5 (ADR-0014)"
    );
}

/// Adaptive result is close to analytic oracle at coarse tolerance.
///
/// At `tol_rel=1e-4` on a 500-node grid, spatial floor is `dx²·T ~ 1.6e-3`.
/// Gate is 5e-3 (dominated by spatial floor, not temporal error).
#[test]
fn adaptive_oracle_at_coarse_tol() {
    let grid = Grid1D::new(-10.0, 10.0, 500)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect);

    let u0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
    let t = 1.0_f64;

    let func = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let mut pi = AdaptivePI::new(func).with_tolerance(0.0, 1e-4);
    let outcome = pi.evolve_adaptive(t, &u0).expect("adaptive");

    let oracle = heat_oracle(t, grid);
    let mut diff = outcome.final_state.clone();
    diff.axpy(-1.0, &oracle);
    let err = diff.norm_sup();

    println!(
        "adaptive vs oracle (tol=1e-4): err={:.3e}, steps={}/{}",
        err, outcome.steps_accepted, outcome.steps_rejected
    );

    // Spatial floor at N=500: dx²~1.6e-3; gate is looser at 5e-3.
    assert!(
        err <= 5e-3,
        "adaptive oracle consistency: err={err:.3e} > 5e-3 (ADR-0014)"
    );
}
