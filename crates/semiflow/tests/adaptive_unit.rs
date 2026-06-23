//! Unit tests for `AdaptivePI<C>` (v0.6.0, ADR-0014).
//!
//! Six tests:
//! 1. `default_construction` — defaults match ADR-0014 for τ-order-2 inner (post-D1, math.md §11.1.bis).
//! 2. `with_tolerance_builder` — builder overrides `tol_abs` and `tol_rel`.
//! 3. `accepts_loose_tol` — at tol_rel=1e-2, smooth heat: few steps, no rejects.
//! 4. `accepts_strict_tol` — at tol_rel=1e-8 (post-D1 p=2), smooth heat: many steps taken.
//! 5. `runaway_protection` — `max_substeps=5` with impossibly tight tol triggers
//!    `AdaptiveStepRejected`.
//! 6. `last_tau_is_finite` — `outcome.last_tau` is always finite after clean run.

use semiflow_core::{
    boundary::InterpKind, AdaptivePI, BoundaryPolicy, Diffusion4thChernoff, DiffusionChernoff,
    Grid1D, GridFn1D, SemiflowError,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn heat_grid(n: usize) -> Grid1D {
    // Explicitly pin to CubicHermite: these AdaptivePI unit tests are calibrated
    // against the v1.0.0 step-count baselines (e.g. ≥200 steps for strict tol).
    // v6.0 changed Grid1D::new default to SepticHermite (ADR-0109), which reduces
    // spatial error and allows larger time-steps — but the step-count gates would
    // then need re-calibration. Pin CubicHermite so the AdaptivePI logic is tested
    // independently of the spatial kernel change.
    Grid1D::new(-10.0, 10.0, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect)
        .with_interp(InterpKind::CubicHermite)
}

fn gaussian_ic(grid: Grid1D) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| libm::exp(-x * x / 2.0))
}

fn d4_const(grid: Grid1D) -> Diffusion4thChernoff {
    Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid)
}

fn d2_const(grid: Grid1D) -> DiffusionChernoff {
    DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid)
}

// ---------------------------------------------------------------------------
// Test 1: default construction
// ---------------------------------------------------------------------------

#[test]
fn default_construction() {
    let grid = heat_grid(100);
    let d4 = d4_const(grid);
    let pi = AdaptivePI::new(d4);

    // τ-order-2 inner (post-D1, math.md §11.1.bis): alpha = 0.7/2 = 0.35, beta = 0.4/2 = 0.2
    let expected_alpha = 0.7 / 2.0;
    let expected_beta = 0.4 / 2.0;

    assert!(
        (pi.alpha() - expected_alpha).abs() < 1e-15,
        "alpha = {} != 0.35",
        pi.alpha()
    );
    assert!(
        (pi.beta() - expected_beta).abs() < 1e-15,
        "beta = {} != 0.2",
        pi.beta()
    );
    assert!((pi.safety - 0.9).abs() < 1e-15);
    assert!((pi.min_ratio - 0.2).abs() < 1e-15);
    assert!((pi.max_ratio - 5.0).abs() < 1e-15);
    assert_eq!(pi.max_substeps, 100_000);
    assert!((pi.tol_abs - 1e-8).abs() < 1e-20);
    assert!((pi.tol_rel - 1e-6).abs() < 1e-18);
}

// ---------------------------------------------------------------------------
// Test 2: with_tolerance builder
// ---------------------------------------------------------------------------

#[test]
fn with_tolerance_builder() {
    let grid = heat_grid(100);
    let d4 = d4_const(grid);
    let pi = AdaptivePI::new(d4).with_tolerance(1e-10, 1e-8);
    assert!((pi.tol_abs - 1e-10).abs() < 1e-22, "tol_abs not overridden");
    assert!((pi.tol_rel - 1e-8).abs() < 1e-20, "tol_rel not overridden");
    // Safety and gains unchanged.
    assert!((pi.safety - 0.9).abs() < 1e-15);
    let _ = pi.alpha(); // smoke: accessor compiles
}

// ---------------------------------------------------------------------------
// Test 3: accepts_loose_tol
// ---------------------------------------------------------------------------

#[test]
fn accepts_loose_tol() {
    let grid = heat_grid(400);
    let d4 = d4_const(grid);
    let mut pi = AdaptivePI::new(d4).with_tolerance(0.0, 1e-2);
    let u0 = gaussian_ic(grid);

    let outcome = pi
        .evolve_adaptive(1.0, &u0)
        .expect("should not error at loose tol");
    assert!(
        outcome.steps_accepted <= 30,
        "loose tol: expected ≤30 accepted steps, got {}",
        outcome.steps_accepted
    );
    assert_eq!(
        outcome.steps_rejected, 0,
        "loose tol: expected 0 rejected steps"
    );
}

// ---------------------------------------------------------------------------
// Test 4: accepts_strict_tol
// ---------------------------------------------------------------------------

#[test]
fn accepts_strict_tol() {
    // tol_rel=1e-10 was too tight for p=2 (divisor=3) within max_substeps=100k.
    // Post-D1: Diffusion4thChernoff reports τ-order 2; Richardson divisor is 3
    // (not 15 as with the old buggy p=4). Use tol_rel=1e-8 — still "strict"
    // (forces ≥200 accepted steps) but achievable in <100k steps. See math.md §11.1.bis.
    let grid = heat_grid(400);
    let d4 = d4_const(grid);
    let mut pi = AdaptivePI::new(d4).with_tolerance(0.0, 1e-8);
    let u0 = gaussian_ic(grid);

    let outcome = pi
        .evolve_adaptive(1.0, &u0)
        .expect("should not error at strict tol");
    assert!(
        outcome.steps_accepted >= 200,
        "strict tol: expected ≥200 accepted steps, got {}",
        outcome.steps_accepted
    );
}

// ---------------------------------------------------------------------------
// Test 5: runaway_protection
// ---------------------------------------------------------------------------

#[test]
fn runaway_protection() {
    let grid = heat_grid(200);
    let d2 = d2_const(grid);
    let mut pi = AdaptivePI::new(d2).with_tolerance(0.0, 1e-12);
    pi.max_substeps = 5;
    let u0 = gaussian_ic(grid);

    let err = pi
        .evolve_adaptive(1.0, &u0)
        .expect_err("should hit runaway cap");
    match err {
        SemiflowError::AdaptiveStepRejected {
            steps_attempted,
            last_tau,
            last_err,
        } => {
            assert_eq!(
                steps_attempted, 5,
                "steps_attempted should equal max_substeps=5, got {steps_attempted}"
            );
            assert!(last_tau.is_finite(), "last_tau must be finite: {last_tau}");
            assert!(last_err.is_finite(), "last_err must be finite: {last_err}");
        }
        other => panic!("expected AdaptiveStepRejected, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Test 6: last_tau_is_finite
// ---------------------------------------------------------------------------

#[test]
fn last_tau_is_finite() {
    let grid = heat_grid(300);
    let d4 = d4_const(grid);
    let mut pi = AdaptivePI::new(d4).with_tolerance(0.0, 1e-4);
    let u0 = gaussian_ic(grid);

    let outcome = pi.evolve_adaptive(1.0, &u0).expect("clean run");
    assert!(
        outcome.last_tau.is_finite() && outcome.last_tau > 0.0,
        "last_tau must be finite and positive: {}",
        outcome.last_tau
    );
    // last_tau should be within [min_ratio * initial_guess, max_ratio * initial_guess]
    // times a few PI controller iterations — just verify it's in a sane range.
    assert!(
        outcome.last_tau < 2.0,
        "last_tau unexpectedly large: {}",
        outcome.last_tau
    );
}
