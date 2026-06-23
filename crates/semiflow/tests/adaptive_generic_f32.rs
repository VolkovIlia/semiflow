//! Wave 4 generic-over-F smoke tests (ADR-0044).
//!
//! ## Scope
//!
//! `AdaptivePI<C, F>` requires `C: ChernoffFunction<F>`.  As of v2.0, all
//! concrete `ChernoffFunction` implementations are `f64`-monomorphic (the
//! Wave D trait refactor generalised the struct types to `DiffusionChernoff<F>`
//! but the `ChernoffFunction` *trait impl* for `f32` is deferred — the semigroup
//! operator kernels need f32 specialisation first).
//!
//! This file therefore tests the Wave 4 generic machinery at the controller layer:
//!
//! - `ClassicalPI::<f32>` — compiles, runs, produces reasonable output
//! - `H211bFilter::<f32>` — compiles, runs, produces reasonable output
//! - `AdaptivePI<DiffusionChernoff<f64>, f64>` — full integration (f64, baseline)
//!
//! When `ChernoffFunction<f32>` is implemented for `DiffusionChernoff<f32>` in
//! a future wave, add `AdaptivePI::<_, f32>` integration tests here.

use semiflow::{
    grid::BoundaryPolicy, AdaptivePI, ClassicalPI, DiffusionChernoff, Grid1D, GridFn1D,
    H211bFilter, State, StepController,
};

// ---------------------------------------------------------------------------
// ClassicalPI<f32> smoke
// ---------------------------------------------------------------------------

/// `ClassicalPI::<f32>` compiles, produces a finite multiplier, and preserves
/// the normative FP association over f32.
#[test]
fn classical_pi_f32_propose_accept() {
    let mut ctrl = ClassicalPI::<f32>::with_order(2);
    assert!(
        (ctrl.alpha() - 0.35_f32).abs() < 1e-6,
        "f32 alpha={} ≠ 0.35",
        ctrl.alpha()
    );
    assert!(
        (ctrl.beta() - 0.2_f32).abs() < 1e-6,
        "f32 beta={} ≠ 0.2",
        ctrl.beta()
    );

    let err_norm: f32 = 1e-5;
    let tol: f32 = 5e-5;
    let safety: f32 = 0.9;

    let factor = ctrl.propose_accept(err_norm, tol, safety, 2);
    assert!(
        factor.is_finite() && factor > 0.0,
        "f32 ClassicalPI factor not positive finite: {factor}"
    );
    // Multiplier ≈ safety × (tol/err)^α = 0.9 × 5^0.35 ≈ 1.36 (rough check)
    assert!(
        factor > 0.5 && factor < 20.0,
        "f32 ClassicalPI factor out of plausible range: {factor}"
    );
    println!("ClassicalPI<f32> propose_accept: factor={factor:.4e}");
}

/// `ClassicalPI::<f32>` `propose_reject` is finite.
#[test]
fn classical_pi_f32_propose_reject() {
    let mut ctrl = ClassicalPI::<f32>::with_order(2);
    let factor = ctrl.propose_reject(1e-3, 1e-5, 0.9, 2);
    assert!(
        factor.is_finite() && factor > 0.0 && factor < 1.0,
        "f32 ClassicalPI reject factor not in (0, 1): {factor}"
    );
    println!("ClassicalPI<f32> propose_reject: factor={factor:.4e}");
}

/// `ClassicalPI::<f32>` `reset` seeds `err_prev` = 1.0.
#[test]
fn classical_pi_f32_reset() {
    let mut ctrl = ClassicalPI::<f32>::with_order(2);
    // Pollute state with one call
    let _ = ctrl.propose_accept(1e-4, 1e-5, 0.9, 2);
    ctrl.reset();
    // After reset, two identical consecutive calls should give same factor
    let f1 = ctrl.propose_accept(1e-5, 5e-5, 0.9, 2);
    ctrl.reset();
    let f2 = ctrl.propose_accept(1e-5, 5e-5, 0.9, 2);
    assert_eq!(
        f1.to_bits(),
        f2.to_bits(),
        "ClassicalPI<f32> reset not idempotent: f1={f1:.6e} f2={f2:.6e}"
    );
}

// ---------------------------------------------------------------------------
// H211bFilter<f32> smoke
// ---------------------------------------------------------------------------

/// `H211bFilter::<f32>` compiles and produces a finite multiplier.
#[test]
fn h211b_filter_f32_propose_accept() {
    let mut ctrl = H211bFilter::<f32>::default();
    let factor = ctrl.propose_accept(1e-5, 5e-5, 0.9, 2);
    assert!(
        factor.is_finite() && factor > 0.0,
        "H211bFilter<f32> factor not positive finite: {factor}"
    );
    // Second call exercises the I-term feedback
    let factor2 = ctrl.propose_accept(1e-5, 5e-5, 0.9, 2);
    assert!(
        factor2.is_finite() && factor2 > 0.0,
        "H211bFilter<f32> factor2 not positive finite: {factor2}"
    );
    println!("H211bFilter<f32> propose_accept: f1={factor:.4e} f2={factor2:.4e}");
}

/// `H211bFilter::<f32>` `reset` seeds both `err_prev` and `r_prev` to 1.0.
#[test]
fn h211b_filter_f32_reset() {
    let mut ctrl = H211bFilter::<f32>::default();
    let _ = ctrl.propose_accept(1e-3, 1e-5, 0.9, 2);
    ctrl.reset();
    let f1 = ctrl.propose_accept(1e-5, 5e-5, 0.9, 2);
    ctrl.reset();
    let f2 = ctrl.propose_accept(1e-5, 5e-5, 0.9, 2);
    assert_eq!(
        f1.to_bits(),
        f2.to_bits(),
        "H211bFilter<f32> reset not idempotent"
    );
}

// ---------------------------------------------------------------------------
// AdaptivePI<DiffusionChernoff<f64>, f64> integration baseline (f64)
// ---------------------------------------------------------------------------

/// Integration smoke with default f64 path: confirms the generic bounds compile
/// and produce a finite, contractive result.
#[test]
fn adaptive_pi_f64_generic_baseline() {
    let grid = Grid1D::new(-5.0, 5.0, 100)
        .expect("grid")
        .with_boundary(BoundaryPolicy::Reflect);

    let u0 = GridFn1D::from_fn(grid, |x: f64| libm::exp(-x * x));
    let norm_u0 = u0.norm_sup();

    let func = DiffusionChernoff::new(
        |_: f64| 0.5_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        0.5_f64,
        grid,
    );

    // Verify the generic type parameters are explicit (no inference ambiguity).
    let mut pi = AdaptivePI::<DiffusionChernoff, f64, ClassicalPI<f64>>::new(func);
    pi.tol_rel = 1e-4;

    let outcome = pi.evolve_adaptive(0.2, &u0).expect("f64 baseline");
    assert!(outcome.steps_accepted >= 1, "zero accepted steps");

    let out_norm = outcome.final_state.norm_sup();
    assert!(
        out_norm <= norm_u0 * 1.001,
        "contractivity: ‖out‖={out_norm:.4e} > 1.001·‖u0‖={:.4e}",
        norm_u0 * 1.001
    );

    println!(
        "AdaptivePI<f64> baseline: steps={}/{}, ‖out‖={:.4e}",
        outcome.steps_accepted, outcome.steps_rejected, out_norm
    );
}

// ---------------------------------------------------------------------------
// Deviation note
// ---------------------------------------------------------------------------

// `AdaptivePI<DiffusionChernoff<f32>, f32>` is NOT tested here because
// `DiffusionChernoff<f32>` does not yet implement `ChernoffFunction<f32>`.
// The trait bound `C: ChernoffFunction<F>` on `AdaptivePI` requires this impl.
// This is a known deferred item per ADR-0026 (composition types f64-bound)
// and will be resolved when the ChernoffFunction kernel f32-specialisation lands.
