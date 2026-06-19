//! Variable-a no-regression test for `TruncatedExp4thDiffusionChernoff` (v0.6.0, ADR-0013).
//!
//! Gate: `‖err_M4‖_∞ ≤ ‖err_M2‖_∞ · 1.05` (no regression vs v0.4.0 `TruncatedExp`).
//!
//! Setup:
//! - Variable a: Liouville `a(x) = (1 + 0.05·x)²` — strictly positive on [−10, 10].
//! - Initial condition: Gaussian `f(x) = exp(−x²)`.
//! - T = 0.5, single-step (n=1) at fixed τ satisfying BOTH CFL constraints:
//!   - `TruncatedExp` v0.4.0:  `2·τ·a_norm` < dx²
//!   - `TruncatedExp4th` v0.6.0: `8·τ·a_norm` < 3·dx²
//!
//!   So τ < min(`dx²/(2·a_norm)`, `3·dx²/(8·a_norm)`) = `3·dx²/(8·a_norm)` [tighter].
//!
//! Oracle: solved by `ChernoffSemigroup` with `TruncatedExpDiffusionChernoff` at large n=4000.
//! Both M4 and M2 are compared against this high-n reference.
//!
//! Honest note (ADR-0013 §3): variable-a spatial order remains O(dx²) for
//! `TruncatedExp4thDiffusionChernoff`. The 5-point stencil reduces the leading dx²-constant
//! by ~3× (Mickens-improvement), but does not lift the order. This test verifies
//! the Mickens-improvement: M4 error ≤ 1.05×M2 error (no regression), optionally
//! ≤ 0.5×M2 (3× tighter, demonstrating the constant reduction).

use semiflow_core::{
    ChernoffSemigroup, Grid1D, GridFn1D, TruncatedExp4thDiffusionChernoff,
    TruncatedExpDiffusionChernoff,
};

// a(x) = (1 + 0.05·x)²
fn a_liouville(x: f64) -> f64 {
    let inner = 1.0 + 0.05 * x;
    inner * inner
}

fn a_zero(_: f64) -> f64 {
    0.0
}

/// Upper bound for ‖a‖_∞ on [-10, 10]: max at x=10: (1+0.5)² = 2.25.
const A_NORM: f64 = 2.25;

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const T_FINAL: f64 = 0.5;

/// Build the high-n reference solution and return the `GridFn1D`.
///
/// Helper for `variable_a_no_regression_vs_trunc_exp_v040` to keep it ≤50 lines.
#[allow(clippy::cast_precision_loss)] // N_REF_STEPS = 8000; well within f64 mantissa
fn build_reference(n_ref: usize, n_ref_steps: usize) -> (Grid1D, crate::GridFn1D) {
    use semiflow_core::GridFn1D;
    let grid_ref = Grid1D::new(X_MIN, X_MAX, n_ref).expect("ref grid valid");
    let dx_ref = grid_ref.dx();
    let tau_ref = T_FINAL / n_ref_steps as f64;
    let cfl_ref = tau_ref / (dx_ref * dx_ref / (2.0 * A_NORM));
    assert!(cfl_ref < 1.0, "REF CFL violated: factor={cfl_ref:.4}");
    let f0_ref = GridFn1D::from_fn(grid_ref, |x| libm::exp(-x * x));
    let ref_magnus =
        TruncatedExpDiffusionChernoff::new(a_liouville, a_zero, a_zero, A_NORM, grid_ref);
    let ref_sem = ChernoffSemigroup::new(ref_magnus, n_ref_steps).expect("ref semigroup");
    let ref_out = ref_sem.evolve(T_FINAL, &f0_ref).expect("ref evolve");
    (grid_ref, ref_out)
}

#[test]
#[allow(clippy::cast_precision_loss)] // N_STEPS/N_REF_STEPS ≤ 8000; well within f64 mantissa
#[allow(clippy::too_many_lines)] // oracle setup + two semigroups; extraction to helpers
fn variable_a_no_regression_vs_trunc_exp_v040() {
    // Grid dimensions and CFL checks:
    // N=400:  dx=20/400=0.05, CFL_M4=3*(0.05)²/(8*2.25)≈4.17e-4, tau=0.5/4000=1.25e-4 ✓
    // N_REF=800: dx=0.025, CFL_M2_ref=(0.025)²/(2*2.25)≈1.39e-4, tau=0.5/8000=6.25e-5 ✓
    const N_SPATIAL: usize = 400;
    const N_STEPS: usize = 4000;
    const N_REF: usize = 800;
    const N_REF_STEPS: usize = 8000;

    let (grid_ref, ref_out) = build_reference(N_REF, N_REF_STEPS);
    let dx_ref = grid_ref.dx();
    let tau_ref = T_FINAL / N_REF_STEPS as f64;

    // Test grid (coarser).
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid valid");
    let dx = grid.dx();
    let tau = T_FINAL / N_STEPS as f64;
    let cfl_m4 = 3.0 * dx * dx / (8.0 * A_NORM);
    let cfl_m2 = dx * dx / (2.0 * A_NORM);
    assert!(
        tau < cfl_m4,
        "M4 CFL violated: tau={tau:.3e} >= {cfl_m4:.3e}"
    );
    assert!(
        tau < cfl_m2,
        "M2 CFL violated: tau={tau:.3e} >= {cfl_m2:.3e}"
    );

    eprintln!("Variable-a no-regression: a(x)=(1+0.05x)²");
    eprintln!("  Test:  N={N_SPATIAL}, n={N_STEPS}, dx={dx:.4e}, tau={tau:.4e}");
    eprintln!("  Ref:   N={N_REF}, n={N_REF_STEPS}, dx={dx_ref:.4e}, tau={tau_ref:.4e}");

    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
    let out4 = ChernoffSemigroup::new(
        TruncatedExp4thDiffusionChernoff::new(a_liouville, a_zero, a_zero, A_NORM, grid),
        N_STEPS,
    )
    .expect("m4 semigroup")
    .evolve(T_FINAL, &f0)
    .expect("m4 evolve");
    let out2 = ChernoffSemigroup::new(
        TruncatedExpDiffusionChernoff::new(a_liouville, a_zero, a_zero, A_NORM, grid),
        N_STEPS,
    )
    .expect("m2 semigroup")
    .evolve(T_FINAL, &f0)
    .expect("m2 evolve");

    let mut err4 = 0.0_f64;
    let mut err2 = 0.0_f64;
    for i in 0..N_SPATIAL {
        let ref_val = ref_out.sample(grid.x_at(i)).expect("ref sample");
        err4 = err4.max((out4.values[i] - ref_val).abs());
        err2 = err2.max((out2.values[i] - ref_val).abs());
    }

    let ratio = err4 / err2.max(1e-16);
    eprintln!("err_M4 = {err4:.4e}  err_M2 = {err2:.4e}  ratio = {ratio:.4} (gate ≤ 1.05)");
    assert!(
        err4 <= err2 * 1.05,
        "Variable-a no-regression FAIL: err_M4={err4:.4e} > 1.05·err_M2={:.4e} (ratio={ratio:.4})",
        err2 * 1.05
    );
    if err4 <= err2 * 0.5 {
        eprintln!("Mickens-improvement CONFIRMED: ratio={ratio:.4}");
    } else {
        eprintln!("Mickens-improvement NOT confirmed: ratio={ratio:.4} > 0.5 (O(dx²) expected)");
    }
}
