//! G9 — variable-coefficient oracle test for `DriftReactionChernoff` (v0.2.2).
//!
//! PDE: `∂_t u = b(x)·∂_x u + c·u` with linear-restoring drift:
//!   - `b(x) = -γx`, `c ≡ -κγ`, γ = 0.3, κ = 1.0
//!   - Gaussian initial data: `u(0,x) = f(x) = exp(-x²/(2σ²))`, σ = 1.0
//!
//! Closed-form oracle (math.md §9.3, ADR-0006 Amendment 5):
//!
//! ```text
//! u(t, x) = exp(-κγt) · exp(-x² · exp(-2γt) / (2σ²))
//! ```
//!
//! Gate (G9): `‖(R(t/n))^n f − u(t,·)‖_∞ < 1e-3`
//! with `n = 256`, `t = 0.5`, grid `[-5, 5]`, `N = 1000` nodes.
//!
//! Reference: `contracts/semiflow-core.math.md §9.3`, `contracts/semiflow-core.traits.yaml` I2.

use semiflow::{ChernoffSemigroup, DriftReactionChernoff, Grid1D, GridFn1D};

// ---------------------------------------------------------------------------
// Oracle parameters (NON-NEGOTIABLE — match math.md §9.3)
// ---------------------------------------------------------------------------

const GAMMA: f64 = 0.3;
const KAPPA: f64 = 1.0;
const SIGMA: f64 = 1.0;
const T_FINAL: f64 = 0.5;
const N_STEPS: usize = 256;
const N_NODES: usize = 1000;

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// Closed-form solution for linear-restoring drift with Gaussian initial data.
///
/// `u(t, x) = exp(-κγt) · exp(-x²·exp(-2γt) / (2σ²))`
///
/// Derivation: math.md §9.3 / `.dev-docs/verification/variable-drift-rk2-derivation.md`.
fn oracle(t: f64, x: f64) -> f64 {
    let decay = (-KAPPA * GAMMA * t).exp();
    let squeeze = (-x * x * (-2.0 * GAMMA * t).exp() / (2.0 * SIGMA * SIGMA)).exp();
    decay * squeeze
}

// ---------------------------------------------------------------------------
// G9
// ---------------------------------------------------------------------------

/// G9: sup-norm accuracy vs. linear-restoring oracle with variable `b(x) = -γx`.
///
/// Gate: `‖(R(T/n))^n f − u(T,·)‖_∞ < 1e-3` (NON-NEGOTIABLE).
#[test]
fn g9_variable_drift_oracle() {
    let grid = Grid1D::new(-5.0, 5.0, N_NODES).expect("grid params valid");
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x / (2.0 * SIGMA * SIGMA)).exp());

    // fn-pointer workaround: capture γ via `static` thread-local.
    // b(x) = -γx, c ≡ -κγ (constant).
    let drift = DriftReactionChernoff::new(|x| -GAMMA * x, |_| -KAPPA * GAMMA, KAPPA * GAMMA, grid);
    let semi = ChernoffSemigroup::new(drift, N_STEPS).expect("n >= 1");
    let u_n = semi.evolve(T_FINAL, &f0).expect("evolve succeeds");

    let mut max_err: f64 = 0.0;
    for i in 0..N_NODES {
        let x = grid.x_at(i);
        let err = (u_n.values[i] - oracle(T_FINAL, x)).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!(
        "G9 variable-drift oracle: ‖err‖_∞ = {max_err:.4e}  (gate: < 1e-3, n={N_STEPS}, N={N_NODES})"
    );

    assert!(
        max_err < 1e-3,
        "G9 FAIL: ‖err‖_∞ = {max_err:.4e} ≥ 1e-3 — escalate to architect"
    );
}
