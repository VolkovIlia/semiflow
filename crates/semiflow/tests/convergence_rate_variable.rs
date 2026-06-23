//! G10 — empirical log-log slope test for variable-coefficient `DriftReactionChernoff`.
//!
//! Same oracle as G9 (linear-restoring drift, `b(x) = -γx`, `c ≡ -κγ`,
//! Gaussian initial data). Sweeps `n ∈ {32, 64, 128, 256, 512}`, iterating
//! `(R(T/n))^n f₀` for each n, and fits the log-log slope of
//! `‖err‖_∞` vs. n in `(ln n, ln err)` coordinates.
//!
//! Gate (G10): `slope ≤ -1.95` from `contracts/semiflow-core.properties.yaml`
//! (`drift_reaction_variable_order2`). v0.2.1 first-order R gives slope ≈ -1.0
//! and would FAIL here; v0.2.2 RK2 gives slope ≈ -2.0.
//!
//! Grid: `[-5, 5]`, `N = 10000` (fine enough to keep the spatial discretization
//! floor below the temporal error across the full n-sweep; same spatial budget
//! as ADR-0006 Amendment 3 approach but for the variable-b oracle).
//!
//! Reference: `contracts/semiflow-core.math.md §9.3`,
//! `contracts/semiflow-core.properties.yaml drift_reaction_variable_order2`.

use semiflow_core::{ChernoffSemigroup, DriftReactionChernoff, Grid1D, GridFn1D};

// ---------------------------------------------------------------------------
// Oracle parameters (must match G9 — NON-NEGOTIABLE)
// ---------------------------------------------------------------------------

const GAMMA: f64 = 0.3;
const KAPPA: f64 = 1.0;
const SIGMA: f64 = 1.0;
const T_FINAL: f64 = 0.5;
const N_VALUES: [usize; 5] = [32, 64, 128, 256, 512];

/// Spatial nodes — fine enough that interpolation floor < temporal error
/// across the full n-sweep. N=100000 on [-5,5] gives dx=1e-4 (same regime
/// as `strang_advdiff` Amendment 4).
const N_NODES: usize = 100_000;

/// Gate from `contracts/semiflow-core.properties.yaml` `drift_reaction_variable_order2`.
const SLOPE_GATE: f64 = -1.95;

// ---------------------------------------------------------------------------
// Oracle (same as G9)
// ---------------------------------------------------------------------------

/// `u(t, x) = exp(-κγt) · exp(-x²·exp(-2γt) / (2σ²))`
fn oracle(t: f64, x: f64) -> f64 {
    let decay = (-KAPPA * GAMMA * t).exp();
    let squeeze = (-x * x * (-2.0 * GAMMA * t).exp() / (2.0 * SIGMA * SIGMA)).exp();
    decay * squeeze
}

// ---------------------------------------------------------------------------
// Global error at fixed n
// ---------------------------------------------------------------------------

/// Sup-norm error of `(R(T/n))^n f₀` vs. oracle at `t = T_FINAL`.
///
/// Iterated application measures global O(τ²) temporal accuracy.
fn error_at_n(n_steps: usize) -> f64 {
    let grid = Grid1D::new(-5.0, 5.0, N_NODES).expect("grid params valid");
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x / (2.0 * SIGMA * SIGMA)).exp());
    let drift = DriftReactionChernoff::new(|x| -GAMMA * x, |_| -KAPPA * GAMMA, KAPPA * GAMMA, grid);
    let semi = ChernoffSemigroup::new(drift, n_steps).expect("n >= 1");
    let u_n = semi.evolve(T_FINAL, &f0).expect("evolve succeeds");

    let mut max_err: f64 = 0.0;
    for i in 0..N_NODES {
        let x = grid.x_at(i);
        let err = (u_n.values[i] - oracle(T_FINAL, x)).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Log-log slope in (ln n, ln err) — OLS
// ---------------------------------------------------------------------------

/// OLS slope of `(ln n_i, ln err_i)` pairs.
///
/// For global O(τ²) = O(n^{-2}): `err ∝ n^{-2}` → slope ≈ -2.
/// Gate: slope ≤ -1.95 (order ≥ 1.95).
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
// sum_x/sum_y/sum_xy: OLS variable names by convention; n ≤ N_VALUES max ≤ 2^16
fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|&x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// G10
// ---------------------------------------------------------------------------

/// G10: empirical log-log slope ≤ -1.95 for variable-coefficient RK2 drift-reaction.
///
/// Gate from `contracts/semiflow-core.properties.yaml` `drift_reaction_variable_order2`.
/// If this fails with slope ≈ -1.0, the RK2 midpoint or trapezoidal step is wrong
/// (see properties.yaml `failure_mode` for diagnosis checklist).
#[test]
fn g10_variable_drift_convergence_slope() {
    let errs: Vec<f64> = N_VALUES.iter().map(|&n| error_at_n(n)).collect();

    for (&n, &e) in N_VALUES.iter().zip(errs.iter()) {
        println!("n={n:5}, err={e:.6e}");
    }

    let slope = log_log_slope(&N_VALUES, &errs);
    println!("G10: log-log slope = {slope:.4}  (gate: <= {SLOPE_GATE}, N_NODES={N_NODES})");

    assert!(
        slope <= SLOPE_GATE,
        "G10 FAIL: slope {slope:.4} > {SLOPE_GATE} — v0.2.2 RK2 not achieving order-2. \
         Diagnosis: if slope ≈ -1.0 check RK2 formula; if slope > 0 check spatial floor."
    );
}
