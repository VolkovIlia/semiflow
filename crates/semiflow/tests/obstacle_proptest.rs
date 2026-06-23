//! Proptest invariants for `ObstacleChernoff` / `Π_g` (math §44.2).
//!
//! Six properties verified via randomised cases:
//!
//!   P1 — Lower bound: `Π_g(W)[i] ≥ g(x_i)` for all i.
//!   P2 — Idempotence: `project(project(W)) == project(W)` (bit-exact).
//!   P3 — Nonexpansiveness: `‖Π_g(W1) − Π_g(W2)‖_∞ ≤ ‖W1 − W2‖_∞` (+eps).
//!   P4 — Monotonicity: `W1 ≤ W2 ⇒ Π_g(W1) ≤ Π_g(W2)`.
//!   P5 — Composite order-preservation: `V0 ≤ W0 ⇒ Π_g(S(τ)V0) ≤ Π_g(S(τ)W0)`.
//!   P6 — Active-set consistency: `active[i] == (W[i] > g(x_i))` and projected
//!        value equals W[i] exactly on active nodes.
//!
//! Cases: 256 per property (fast).

// Integration test: allows for numerical / binding wrapper patterns.
#![allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::needless_range_loop
)]

use proptest::prelude::*;
use semiflow::chernoff::ChernoffFunction;
use semiflow::{
    ClosureObstacle, ConstantObstacle, DiffusionChernoff, Grid1D, GridFn1D, Obstacle,
    ObstacleChernoff, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Uniform grid [0,1] with `n` nodes.
fn unit_grid(n: usize) -> Grid1D<f64> {
    Grid1D::new(0.0_f64, 1.0, n).expect("unit_grid")
}

/// Sup-norm of the pointwise difference of two `GridFn1D` slices.
fn sup_diff(a: &GridFn1D<f64>, b: &GridFn1D<f64>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Pointwise `a ≤ b` (within rounding eps).
fn pointwise_le(a: &GridFn1D<f64>, b: &GridFn1D<f64>, eps: f64) -> bool {
    a.values
        .iter()
        .zip(b.values.iter())
        .all(|(x, y)| *x <= *y + eps)
}

/// Project `state` with `ConstantObstacle(level)` and return result.
fn project_const(state: &GridFn1D<f64>, level: f64) -> GridFn1D<f64> {
    let obs = ConstantObstacle::new(level).expect("level finite");
    let mut out = state.clone();
    obs.project_in_place(&mut out).expect("project ok");
    out
}

// ---------------------------------------------------------------------------
// P1 — Lower bound
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// P1: every projected value ≥ obstacle level.
    #[test]
    fn p1_lower_bound(
        n in 8usize..64,
        seed in any::<u64>(),
        level in -1.0_f64..0.5,
    ) {
        let grid = unit_grid(n);
        // Random state: values in [-2, 2].
        let state = GridFn1D::from_fn(grid, |x| {
            let h = (seed as f64 * 6.283 * x).sin();
            2.0 * h
        });
        let projected = project_const(&state, level);
        for &v in &projected.values {
            prop_assert!(v >= level - 1e-14,
                "P1: projected value {v} < obstacle {level}");
        }
    }
}

// ---------------------------------------------------------------------------
// P2 — Idempotence (bit-exact)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// P2: projecting twice gives the same result as projecting once (bit-exact).
    #[test]
    fn p2_idempotence(
        n in 8usize..64,
        seed in any::<u64>(),
        level in -1.0_f64..0.5,
    ) {
        let grid = unit_grid(n);
        let state = GridFn1D::from_fn(grid, |x| {
            2.0 * ((seed as f64 * 3.7 * x).cos())
        });
        let once = project_const(&state, level);
        let twice = project_const(&once, level);
        for (a, b) in once.values.iter().zip(twice.values.iter()) {
            prop_assert_eq!(*a, *b, "P2: idempotence violated");
        }
    }
}

// ---------------------------------------------------------------------------
// P3 — Nonexpansiveness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// P3: ‖Π(W1) − Π(W2)‖_∞ ≤ ‖W1 − W2‖_∞  (+2e-14 rounding).
    #[test]
    fn p3_nonexpansive(
        n in 8usize..64,
        seed1 in any::<u64>(),
        seed2 in any::<u64>(),
        level in -1.0_f64..0.5,
    ) {
        let grid = unit_grid(n);
        let w1 = GridFn1D::from_fn(grid, |x| (seed1 as f64 * 2.1 * x).sin());
        let w2 = GridFn1D::from_fn(grid, |x| (seed2 as f64 * 1.7 * x).cos());
        let p1 = project_const(&w1, level);
        let p2 = project_const(&w2, level);
        let before = sup_diff(&w1, &w2);
        let after  = sup_diff(&p1, &p2);
        prop_assert!(after <= before + 2e-14,
            "P3: nonexpansiveness violated: {after:.6e} > {before:.6e}");
    }
}

// ---------------------------------------------------------------------------
// P4 — Monotonicity
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// P4: W1 ≤ W2 pointwise ⇒ Π(W1) ≤ Π(W2).
    #[test]
    fn p4_monotone(
        n in 8usize..64,
        seed in any::<u64>(),
        level in -1.0_f64..0.5,
        delta in 0.0_f64..0.5,
    ) {
        let grid = unit_grid(n);
        let w1 = GridFn1D::from_fn(grid, |x| (seed as f64 * 2.3 * x).sin());
        // W2 = W1 + delta ≥ W1 everywhere.
        let w2 = GridFn1D::from_fn(grid, |x| (seed as f64 * 2.3 * x).sin() + delta);
        let p1 = project_const(&w1, level);
        let p2 = project_const(&w2, level);
        prop_assert!(pointwise_le(&p1, &p2, 1e-14),
            "P4: monotonicity violated");
    }
}

// ---------------------------------------------------------------------------
// P5 — Composite order-preservation (with DiffusionChernoff inner)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// P5: V0 ≤ W0 ⇒ ObstacleChernoff.apply(V0) ≤ ObstacleChernoff.apply(W0).
    #[test]
    fn p5_composite_order_preserving(
        n in 8usize..48,
        seed in any::<u64>(),
        level in -0.5_f64..0.0,  // g ≤ 0 so inner sub-Markov regime is clean
        tau in 1e-4_f64..5e-3,
        delta in 0.0_f64..0.3,
    ) {
        let grid = unit_grid(n);
        let inner = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let obs = ConstantObstacle::new(level).expect("level finite");
        let kernel = ObstacleChernoff::new(inner, obs).expect("kernel ok");
        let v0 = GridFn1D::from_fn(grid, |x| 0.3 * (seed as f64 * 2.1 * x).sin());
        // W0 = V0 + delta ≥ V0.
        let w0 = GridFn1D::from_fn(grid, |x| 0.3 * (seed as f64 * 2.1 * x).sin() + delta);
        let mut out_v = v0.zeroed_like();
        let mut out_w = w0.zeroed_like();
        let mut scratch = ScratchPool::new();
        kernel.apply_into(tau, &v0, &mut out_v, &mut scratch).expect("apply v0");
        kernel.apply_into(tau, &w0, &mut out_w, &mut scratch).expect("apply w0");
        prop_assert!(pointwise_le(&out_v, &out_w, 1e-13),
            "P5: composite order-preservation violated");
    }
}

// ---------------------------------------------------------------------------
// P6 — Active-set consistency
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// P6: active[i] == (W[i] > g(x_i)) and Π(W)[i] == W[i] on active nodes.
    #[test]
    fn p6_active_set_consistency(
        n in 8usize..64,
        seed in any::<u64>(),
    ) {
        let grid = unit_grid(n);
        // Use a closure obstacle: g(x) = 0.2 * sin(3πx) (bounded, possibly positive).
        let obs = ClosureObstacle::new(|x: f64| 0.2 * (3.0 * core::f64::consts::PI * x).sin());
        let state = GridFn1D::from_fn(grid, |x| {
            0.5 * ((seed as f64 * 2.7 * x).sin())
        });
        // Compute active set.
        let mut active = vec![false; n];
        obs.active_set_into(&state, &mut active).expect("active_set_into ok");
        // Project.
        let mut projected = state.clone();
        obs.project_in_place(&mut projected).expect("project ok");
        for i in 0..n {
            let x = grid.x_at(i);
            let g_val = 0.2 * (3.0 * core::f64::consts::PI * x).sin();
            let w_val = state.values[i];
            // Active iff strictly above obstacle.
            let expected_active = w_val > g_val;
            prop_assert!(active[i] == expected_active,
                "P6: active[{}]={} but w={:.6} g={:.6}", i, active[i], w_val, g_val);
            // On active nodes projection is identity.
            if active[i] {
                prop_assert!(
                    projected.values[i] == w_val,
                    "P6: projected != state on active node {}", i
                );
            }
            // On all nodes projected >= g.
            prop_assert!(projected.values[i] >= g_val - 1e-14,
                "P6: projected value < obstacle at node {}", i);
        }
    }
}
