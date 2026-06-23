//! G11 — proptest regression-safety for constant-coefficient `DriftReactionChernoff`.
//!
//! Invariant I3 (traits.yaml): for constant `b ≡ b₀`, `c ≡ c₀`, the v0.2.2
//! RK2 formula MUST return values bit-equal (within ≤ 4 ULP, tolerance 1e-13)
//! to the v0.2.1 closed form `exp(τ·c₀) · f.sample(x + τ·b₀)`.
//!
//! The reduction-to-constant lemma (math.md §9.3) proves this algebraically:
//! - constant b: `b_mid = b₀`, so `x_foot = x + τ·b₀` (exact).
//! - constant c: `c0 = c1 = c₀`, so `(τ/2)·(c₀ + c₀) = τ·c₀` (≤ 2 ULP).
//!
//! 1000 proptest cases over `(τ ∈ [0, 1], b₀ ∈ [-2, 2], c₀ ∈ [-1, 1])`.
//! Tolerance 1e-13 is ≈ 10⁴ ULP at scale 1.0 — generous for rearrangement,
//! tight enough to detect sign errors or missing factors.
//!
//! `fn`-pointer restriction: `DriftReactionChernoff` stores `fn(f64) -> f64`
//! (no capturing closures). Use thread-local `Cell<f64>` to pass `b₀`, `c₀`
//! per proptest case — the standard workaround from properties.yaml §`drift_reaction_variable_order2`.
//!
//! Reference: `contracts/semiflow-core.properties.yaml`
//! `drift_reaction_constant_fast_path_exact`; traits.yaml I3.

use core::cell::Cell;

use proptest::prelude::*;
use semiflow::{chernoff::ApplyChernoffExt, DriftReactionChernoff, Grid1D, GridFn1D};

// Thread-local slots for the constant coefficient values per proptest case.
thread_local! {
    static B0_CELL: Cell<f64> = const { Cell::new(0.0) };
    static C0_CELL: Cell<f64> = const { Cell::new(0.0) };
}

/// Constant drift `b(x) = B0_CELL` — reads from the thread-local slot.
fn b_const(_x: f64) -> f64 {
    B0_CELL.with(Cell::get)
}

/// Constant reaction `c(x) = C0_CELL` — reads from the thread-local slot.
fn c_const(_x: f64) -> f64 {
    C0_CELL.with(Cell::get)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]

    /// G11: RK2 formula bit-equal to v0.2.1 closed form for constant b, c.
    ///
    /// Checks 5 nodes per case for breadth. Tolerance 1e-13 (I3 / properties.yaml).
    #[test]
    fn g11_constant_coeff_regression_safety(
        tau in 0.0f64..=1.0f64,
        b0 in -2.0f64..=2.0f64,
        c0 in -1.0f64..=1.0f64,
    ) {
        // Set the per-case constants via thread-locals.
        B0_CELL.with(|cell| cell.set(b0));
        C0_CELL.with(|cell| cell.set(c0));

        let grid = Grid1D::new(-10.0, 10.0, 1000).expect("grid valid");
        // Gaussian initial data — smooth and bounded.
        let f = GridFn1D::from_fn(grid, |x| (-x * x / 2.0).exp());

        let r = DriftReactionChernoff::new(b_const, c_const, c0.abs(), grid);
        let result = r.apply_chernoff(tau, &f).expect("apply succeeds");

        // Check 5 nodes uniformly (per properties.yaml).
        let indices = [0, grid.n / 4, grid.n / 2, 3 * grid.n / 4, grid.n - 1];
        for i in indices {
            let x = grid.x_at(i);
            // v0.2.1 closed form: exp(τ·c₀) · f.sample(x + τ·b₀).
            let expected = (tau * c0).exp() * f.sample(x + tau * b0).expect("sample ok");
            let actual = result.values[i];
            let scale = expected.abs().max(1.0);
            prop_assert!(
                (actual - expected).abs() <= 1e-13 * scale,
                "G11 I3 violated at i={i}: actual={actual:.15e}, expected={expected:.15e}, \
                 b0={b0}, c0={c0}, tau={tau}, diff={:.4e}",
                (actual - expected).abs()
            );
        }
    }
}
