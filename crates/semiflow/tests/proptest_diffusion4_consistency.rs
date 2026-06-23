//! Property tests for `Diffusion4thChernoff` (v0.6.0, ADR-0013).
//!
//! P1 — constant-a bit-equal:
//!   For `a_prime ≡ 0 ∧ a_double_prime ≡ 0`, `Diffusion4thChernoff` output
//!   must match `DiffusionChernoff` output bit-exactly (same bits, tolerance 0).
//!   Gate: Z⁴_const-a (sympy-proven).
//!   1000 cases over `a₀ ∈ [0.01, 5.0]`, `τ ∈ [1e-6, 0.1]`, Gaussian `f`.
//!
//! P2 — monotone-a non-explosion:
//!   For `a(x) = a₀ + 0.01·(x - 5)`, `a_prime ≡ 0.01`, `a_double_prime ≡ 0`,
//!   single-step output stays within `[−1.5·||f||_∞, 1.5·||f||_∞]`.
//!   Gate: numerical stability in the production regime (`dx ≈ 0.04`,
//!   `τ ≤ 0.1`, single step).
//!   500 cases.
//!
//! `fn`-pointer restriction: use thread-local `Cell<f64>` pattern.

use core::cell::Cell;

use proptest::prelude::*;
use semiflow::{
    chernoff::ApplyChernoffExt, Diffusion4thChernoff, DiffusionChernoff, Grid1D, GridFn1D,
};

// Thread-local slots.
thread_local! {
    static A0_CELL: Cell<f64> = const { Cell::new(1.0) };
    static AP_CELL: Cell<f64> = const { Cell::new(0.0) };
    static APP_CELL: Cell<f64> = const { Cell::new(0.0) };
}

fn a_const(_: f64) -> f64 {
    A0_CELL.with(Cell::get)
}
fn a_zero(_: f64) -> f64 {
    0.0
}
// Linear a for P2: a(x) = A0_CELL + AP_CELL*(x - 5)
fn a_linear(x: f64) -> f64 {
    A0_CELL.with(Cell::get) + AP_CELL.with(Cell::get) * (x - 5.0)
}
fn a_prime_const(_: f64) -> f64 {
    AP_CELL.with(Cell::get)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]

    /// P1: `Diffusion4thChernoff` with constant `a` is bit-equal to `DiffusionChernoff`.
    ///
    /// The ζ⁴ correction vanishes for `a' ≡ 0 ∧ a'' ≡ 0` (Z⁴_const-a gate).
    /// The γ-A baseline is identical to v0.5.0 DiffusionChernoff (BIT-EQUAL).
    /// Therefore `Diffusion4thChernoff.apply(τ, f) == DiffusionChernoff.apply(τ, f)`.
    #[test]
    fn p1_constant_a_bit_equal(
        a0 in 0.01f64..=5.0f64,
        tau in 1.0e-6f64..=0.1f64,
        amplitude in 0.5f64..=2.0f64,
        mu in -2.0f64..=2.0f64,
        sigma_sq in 0.1f64..=2.0f64,
    ) {
        A0_CELL.with(|c| c.set(a0));

        let grid = Grid1D::new(-10.0, 10.0, 500).expect("grid");
        let f = GridFn1D::from_fn(grid, |x| {
            amplitude * libm::exp(-(x - mu) * (x - mu) / (2.0 * sigma_sq))
        });

        let d4 = Diffusion4thChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let d2 = DiffusionChernoff::new(a_const, a_zero, a_zero, a0, grid);

        let out4 = d4.apply_chernoff(tau, &f).expect("d4 apply");
        let out2 = d2.apply_chernoff(tau, &f).expect("d2 apply");

        // Check 5 interior nodes for bit equality.
        let indices = [50, 150, 250, 350, 450];
        for i in indices {
            let v4 = out4.values[i];
            let v2 = out2.values[i];
            prop_assert!(
                v4.to_bits() == v2.to_bits(),
                "P1 bit-equal violated at i={}: d4={:.15e} d2={:.15e} (a0={:.4}, tau={:.4e})",
                i, v4, v2, a0, tau
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 500, ..ProptestConfig::default() })]

    /// P2: `Diffusion4thChernoff` with mildly variable `a` stays bounded.
    ///
    /// Single-step output must stay within `[−1.5·||f||_∞, 1.5·||f||_∞]`
    /// (growth bound — Chernoff function is not strictly a contraction for variable a,
    /// but should not amplify by more than 50% in one step in the production regime).
    ///
    /// Setup: linear `a(x) = a₀ + 0.01·(x-5)` on [-10, 10], `a' ≡ 0.01`, `a'' ≡ 0`.
    /// Grid: 250 nodes, `dx ≈ 0.08`. `τ ∈ [1e-4, 0.02]`.
    #[test]
    fn p2_variable_a_bounded_one_step(
        a0 in 0.5f64..=3.0f64,
        tau in 1.0e-4f64..=0.02f64,
        amplitude in 0.5f64..=2.0f64,
        mu in -2.0f64..=2.0f64,
        sigma_sq in 0.5f64..=2.0f64,
    ) {
        A0_CELL.with(|c| c.set(a0));
        AP_CELL.with(|c| c.set(0.01));

        // Ensure a(x) > 0 everywhere on [-10, 10]:
        // a_min = a0 + 0.01*(−10 − 5) = a0 − 0.15 ≥ 0.5 − 0.15 = 0.35 > 0 ✓
        let a_norm = a0 + 0.01 * (10.0 - 5.0_f64).abs(); // upper bound

        let grid = Grid1D::new(-10.0, 10.0, 250).expect("grid");
        let f = GridFn1D::from_fn(grid, |x| {
            amplitude * libm::exp(-(x - mu) * (x - mu) / (2.0 * sigma_sq))
        });

        let d4 = Diffusion4thChernoff::new(a_linear, a_prime_const, a_zero, a_norm, grid);

        AP_CELL.with(|c| c.set(0.01));
        A0_CELL.with(|c| c.set(a0));
        let out4 = d4.apply_chernoff(tau, &f).expect("d4 apply");

        let norm_f: f64 = f.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let norm_out: f64 = out4.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let bound = 1.5 * norm_f;

        prop_assert!(
            norm_out <= bound,
            "P2 growth bound violated: ||out||={norm_out:.4e} > 1.5·||f||={bound:.4e} \
             (a0={a0:.3}, tau={tau:.4e})"
        );
    }
}
