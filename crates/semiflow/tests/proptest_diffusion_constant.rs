//! G12 — `diffusion_chernoff_constant_fast_path_exact` (v0.3.0, ADR-0008).
//!
//! Invariant: for constant `a ≡ a₀` (pass `|_| 0.0_f64` for `a_prime` and
//! `a_double_prime`), the ζ-A formula MUST produce output bit-equal (within
//! 1e-13, ≈ 4 ULP) to the v0.2.2 closed form:
//!
//! ```text
//! (7/12)·f(x) + (3/16)·(f(x + 2√(a₀τ)) + f(x - 2√(a₀τ)))
//!            + (1/48)·(f(x + 2√(3a₀τ)) + f(x - 2√(3a₀τ)))
//! ```
//!
//! `fn`-pointer restriction: use thread-local `Cell<f64>` to pass `a₀` per
//! case — the standard workaround from `proptest_drift_consistency.rs`.
//!
//! 1000 cases over `a₀ ∈ [0.01, 5.0]`, `τ ∈ [1e-6, 0.1]`, Gaussian `f`.
//! Tolerance 1e-13 includes 4 ULP floor at scale 1.
//!
//! Reference: `contracts/semiflow-core.properties.yaml`
//! `diffusion_chernoff_constant_fast_path_exact`.

use core::cell::Cell;

use proptest::prelude::*;
use semiflow::{DiffusionChernoff, Grid1D, GridFn1D};

// Fourier-symbol weights (match diffusion.rs constants).
const W0: f64 = 7.0 / 12.0;
const W1: f64 = 3.0 / 16.0;
const W2: f64 = 1.0 / 48.0;

// Thread-local slot for the constant diffusion coefficient per case.
thread_local! {
    static A0_CELL: Cell<f64> = const { Cell::new(1.0) };
}

/// Constant diffusion `a(x) = A0_CELL`.
fn a_const(_x: f64) -> f64 {
    A0_CELL.with(Cell::get)
}

/// Zero derivative (constant `a` has `a' ≡ 0`).
fn a_zero(_x: f64) -> f64 {
    0.0
}

/// v0.2.2 closed-form oracle for constant-a diffusion at one node `x`.
fn v022_oracle(a0: f64, tau: f64, f: &GridFn1D, x: f64) -> f64 {
    let h = 2.0 * libm::sqrt(a0 * tau);
    let h3 = 2.0 * libm::sqrt(3.0 * a0 * tau);
    let center = W0 * f.sample(x).unwrap_or(0.0);
    let near = W1 * (f.sample(x + h).unwrap_or(0.0) + f.sample(x - h).unwrap_or(0.0));
    let far = W2 * (f.sample(x + h3).unwrap_or(0.0) + f.sample(x - h3).unwrap_or(0.0));
    center + near + far
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]

    /// G12: ζ-A constant-a output bit-equal to v0.2.2 closed form.
    ///
    /// Checks 5 uniformly-spaced interior nodes per case.
    #[test]
    fn g12_constant_fast_path_exact(
        a0 in 0.01f64..=5.0f64,
        tau in 1.0e-6f64..=0.1f64,
        amplitude in 0.5f64..=2.0f64,
        mu in -2.0f64..=2.0f64,
        sigma_sq in 0.1f64..=2.0f64,
    ) {
        A0_CELL.with(|cell| cell.set(a0));

        let grid = Grid1D::new(-10.0, 10.0, 1000).expect("grid valid");
        let f = GridFn1D::from_fn(grid, |x| {
            amplitude * (-(x - mu).powi(2) / (2.0 * sigma_sq)).exp()
        });

        let dc = DiffusionChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let result = dc.apply_chernoff(tau, &f).expect("apply succeeds");

        let indices = [100, 300, 500, 700, 900];
        for i in indices {
            let x = grid.x_at(i);
            let expected = v022_oracle(a0, tau, &f, x);
            let actual = result.values[i];
            let scale = expected.abs().max(1.0);
            prop_assert!(
                (actual - expected).abs() <= 1e-13 * scale,
                "G12 Z_const-a violated at i={i}: actual={actual:.15e}, \
                 expected={expected:.15e}, a0={a0}, tau={tau}, \
                 diff={:.4e}",
                (actual - expected).abs()
            );
        }
    }
}
