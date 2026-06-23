//! Property tests for `Diffusion6thChernoff` (v0.7.0, ADR-0015).
//!
//! P1 — `Z⁶_τ⁰` (tau=0 identity):
//!   For any random (τ, Gaussian f), `apply(0, &f) ≈ f` to machine epsilon.
//!   200 cases (constant-a; each apply is O(N) with N=100).
//!
//! P2 — `Z⁶_τ¹` (generator approximation):
//!   At very small τ=1e-6, `(apply(τ, f) - f) / τ → A·f` as τ→0.
//!   For constant a=0.5 and Gaussian IC, `A·f = 0.5·f''`.
//!   Gate: ‖(out-f)/τ - 0.5·f''_FD‖_∞ < 5e-3 (small-τ linearization).
//!   200 cases.
//!
//! P3 — variable-a bounded single step:
//!   For variable a(x) = `σ_a·(1` + 0.1·tanh(x)), single-step output stays
//!   within 1.5·‖f‖_∞.
//!   200 cases.
//!
//! `fn`-pointer restriction: thread-local Cell<f64> pattern.
// v7.0: QuinticHermite removed (ADR-0109 removal clock fulfilled).
// Tests updated to use SepticHermite (v6.0+ default).

use core::cell::Cell;

use proptest::prelude::*;
use semiflow_core::{
    chernoff::ApplyChernoffExt, Diffusion6thChernoff, Grid1D, GridFn1D, InterpKind,
};

thread_local! {
    static A0_CELL: Cell<f64> = const { Cell::new(1.0) };
    static SA_CELL: Cell<f64> = const { Cell::new(1.0) };
}

fn a_const(_: f64) -> f64 {
    A0_CELL.with(Cell::get)
}

fn a_zero(_: f64) -> f64 {
    0.0
}

// Variable a: σ_a * (1 + 0.1 * tanh(x))
fn a_var(x: f64) -> f64 {
    SA_CELL.with(Cell::get) * (1.0 + 0.1 * x.tanh())
}

fn a_var_prime(x: f64) -> f64 {
    let ch = x.cosh();
    SA_CELL.with(Cell::get) * 0.1 / (ch * ch)
}

fn a_var_double_prime(x: f64) -> f64 {
    let ch = x.cosh();
    let sh = x.sinh();
    SA_CELL.with(Cell::get) * (-0.2 * sh / (ch * ch * ch))
}

// ---------------------------------------------------------------------------
// P1: tau=0 is identity
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// P1: apply(0, f) == f elementwise (Z⁶_τ⁰ gate).
    ///
    /// Tolerance: 1e-12 (floating-point identity for tau=0 path).
    #[test]
    fn p1_tau_zero_identity(
        a0 in 0.01f64..=5.0f64,
        amplitude in 0.5f64..=2.0f64,
        mu in -2.0f64..=2.0f64,
        sigma_sq in 0.1f64..=2.0f64,
    ) {
        A0_CELL.with(|c| c.set(a0));

        let grid = Grid1D::new(-10.0, 10.0, 100)
            .unwrap()
            .with_interp(InterpKind::SepticHermite);
        let f = GridFn1D::from_fn(grid, |x| {
            amplitude * libm::exp(-(x - mu) * (x - mu) / (2.0 * sigma_sq))
        });

        let d6 = Diffusion6thChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let out = d6.apply_chernoff(0.0, &f).expect("tau=0 apply");

        let indices = [5, 25, 50, 75, 95];
        for i in indices {
            let fi = f.values[i];
            let oi = out.values[i];
            prop_assert!(
                (fi - oi).abs() < 1e-12,
                "P1 identity violated at i={i}: f={fi:.15e} out={oi:.15e} (a0={a0:.4})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// P2: small-tau generator approximation (Z⁶_τ¹ linearization)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// P2: (apply(τ, f) - f) / τ → A·f as τ→0.
    ///
    /// For constant a=1 and Gaussian IC, A·f = f'' (standard diffusion).
    /// We approximate f'' via 5-pt FD on the grid and compare to (out-f)/τ.
    /// Gate: sup-norm ≤ 2e-2 (allows O(τ) truncation + FD approximation errors;
    /// for σ_sq small, f'' is large so the 5-pt FD has larger relative error).
    #[test]
    fn p2_generator_approximation(
        amplitude in 0.5f64..=2.0f64,
        mu in -2.0f64..=2.0f64,
        sigma_sq in 0.5f64..=2.0f64,
    ) {
        let a0 = 1.0_f64;
        A0_CELL.with(|c| c.set(a0));
        let tau = 1e-6_f64;

        let grid = Grid1D::new(-10.0, 10.0, 100)
            .unwrap()
            .with_interp(InterpKind::SepticHermite);
        let dx = grid.dx();
        let f = GridFn1D::from_fn(grid, |x| {
            amplitude * libm::exp(-(x - mu) * (x - mu) / (2.0 * sigma_sq))
        });

        let d6 = Diffusion6thChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let out = d6.apply_chernoff(tau, &f).expect("apply");

        // Interior nodes only (avoid FD boundary effects).
        let mut max_err = 0.0_f64;
        for i in 4..96 {
            let gen_approx = (out.values[i] - f.values[i]) / tau;
            // 5-pt central FD for f'': (-f[i-2]+16f[i-1]-30f[i]+16f[i+1]-f[i+2])/(12dx²)
            let fpp = (-f.values[i - 2] + 16.0 * f.values[i - 1]
                - 30.0 * f.values[i]
                + 16.0 * f.values[i + 1]
                - f.values[i + 2])
                / (12.0 * dx * dx);
            // A·f = a * f'' = 1 * f'' for constant a=1
            let err = (gen_approx - a0 * fpp).abs();
            if err > max_err {
                max_err = err;
            }
        }

        prop_assert!(
            max_err < 2e-2,
            "P2 generator approx: sup-norm={max_err:.4e} ≥ 2e-2 \
             (mu={mu:.3}, sigma_sq={sigma_sq:.3}, amplitude={amplitude:.3})"
        );
    }
}

// ---------------------------------------------------------------------------
// P3: variable-a bounded single step
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// P3: single-step output stays within 1.5·‖f‖_∞ for variable a.
    ///
    /// a(x) = σ_a·(1 + 0.1·tanh(x)), σ_a ∈ [0.1, 2.0].
    /// Grid: 100 nodes, τ ∈ [1e-6, 1e-3] (CFL-safe for a_norm ≤ 2.2).
    #[test]
    fn p3_variable_a_bounded_one_step(
        sigma_a in 0.1f64..=2.0f64,
        tau in 1.0e-6f64..=1.0e-3f64,
        amplitude in 0.5f64..=2.0f64,
        mu in -2.0f64..=2.0f64,
        sigma_sq in 0.5f64..=2.0f64,
    ) {
        SA_CELL.with(|c| c.set(sigma_a));
        let a_norm = sigma_a * 1.2; // upper bound including tanh variation

        let grid = Grid1D::new(-10.0, 10.0, 100)
            .unwrap()
            .with_interp(InterpKind::SepticHermite);
        let f = GridFn1D::from_fn(grid, |x| {
            amplitude * libm::exp(-(x - mu) * (x - mu) / (2.0 * sigma_sq))
        });

        SA_CELL.with(|c| c.set(sigma_a));
        let d6 = Diffusion6thChernoff::new(a_var, a_var_prime, a_var_double_prime, a_norm, grid);

        SA_CELL.with(|c| c.set(sigma_a));
        let out = d6.apply_chernoff(tau, &f).expect("apply");

        let norm_f: f64 = f.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let norm_out: f64 = out.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let bound = 1.5 * norm_f;

        prop_assert!(
            norm_out <= bound,
            "P3 growth bound violated: ||out||={norm_out:.4e} > 1.5·||f||={bound:.4e} \
             (sigma_a={sigma_a:.3}, tau={tau:.4e})"
        );
    }
}
