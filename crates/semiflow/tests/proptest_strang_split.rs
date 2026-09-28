//! Property tests for the library `StrangSplit` composition (issue #29).
//!
//! Implements two contract properties that previously had no test
//! (`contracts/semiflow-core.properties.yaml`):
//!
//! - `strang_split_palindrome_consistency` (100 cases): `StrangSplit::apply_into`
//!   equals the hand-composed `D(τ/2) ∘ R(τ) ∘ D(τ/2)`. True by construction —
//!   which is exactly what it guards: a composition wired in the wrong order.
//! - `truncated_exp_strang_quasi_contractivity` (10000 cases): for
//!   `StrangSplit<TruncatedExpDiffusionChernoff, DriftReactionChernoff>`,
//!   `‖Φ(τ)f‖_∞ ≤ (1 + |c|·τ + 20·τ²)·‖f‖_∞` (math.md §9.6).
//!
//! Unlike `proptest_contractivity.rs::g4_strang_quasi_contractivity`, which
//! re-implements the kernels inline, both properties drive the library types
//! through the ADR-0074 `apply_into` API.
//!
//! `TruncatedExpDiffusionChernoff::new` only accepts `fn` pointers, so the
//! proptest-drawn diffusion coefficient reaches it through a thread-local
//! `Cell` (same pattern as `proptest_truncated_exp4_consistency.rs`).
//! Generators follow the contract's `generators:` block and
//! `standard_grid = Grid1D::new(-10.0, 10.0, 1000)`.

use core::cell::Cell;

use proptest::{prelude::*, test_runner::TestCaseError};
use semiflow::{
    chernoff::ChernoffFunction, scratch::ScratchPool, DiffusionChernoff, DriftReactionChernoff,
    Grid1D, GridFn1D, State, StrangSplit, TruncatedExpDiffusionChernoff,
};

fn standard_grid() -> Grid1D {
    Grid1D::new(-10.0, 10.0, 1000).expect("standard_grid")
}

fn gaussian_state(grid: Grid1D, amplitude: f64, mu: f64, sigma_sq: f64) -> GridFn1D<f64> {
    GridFn1D::from_fn(grid, |x| {
        amplitude * (-(x - mu) * (x - mu) / (2.0 * sigma_sq)).exp()
    })
}

/// `op.apply_into(tau, src, ·)` into a fresh buffer.
fn step<C>(op: &C, tau: f64, src: &GridFn1D<f64>) -> Result<GridFn1D<f64>, TestCaseError>
where
    C: ChernoffFunction<f64, S = GridFn1D<f64>>,
{
    let mut dst = src.zeroed_like();
    op.apply_into(tau, src, &mut dst, &mut ScratchPool::new())
        .map_err(|e| TestCaseError::fail(format!("apply_into failed: {e:?}")))?;
    Ok(dst)
}

thread_local! {
    static A_CELL: Cell<f64> = const { Cell::new(1.0) };
}

fn a_cell(_: f64) -> f64 {
    A_CELL.with(Cell::get)
}

fn zero(_: f64) -> f64 {
    0.0
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    /// `strang_split_palindrome_consistency`: `StrangSplit` is exactly D(τ/2)·R(τ)·D(τ/2).
    #[test]
    fn strang_split_palindrome_consistency(
        a in 0.01_f64..=5.0,
        b in -5.0_f64..=5.0,
        c in -5.0_f64..=5.0,
        tau in 1e-6_f64..=0.1,
        amplitude in 0.5_f64..=2.0,
        mu in -2.0_f64..=2.0,
        sigma_sq in 0.1_f64..=2.0,
    ) {
        let grid = standard_grid();
        let f = gaussian_state(grid, amplitude, mu, sigma_sq);
        let d = DiffusionChernoff::with_closure(move |_| a, |_| 0.0, |_| 0.0, a, grid);
        let r = DriftReactionChernoff::with_closure(move |_| b, move |_| c, c.abs(), grid);
        let phi = StrangSplit::new(d.clone(), r.clone());

        let result_phi = step(&phi, tau, &f)?;
        let manual_step1 = step(&d, tau / 2.0, &f)?;
        let manual_step2 = step(&r, tau, &manual_step1)?;
        let manual_result = step(&d, tau / 2.0, &manual_step2)?;

        let diff = result_phi
            .values
            .iter()
            .zip(&manual_result.values)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(
            diff <= 1e-12 * (1.0 + f.norm_sup()),
            "palindrome FAIL: ‖Φ(τ)f − D(τ/2)R(τ)D(τ/2)f‖_∞ = {diff:.3e} \
             (a={a}, b={b}, c={c}, tau={tau})"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 10_000, ..ProptestConfig::default() })]

    /// `truncated_exp_strang_quasi_contractivity`: the G4-strang growth bound with C = 20.
    ///
    /// The contract draws `tau` from `small_tau = [1e-6, 0.1]` and rejects CFL
    /// violations. On `standard_grid` (dx ≈ 0.02) nearly every such draw
    /// violates `τ·a < dx²/2`, which would exhaust proptest's global reject
    /// budget. So `tau` is drawn as a fraction of the CFL limit instead; the
    /// reject guard is kept as the contract requires, but never fires.
    /// `‖f‖_∞` on the right-hand side is the continuous datum's sup (see below).
    #[test]
    fn truncated_exp_strang_quasi_contractivity(
        a in 0.01_f64..=5.0,
        b in -5.0_f64..=5.0,
        c in -5.0_f64..=5.0,
        cfl_frac in 0.01_f64..0.99,
        amplitude in 0.5_f64..=2.0,
        mu in -2.0_f64..=2.0,
        sigma_sq in 0.1_f64..=2.0,
    ) {
        let grid = standard_grid();
        let dx = grid.dx();
        let tau = (cfl_frac * 0.5 * dx * dx / a).max(1e-6);
        if tau * a >= 0.5 * dx * dx {
            return Err(TestCaseError::reject("CFL violation"));
        }
        let f = gaussian_state(grid, amplitude, mu, sigma_sq);
        A_CELL.with(|cell| cell.set(a));
        let m = TruncatedExpDiffusionChernoff::new(a_cell, zero, zero, a, grid);
        let r = DriftReactionChernoff::with_closure(move |_| b, move |_| c, c.abs(), grid);
        let phi = StrangSplit::new(m, r);

        let lhs = step(&phi, tau, &f)?.norm_sup();
        // ‖f‖_∞ is the sup of the Gaussian datum (= amplitude), not of its grid
        // samples: with the peak between nodes, R's sub-cell shift (|b·τ| < dx)
        // can move it onto a node, and the grid max grows by up to
        // ~(dx/2)²/(2σ²) without any overshoot. Shrunk counterexample under
        // the grid norm: a=0.01, b≈3.63, c=0, mu=0, sigma²=0.1 →
        // 0.4997746 > 0.4997500 = grid max, < 0.5 = amplitude.
        let f_sup = amplitude.max(f.norm_sup());
        let rhs = (1.0 + c.abs() * tau + 20.0 * tau * tau) * f_sup;
        prop_assert!(
            lhs <= rhs,
            "TruncatedExp G4-strang FAIL: ‖Φ(τ)f‖_∞ = {lhs:.6e} > {rhs:.6e} \
             (a={a}, b={b}, c={c}, tau={tau:.3e})"
        );
    }
}
