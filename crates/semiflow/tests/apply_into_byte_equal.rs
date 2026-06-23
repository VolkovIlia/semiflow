//! Proptest: `apply_into` is bit-identical to `apply_chernoff` for all Wave 1 kernel overrides.
//!
//! ADR-0041 Wave 1 math-fidelity requirement: for every (tau, src) pair the
//! output of `apply_into` must equal the output of `apply_chernoff` bit-for-bit (IEEE 754).
//!
//! In v3.0, `apply_chernoff` (from `ApplyChernoffExt`) replaces the removed `apply` method
//! as the allocating interface.
//!
//! Kernels covered:
//! - `DiffusionChernoff`
//! - `Diffusion4thChernoff`
//! - `Diffusion6thChernoff`
//! - `TruncatedExpDiffusionChernoff`
//! - `TruncatedExp4thDiffusionChernoff`
//! - `TruncatedExp4WithCache`
//!
//! fn-pointer restriction: thread-local `Cell<f64>` pattern (see `proptest_diffusion_constant.rs`).

use core::cell::Cell;

use proptest::prelude::*;
use semiflow::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    scratch::ScratchPool,
    Diffusion4thChernoff, Diffusion6thChernoff, DiffusionChernoff, Grid1D, GridFn1D,
    TruncatedExp4WithCache, TruncatedExp4thDiffusionChernoff, TruncatedExpDiffusionChernoff,
};

// ---------------------------------------------------------------------------
// Thread-local slots for fn-pointer capture
// ---------------------------------------------------------------------------

thread_local! {
    static A0_CELL: Cell<f64> = const { Cell::new(1.0) };
}

fn a_const(_: f64) -> f64 {
    A0_CELL.with(Cell::get)
}

fn a_zero(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn gaussian_values(n: usize) -> Vec<f64> {
    let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
    (0..n)
        .map(|i| {
            let x = grid.x_at(i);
            (-x * x).exp()
        })
        .collect()
}

fn grid_n() -> impl Strategy<Value = usize> {
    (8_usize..=64).prop_map(|n| n)
}

fn tau_and_a0() -> impl Strategy<Value = (f64, f64)> {
    (1e-5_f64..=0.05_f64, 0.05_f64..=2.0_f64)
}

// ---------------------------------------------------------------------------
// DiffusionChernoff
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]
    #[test]
    fn diffusion_apply_into_byte_equal(n in grid_n(), (raw_tau, a0) in tau_and_a0()) {
        let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
        let dx = grid.dx();
        let tau = raw_tau.min(0.4 * dx * dx / a0);

        A0_CELL.with(|c| c.set(a0));
        let op = DiffusionChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let f = GridFn1D { values: gaussian_values(n), grid };

        let expected = op.apply_chernoff(tau, &f).unwrap();
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        op.apply_into(tau, &f, &mut dst, &mut scratch).unwrap();

        assert_eq!(
            expected.values, dst.values,
            "DiffusionChernoff: apply_chernoff not bit-identical to apply_into (tau={tau}, a0={a0}, n={n})"
        );
    }
}

// ---------------------------------------------------------------------------
// Diffusion4thChernoff
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]
    #[test]
    fn diffusion4_apply_into_byte_equal(n in grid_n(), (raw_tau, a0) in tau_and_a0()) {
        let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
        let dx = grid.dx();
        // 4th-order CFL: tau < 3*dx²/(8·a0)
        let tau = raw_tau.min(0.3 * dx * dx / a0);

        A0_CELL.with(|c| c.set(a0));
        let op = Diffusion4thChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let f = GridFn1D { values: gaussian_values(n), grid };

        let expected = op.apply_chernoff(tau, &f).unwrap();
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        op.apply_into(tau, &f, &mut dst, &mut scratch).unwrap();

        assert_eq!(
            expected.values, dst.values,
            "Diffusion4thChernoff: apply_chernoff not bit-identical to apply_into (tau={tau}, a0={a0}, n={n})"
        );
    }
}

// ---------------------------------------------------------------------------
// Diffusion6thChernoff
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]
    #[test]
    fn diffusion6_apply_into_byte_equal(n in grid_n(), (raw_tau, a0) in tau_and_a0()) {
        let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
        let dx = grid.dx();
        // 6th-order CFL: conservative 20% of dx²/(2a)
        let tau = raw_tau.min(0.2 * dx * dx / a0);

        A0_CELL.with(|c| c.set(a0));
        let op = Diffusion6thChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let f = GridFn1D { values: gaussian_values(n), grid };

        let expected = op.apply_chernoff(tau, &f).unwrap();
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        op.apply_into(tau, &f, &mut dst, &mut scratch).unwrap();

        assert_eq!(
            expected.values, dst.values,
            "Diffusion6thChernoff: apply_chernoff not bit-identical to apply_into (tau={tau}, a0={a0}, n={n})"
        );
    }
}

// ---------------------------------------------------------------------------
// TruncatedExpDiffusionChernoff
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]
    #[test]
    fn truncated_exp_apply_into_byte_equal(n in grid_n(), (raw_tau, a0) in tau_and_a0()) {
        let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
        let dx = grid.dx();
        let tau = raw_tau.min(0.4 * dx * dx / a0);

        A0_CELL.with(|c| c.set(a0));
        let op = TruncatedExpDiffusionChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let f = GridFn1D { values: gaussian_values(n), grid };

        let expected = op.apply_chernoff(tau, &f).unwrap();
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        op.apply_into(tau, &f, &mut dst, &mut scratch).unwrap();

        assert_eq!(
            expected.values, dst.values,
            "TruncatedExp: apply_chernoff not bit-identical to apply_into (tau={tau}, a0={a0}, n={n})"
        );
    }
}

// ---------------------------------------------------------------------------
// TruncatedExp4thDiffusionChernoff
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]
    #[test]
    fn truncated_exp4_apply_into_byte_equal(n in grid_n(), (raw_tau, a0) in tau_and_a0()) {
        let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
        let dx = grid.dx();
        let tau = raw_tau.min(0.3 * dx * dx / a0);

        A0_CELL.with(|c| c.set(a0));
        let op = TruncatedExp4thDiffusionChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let f = GridFn1D { values: gaussian_values(n), grid };

        let expected = op.apply_chernoff(tau, &f).unwrap();
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        op.apply_into(tau, &f, &mut dst, &mut scratch).unwrap();

        assert_eq!(
            expected.values, dst.values,
            "TruncatedExp4th: apply_chernoff not bit-identical to apply_into (tau={tau}, a0={a0}, n={n})"
        );
    }
}

// ---------------------------------------------------------------------------
// TruncatedExp4WithCache
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]
    #[test]
    fn truncated_exp4_cached_apply_into_byte_equal(n in grid_n(), (raw_tau, a0) in tau_and_a0()) {
        let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
        let dx = grid.dx();
        let tau = raw_tau.min(0.3 * dx * dx / a0);

        A0_CELL.with(|c| c.set(a0));
        let op = TruncatedExp4WithCache::with_cached_coefficients(a_const, a_zero, a_zero, a0, grid);
        let f = GridFn1D { values: gaussian_values(n), grid };

        let expected = op.apply_chernoff(tau, &f).unwrap();
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        op.apply_into(tau, &f, &mut dst, &mut scratch).unwrap();

        assert_eq!(
            expected.values, dst.values,
            "TruncatedExp4WithCache: apply_chernoff not bit-identical to apply_into (tau={tau}, a0={a0}, n={n})"
        );
    }
}
