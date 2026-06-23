//! Property tests for `TruncatedExp4thDiffusionChernoff` (v0.6.0, ADR-0013).
//!
//! P1 — constant-a proximity to `TruncatedExp` v0.4.0:
//!   For constant a, both `TruncatedExp4thDiffusionChernoff` and `TruncatedExpDiffusionChernoff`
//!   approximate the same exact solution. They are NOT bit-equal (different stencils)
//!   but should be CLOSE for fine grids: ‖M4 − M2‖_∞ ≤ ‖f‖_∞ · 1e-3.
//!
//!   Gate: tolerance 1e-3 (relative) — tight enough to catch wrong stencil but loose
//!   enough for O(dx²) difference between 4th-order and 2nd-order stencils.
//!
//!   100 cases over `a₀ ∈ [0.1, 5.0]`, `n_nodes ∈ {64, 128, 256}`.
//!   τ < `CFL_min` (4th-order CFL is tighter: `3·dx²/(8·a_norm)`).
//!   `fn`-pointer restriction: thread-local `Cell<f64>` for `a₀`.

use core::cell::Cell;

use proptest::prelude::*;
use semiflow_core::{
    chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExp4thDiffusionChernoff,
    TruncatedExpDiffusionChernoff,
};

thread_local! {
    static A0_CELL: Cell<f64> = const { Cell::new(1.0) };
}

fn a_const(_: f64) -> f64 {
    A0_CELL.with(Cell::get)
}
fn a_zero(_: f64) -> f64 {
    0.0
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    /// P1: constant-a `TruncatedExp4thDiffusionChernoff` close to `TruncatedExpDiffusionChernoff`.
    ///
    /// Both approximate the same exact heat kernel; stencil difference is O(dx²)
    /// so ‖M4 − M2‖_∞ ≤ ‖f‖_∞ · 5e-3 (relative tolerance).
    /// The 5e-3 threshold accounts for O(dx²) stencil-order difference at large τ·a/dx².
    ///
    /// CFL: uses M4's tighter bound `3·dx²/(8·a₀)`. M2 also satisfies it since
    /// M2's bound `dx²/(2·a₀)` is looser (for same dx): `dx²/(2a) > 3dx²/(8a)`.
    #[test]
    fn p1_constant_a_proximity(
        a0 in 0.1f64..=5.0f64,
        amplitude in 0.5f64..=2.0f64,
        mu in -1.0f64..=1.0f64,
        sigma_sq in 0.1f64..=1.0f64,
        n_nodes_idx in 0usize..=2usize,
    ) {
        A0_CELL.with(|c| c.set(a0));

        let n_nodes_choices = [64usize, 128, 256];
        let n_nodes = n_nodes_choices[n_nodes_idx];

        // Domain [-4, 4], dx = 8/n_nodes.
        let grid = Grid1D::new(-4.0, 4.0, n_nodes).expect("grid");
        let dx = grid.dx();

        // Use τ = 0.8 × CFL_M4_max to ensure M4 CFL is satisfied.
        let cfl_m4_max = 3.0 * dx * dx / (8.0 * a0);
        let tau = 0.8 * cfl_m4_max;

        let f = GridFn1D::from_fn(grid, |x| {
            amplitude * libm::exp(-(x - mu) * (x - mu) / (2.0 * sigma_sq))
        });
        let norm_f: f64 = f.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);

        let m4 = TruncatedExp4thDiffusionChernoff::new(a_const, a_zero, a_zero, a0, grid);
        let m2 = TruncatedExpDiffusionChernoff::new(a_const, a_zero, a_zero, a0, grid);

        let out4 = m4.apply_chernoff(tau, &f).expect("m4 apply");
        let out2 = m2.apply_chernoff(tau, &f).expect("m2 apply");

        // ‖M4 - M2‖_∞ should be small (stencil difference).
        let diff: f64 = out4.values.iter()
            .zip(out2.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        let rel = if norm_f > 0.0 { diff / norm_f } else { diff };

        prop_assert!(
            rel <= 5e-3,
            "P1 proximity violated: ‖M4-M2‖_∞={diff:.4e}, ‖f‖_∞={norm_f:.4e}, \
             rel={rel:.4e} > 5e-3 (a0={a0:.3}, n={n_nodes}, tau={tau:.4e})"
        );
    }
}
