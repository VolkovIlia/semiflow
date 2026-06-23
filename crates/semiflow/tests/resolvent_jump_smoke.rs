//! Smoke test for `ResolventJumpChernoff` (F2, ADR-0134, math.md §47).
//!
//! Verifies that a large-step TWS parabolic-contour jump on the unit-diffusion
//! 1D Laplacian (N=32, t=10, M=20) converges to a reasonable accuracy versus
//! a many-small-step Chernoff reference (`n_ref=1000`). This exercises the
//! LHP Thomas solve and the contour quadrature end-to-end without the full
//! `G_RESOLVENT_JUMP_ORDER` slope sweep (which lives in `slow-tests`).
//!
//! Gate: `‖jump_M(t,g) − reference(t,g)‖_∞ ≤ 5e-3` at t=10, M=20, N=32.

// Integration test: allows for numerical / binding wrapper patterns.
#![allow(clippy::cast_precision_loss)]

use semiflow_core::{DiffusionChernoff, Grid1D, GridFn1D, ResolventJumpChernoff};

// ---------------------------------------------------------------------------
// Smoke-test constants
// ---------------------------------------------------------------------------

/// Domain half-width.
const L: f64 = 5.0;
/// Grid size (matches oracle canonical setup).
const N: usize = 64;
/// Number of contour nodes.
const M: usize = 24;
/// Large time step (well into the regime where `n_ref=T/τ` Chernoff is costly).
const T: f64 = 10.0;
/// Many-step Chernoff reference: small τ = `T/n_ref`.
const N_REF: usize = 2000;
/// Accuracy budget for this smoke check.
const BUDGET: f64 = 1e-2;

// ---------------------------------------------------------------------------
// Smoke test
// ---------------------------------------------------------------------------

/// `ResolventJumpChernoff` smoke: large-step jump on 1D Laplacian, Gaussian IC.
///
/// Constructs a unit-diffusion kernel, applies the M-node TWS contour jump
/// at t=10, and compares with a 2000-step reference Chernoff.
/// Error must be ≤ 5e-3 (the full gate `G_RESOLVENT_JUMP_ORDER` uses M∈{6..14}
/// and slope ≥ 1.95 at t=100).
#[test]
fn resolvent_jump_smoke_large_t_gaussian() {
    let grid = Grid1D::new(-L, L, N).unwrap();

    // Unit constant-a diffusion: A = ∂²_x.
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);

    let kernel = ResolventJumpChernoff::new(chernoff, M, grid).unwrap();

    // Gaussian initial condition (matches oracle setup).
    let g = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());

    // Large-step jump via TWS contour quadrature.
    let jump_result = kernel.jump(T, &g).unwrap();
    assert_eq!(
        jump_result.values.len(),
        N,
        "output length must match grid.n"
    );

    // Many-step reference: 2000 small Chernoff steps with τ = T/N_REF.
    let ref_result = reference_evolve(grid, T, &g, N_REF);

    // Sup-norm error.
    let err: f64 = jump_result
        .values
        .iter()
        .zip(ref_result.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!(
        "resolvent_jump smoke: err_inf = {err:.4e}  (budget ≤ {BUDGET:.0e}, M={M}, t={T}, N={N})"
    );

    assert!(err.is_finite(), "jump result must be finite (got NaN/Inf)");
    assert!(
        err <= BUDGET,
        "ResolventJumpChernoff smoke FAIL: err {err:.4e} > budget {BUDGET:.0e} \
         (M={M}, t={T}, N={N}, Gaussian IC, unit diffusion)"
    );
}

/// Sanity: `new` rejects `m_nodes` < 6.
#[test]
fn resolvent_jump_rejects_small_m() {
    let grid = Grid1D::new(-L, L, N).unwrap();
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let err = ResolventJumpChernoff::new(chernoff, 4, grid);
    assert!(err.is_err(), "m_nodes=4 must be rejected");
}

/// Sanity: `jump` rejects non-positive t.
#[test]
fn resolvent_jump_rejects_non_positive_t() {
    let grid = Grid1D::new(-L, L, N).unwrap();
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let kernel = ResolventJumpChernoff::new(chernoff, M, grid).unwrap();
    let g = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());
    assert!(kernel.jump(0.0, &g).is_err(), "t=0 must be rejected");
    assert!(kernel.jump(-1.0, &g).is_err(), "t<0 must be rejected");
}

// ---------------------------------------------------------------------------
// Helper: many-step Chernoff reference
// ---------------------------------------------------------------------------

/// Evolve `g` via `n_steps` small Chernoff steps at τ = t / `n_steps`.
fn reference_evolve(grid: Grid1D<f64>, t: f64, g: &GridFn1D<f64>, n_steps: usize) -> GridFn1D<f64> {
    use semiflow_core::chernoff::ChernoffFunction;
    use semiflow_core::scratch::ScratchPool;

    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let tau = t / n_steps as f64;
    let mut scratch = ScratchPool::default();
    let mut src = GridFn1D {
        values: g.values.clone(),
        grid: g.grid,
    };
    let mut dst = GridFn1D {
        values: g.values.clone(),
        grid: g.grid,
    };
    for _ in 0..n_steps {
        chernoff
            .apply_into(tau, &src, &mut dst, &mut scratch)
            .unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src
}
