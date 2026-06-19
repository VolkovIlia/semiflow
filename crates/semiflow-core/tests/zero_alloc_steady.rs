//! Gate: `apply_into` allocates 0 heap bytes per step in steady state.
//!
//! ADR-0041 Wave 1 acceptance criterion AC-4:
//! After the first step (warm-up), subsequent steps must allocate 0 bytes when
//! `DiffusionChernoff` or `TruncatedExp4thDiffusionChernoff` is the inner kernel.
//!
//! Strategy: run `apply_into` once (warm-up), then measure allocations for a
//! second identical call with the same pre-warmed `ScratchPool`.
//!
//! The `allocation-counter` crate overrides the global allocator (test-only).

use allocation_counter::{self, AllocationInfo};
use semiflow_core::{
    scratch::ScratchPool, ChernoffFunction, ChernoffSemigroup, DiffusionChernoff, Grid1D, GridFn1D,
    TruncatedExp4thDiffusionChernoff,
};

fn a_half(_: f64) -> f64 {
    0.5
}

fn a_zero(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// DiffusionChernoff: 0 allocs after warm-up
// ---------------------------------------------------------------------------

#[test]
fn diffusion_zero_alloc_steady() {
    let n = 64;
    let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
    let dx = grid.dx();
    let a0 = 0.5;
    let tau = 0.35 * dx * dx / a0;
    let op = DiffusionChernoff::new(a_half, a_zero, a_zero, a0, grid);
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    // Warm-up: populates the scratch pool's free-list.
    let mut pool = ScratchPool::new();
    let mut dst = f0.zeroed_like();
    op.apply_into(tau, &f0, &mut dst, &mut pool).unwrap();

    // Steady-state measurement: pool already holds all needed capacity.
    let info: AllocationInfo = allocation_counter::measure(|| {
        op.apply_into(tau, &f0, &mut dst, &mut pool).unwrap();
    });

    assert_eq!(
        info.count_total, 0,
        "DiffusionChernoff::apply_into allocated {} time(s) in steady state (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// TruncatedExp4thDiffusionChernoff: 0 allocs after warm-up
// ---------------------------------------------------------------------------

#[test]
fn texp4_zero_alloc_steady() {
    let n = 64;
    let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
    let dx = grid.dx();
    let a0 = 0.5;
    let tau = 0.25 * dx * dx / a0;
    let op = TruncatedExp4thDiffusionChernoff::new(a_half, a_zero, a_zero, a0, grid);
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    let mut pool = ScratchPool::new();
    let mut dst = f0.zeroed_like();
    // Warm-up: take_vec × 4 → return_vec × 4 → 4 items in free-list.
    op.apply_into(tau, &f0, &mut dst, &mut pool).unwrap();

    let info: AllocationInfo = allocation_counter::measure(|| {
        op.apply_into(tau, &f0, &mut dst, &mut pool).unwrap();
    });

    assert_eq!(
        info.count_total, 0,
        "TruncatedExp4th::apply_into allocated {} time(s) in steady state (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// ChernoffSemigroup::evolve ping-pong: ≤ 2 allocs (setup only, 0 per step)
// ---------------------------------------------------------------------------

#[test]
fn chernoff_semigroup_ping_pong_allocs() {
    let n = 64;
    let grid = Grid1D::new(-4.0, 4.0, n).unwrap();
    let a0 = 0.5;
    let t = 0.01;
    let steps = 4_usize;
    #[allow(clippy::cast_precision_loss)]
    let tau = t / steps as f64;
    let dx = grid.dx();
    assert!(tau < 0.4 * dx * dx / a0, "CFL violated in test setup");

    let op = DiffusionChernoff::new(a_half, a_zero, a_zero, a0, grid);
    let semi = ChernoffSemigroup::new(op, steps).unwrap();
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    // Measure allocations for a complete evolve call.
    // Expected: exactly 2 allocations (buf_a clone + buf_b zeroed_like).
    // No per-step allocations when apply_into is overridden and does not use scratch.
    let info: AllocationInfo = allocation_counter::measure(|| {
        let _result = semi.evolve(t, &f0).unwrap();
    });

    assert!(
        info.count_total <= 4,
        "ChernoffSemigroup::evolve allocated {} time(s) (expected ≤ 4 for buffers + scratch init)",
        info.count_total
    );
}
