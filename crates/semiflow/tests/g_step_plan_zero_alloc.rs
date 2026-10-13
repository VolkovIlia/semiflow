//! `G_STEP_PLAN_ZERO_ALLOC` (`RELEASE_BLOCKING`, ADR-0215): τ-keyed plans live in
//! caller-owned scratch buffers.
//!
//! * A planned `Evolver::evolve_into` allocates per CALL only (the two ping-pong
//!   state clones): the count is the same for 2 and 50 steps and equals 2 once the
//!   caller's pool is warm.
//! * A planned `Strang2D` step (serial sweeps; `16²` stays below the threaded
//!   threshold) allocates nothing in steady state.
//! * `ChernoffSemigroup::evolve` (local pool) allocates at most 3 times per call:
//!   two states and the plan, which replaces the per-step ghost-table buffer.

use allocation_counter::AllocationInfo;
use semiflow::{
    ChernoffFunction, ChernoffSemigroup, DiffusionChernoff, Evolver, Grid1D, Grid2D, GridFn1D,
    GridFn2D, ScratchPool, Strang2D,
};

fn a_half(_: f64) -> f64 {
    0.5
}
fn zero(_: f64) -> f64 {
    0.0
}

fn engine(grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new(a_half, zero, zero, 0.5, grid)
}

/// Allocations of one warm `evolve_into` with `n` steps.
fn evolver_allocs(n: usize) -> u64 {
    let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
    let ev = Evolver::new(engine(grid), n).unwrap();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let (mut dst, mut pool) = (u0.clone(), ScratchPool::new());
    ev.evolve_into(0.01, &u0, &mut dst, &mut pool).unwrap();
    let info: AllocationInfo = allocation_counter::measure(|| {
        ev.evolve_into(0.01, &u0, &mut dst, &mut pool).unwrap();
    });
    info.count_total
}

#[test]
fn g_step_plan_zero_alloc_evolver() {
    let (two, fifty) = (evolver_allocs(2), evolver_allocs(50));
    eprintln!("G_STEP_PLAN_ZERO_ALLOC evolver: n=2 → {two}, n=50 → {fifty}");
    assert_eq!(two, fifty, "per-step allocations");
    assert!(two <= 2, "{two}");
}

#[test]
fn g_step_plan_zero_alloc_strang2d() {
    let g = Grid1D::new(-2.0, 2.0, 16).unwrap();
    let g2 = Grid2D::new(g, g);
    let s = Strang2D::new(engine(g), engine(g));
    let f0 = GridFn2D::from_fn(g2, |x, y| (-x * x - y * y).exp());
    let (mut dst, mut pool) = (f0.clone(), ScratchPool::new());
    s.apply_into(0.01, &f0, &mut dst, &mut pool).unwrap();
    let info: AllocationInfo = allocation_counter::measure(|| {
        s.apply_into(0.01, &f0, &mut dst, &mut pool).unwrap();
    });
    eprintln!("G_STEP_PLAN_ZERO_ALLOC strang2d: {}", info.count_total);
    assert_eq!(info.count_total, 0);
}

#[test]
fn g_step_plan_zero_alloc_semigroup() {
    let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
    let semi = ChernoffSemigroup::new(engine(grid), 20).unwrap();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let info: AllocationInfo = allocation_counter::measure(|| {
        let _ = semi.evolve(0.01, &u0).unwrap();
    });
    eprintln!("G_STEP_PLAN_ZERO_ALLOC semigroup: {}", info.count_total);
    assert!(info.count_total <= 3, "{}", info.count_total);
}
