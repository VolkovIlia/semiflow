//! `G_ND_SWEEP_ZERO_ALLOC` (ADR-0214): `SmolyakGridND::apply_into` and
//! `AnisotropicShiftChernoffND::apply_into` allocate nothing in steady state —
//! the separable path (constant coefficients) takes every buffer from the
//! `ScratchPool` and the quadrature list is built at construction. The per-node
//! path is measured in the serial build only (`parallel` spawns worker threads).
//!
//! Strategy: one warm-up `apply_into`, then count the allocations of a second
//! identical call with the same pool (`allocation-counter`, test-only allocator).

use allocation_counter::AllocationInfo;
use semiflow::{
    grid_nd::{GridFnND, GridND},
    shift_nd::{AnisotropicShiftChernoffND, SquareMatrix},
    smolyak::SmolyakGridND,
    ChernoffFunction, Grid1D, ScratchPool,
};

fn a_const(_x: &[f64; 2], a: &mut SquareMatrix<f64, 2>) {
    a.set(0, 0, 0.6);
    a.set(1, 1, 0.5);
    a.set(0, 1, 0.1);
    a.set(1, 0, 0.1);
}

fn b_const(_x: &[f64; 2], b: &mut [f64; 2]) {
    *b = [0.2, -0.1];
}

#[cfg_attr(feature = "parallel", allow(dead_code))] // used by the serial-only test
fn b_var(x: &[f64; 2], b: &mut [f64; 2]) {
    *b = [0.2 * x[1], -0.1];
}

fn c_var(x: &[f64; 2]) -> f64 {
    0.1 * x[0]
}

fn field() -> GridFnND<f64, 2> {
    let ax = |n| Grid1D::new(-1.0, 1.0, n).unwrap();
    GridFnND::from_fn(GridND::new([ax(9), ax(7)]).unwrap(), |x| {
        (x[0] - 0.3 * x[1]).cos()
    })
}

/// Allocations of the second of two identical `apply_into` calls.
fn steady_allocs<C: ChernoffFunction<f64, S = GridFnND<f64, 2>>>(engine: &C) -> u64 {
    let src = field();
    let mut dst = src.clone();
    let mut pool = ScratchPool::new();
    engine.apply_into(0.05, &src, &mut dst, &mut pool).unwrap();
    let info: AllocationInfo = allocation_counter::measure(|| {
        engine.apply_into(0.05, &src, &mut dst, &mut pool).unwrap();
    });
    info.count_total
}

#[test]
fn g_nd_sweep_zero_alloc_separable() {
    let g = field().grid;
    let smolyak = SmolyakGridND::with_level(a_const, b_const, c_var, g.clone(), 5).unwrap();
    assert_eq!(steady_allocs(&smolyak), 0, "Smolyak, separable path");
    let aniso = AnisotropicShiftChernoffND::new(a_const, b_const, c_var, g).unwrap();
    assert_eq!(steady_allocs(&aniso), 0, "anisotropic, separable path");
}

#[cfg(not(feature = "parallel"))]
#[test]
fn g_nd_sweep_zero_alloc_per_node() {
    let g = field().grid;
    let smolyak = SmolyakGridND::with_level(a_const, b_var, c_var, g.clone(), 5).unwrap();
    assert_eq!(steady_allocs(&smolyak), 0, "Smolyak, per-node path");
    let aniso = AnisotropicShiftChernoffND::new(a_const, b_var, c_var, g).unwrap();
    assert_eq!(steady_allocs(&aniso), 0, "anisotropic, per-node path");
}
