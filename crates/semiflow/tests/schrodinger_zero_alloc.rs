//! R4 zero-alloc gate: `SchrodingerChernoff::apply_into` in steady state.
//!
//! Contract (wave-b-advanced-semigroups.md §3.4): the 12 f64 working buffers for
//! the Crank-Nicolson kinetic step are pre-allocated in `SchrodingerChernoff::new()`
//! and reused via `RefCell` on every `apply_into` call. Zero heap allocations per
//! step in steady state.
//!
//! Strategy: construct `SchrodingerChernoff` + `SchrodingerState` outside the
//! measured block, warm-up once (this ensures the `RefCell`'s internal Vecs have
//! sufficient capacity for size N), then measure allocations for a second identical
//! call.
//!
//! See ADR-0057 Amendment 1 and R4 zero-alloc invariant.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use allocation_counter::{self, AllocationInfo};

use semiflow::diffusion4::Diffusion4thChernoff;
use semiflow::{
    ChernoffFunction, Grid1D, GridFn1D, SchrodingerChernoff, SchrodingerState, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_schr_f64(n: usize) -> SchrodingerChernoff<f64> {
    let grid = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x).unwrap()
}

fn make_state_f64(schr: &SchrodingerChernoff<f64>) -> SchrodingerState<f64> {
    let grid = schr.kinetic().grid;
    let n = grid.n;
    let psi_re = GridFn1D::from_fn(grid, |x: f64| (-x * x / 8.0).exp());
    let psi_im = GridFn1D {
        values: alloc::vec![0.0_f64; n],
        grid,
    };
    SchrodingerState { psi_re, psi_im }
}

extern crate alloc;

// ---------------------------------------------------------------------------
// 0 allocs after warm-up (f64)
// ---------------------------------------------------------------------------

#[test]
fn schrodinger_apply_into_zero_alloc_steady_f64() {
    let n = 64usize;
    let schr = make_schr_f64(n);
    let src = make_state_f64(&schr);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    // Warm-up: the RefCell's pre-allocated Vecs gain capacity for N nodes.
    schr.apply_into(0.01_f64, &src, &mut dst, &mut pool)
        .expect("warm-up");

    // Steady-state measurement: no new allocations expected.
    let info: AllocationInfo = allocation_counter::measure(|| {
        schr.apply_into(0.01_f64, &src, &mut dst, &mut pool)
            .expect("steady-state");
    });

    assert_eq!(
        info.count_total, 0,
        "SchrodingerChernoff f64 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// 0 allocs after warm-up (f32 public API, f64 internal)
// ---------------------------------------------------------------------------

#[test]
fn schrodinger_apply_into_zero_alloc_steady_f32() {
    let n = 64usize;
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, n).unwrap();
    let kinetic = Diffusion4thChernoff::<f32>::new_generic(
        (|_: f32| 0.5_f32) as fn(f32) -> f32,
        (|_: f32| 0.0_f32) as fn(f32) -> f32,
        (|_: f32| 0.0_f32) as fn(f32) -> f32,
        0.5,
        grid,
    );
    let schr = SchrodingerChernoff::<f32>::new(kinetic, |x: f32| 0.5_f32 * x * x).unwrap();

    let psi_re = GridFn1D::<f32>::from_fn_generic(grid, |x: f32| (-x * x / 8.0_f32).exp());
    let psi_im = GridFn1D::<f32> {
        values: alloc::vec![0.0_f32; n],
        grid,
    };
    let src = SchrodingerState::<f32> { psi_re, psi_im };
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f32>::new();

    // Warm-up: f64 internal Vecs gain capacity for N nodes.
    schr.apply_into(0.01_f32, &src, &mut dst, &mut pool)
        .expect("warm-up");

    // Steady-state measurement.
    let info: AllocationInfo = allocation_counter::measure(|| {
        schr.apply_into(0.01_f32, &src, &mut dst, &mut pool)
            .expect("steady-state");
    });

    assert_eq!(
        info.count_total, 0,
        "SchrodingerChernoff f32 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}
