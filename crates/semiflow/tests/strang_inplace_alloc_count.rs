//! Gate: `Strang2D::apply_into` and `Strang3D::apply_into` allocate 0 heap
//! bytes per step in steady state.
//!
//! ADR-0042 acceptance criterion 6: after 3 warmup steps, subsequent steps
//! must allocate 0 bytes when `DiffusionChernoff` is the inner kernel.
//!
//! The `allocation-counter` crate overrides the global allocator (test-only).

use allocation_counter::{self, AllocationInfo};
use semiflow::{
    diffusion::DiffusionChernoff, grid::Grid1D, grid2d::Grid2D, grid3d::Grid3D,
    grid_fn2d::GridFn2D, grid_fn3d::GridFn3D, scratch::ScratchPool, ChernoffFunction, Strang2D,
    Strang3D,
};

fn diff(grid: Grid1D) -> DiffusionChernoff {
    DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid)
}

// ---------------------------------------------------------------------------
// Strang2D: 0 allocs after 3 warmup steps
// ---------------------------------------------------------------------------

#[test]
fn strang2d_zero_alloc_steady() {
    let nx = 32;
    let ny = 24;
    let gx = Grid1D::new(-4.0, 4.0, nx).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, ny).unwrap();
    let g2 = Grid2D::new(gx, gy);
    let tau = 0.01;
    let s2d = Strang2D::new(diff(gx), diff(gy));
    let f0 = GridFn2D::from_fn(g2, |x, y| (-(x * x + y * y)).exp());

    let mut pool = ScratchPool::new();
    let mut dst = f0.zeroed_like();

    // 3 warmup steps: pool grows to steady-state capacity.
    for _ in 0..3 {
        s2d.apply_into(tau, &f0, &mut dst, &mut pool).unwrap();
    }

    // Steady-state measurement.
    let info: AllocationInfo = allocation_counter::measure(|| {
        s2d.apply_into(tau, &f0, &mut dst, &mut pool).unwrap();
    });

    assert_eq!(
        info.count_total, 0,
        "Strang2D::apply_into allocated {} time(s) in steady state (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// Strang3D: 0 allocs after 3 warmup steps
// ---------------------------------------------------------------------------

#[test]
fn strang3d_zero_alloc_steady() {
    let nx = 12;
    let ny = 10;
    let nz = 8;
    let gx = Grid1D::new(-4.0, 4.0, nx).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, ny).unwrap();
    let gz = Grid1D::new(-4.0, 4.0, nz).unwrap();
    let g3 = Grid3D::new(gx, gy, gz).unwrap();
    let tau = 0.005;
    let s3d = Strang3D::new(diff(gx), diff(gy), diff(gz));
    let f0 = GridFn3D::from_fn(g3, |x, y, z| (-(x * x + y * y + z * z)).exp());

    let mut pool = ScratchPool::new();
    let mut dst = f0.zeroed_like();

    // 3 warmup steps.
    for _ in 0..3 {
        s3d.apply_into(tau, &f0, &mut dst, &mut pool).unwrap();
    }

    let info: AllocationInfo = allocation_counter::measure(|| {
        s3d.apply_into(tau, &f0, &mut dst, &mut pool).unwrap();
    });

    assert_eq!(
        info.count_total, 0,
        "Strang3D::apply_into allocated {} time(s) in steady state (expected 0)",
        info.count_total
    );
}
