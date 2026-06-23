//! Wave 2 (ADR-0042) byte-equality tests: `apply_into` must produce
//! bit-identical results to `apply` for `Strang2D`, `Strang3D`, `AxisLift`, `AxisLift3D`.
//!
//! Uses proptest to sample random `(τ, grid sizes, initial data)` combinations.
//! See ADR-0042 acceptance criterion 5.

use proptest::prelude::*;
use semiflow_core::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid2d::Grid2D,
    grid3d::Grid3D,
    grid_fn2d::GridFn2D,
    grid_fn3d::GridFn3D,
    scratch::ScratchPool,
    Axis, AxisLift, AxisLift3D, Strang2D, Strang3D,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_diff(grid: Grid1D) -> DiffusionChernoff {
    DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid)
}

fn grid1d(n: usize) -> Grid1D {
    Grid1D::new(-3.0, 3.0, n).unwrap()
}

fn grid2d(nx: usize, ny: usize) -> Grid2D {
    Grid2D::new(grid1d(nx), grid1d(ny))
}

fn grid3d(nx: usize, ny: usize, nz: usize) -> Grid3D {
    Grid3D::new(grid1d(nx), grid1d(ny), grid1d(nz)).unwrap()
}

fn f2d(g: Grid2D, seed: f64) -> GridFn2D<f64> {
    let s = seed.abs().max(0.1);
    GridFn2D::from_fn(g, move |x, y| (-(x * x + y * y) * s).exp())
}

fn f3d(g: Grid3D, seed: f64) -> GridFn3D<f64> {
    let s = seed.abs().max(0.1);
    GridFn3D::from_fn(g, move |x, y, z| (-(x * x + y * y + z * z) * s).exp())
}

fn assert_slices_eq(a: &[f64], b: &[f64], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            ai.to_bits(),
            bi.to_bits(),
            "{label}: bit diff at [{i}]: apply={ai} apply_into={bi}"
        );
    }
}

// ---------------------------------------------------------------------------
// AxisLift (2D) byte-equality
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn axislift_x_byte_equal(
        tau in 0.001f64..0.3,
        nx in 4usize..20,
        ny in 4usize..20,
        seed in 0.1f64..2.0,
    ) {
        let g2 = grid2d(nx, ny);
        let src = f2d(g2, seed);
        let lift = AxisLift::new(make_diff(grid1d(nx)), Axis::X);

        let expected = lift.apply_chernoff(tau, &src).unwrap();
        let mut dst = src.zeroed_like();
        let mut pool = ScratchPool::new();
        lift.apply_into(tau, &src, &mut dst, &mut pool).unwrap();

        assert_slices_eq(&expected.values, &dst.values, "AxisLift::X apply_into");
    }

    #[test]
    fn axislift_y_byte_equal(
        tau in 0.001f64..0.3,
        nx in 4usize..20,
        ny in 4usize..20,
        seed in 0.1f64..2.0,
    ) {
        let g2 = grid2d(nx, ny);
        let src = f2d(g2, seed);
        let lift = AxisLift::new(make_diff(grid1d(ny)), Axis::Y);

        let expected = lift.apply_chernoff(tau, &src).unwrap();
        let mut dst = src.zeroed_like();
        let mut pool = ScratchPool::new();
        lift.apply_into(tau, &src, &mut dst, &mut pool).unwrap();

        assert_slices_eq(&expected.values, &dst.values, "AxisLift::Y apply_into");
    }

    // ---------------------------------------------------------------------------
    // Strang2D byte-equality
    // ---------------------------------------------------------------------------

    #[test]
    fn strang2d_byte_equal(
        tau in 0.001f64..0.2,
        nx in 4usize..16,
        ny in 4usize..16,
        seed in 0.1f64..2.0,
    ) {
        let g2 = grid2d(nx, ny);
        let src = f2d(g2, seed);
        let s2d = Strang2D::new(make_diff(grid1d(nx)), make_diff(grid1d(ny)));

        let expected = s2d.apply_chernoff(tau, &src).unwrap();
        let mut dst = src.zeroed_like();
        let mut pool = ScratchPool::new();
        s2d.apply_into(tau, &src, &mut dst, &mut pool).unwrap();

        assert_slices_eq(&expected.values, &dst.values, "Strang2D apply_into");
    }

    // ---------------------------------------------------------------------------
    // AxisLift3D byte-equality
    // ---------------------------------------------------------------------------

    #[test]
    fn axislift3d_x_byte_equal(
        tau in 0.001f64..0.3,
        nx in 4usize..10,
        ny in 4usize..8,
        nz in 4usize..8,
        seed in 0.1f64..2.0,
    ) {
        let g3 = grid3d(nx, ny, nz);
        let src = f3d(g3, seed);
        let lift = AxisLift3D::new(make_diff(grid1d(nx)), Axis::X);

        let expected = lift.apply_chernoff(tau, &src).unwrap();
        let mut dst = src.zeroed_like();
        let mut pool = ScratchPool::new();
        lift.apply_into(tau, &src, &mut dst, &mut pool).unwrap();

        assert_slices_eq(&expected.values, &dst.values, "AxisLift3D::X apply_into");
    }

    #[test]
    fn axislift3d_y_byte_equal(
        tau in 0.001f64..0.3,
        nx in 4usize..10,
        ny in 4usize..8,
        nz in 4usize..8,
        seed in 0.1f64..2.0,
    ) {
        let g3 = grid3d(nx, ny, nz);
        let src = f3d(g3, seed);
        let lift = AxisLift3D::new(make_diff(grid1d(ny)), Axis::Y);

        let expected = lift.apply_chernoff(tau, &src).unwrap();
        let mut dst = src.zeroed_like();
        let mut pool = ScratchPool::new();
        lift.apply_into(tau, &src, &mut dst, &mut pool).unwrap();

        assert_slices_eq(&expected.values, &dst.values, "AxisLift3D::Y apply_into");
    }

    #[test]
    fn axislift3d_z_byte_equal(
        tau in 0.001f64..0.3,
        nx in 4usize..10,
        ny in 4usize..8,
        nz in 4usize..8,
        seed in 0.1f64..2.0,
    ) {
        let g3 = grid3d(nx, ny, nz);
        let src = f3d(g3, seed);
        let lift = AxisLift3D::new(make_diff(grid1d(nz)), Axis::Z);

        let expected = lift.apply_chernoff(tau, &src).unwrap();
        let mut dst = src.zeroed_like();
        let mut pool = ScratchPool::new();
        lift.apply_into(tau, &src, &mut dst, &mut pool).unwrap();

        assert_slices_eq(&expected.values, &dst.values, "AxisLift3D::Z apply_into");
    }

    // ---------------------------------------------------------------------------
    // Strang3D byte-equality
    // ---------------------------------------------------------------------------

    #[test]
    fn strang3d_byte_equal(
        tau in 0.001f64..0.15,
        nx in 4usize..10,
        ny in 4usize..8,
        nz in 4usize..8,
        seed in 0.1f64..2.0,
    ) {
        let g3 = grid3d(nx, ny, nz);
        let src = f3d(g3, seed);
        let s3d = Strang3D::new(
            make_diff(grid1d(nx)),
            make_diff(grid1d(ny)),
            make_diff(grid1d(nz)),
        );

        let expected = s3d.apply_chernoff(tau, &src).unwrap();
        let mut dst = src.zeroed_like();
        let mut pool = ScratchPool::new();
        s3d.apply_into(tau, &src, &mut dst, &mut pool).unwrap();

        assert_slices_eq(&expected.values, &dst.values, "Strang3D apply_into");
    }
}
