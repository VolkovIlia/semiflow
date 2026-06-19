//! G6 — `BoundaryPolicy::Periodic` round-trip and shift-invariance tests.
//!
//! Test 1: `f(x) = sin(2π k (x − xmin) / L)` for `k ∈ {1, 3}`.
//!   Samples at all interior nodes agree with the analytical value to 1e-12.
//!
//! Test 2: `f.sample(x + L) ≈ f.sample(x)` to 1e-12 for interior `x`.

use std::f64::consts::PI;

use semiflow_core::{BoundaryPolicy, Grid1D, GridFn1D};

// Grid parameters shared by both tests.
const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
const N: usize = 1000;
const L: f64 = XMAX - XMIN;

fn periodic_grid() -> Grid1D {
    Grid1D::new(XMIN, XMAX, N)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic)
}

// ---------------------------------------------------------------------------
// Test 1 — sine round-trip at interior nodes.
// ---------------------------------------------------------------------------

/// Verify that `Grid1D{Periodic}` samples `sin(2π k x / L)` exactly at each
/// interior node (no interpolation error on-node; only f64 round-off applies).
#[allow(clippy::cast_precision_loss)]
// k ≤ 3, N = 1000; well within f64 52-bit mantissa
fn sine_interior_nodes(k: usize) {
    let grid = periodic_grid();
    let f = GridFn1D::from_fn(grid, |x| (2.0 * PI * k as f64 * (x - XMIN) / L).sin());
    for i in 1..(N - 1) {
        let x = grid.x_at(i);
        let actual = f.sample(x).unwrap();
        let expected = (2.0 * PI * k as f64 * (x - XMIN) / L).sin();
        assert!(
            (actual - expected).abs() <= 1e-12,
            "k={k}, i={i}: actual={actual:.15e}, expected={expected:.15e}",
        );
    }
}

#[test]
fn g6_periodic_sine_k1_interior_nodes() {
    sine_interior_nodes(1);
}

#[test]
fn g6_periodic_sine_k3_interior_nodes() {
    sine_interior_nodes(3);
}

// ---------------------------------------------------------------------------
// Test 2 — shift-invariance: f.sample(x + L) ≈ f.sample(x).
// ---------------------------------------------------------------------------

/// For `x ∈ (xmin, xmax)` interior, `f.sample(x + L) ≈ f.sample(x)` to 1e-12
/// (periodic round-trip).
#[test]
fn g6_periodic_shift_by_period() {
    let grid = periodic_grid();
    // Use k=1 sine (smooth, period L, f(xmin) ≈ 0).
    let f = GridFn1D::from_fn(grid, |x| (2.0 * PI * (x - XMIN) / L).sin());

    // Sample at 50 interior points (avoid end nodes to stay away from the seam).
    let dx = grid.dx();
    #[allow(clippy::cast_precision_loss)]
    // j ≤ 50, N/52 ≤ 19; product ≤ 950; well within f64 52-bit mantissa
    let x_values: Vec<f64> = (1..=50)
        .map(|j| XMIN + dx * (j * (N / 52)) as f64)
        .collect();

    for x in x_values {
        if x + L >= XMAX {
            continue; // skip if shifted point hits the right edge
        }
        let v0 = f.sample(x).unwrap();
        let vl = f.sample(x + L).unwrap();
        assert!(
            (v0 - vl).abs() <= 1e-12 * (1.0 + v0.abs()),
            "shift by L: x={x:.4}, v0={v0:.15e}, vL={vl:.15e}",
        );
    }
}
