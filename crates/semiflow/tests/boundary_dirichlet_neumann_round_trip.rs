//! v2.6 — boundary policy semantics for `Dirichlet` and `Neumann` variants.
//!
//! Validates the new `BoundaryPolicy::Dirichlet { value }` and `BoundaryPolicy::Neumann`
//! variants through the public `GridFn1D::sample` surface (ADR-0068, math.md §3.5.bis).
//!
//! ## Why `oob_offset >= 1.0`
//!
//! The Catmull-Rom stencil uses 4 control points at indices
//! `(idx-1, idx, idx+1, idx+2)`. For the sample to "see" only ghost nodes,
//! ALL four must be out of range. For a grid on `[0, 1]` with `n >= 4` nodes,
//! `dx = 1/(n-1) <= 1/3`. With `oob_offset >= 1.0`, the stencil index
//! `idx+2 = floor(-oob_offset/dx) + 2 < floor(-1/(1/3)) + 2 = -3 + 2 = -1 < 0`.
//! So all four control points return the ghost value → `sample` returns exactly
//! the boundary-policy value. This holds symmetrically for the right OOB region.
//!
//! Tests that only probe the OOB region at offsets where the stencil mixes ghost
//! and interior nodes would need to assert the Catmull-Rom formula exactly —
//! that is tested at the `bc_value` level in `src/boundary.rs` unit tests.

use proptest::prelude::*;
use semiflow::{boundary::InterpKind, BoundaryPolicy, Grid1D, GridFn1D};

// Grid parameters: [0, 1] with n nodes.
// n >= 4 required by the Catmull-Rom stencil invariant.

// ---------------------------------------------------------------------------
// Helper: build Grid1D<f64> on [0, 1] with n nodes and the given policy.
// ---------------------------------------------------------------------------

fn unit_grid(n: usize, policy: BoundaryPolicy<f64>) -> Grid1D<f64> {
    // Explicitly pin to CubicHermite (Catmull-Rom stencil ±1).
    // The tests use `oob_offset >= 1.0` to ensure ALL stencil points are OOB.
    // This invariant holds for the 4-pt Catmull-Rom stencil (requires idx+2 < 0)
    // but NOT for SepticHermite's 9-pt FD stencil (requires idx+4 < 0 → needs
    // oob_offset > 4*dx, which may exceed 1.0 for small n).
    // v6.0 changed Grid1D::new default to SepticHermite (ADR-0109); pin here
    // to keep the boundary-policy round-trip tests independent of kernel choice.
    Grid1D::new(0.0, 1.0, n)
        .unwrap()
        .with_boundary(policy)
        .with_interp(InterpKind::CubicHermite)
}

// ---------------------------------------------------------------------------
// proptest: Dirichlet OOB → returns `value` (500 cases, offset ≥ 1.0)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 500, ..ProptestConfig::default() })]

    /// Sampling well outside [0,1] with `Dirichlet { value }` must return `value`.
    ///
    /// "Well outside" means `oob_offset >= 1.0` (see module-level rationale):
    /// the Catmull-Rom stencil uses only ghost nodes in this regime, each of which
    /// `bc_value` maps to `value`, so Catmull-Rom evaluates a constant → `value`.
    ///
    /// Tolerance 1e-9: the value passes through Catmull-Rom arithmetic with zero
    /// variance (constant-function kernel) → round-trip error ≤ machine epsilon.
    #[test]
    fn dirichlet_returns_fixed_value_far_outside_grid(
        n in 4usize..=128,
        oob_offset in 1.0f64..=5.0f64,
        value in -1e3f64..=1e3f64,
    ) {
        let grid = unit_grid(n, BoundaryPolicy::Dirichlet { value });
        let f = GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin());

        let x_left  = -oob_offset;
        let x_right = 1.0 + oob_offset;

        let got_l = f.sample(x_left).unwrap();
        let got_r = f.sample(x_right).unwrap();

        prop_assert!(
            (got_l - value).abs() < 1e-9,
            "Dirichlet left OOB (n={}, offset={}): got={}, want={}",
            n,
            oob_offset,
            got_l,
            value
        );
        prop_assert!(
            (got_r - value).abs() < 1e-9,
            "Dirichlet right OOB (n={}, offset={}): got={}, want={}",
            n,
            oob_offset,
            got_r,
            value
        );
    }

    /// Sampling well outside [0,1] with `Neumann` returns the nearest boundary-node value.
    ///
    /// For `oob_offset >= 1.0`: all four Catmull-Rom control points are mapped
    /// to `Inside(0)` (left) or `Inside(n-1)` (right) by the Neumann clamping.
    /// A constant-function Catmull-Rom gives back the same constant.
    ///
    /// `f(x) = x` on the grid, so:
    ///   - left boundary node value = `grid.x_at(0) = 0.0`
    ///   - right boundary node value = `grid.x_at(n-1) = 1.0`
    #[test]
    fn neumann_clamps_to_boundary_node_far_outside(
        n in 4usize..=128,
        oob_offset in 1.0f64..=5.0f64,
    ) {
        let grid = unit_grid(n, BoundaryPolicy::Neumann);
        // f(x_i) = x_i. Left boundary = 0.0, right boundary = 1.0.
        let f = GridFn1D::from_fn(grid, |x| x);

        let x_left  = -oob_offset;
        let x_right = 1.0 + oob_offset;

        let got_l = f.sample(x_left).unwrap();
        let got_r = f.sample(x_right).unwrap();

        // All stencil points map to Inside(0) → all return f(0) = 0.0
        prop_assert!(
            got_l.abs() < 1e-9,
            "Neumann left OOB (n={}, offset={}): got={}, want=0.0",
            n,
            oob_offset,
            got_l
        );
        // All stencil points map to Inside(n-1) → all return f(1) = 1.0
        prop_assert!(
            (got_r - 1.0).abs() < 1e-9,
            "Neumann right OOB (n={}, offset={}): got={}, want=1.0",
            n,
            oob_offset,
            got_r
        );
    }
}

// ---------------------------------------------------------------------------
// Deterministic unit tests
// ---------------------------------------------------------------------------

/// Querying at an exact interior grid node returns the node value for both policies.
#[test]
fn dirichlet_and_neumann_inside_match_interior() {
    let n = 16usize;
    let d_grid = unit_grid(n, BoundaryPolicy::Dirichlet { value: -999.0 });
    let n_grid = unit_grid(n, BoundaryPolicy::Neumann);

    let f_d = GridFn1D::from_fn(d_grid, |x| x * x);
    let f_n = GridFn1D::from_fn(n_grid, |x| x * x);

    // Sample at node 5 (interior): should be x_5^2 regardless of OOB policy.
    let x5 = d_grid.x_at(5);
    let expected = x5 * x5;

    let got_d = f_d.sample(x5).unwrap();
    let got_n = f_n.sample(x5).unwrap();

    assert!(
        (got_d - expected).abs() < 1e-10,
        "Dirichlet interior: got {got_d}, want {expected}"
    );
    assert!(
        (got_n - expected).abs() < 1e-10,
        "Neumann interior: got {got_n}, want {expected}"
    );
}

/// `Dirichlet { value: 0.0 }` far outside the grid returns 0 — same as `ZeroExtend`.
///
/// Both policies produce the same ghost-node value (0) for OOB indices. When
/// all four Catmull-Rom stencil points are OOB (offset >= 1.0 for n >= 4),
/// both kernels evaluate a constant-zero function → both return 0.0.
#[test]
fn dirichlet_zero_value_matches_zero_extend_far_outside() {
    let n = 32usize;
    let d_grid = unit_grid(n, BoundaryPolicy::Dirichlet { value: 0.0 });
    let z_grid = unit_grid(n, BoundaryPolicy::ZeroExtend);

    let f_d = GridFn1D::from_fn(d_grid, |x| (x - 0.5).powi(2));
    let f_z = GridFn1D::from_fn(z_grid, |x| (x - 0.5).powi(2));

    // Use offsets >= 1.0 so all stencil points are OOB.
    let test_xs = [-1.5_f64, -1.0, 1.0 + 1.0, 1.0 + 2.0, 1.0 + 5.0];
    for x in test_xs {
        let got_d = f_d.sample(x).unwrap();
        let got_z = f_z.sample(x).unwrap();
        assert!(
            (got_d - got_z).abs() < 1e-14,
            "Dirichlet(0) vs ZeroExtend at x={x}: got_d={got_d}, got_z={got_z}"
        );
    }
}
