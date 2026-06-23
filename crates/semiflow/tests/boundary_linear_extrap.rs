//! G7 — `BoundaryPolicy::LinearExtrapolate` tests.
//!
//! Test 1: affine `f(x) = 2x + 1` is reproduced exactly (machine ε) at
//!   `x ∈ {xmin − 0.5·dx, xmin − 2·dx, xmax + 1.5·dx}`.
//!
//! Test 2: noise bound — random grid data with `‖f‖_∞ ≤ M`; extrapolated
//!   values bounded by `M + C` where C is a generous constant absorbing the
//!   3-point slope formula's possible amplification at the boundary stencil.

use semiflow::{BoundaryPolicy, Grid1D, GridFn1D};

// Grid parameters.
const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
const N: usize = 1000;

fn extrap_grid() -> Grid1D {
    Grid1D::new(XMIN, XMAX, N)
        .unwrap()
        .with_boundary(BoundaryPolicy::LinearExtrapolate)
}

// ---------------------------------------------------------------------------
// Test 1 — affine exactness.
// ---------------------------------------------------------------------------

/// Affine function `f(x) = alpha*x + beta` is reproduced machine-exactly by
/// `LinearExtrapolate` at any `x ∈ ℝ`, including outside `[xmin, xmax]`.
///
/// The tolerance is 16·ε·scale (derivation: boundary-policies-derivation.md §3.5).
#[test]
fn g7_linear_extrap_affine_exact() {
    let grid = extrap_grid();
    let dx = grid.dx();
    let alpha = 2.0_f64;
    let beta = 1.0_f64;
    let f = GridFn1D::from_fn(grid, |x| alpha * x + beta);

    // Points outside the domain (left and right).
    let test_xs = [XMIN - 0.5 * dx, XMIN - 2.0 * dx, XMAX + 1.5 * dx];

    for x in test_xs {
        let actual = f.sample(x).unwrap();
        let expected = alpha * x + beta;
        let scale = (alpha * x).abs().max(beta.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= 16.0 * f64::EPSILON * scale,
            "affine-exact: x={x:.6}, actual={actual:.15e}, expected={expected:.15e}",
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2 — noise bound: extrapolated |value| ≤ M + C.
// ---------------------------------------------------------------------------

/// For data with `‖f‖_∞ ≤ M`, extrapolated values are bounded.
///
/// The 3-point boundary slope formula `-3f0 + 4f1 - f2` has coefficients
/// summing to `|−3|+|4|+|1|=8`. For `d` steps outside, the bound is
/// `M + d * 0.5 * 8 * M = M(1 + 4d)`. We test `d <= 3` (stencil extent).
#[test]
fn g7_linear_extrap_noise_bound() {
    let grid = extrap_grid();
    let dx = grid.dx();

    // Data with ‖f‖_∞ ≤ M = 2.0 (alternating ±1 — worst-case for the slope).
    // Boundary stencil: f[0]=1, f[1]=-1, f[2]=1 → slope_combo = -3-4+1 = -6.
    // For d=1 left: 1 - 1*0.5*(-6) = 1 + 3 = 4.  Bound: M*(1+4*1) = 10.
    let values: std::vec::Vec<f64> = (0..N)
        .map(|i| if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 })
        .collect();
    let f = GridFn1D::new(grid, values).unwrap();
    let cap_m = 1.0_f64;

    // Test extrapolation at 1, 2, 3 dx outside on both sides.
    for d in 1_u32..=3 {
        let x_left = XMIN - f64::from(d) * dx;
        let x_right = XMAX + f64::from(d) * dx;
        let bound = cap_m * (1.0 + 4.0 * f64::from(d));

        let vl = f.sample(x_left).unwrap();
        let vr = f.sample(x_right).unwrap();
        assert!(
            vl.abs() <= bound,
            "noise bound left: d={d}, x={x_left:.6}, |value|={:.6}, bound={bound:.6}",
            vl.abs()
        );
        assert!(
            vr.abs() <= bound,
            "noise bound right: d={d}, x={x_right:.6}, |value|={:.6}, bound={bound:.6}",
            vr.abs()
        );
    }
}
