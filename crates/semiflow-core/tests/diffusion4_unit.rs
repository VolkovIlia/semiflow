//! Unit tests for `Diffusion4thChernoff` (v0.6.0, ADR-0013).
//!
//! C1 tests:
//! 1. `order()` returns 2 (τ-axis post-D1, math.md §11.1.bis), `growth()` returns (1.0, 0.0).
//! 2. 7-point FD polynomial-exactness: for `f(x) = x^k`, k ∈ {0..6},
//!    the FD stencil recovers the analytic derivative to within 1e-10.
//! 3. Constant-a fast path: `Diffusion4thChernoff` output matches
//!    `DiffusionChernoff` bit-equal (tolerance 0.0) when a' ≡ a'' ≡ 0.

use core::cell::Cell;

use semiflow_core::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    BoundaryPolicy, Diffusion4thChernoff, DiffusionChernoff, Grid1D, GridFn1D,
};

// Thread-local for constant diffusion coefficient (fn-pointer compatibility).
thread_local! {
    static A0_CELL: Cell<f64> = const { Cell::new(1.0) };
}

fn a_const(_: f64) -> f64 {
    A0_CELL.with(Cell::get)
}
fn a_zero(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Helpers: 7-point FD stencil weights (mirror diffusion4.rs constants)
// ---------------------------------------------------------------------------

const C1: [f64; 7] = [
    -1.0 / 60.0,
    3.0 / 20.0,
    -3.0 / 4.0,
    0.0,
    3.0 / 4.0,
    -3.0 / 20.0,
    1.0 / 60.0,
];
const C2: [f64; 7] = [
    1.0 / 90.0,
    -3.0 / 20.0,
    3.0 / 2.0,
    -49.0 / 18.0,
    3.0 / 2.0,
    -3.0 / 20.0,
    1.0 / 90.0,
];
const C3: [f64; 7] = [
    1.0 / 8.0,
    -1.0,
    13.0 / 8.0,
    0.0,
    -13.0 / 8.0,
    1.0,
    -1.0 / 8.0,
];

/// Apply 7-point FD stencil with given weights, divide by `delta^deriv`.
fn fd7_scalar(f_vals: &[f64; 7], delta: f64, coeffs: &[f64; 7], deriv: u32) -> f64 {
    let sum: f64 = coeffs.iter().zip(f_vals.iter()).map(|(&c, &v)| c * v).sum();
    sum / libm::pow(delta, f64::from(deriv))
}

// ---------------------------------------------------------------------------
// Test 1: order and growth metadata
// ---------------------------------------------------------------------------

#[test]
fn order_is_4() {
    let grid = Grid1D::new(-5.0, 5.0, 100).unwrap();
    let d4 = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    assert_eq!(d4.order(), 2, "Diffusion4thChernoff::order() must return 2 (τ-axis); spatial dx⁴ accuracy is verified by G3⁴, not order(). See math.md §11.1.bis.");
}

#[test]
fn growth_is_1_0() {
    let grid = Grid1D::new(-5.0, 5.0, 100).unwrap();
    let d4 = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let g = d4.growth();
    assert_eq!(
        (g.multiplier, g.omega),
        (1.0, 0.0),
        "Diffusion4thChernoff::growth() must return (1.0, 0.0)"
    );
}

// ---------------------------------------------------------------------------
// Test 2: 7-point FD polynomial-exactness
// ---------------------------------------------------------------------------

/// Evaluate 7-point FD at x=0 on x^k polynomial (analytic sample, no grid needed).
fn fd7_at_zero_for_xk(k: u32, delta: f64, coeffs: &[f64; 7], deriv: u32) -> f64 {
    let ks: [f64; 7] = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let f_vals: [f64; 7] = core::array::from_fn(|j| libm::pow(ks[j] * delta, f64::from(k)));
    fd7_scalar(&f_vals, delta, coeffs, deriv)
}

/// Analytic derivative of x^k at x=0: k!/(k-d)! * 0^(k-d) — only nonzero when k==d.
fn analytic_xk_at_zero(k: u32, deriv: u32) -> f64 {
    if k < deriv {
        return 0.0;
    }
    if k == deriv {
        // k! / 0! = k!
        (1..=k).map(f64::from).product::<f64>()
    } else {
        0.0 // x^(k-d) at x=0 is 0 for k > d
    }
}

#[test]
fn fd7_first_deriv_polynomial_exactness() {
    let delta = 0.1;
    for k in 0u32..=6 {
        let fd = fd7_at_zero_for_xk(k, delta, &C1, 1);
        let analytic = analytic_xk_at_zero(k, 1);
        let err = (fd - analytic).abs();
        assert!(
            err < 1.0e-10,
            "f' FD on x^{k}: fd={fd:.6e}, analytic={analytic:.6e}, err={err:.4e} > 1e-10"
        );
    }
}

#[test]
fn fd7_second_deriv_polynomial_exactness() {
    let delta = 0.1;
    for k in 0u32..=6 {
        let fd = fd7_at_zero_for_xk(k, delta, &C2, 2);
        let analytic = analytic_xk_at_zero(k, 2);
        let err = (fd - analytic).abs();
        assert!(
            err < 1.0e-10,
            "f'' FD on x^{k}: fd={fd:.6e}, analytic={analytic:.6e}, err={err:.4e} > 1e-10"
        );
    }
}

#[test]
fn fd7_third_deriv_polynomial_exactness() {
    let delta = 0.1;
    for k in 0u32..=6 {
        let fd = fd7_at_zero_for_xk(k, delta, &C3, 3);
        let analytic = analytic_xk_at_zero(k, 3);
        let err = (fd - analytic).abs();
        assert!(
            err < 1.0e-10,
            "f''' FD on x^{k}: fd={fd:.6e}, analytic={analytic:.6e}, err={err:.4e} > 1e-10"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: constant-a bit-equal regression vs DiffusionChernoff
// ---------------------------------------------------------------------------

/// Gaussian IC for comparison tests.
fn gaussian(grid: Grid1D, sigma: f64) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| libm::exp(-(x * x) / (2.0 * sigma * sigma)))
}

/// Check bit-equal between `Diffusion4thChernoff` and `DiffusionChernoff` for constant a.
fn constant_a_bit_equal_check(a0: f64, tau: f64, n: usize) {
    A0_CELL.with(|c| c.set(a0));
    let grid = Grid1D::new(-5.0, 5.0, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect);
    let f0 = gaussian(grid, 1.0);

    let d4 = Diffusion4thChernoff::new(a_const, a_zero, a_zero, a0, grid);
    let d2 = DiffusionChernoff::new(a_const, a_zero, a_zero, a0, grid);

    let out4 = d4.apply_chernoff(tau, &f0).expect("d4 apply");
    let out2 = d2.apply_chernoff(tau, &f0).expect("d2 apply");

    for (i, (&v4, &v2)) in out4.values.iter().zip(out2.values.iter()).enumerate() {
        assert_eq!(
            v4.to_bits(),
            v2.to_bits(),
            "constant-a bit-equal failed at i={i}: d4={v4:.15e} d2={v2:.15e} \
             (a0={a0}, tau={tau}, n={n})"
        );
    }
}

#[test]
fn constant_a_bit_equal_a05_tau001_n200() {
    constant_a_bit_equal_check(0.5, 1e-2, 200);
}

#[test]
fn constant_a_bit_equal_a10_tau001_n100() {
    constant_a_bit_equal_check(1.0, 1e-2, 100);
}

#[test]
fn constant_a_bit_equal_a20_tau001_n200() {
    constant_a_bit_equal_check(2.0, 1e-2, 200);
}

#[test]
fn constant_a_bit_equal_a05_tau1e3_n100() {
    constant_a_bit_equal_check(0.5, 1e-3, 100);
}
