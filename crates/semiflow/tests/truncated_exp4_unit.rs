//! Unit tests for `TruncatedExp4thDiffusionChernoff` (v0.6.0, ADR-0013).
//!
//! - `order_is_2` — `order()` returns 2 (τ-axis; post-D1, math.md §11.1.bis).
//! - `growth_is_1_0` — `growth()` returns `(1.0, 0.0)`.
//! - `g4_stencil_polynomial_exactness` — constant-a G⁴ matches analytic f'' within 1e-10.
//! - `g4_stencil_variable_a_spot_check` — G⁴ at one node matches (a·f')' within 1e-6.
//! - `cfl_boundary_accept` — apply at 0.95×CFL returns Ok.
//! - `cfl_boundary_reject` — apply at 1.05×CFL returns Err(CflViolated).
//! - `constant_a_steady_state_convergence` — both `TruncatedExp` types converge to oracle.

use semiflow_core::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    ChernoffSemigroup, Grid1D, GridFn1D, SemiflowError, State, TruncatedExp4thDiffusionChernoff,
    TruncatedExpDiffusionChernoff,
};

fn a_const_one(_: f64) -> f64 {
    1.0
}
fn a_const_half(_: f64) -> f64 {
    0.5
}
fn a_zero(_: f64) -> f64 {
    0.0
}
fn a_vary(x: f64) -> f64 {
    1.0 + 0.2 * x * x
}

/// Apply the G⁴ stencil at `x_i` for arbitrary `a` and `f` closures.
/// `h_p2/h_p1/h_m1/h_m2` and `a_p3h/a_p1h/a_m1h/a_m3h` are stencil names (math convention).
#[allow(clippy::similar_names)]
fn g4_at_with(x_i: f64, dx: f64, f: impl Fn(f64) -> f64, a: impl Fn(f64) -> f64) -> f64 {
    let dx_sq = dx * dx;
    let (h_p2, h_p1, h_i, h_m1, h_m2) = (
        f(x_i + 2.0 * dx),
        f(x_i + dx),
        f(x_i),
        f(x_i - dx),
        f(x_i - 2.0 * dx),
    );
    let (a_p3h, a_p1h, a_m1h, a_m3h) = (
        a(x_i + 1.5 * dx),
        a(x_i + 0.5 * dx),
        a(x_i - 0.5 * dx),
        a(x_i - 1.5 * dx),
    );
    (-a_p3h * (h_p2 - h_p1) / 12.0 + 5.0 * a_p1h * (h_p1 - h_i) / 4.0
        - 5.0 * a_m1h * (h_i - h_m1) / 4.0
        + a_m3h * (h_m1 - h_m2) / 12.0)
        / dx_sq
}

/// For constant a ≡ 1, G⁴ is just the standard 4th-order Laplacian.
fn g4_const_a_at(x_i: f64, dx: f64, f: impl Fn(f64) -> f64) -> f64 {
    g4_at_with(x_i, dx, f, |_| 1.0)
}

#[test]
fn order_is_2() {
    let m4 = TruncatedExp4thDiffusionChernoff::new(
        a_const_one,
        a_zero,
        a_zero,
        1.0,
        Grid1D::new(-5.0, 5.0, 100).expect("grid"),
    );
    assert_eq!(m4.order(), 2, "TruncatedExp4thDiffusionChernoff::order() returns τ-axis 2; dx⁴ verified by G3⁴ (math.md §11.1.bis).");
}

#[test]
fn growth_is_1_0() {
    let m4 = TruncatedExp4thDiffusionChernoff::new(
        a_const_one,
        a_zero,
        a_zero,
        1.0,
        Grid1D::new(-5.0, 5.0, 100).expect("grid"),
    );
    let g = m4.growth();
    assert_eq!((g.multiplier, g.omega), (1.0, 0.0));
}

/// Constant-a G⁴ polynomial-exactness: f = x^k (k=0..4), G⁴f|_i = f''(`x_i`).
#[test]
fn g4_stencil_polynomial_exactness() {
    let dx = 0.01_f64;
    // k=0: 1 → f''=0
    assert!(g4_const_a_at(0.0, dx, |_| 1.0).abs() < 1e-10, "k=0 failed");
    // k=1: x → f''=0
    assert!(g4_const_a_at(0.0, dx, |x| x).abs() < 1e-10, "k=1 failed");
    // k=2: x² → f''=2
    assert!(
        (g4_const_a_at(0.0, dx, |x| x * x) - 2.0).abs() < 1e-10,
        "k=2 failed"
    );
    // k=3: x³ → f''(x)=6x; at x=1: f''=6
    assert!(
        (g4_const_a_at(1.0, dx, |x| x * x * x) - 6.0).abs() < 1e-10,
        "k=3 failed"
    );
    // k=4: x⁴ → f''(x)=12x²; at x=1: f''=12
    assert!(
        (g4_const_a_at(1.0, dx, |x| x * x * x * x) - 12.0).abs() < 1e-8,
        "k=4 failed"
    );
}

/// Variable-a spot check: a(x)=1+0.2x², f=sin(x).
/// (a·f')' = 0.4x·cos(x) + (1+0.2x²)·(-sin(x)).
#[test]
fn g4_stencil_variable_a_spot_check() {
    let x_i = 0.5_f64;
    let dx = 1e-3_f64;
    let g4_val = g4_at_with(x_i, dx, f64::sin, a_vary);
    let analytic = 0.4 * x_i * x_i.cos() + (1.0 + 0.2 * x_i * x_i) * (-x_i.sin());
    assert!(
        (g4_val - analytic).abs() < 1e-6,
        "variable-a G⁴: got {g4_val:.8e}, expected {analytic:.8e}"
    );
}

#[test]
fn cfl_boundary_accept() {
    let grid = Grid1D::new(-5.0, 5.0, 64).expect("grid");
    let dx = grid.dx();
    let tau = 0.95 * 3.0 * dx * dx / 8.0; // a_norm=1
    let m4 = TruncatedExp4thDiffusionChernoff::new(a_const_one, a_zero, a_zero, 1.0, grid);
    let result = m4.apply_chernoff(tau, &GridFn1D::from_fn(grid, |x| libm::exp(-x * x)));
    assert!(result.is_ok(), "0.95×CFL should succeed");
}

#[test]
fn cfl_boundary_reject() {
    let grid = Grid1D::new(-5.0, 5.0, 64).expect("grid");
    let dx = grid.dx();
    let tau = 1.05 * 3.0 * dx * dx / 8.0; // a_norm=1
    let m4 = TruncatedExp4thDiffusionChernoff::new(a_const_one, a_zero, a_zero, 1.0, grid);
    let result = m4.apply_chernoff(tau, &GridFn1D::from_fn(grid, |x| libm::exp(-x * x)));
    assert!(
        matches!(result, Err(SemiflowError::CflViolated { .. })),
        "1.05×CFL must return Err(CflViolated); got {result:?}",
    );
}

/// Both `TruncatedExp` types converge to oracle; NOT bit-equal (different stencils).
#[test]
fn constant_a_steady_state_convergence() {
    let grid = Grid1D::new(-10.0, 10.0, 256).expect("grid");
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
    // Oracle: u(1,x) = 3^{-1/2}·exp(-x²/3) for a=0.5, IC=exp(-x²), T=1.
    let oracle = GridFn1D::from_fn(grid, |x| (3.0_f64).sqrt().recip() * libm::exp(-x * x / 3.0));

    // n=800 steps: tau=0.00125 < CFL_M4_max≈0.00458 ✓
    let m4 = TruncatedExp4thDiffusionChernoff::new(a_const_half, a_zero, a_zero, 0.5, grid);
    let m2 = TruncatedExpDiffusionChernoff::new(a_const_half, a_zero, a_zero, 0.5, grid);
    let out4 = ChernoffSemigroup::new(m4, 800)
        .expect("sem4")
        .evolve(1.0, &f0)
        .expect("m4");
    let out2 = ChernoffSemigroup::new(m2, 800)
        .expect("sem2")
        .evolve(1.0, &f0)
        .expect("m2");

    let err = |out: &GridFn1D| {
        let mut d = out.clone();
        d.axpy(-1.0, &oracle);
        d.norm_sup()
    };
    let (err4, err2) = (err(&out4), err(&out2));
    assert!(err4 < 0.1, "M4 error too large: {err4:.3e}");
    assert!(err2 < 0.1, "M2 error too large: {err2:.3e}");

    // NOT bit-equal: different stencil orders.
    let mut dc = out4.clone();
    dc.axpy(-1.0, &out2);
    assert!(dc.norm_sup() > 0.0, "should NOT be bit-equal");
    eprintln!(
        "err4={err4:.3e}  err2={err2:.3e}  cross={:.3e}",
        dc.norm_sup()
    );
}
