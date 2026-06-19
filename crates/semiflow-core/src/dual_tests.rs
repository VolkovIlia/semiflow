// Unit tests for `Dual<F>` forward-mode AD (§46, ADR-0133).
// Included into `dual.rs` via `include!` inside `#[cfg(test)] mod tests`.

use super::*;

// fn-ptrs for dual_kernels_compile_and_run (closures cannot capture + coerce).
fn a_theta(x: Dual<f64>) -> Dual<f64> {
    let _ = x;
    Dual::variable(0.5)
}
fn zero_fn(x: Dual<f64>) -> Dual<f64> {
    Dual::constant(x.value * 0.0)
}

/// Basic arithmetic per §46.2.
#[test]
fn arithmetic_rules() {
    let u = Dual::new(2.0_f64, 1.0);
    let v = Dual::new(3.0_f64, 0.0);
    assert_eq!((u + v).tangent, 1.0);
    assert_eq!((u * v).tangent, 3.0); // product rule: 1·3 + 2·0
    let q = u / v;
    assert!((q.tangent - 1.0_f64 / 3.0).abs() < 1e-15); // quotient rule
}

/// `PartialOrd` compares value only (§46.2).
#[test]
fn ordering_value_only() {
    let a = Dual::new(1.0_f64, 100.0);
    let b = Dual::new(2.0_f64, -100.0);
    assert!(a < b);
}

/// `Dual<Dual<f64>>` compiles — hyper-dual Γ path (§46.4).
#[test]
fn hyper_dual_compiles() {
    let x: Dual<Dual<f64>> = Dual::variable(Dual::variable(1.0_f64));
    let y = x.exp();
    assert!((y.value.value - core::f64::consts::E).abs() < 1e-12);
}

/// Kernels compile and run at `F = Dual<f64>` (ADR-0133 acceptance criterion).
///
/// Uses generic scalar paths (`apply_f`, `new_generic`) — SIMD-optimised
/// `ChernoffFunction<f64>` remains f64-only per ADR-0018. The manual
/// triple-step proves `StrangSplit` composition compiles at F = Dual<f64>.
#[test]
fn dual_kernels_compile_and_run() {
    check_diffusion_chernoff();
    check_diffusion4th_chernoff();
    check_strang_triple_step();
}

fn check_diffusion_chernoff() {
    use crate::{DiffusionChernoff, Grid1D, GridFn1D};
    let n = 32usize;
    let grid = Grid1D::<Dual<f64>>::new_generic(Dual::constant(-5.0), Dual::constant(5.0), n)
        .expect("valid grid");
    let diff = DiffusionChernoff::<Dual<f64>>::new(
        a_theta as fn(Dual<f64>) -> Dual<f64>,
        zero_fn as fn(Dual<f64>) -> Dual<f64>,
        zero_fn as fn(Dual<f64>) -> Dual<f64>,
        1.0_f64,
        grid,
    );
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau = Dual::constant(0.01);
    let u1 = diff.apply_f(tau, &u0).expect("DiffusionChernoff::apply_f");
    assert!(u1.values[n / 2].value.is_finite(), "value finite");
    assert!(u1.values[n / 2].tangent.is_finite(), "tangent finite");
}

fn check_diffusion4th_chernoff() {
    use crate::{Diffusion4thChernoff, Grid1D, GridFn1D};
    let n = 32usize;
    let grid = Grid1D::<Dual<f64>>::new_generic(Dual::constant(-5.0), Dual::constant(5.0), n)
        .expect("valid grid");
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau = Dual::constant(0.01);
    let diff4 = Diffusion4thChernoff::<Dual<f64>>::new_generic(
        a_theta as fn(Dual<f64>) -> Dual<f64>,
        zero_fn as fn(Dual<f64>) -> Dual<f64>,
        zero_fn as fn(Dual<f64>) -> Dual<f64>,
        1.0_f64,
        grid,
    );
    let u1_4 = diff4
        .apply_f(tau, &u0)
        .expect("Diffusion4thChernoff::apply_f");
    assert!(u1_4.values[n / 2].value.is_finite());
    assert!(u1_4.values[n / 2].tangent.is_finite());
}

fn check_strang_triple_step() {
    use crate::{DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D};
    let n = 32usize;
    let grid = Grid1D::<Dual<f64>>::new_generic(Dual::constant(-5.0), Dual::constant(5.0), n)
        .expect("valid grid");
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau = Dual::constant(0.01);
    let diff = DiffusionChernoff::<Dual<f64>>::new(
        a_theta as fn(Dual<f64>) -> Dual<f64>,
        zero_fn as fn(Dual<f64>) -> Dual<f64>,
        zero_fn as fn(Dual<f64>) -> Dual<f64>,
        1.0_f64,
        grid,
    );
    let drift = DriftReactionChernoff::<Dual<f64>>::new_generic(
        zero_fn as fn(Dual<f64>) -> Dual<f64>,
        zero_fn as fn(Dual<f64>) -> Dual<f64>,
        0.0_f64,
        grid,
    );
    // StrangSplit D(τ/2)·R(τ)·D(τ/2) manual triple-step.
    let tau2 = tau / Dual::constant(2.0);
    let u_end = diff
        .apply_f(tau2, &u0)
        .and_then(|h| drift.apply_f(tau, &h))
        .and_then(|m| diff.apply_f(tau2, &m))
        .expect("strang triple-step");
    assert!(u_end.values[n / 2].value.is_finite(), "strang value finite");
    assert!(
        u_end.values[n / 2].tangent.is_finite(),
        "strang tangent finite"
    );
}
