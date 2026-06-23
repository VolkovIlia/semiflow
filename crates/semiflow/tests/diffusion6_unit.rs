//! Unit tests for `Diffusion6thChernoff` (v0.7.0, ADR-0015).
//!
//! C1 tests (mirroring `diffusion4_unit.rs`, adapted to ζ⁶):
//!
//! 1. `tau_zero_is_identity` — apply(0.0, &f) returns f element-wise.
//! 2. `order_is_2` — `order()` == 2 (τ-axis; D1 audit lesson; spatial dx⁶ is gate G3⁶).
//! 3. `growth_is_1_0` — `growth()` == (1.0, 0.0).
//! 4. `negative_tau_returns_error` — `validate_tau` rejects tau < 0.
//! 5. `zero_a_returns_error` — a(x) == 0 triggers `DomainViolation` (sqrt path).
//! 6. `g6_stencil_variable_a_spot_check` — single-step ζ⁶ stays finite, close to f.
//! 7. `constant_a_steady_state_convergence` — long-time integration converges to oracle.
//! 8. `c7_caller_invariant_smoke` — smooth a ∈ C⁷ completes one step without panic.
// v7.0: QuinticHermite removed (ADR-0109 removal clock fulfilled).
// Tests updated to use SepticHermite (v6.0+ default).

use semiflow::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    Diffusion6thChernoff, Grid1D, GridFn1D, InterpKind,
};

// ---------------------------------------------------------------------------
// Constant fn pointers (fn-pointer restriction: closures that capture locals
// cannot be coerced to fn pointers; use module-level fns instead).
// ---------------------------------------------------------------------------

fn a_half(_: f64) -> f64 {
    0.5
}
fn a_zero(_: f64) -> f64 {
    0.0
}
// Negative a — violates strict ellipticity (a < 0 rejected by validate_a_x).
fn a_negative(_: f64) -> f64 {
    -0.5
}
fn a_tanh_1(x: f64) -> f64 {
    1.0 + 0.1 * x.tanh()
}
fn a_tanh_1_prime(x: f64) -> f64 {
    let ch = x.cosh();
    0.1 / (ch * ch)
}
fn a_tanh_1_double_prime(x: f64) -> f64 {
    let ch = x.cosh();
    let sh = x.sinh();
    -0.2 * sh / (ch * ch * ch)
}

fn gaussian(grid: Grid1D, sigma: f64) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| libm::exp(-(x * x) / (2.0 * sigma * sigma)))
}

// ---------------------------------------------------------------------------
// Test 1: tau == 0 is identity
// ---------------------------------------------------------------------------

#[test]
fn tau_zero_is_identity() {
    let grid = Grid1D::new(-10.0, 10.0, 200)
        .unwrap()
        .with_interp(InterpKind::SepticHermite);
    let d6 = Diffusion6thChernoff::new(a_half, a_zero, a_zero, 0.5, grid);
    let f = gaussian(grid, 1.0);
    let out = d6.apply_chernoff(0.0, &f).expect("tau=0 apply");
    for (i, (&fi, &oi)) in f.values.iter().zip(out.values.iter()).enumerate() {
        assert!(
            (fi - oi).abs() < 1e-13,
            "tau=0 identity failed at i={i}: f={fi:.15e} out={oi:.15e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: order() == 2  (τ-axis; see math.md §11.1.bis)
// ---------------------------------------------------------------------------

#[test]
fn order_is_2() {
    let grid = Grid1D::new(-5.0, 5.0, 100).unwrap();
    let d6 = Diffusion6thChernoff::new(a_half, a_zero, a_zero, 0.5, grid);
    assert_eq!(
        d6.order(),
        2,
        "Diffusion6thChernoff::order() must return 2 (τ-axis). \
         Spatial dx⁶ is verified by gate G3⁶, not order(). \
         See math.md §11.1.bis and audit D1."
    );
}

// ---------------------------------------------------------------------------
// Test 3: growth() == (1.0, 0.0)
// ---------------------------------------------------------------------------

#[test]
fn growth_is_1_0() {
    let grid = Grid1D::new(-5.0, 5.0, 100).unwrap();
    let d6 = Diffusion6thChernoff::new(a_half, a_zero, a_zero, 0.5, grid);
    let g = d6.growth();
    assert_eq!(
        (g.multiplier, g.omega),
        (1.0, 0.0),
        "Diffusion6thChernoff::growth() must return (1.0, 0.0)"
    );
}

// ---------------------------------------------------------------------------
// Test 4: negative tau returns error
// ---------------------------------------------------------------------------

#[test]
fn negative_tau_returns_error() {
    let grid = Grid1D::new(-5.0, 5.0, 100).unwrap();
    let d6 = Diffusion6thChernoff::new(a_half, a_zero, a_zero, 0.5, grid);
    let f = gaussian(grid, 1.0);
    let res = d6.apply_chernoff(-1e-6, &f);
    assert!(
        res.is_err(),
        "negative tau must return Err(DomainViolation); got Ok"
    );
}

// ---------------------------------------------------------------------------
// Test 5: negative a(x) triggers DomainViolation (sqrt of negative fails)
// ---------------------------------------------------------------------------

/// `validate_a_x` in `diffusion6.rs` rejects `a_x < 0` with `DomainViolation`.
/// `a ≡ 0` is allowed (sqrt(0) = 0, valid K7 kernel with zero shifts).
/// `a < 0` is rejected immediately.
#[test]
fn negative_a_returns_error() {
    let grid = Grid1D::new(-5.0, 5.0, 100)
        .unwrap()
        .with_interp(InterpKind::SepticHermite);
    // a ≡ -0.5 violates strict ellipticity (validate_a_x rejects a < 0)
    let d6 = Diffusion6thChernoff::new(a_negative, a_zero, a_zero, 0.5, grid);
    let f = gaussian(grid, 1.0);
    let res = d6.apply_chernoff(1e-3, &f);
    assert!(
        res.is_err(),
        "a(x)=-0.5 must return Err(DomainViolation) (a < 0 rejected); got Ok"
    );
}

// ---------------------------------------------------------------------------
// Test 6: variable-a spot check — single step stays finite and close to f
// ---------------------------------------------------------------------------

/// a(x) = 1 + 0.1·tanh(x) — C^∞, strictly positive, smooth.
/// Single step at τ=1e-4 should stay close to IC (small diffusion).
#[test]
fn g6_stencil_variable_a_spot_check() {
    let grid = Grid1D::new(-5.0, 5.0, 200)
        .unwrap()
        .with_interp(InterpKind::SepticHermite);
    let d6 = Diffusion6thChernoff::new(a_tanh_1, a_tanh_1_prime, a_tanh_1_double_prime, 1.2, grid);
    let f = gaussian(grid, 1.0);
    let tau = 1e-4;
    let out = d6.apply_chernoff(tau, &f).expect("variable-a apply");

    // At τ=1e-4, diffusion is small — output should be close to input.
    let max_diff: f64 = f
        .values
        .iter()
        .zip(out.values.iter())
        .map(|(&fi, &oi)| (fi - oi).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-2,
        "variable-a single step: ||out - f||_inf = {max_diff:.4e} >= 1e-2 (τ={tau:.1e})"
    );

    for (i, &v) in out.values.iter().enumerate() {
        assert!(v.is_finite(), "NaN/Inf at node {i}: {v}");
    }
}

// ---------------------------------------------------------------------------
// Test 7: constant-a steady-state convergence (long-time heat equation)
// ---------------------------------------------------------------------------

/// Heat equation `u_t` = `0.5·u_xx`, T=2, IC = Gaussian(0, σ=1).
/// Oracle at T: Gaussian(0, `σ_T²`) with `σ_T` = sqrt(1 + 2*0.5*2) = sqrt(3).
///
/// Gate: sup-norm error < 0.1 (sanity; not the G3⁶ convergence gate).
#[test]
fn constant_a_steady_state_convergence() {
    let t_final = 2.0_f64;
    let sigma_ic = 1.0_f64;
    // σ_T = sqrt(σ_ic² + 2·a·T) = sqrt(1 + 2*0.5*2) = sqrt(3)
    let sigma_t = libm::sqrt(sigma_ic * sigma_ic + 2.0 * 0.5 * t_final);
    let n_steps = 2000_usize;
    // n_steps=2000 fits f64 mantissa (52 bits > 11 bits needed); no precision loss.
    #[allow(clippy::cast_precision_loss)]
    let tau = t_final / n_steps as f64;

    let grid = Grid1D::new(-10.0, 10.0, 400)
        .unwrap()
        .with_interp(InterpKind::SepticHermite);
    // a=0.5 constant
    let d6 = Diffusion6thChernoff::new(a_half, a_zero, a_zero, 0.5, grid);
    let mut state = gaussian(grid, sigma_ic);

    for _ in 0..n_steps {
        state = d6.apply_chernoff(tau, &state).expect("apply");
    }

    // Oracle: Gaussian with broadened sigma_t, normalized to match IC mass.
    let norm = (2.0 * core::f64::consts::PI * sigma_t * sigma_t)
        .sqrt()
        .recip();
    // (IC was not normalised — it's exp(-x²/(2σ²)); oracle must match that scale)
    let oracle = GridFn1D::from_fn(grid, |x| {
        libm::exp(-(x * x) / (2.0 * sigma_t * sigma_t))
            / (2.0 * core::f64::consts::PI * sigma_t * sigma_t).sqrt()
            * (2.0 * core::f64::consts::PI * sigma_ic * sigma_ic).sqrt()
    });

    let err: f64 = state
        .values
        .iter()
        .zip(oracle.values.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("steady-state err = {err:.4e} (σ_T={sigma_t:.4}, gate < 0.1)");
    assert!(
        err < 0.1,
        "steady-state: ||u_num - oracle||_inf = {err:.4e} >= 0.1"
    );
    let _ = norm; // suppress unused warning
}

// ---------------------------------------------------------------------------
// Test 8: c7_caller_invariant_smoke — smooth C⁷ a(x) completes step OK
// ---------------------------------------------------------------------------

/// a(x) = 1 + 0.1·tanh(x) satisfies a ∈ C^∞ ⊂ C⁷ with bounded derivatives.
/// This smoke test verifies the type accepts the function and runs without panic.
#[test]
fn c7_caller_invariant_smoke() {
    let grid = Grid1D::new(-5.0, 5.0, 100)
        .unwrap()
        .with_interp(InterpKind::SepticHermite);
    let d6 = Diffusion6thChernoff::new(a_tanh_1, a_tanh_1_prime, a_tanh_1_double_prime, 1.1, grid);
    let f = gaussian(grid, 1.0);
    let out = d6.apply_chernoff(1e-3, &f).expect("c7 smoke apply");
    for (i, &v) in out.values.iter().enumerate() {
        assert!(v.is_finite(), "NaN/Inf at node {i}: {v}");
    }
}
