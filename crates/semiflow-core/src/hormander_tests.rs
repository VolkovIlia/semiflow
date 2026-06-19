// Tests for hormander.rs — moved from hormander.rs (batch H6).
// Exact float comparisons in tests verify round-trip identity or sentinel values.
use super::*;

// ── Kolmogorov drift ──────────────────────────────────────────────────────

#[test]
fn kolmogorov_drift_at_unit_velocity() {
    let x0 = KolmogorovDriftX0::<f64> { _f: PhantomData };
    let mut out = [0.0_f64; 2];
    x0.evaluate(&[1.0, 2.0], &mut out).unwrap();
    // X₀(x=1, v=2) = (v, 0) = (2, 0)
    assert_eq!(out, [2.0, 0.0]);
}

#[test]
fn kolmogorov_drift_at_origin() {
    let x0 = KolmogorovDriftX0::<f64> { _f: PhantomData };
    let mut out = [0.0_f64; 2];
    x0.evaluate(&[0.0, 0.0], &mut out).unwrap();
    assert_eq!(out, [0.0, 0.0]);
}

#[test]
fn kolmogorov_diffusion_is_constant_unit() {
    let x1 = KolmogorovDiffusionX1::<f64> { _f: PhantomData };
    let mut out = [0.0_f64; 2];
    x1.evaluate(&[3.0, 5.0], &mut out).unwrap();
    // X₁(any) = (0, 1)
    assert_eq!(out, [0.0, 1.0]);
}

#[test]
fn kolmogorov_diffusion_f32_unit() {
    let x1 = KolmogorovDiffusionX1::<f32> { _f: PhantomData };
    let mut out = [0.0_f32; 2];
    x1.evaluate(&[-1.0, 7.0], &mut out).unwrap();
    assert_eq!(out, [0.0_f32, 1.0_f32]);
}

// ── Heisenberg fields ─────────────────────────────────────────────────────

#[test]
fn heisenberg_x_at_origin() {
    let hx = HeisenbergX::<f64> { _f: PhantomData };
    let mut out = [0.0_f64; 3];
    hx.evaluate(&[0.0, 0.0, 0.0], &mut out).unwrap();
    // X₁(0,0,0) = (1, 0, 0) since −y/2 = 0
    assert_eq!(out, [1.0, 0.0, 0.0]);
}

#[test]
fn heisenberg_x_off_axis() {
    let hx = HeisenbergX::<f64> { _f: PhantomData };
    let mut out = [0.0_f64; 3];
    hx.evaluate(&[1.0, 4.0, 0.0], &mut out).unwrap();
    // X₁(1, 4, 0) = (1, 0, −y/2) = (1, 0, −2)
    assert!((out[0] - 1.0).abs() < 1e-15);
    assert!((out[1] - 0.0).abs() < 1e-15);
    assert!((out[2] - (-2.0)).abs() < 1e-15);
}

#[test]
fn heisenberg_y_off_axis() {
    let hy = HeisenbergY::<f64> { _f: PhantomData };
    let mut out = [0.0_f64; 3];
    hy.evaluate(&[3.0, 0.0, 0.0], &mut out).unwrap();
    // X₂(3, 0, 0) = (0, 1, x/2) = (0, 1, 1.5)
    assert!((out[0] - 0.0).abs() < 1e-15);
    assert!((out[1] - 1.0).abs() < 1e-15);
    assert!((out[2] - 1.5).abs() < 1e-15);
}

#[test]
fn heisenberg_y_at_origin() {
    let hy = HeisenbergY::<f64> { _f: PhantomData };
    let mut out = [0.0_f64; 3];
    hy.evaluate(&[0.0, 0.0, 0.0], &mut out).unwrap();
    // X₂(0, 0, 0) = (0, 1, 0) since x/2 = 0
    assert_eq!(out, [0.0, 1.0, 0.0]);
}

// ── Lie bracket default (numerical) ──────────────────────────────────────

/// [X₁, X₀]_Kolmogorov ≈ (+1, 0) = +∂_x (generates missing x-direction).
///
/// Analytical: [X₁, X₀]f = (X₁∘X₀ − X₀∘X₁)f
///           = ∂_v(v·∂_x f) − v·∂_x(∂_v f)
///           = ∂_x f + v·∂_{vx} f − v·∂_{xv} f = ∂_x f
/// So [X₁, X₀] = ∂_x = (+1, 0) — step-2 Carnot (spans missing x-direction).
#[test]
fn kolmogorov_lie_bracket_x1_x0_numerical() {
    let x0: KolmogorovDriftX0<f64> = KolmogorovPhaseSpace::x0_drift();
    let x1: KolmogorovDiffusionX1<f64> = KolmogorovPhaseSpace::x1_diffusion();
    let pt = [0.5_f64, 1.0_f64];
    let mut out = [0.0_f64; 2];
    // [X₁, X₀] numerically via central-difference Jacobians
    x1.bracket_with(&x0, &pt, &mut out).unwrap();
    // [X₁, X₀] = [∂_v, v·∂_x] = +∂_x: out ≈ (+1, 0)
    assert!((out[0] - 1.0).abs() < 1e-4, "out[0]={}", out[0]);
    assert!(out[1].abs() < 1e-4, "out[1]={}", out[1]);
}
