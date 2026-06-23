//! Unit tests for `manifold.rs`.
//!
//! Extracted from `manifold.rs` to keep that file under 500 lines.

// Exact float comparisons in tests verify round-trip identity or sentinel values.
#![allow(clippy::float_cmp)]

use super::*;

// --- Torus ---

#[test]
fn torus_scalar_curvature_is_zero() {
    let t = Torus::<f64, 2>::unit();
    assert_eq!(t.scalar_curvature(&[0.3, 0.7]), 0.0);
}

#[test]
fn torus_exp_map_wraps_into_unit_cell() {
    let t = Torus::<f64, 2>::unit();
    let mut out = [0.0f64; 2];
    t.exp_map(&[0.5, 0.5], &[0.8, 0.8], &mut out).unwrap();
    // (0.5 + 0.8) mod 1.0 = 0.3
    assert!((out[0] - 0.3).abs() < 1e-12, "out[0] = {}", out[0]);
    assert!((out[1] - 0.3).abs() < 1e-12, "out[1] = {}", out[1]);
}

#[test]
fn torus_exp_map_identity_at_zero_v() {
    let t = Torus::<f64, 2>::unit();
    let mut out = [0.0f64; 2];
    t.exp_map(&[0.3, 0.7], &[0.0, 0.0], &mut out).unwrap();
    assert!((out[0] - 0.3).abs() < 1e-15);
    assert!((out[1] - 0.7).abs() < 1e-15);
}

#[test]
fn torus_injectivity_radius_is_half_min_period() {
    let t = Torus::<f64, 3>::with_period([0.5, 1.0, 0.7]).unwrap();
    assert!((t.injectivity_radius() - 0.25).abs() < 1e-12);
}

#[test]
fn torus_parallel_transport_is_identity() {
    let t = Torus::<f64, 2>::unit();
    let v = [0.3, -0.7];
    let mut out = [0.0f64; 2];
    t.parallel_transport(&[0.1, 0.2], &[0.4, 0.6], &v, &mut out)
        .unwrap();
    assert!((out[0] - v[0]).abs() < 1e-15);
    assert!((out[1] - v[1]).abs() < 1e-15);
}

#[test]
fn torus_volume_element_log_is_zero() {
    let t = Torus::<f64, 2>::unit();
    assert_eq!(t.volume_element_log(&[0.3, 0.7]), 0.0);
}

#[test]
fn torus_with_period_rejects_non_positive() {
    assert!(Torus::<f64, 2>::with_period([0.5, 0.0]).is_err());
    assert!(Torus::<f64, 2>::with_period([-1.0, 1.0]).is_err());
    assert!(Torus::<f64, 2>::with_period([f64::NAN, 1.0]).is_err());
}

// --- Sphere2 ---

#[test]
fn sphere2_scalar_curvature_unit() {
    let s = Sphere2::<f64>::unit();
    assert!((s.scalar_curvature(&[0.5, 1.0]) - 2.0).abs() < 1e-12);
}

#[test]
fn sphere2_scalar_curvature_radius5() {
    let s = Sphere2::<f64>::with_radius(5.0).unwrap();
    // R = 2 / 25 = 0.08
    assert!((s.scalar_curvature(&[0.5, 1.0]) - 0.08).abs() < 1e-12);
}

#[test]
fn sphere2_exp_map_zero_v_is_identity() {
    let s = Sphere2::<f64>::unit();
    let mut out = [0.0f64; 2];
    s.exp_map(&[0.5, 1.0], &[0.0, 0.0], &mut out).unwrap();
    assert!((out[0] - 0.5).abs() < 1e-12, "theta = {}", out[0]);
    assert!((out[1] - 1.0).abs() < 1e-12, "phi = {}", out[1]);
}

#[test]
fn sphere2_exp_map_north_pole_quarter_great_circle() {
    // exp at north pole (0, 0) with v = (π/2, 0) should reach equator θ = π/2
    let s = Sphere2::<f64>::unit();
    let pi_half = core::f64::consts::FRAC_PI_2;
    let mut out = [0.0f64; 2];
    s.exp_map(&[0.01, 0.0], &[pi_half, 0.0], &mut out).unwrap();
    // Should end up near equator: θ ≈ π/2 + 0.01
    let expected_theta = pi_half + 0.01;
    assert!((out[0] - expected_theta).abs() < 1e-6, "theta={}", out[0]);
}

#[test]
fn sphere2_with_radius_rejects_invalid() {
    assert!(Sphere2::<f64>::with_radius(f64::NAN).is_err());
    assert!(Sphere2::<f64>::with_radius(0.0).is_err());
    assert!(Sphere2::<f64>::with_radius(-1.0).is_err());
}

#[test]
fn sphere2_injectivity_radius() {
    let s = Sphere2::<f64>::unit();
    let pi = core::f64::consts::PI;
    assert!((s.injectivity_radius() - pi).abs() < 1e-12);
}

// --- Hyperbolic2 ---

#[test]
fn hyperbolic2_scalar_curvature_unit() {
    let h = Hyperbolic2::<f64>::unit();
    assert!((h.scalar_curvature(&[0.3, 0.4]) - (-2.0)).abs() < 1e-12);
}

#[test]
fn hyperbolic2_scalar_curvature_scale2() {
    let h = Hyperbolic2::<f64>::with_scale(2.0).unwrap();
    // R = -2 / 4 = -0.5
    assert!((h.scalar_curvature(&[0.1, 0.2]) - (-0.5)).abs() < 1e-12);
}

#[test]
fn hyperbolic2_exp_map_origin_zero_v_is_origin() {
    let h = Hyperbolic2::<f64>::unit();
    let mut out = [0.0f64; 2];
    h.exp_map(&[0.0, 0.0], &[0.0, 0.0], &mut out).unwrap();
    assert!(out[0].abs() < 1e-12);
    assert!(out[1].abs() < 1e-12);
}

#[test]
fn hyperbolic2_exp_map_origin_stays_in_disk() {
    let h = Hyperbolic2::<f64>::unit();
    let mut out = [0.0f64; 2];
    h.exp_map(&[0.0, 0.0], &[0.5, 0.3], &mut out).unwrap();
    let r_sq = out[0] * out[0] + out[1] * out[1];
    assert!(r_sq < 1.0, "result outside disk: r² = {r_sq}");
}

#[test]
fn hyperbolic2_with_scale_rejects_invalid() {
    assert!(Hyperbolic2::<f64>::with_scale(f64::NAN).is_err());
    assert!(Hyperbolic2::<f64>::with_scale(0.0).is_err());
    assert!(Hyperbolic2::<f64>::with_scale(-1.0).is_err());
}

#[test]
fn hyperbolic2_injectivity_radius_is_infinity() {
    let h = Hyperbolic2::<f64>::unit();
    assert!(h.injectivity_radius().is_infinite());
}
