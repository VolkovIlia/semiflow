// Unit tests for `resolvent_jump_nd` — included via `include!` in `resolvent_jump_nd.rs`.

use super::*;
use crate::{Grid1D, Grid2D, Grid3D};

/// Smoke: `ResolventJumpChernoff2D` constructs and rejects `m_nodes` < 6.
#[test]
fn rj2d_construction_guard() {
    let gx = Grid1D::new(-5.0, 5.0, 8).unwrap();
    let gy = Grid1D::new(-5.0, 5.0, 8).unwrap();
    let g2 = Grid2D::new(gx, gy);
    assert!(ResolventJumpChernoff2D::new(g2, 5).is_err());
    assert!(ResolventJumpChernoff2D::new(g2, 6).is_ok());
}

/// Smoke: `ResolventJumpChernoff3D` constructs and rejects `m_nodes` < 6.
#[test]
fn rj3d_construction_guard() {
    let gx = Grid1D::new(-1.0, 1.0, 4).unwrap();
    let g3 = Grid3D::new(gx, gx, gx).unwrap();
    assert!(ResolventJumpChernoff3D::new(g3, 5).is_err());
    assert!(ResolventJumpChernoff3D::new(g3, 6).is_ok());
}

/// Smoke: 2D jump returns correct shape and finite values.
#[test]
fn rj2d_jump_smoke() {
    let gx = Grid1D::new(-5.0, 5.0, 8).unwrap();
    let gy = Grid1D::new(-5.0, 5.0, 8).unwrap();
    let g2 = Grid2D::new(gx, gy);
    let rj = ResolventJumpChernoff2D::new(g2, 8).unwrap();
    let g = GridFn2D::from_fn(g2, |x: f64, y: f64| (-x * x - y * y).exp());
    let out = rj.jump(1.0_f64, &g).unwrap();
    assert_eq!(out.values.len(), 64);
    assert!(out.values.iter().all(|v| v.is_finite()));
}

/// Smoke: 3D jump returns correct shape and finite values.
#[test]
fn rj3d_jump_smoke() {
    let gx = Grid1D::new(-1.0, 1.0, 4).unwrap();
    let g3 = Grid3D::new(gx, gx, gx).unwrap();
    let rj = ResolventJumpChernoff3D::new(g3, 6).unwrap();
    let g = GridFn3D::from_fn(g3, |x: f64, y: f64, z: f64| (-x * x - y * y - z * z).exp());
    let out = rj.jump(1.0_f64, &g).unwrap();
    assert_eq!(out.values.len(), 64);
    assert!(out.values.iter().all(|v| v.is_finite()));
}
