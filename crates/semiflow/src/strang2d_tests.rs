// Unit tests for [`Strang2D`] (extracted per suckless ≤500-line cap).
use super::*;
use crate::{diffusion::DiffusionChernoff, grid::Grid1D, grid2d::Grid2D, state::State};

fn make_strang() -> (
    Strang2D<DiffusionChernoff, DiffusionChernoff, f64>,
    GridFn2D<f64>,
) {
    let gx = Grid1D::new(-4.0, 4.0, 20).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, 16).unwrap();
    let g2 = Grid2D::new(gx, gy);
    let f = GridFn2D::from_fn(g2, |x, y| (-x * x - y * y).exp());
    let dx = DiffusionChernoff::new(|_| 0.25, |_| 0.0, |_| 0.0, 0.25, gx);
    let dy = DiffusionChernoff::new(|_| 0.25, |_| 0.0, |_| 0.0, 0.25, gy);
    let s = Strang2D::new(dx, dy);
    (s, f)
}

#[test]
fn order_is_2_for_order2_inner() {
    let (s, _) = make_strang();
    assert_eq!(s.order(), 2);
}

// reason: d4_xaxis/d4_yaxis differ only by the axis identifier, which is
// precisely the information needed; renaming to unrelated names obscures
// the 2D splitting intent.
#[allow(clippy::similar_names)]
#[test]
fn order_is_2_for_4th_order_spatial_inner() {
    use crate::diffusion4::Diffusion4thChernoff;
    let gx = Grid1D::new(-4.0, 4.0, 20).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, 16).unwrap();
    let d4_xaxis = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let d4_yaxis = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
    let s4 = Strang2D::new(d4_xaxis, d4_yaxis);
    assert_eq!(s4.order(), 2);
}

#[test]
fn growth_unit_bounded() {
    let (s, _) = make_strang();
    let g = s.growth();
    let (m, omega) = (g.multiplier, g.omega);
    assert!((m - 1.0).abs() < 1e-14, "m={m}");
    assert!(omega.abs() < 1e-14, "omega={omega}");
}

#[test]
fn apply_preserves_shape() {
    use crate::chernoff::ApplyChernoffExt;
    let (s, f) = make_strang();
    let out = s.apply_chernoff(0.01, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
    assert_eq!(out.grid, f.grid);
}

#[test]
fn apply_tau_zero_is_near_identity() {
    use crate::chernoff::ApplyChernoffExt;
    let (s, f) = make_strang();
    let out = s.apply_chernoff(0.0, &f).unwrap();
    let mut diff = out.clone();
    diff.axpy(-1.0, &f);
    assert!(
        diff.norm_sup() < 1e-10,
        "tau=0 deviation = {}",
        diff.norm_sup()
    );
}

// Fused-order confirmation tests moved to integration test
// `tests/strang_fused_order_2d.rs` in Wave 2 (ADR-0042) to keep this
// file within the 500-LoC constitution cap.
