// Unit tests for `Strang3D` and `AxisLift3D`.
// Included into `strang3d.rs` via `include!` inside `#[cfg(test)] mod tests`.
// Moved from `strang3d.rs` (batch H4 suckless split).

use super::*;
use crate::{diffusion::DiffusionChernoff, grid::Grid1D, grid3d::Grid3D, state::State};

fn make_strang() -> (
    Strang3D<DiffusionChernoff, DiffusionChernoff, DiffusionChernoff, f64>,
    GridFn3D<f64>,
) {
    let gx = Grid1D::new(-4.0, 4.0, 12).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, 10).unwrap();
    let gz = Grid1D::new(-4.0, 4.0, 8).unwrap();
    let g3 = Grid3D::new(gx, gy, gz).unwrap();
    let f = GridFn3D::from_fn(g3, |x, y, z| (-x * x - y * y - z * z).exp());
    let dx = DiffusionChernoff::new(|_| 0.25, |_| 0.0, |_| 0.0, 0.25, gx);
    let dy = DiffusionChernoff::new(|_| 0.25, |_| 0.0, |_| 0.0, 0.25, gy);
    let dz = DiffusionChernoff::new(|_| 0.25, |_| 0.0, |_| 0.0, 0.25, gz);
    let s = Strang3D::new(dx, dy, dz);
    (s, f)
}

#[test]
fn order_is_min_per_axis() {
    let (s, _) = make_strang();
    assert_eq!(s.order(), 2);
}

#[test]
fn palindromic_apply_doesnt_panic() {
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

#[test]
fn lift_x_walks_strided_1d() {
    use crate::chernoff::ApplyChernoffExt;
    let gx = Grid1D::new(-3.0, 3.0, 12).unwrap();
    let gy = Grid1D::new(-3.0, 3.0, 8).unwrap();
    let gz = Grid1D::new(-3.0, 3.0, 6).unwrap();
    let g3 = Grid3D::new(gx, gy, gz).unwrap();
    let f = GridFn3D::from_fn(g3, |x, _y, _z| (-x * x).exp());
    let dx = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx);
    let lift = AxisLift3D::new(dx, Axis::X);
    let out = lift.apply_chernoff(0.01_f64, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
}

#[test]
fn lift_y_strided() {
    use crate::chernoff::ApplyChernoffExt;
    let gx = Grid1D::new(-3.0, 3.0, 8).unwrap();
    let gy = Grid1D::new(-3.0, 3.0, 10).unwrap();
    let gz = Grid1D::new(-3.0, 3.0, 6).unwrap();
    let g3 = Grid3D::new(gx, gy, gz).unwrap();
    let f = GridFn3D::from_fn(g3, |_x, y, _z| (-y * y).exp());
    let dy = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gy);
    let lift = AxisLift3D::new(dy, Axis::Y);
    let out = lift.apply_chernoff(0.01_f64, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
}

#[test]
fn lift_z_strided() {
    use crate::chernoff::ApplyChernoffExt;
    let gx = Grid1D::new(-3.0, 3.0, 8).unwrap();
    let gy = Grid1D::new(-3.0, 3.0, 8).unwrap();
    let gz = Grid1D::new(-3.0, 3.0, 10).unwrap();
    let g3 = Grid3D::new(gx, gy, gz).unwrap();
    let f = GridFn3D::from_fn(g3, |_x, _y, z| (-z * z).exp());
    let dz = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gz);
    let lift = AxisLift3D::new(dz, Axis::Z);
    let out = lift.apply_chernoff(0.01_f64, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
}

#[test]
fn fused_apply_shape_3d() {
    let (s_small, f_small) = make_strang();
    let out = s_small.apply_fused(0.01, &f_small).unwrap();
    assert_eq!(out.values.len(), f_small.values.len());
    let out_zero = s_small.apply_fused(0.0, &f_small).unwrap();
    let mut diff = out_zero.clone();
    diff.axpy(-1.0, &f_small);
    assert!(
        diff.norm_sup() < 1e-10,
        "fused tau=0 deviation = {}",
        diff.norm_sup()
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn strang_fused_order_confirmation_3d() {
    let errs = fused_vs_full_3d_diff();
    let slope = log_log_slope_3d(&[0.1, 0.05, 0.025, 0.0125], &errs);
    assert!(
        slope > -1.0,
        "ADR-0039 abort gate: slope {slope:.3} unexpectedly ≤ -1.0"
    );
}

#[allow(clippy::cast_precision_loss)]
fn fused_vs_full_3d_diff() -> Vec<f64> {
    let n = 8_usize;
    let g = Grid1D::new(-4.0, 4.0, n).unwrap();
    let grid3 = Grid3D::new(g, g, g).unwrap();
    let f0 = GridFn3D::from_fn(grid3, |x, y, z| (-(x * x + y * y + z * z)).exp());
    let taus = [0.1_f64, 0.05, 0.025, 0.0125];
    let mut errs = Vec::with_capacity(taus.len());
    for &tau in &taus {
        let d = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, g);
        let s = Strang3D::new(d.clone(), d.clone(), d.clone());
        let u_fused = s.apply_fused(tau, &f0).unwrap();
        let u_full = apply_strang3d_full(tau, &f0, &s.x.inner, &s.y.inner, &s.z.inner).unwrap();
        let max_diff = u_fused
            .values
            .iter()
            .zip(u_full.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        errs.push(max_diff);
    }
    errs
}

#[allow(clippy::cast_precision_loss)]
fn log_log_slope_3d(taus: &[f64], errs: &[f64]) -> f64 {
    let m = taus.len() as f64;
    let xs: Vec<f64> = taus.iter().map(|&t| t.ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|&x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sxy - sx * sy) / (m * sxx - sx * sx)
}
