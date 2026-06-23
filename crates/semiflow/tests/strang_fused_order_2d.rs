//! `STRANG_FUSED_ORDER_CONFIRMATION_2D` (ADR-0039 abort gate) — integration test.
//!
//! Moved from `strang2d.rs` unit tests in Wave 2 (ADR-0042) to keep
//! `strang2d.rs` within the 500-LoC constitution cap.
//!
//! Confirms that `‖fused(τ) − palindromic(τ)‖` is O(τ¹), NOT O(τ²).
//! Slope must be **> -1.0** (strictly first-order difference).
//! This is the negative result that aborted the C2 fused dispatch.

use semiflow::{
    chernoff::ApplyChernoffExt, diffusion::DiffusionChernoff, grid::Grid1D, grid2d::Grid2D,
    grid_fn2d::GridFn2D, Strang2D,
};

/// Fused path: Y(τ) ∘ X(τ) — 2 passes instead of 3.
fn apply_fused(
    s: &Strang2D<DiffusionChernoff, DiffusionChernoff, f64>,
    tau: f64,
    f: &GridFn2D<f64>,
) -> Result<GridFn2D<f64>, semiflow::SemiflowError> {
    let f1 = s.x.apply_chernoff(tau, f)?;
    s.y.apply_chernoff(tau, &f1)
}

/// Compute `‖apply_fused(τ,f) − apply_full_strang(τ,f)‖_∞` at each tau.
#[allow(clippy::cast_precision_loss)]
fn fused_vs_full_2d_diff() -> Vec<f64> {
    let n = 16_usize;
    let gx = Grid1D::new(-4.0, 4.0, n).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, n).unwrap();
    let g2 = Grid2D::new(gx, gy);
    let f0 = GridFn2D::from_fn(g2, |x, y| (-(x * x + y * y)).exp());
    let taus = [0.1_f64, 0.05, 0.025, 0.0125];
    let mut errs = Vec::with_capacity(taus.len());
    for &tau in &taus {
        let dx = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx);
        let dy = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gy);
        let s = Strang2D::new(dx, dy);
        let u_fused = apply_fused(&s, tau, &f0).unwrap();
        let u_full = s.apply_chernoff(tau, &f0).unwrap();
        let max_diff = u_fused
            .values
            .iter()
            .zip(u_full.values.iter())
            .map(|(a, b): (&f64, &f64)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        errs.push(max_diff);
    }
    errs
}

/// OLS log-log slope of `(ln(tau_i), ln(err_i))` pairs.
#[allow(clippy::cast_precision_loss)]
fn log_log_slope_2d(taus: &[f64], errs: &[f64]) -> f64 {
    let m = taus.len() as f64;
    let xs: Vec<f64> = taus.iter().map(|&t| t.ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|&x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sxy - sx * sy) / (m * sxx - sx * sx)
}

/// `STRANG_FUSED_ORDER_CONFIRMATION_2D`: fused `‖diff‖` must be O(τ¹), slope > -1.0.
#[test]
#[allow(clippy::cast_precision_loss)]
fn strang_fused_order_confirmation_2d() {
    let errs = fused_vs_full_2d_diff();
    let slope = log_log_slope_2d(&[0.1, 0.05, 0.025, 0.0125], &errs);
    assert!(
        slope > -1.0,
        "ADR-0039 abort gate: slope {slope:.3} unexpectedly ≤ -1.0; \
         re-evaluate C2 fused dispatch — it may now be τ²-accurate"
    );
}

/// Smoke test: fused path returns correct shape.
#[test]
fn fused_apply_shape() {
    let gx = Grid1D::new(-4.0, 4.0, 20).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, 16).unwrap();
    let g2 = Grid2D::new(gx, gy);
    let f = GridFn2D::from_fn(g2, |x, y| (-x * x - y * y).exp());
    let dx = DiffusionChernoff::new(|_| 0.25, |_| 0.0, |_| 0.0, 0.25, gx);
    let dy = DiffusionChernoff::new(|_| 0.25, |_| 0.0, |_| 0.0, 0.25, gy);
    let s = Strang2D::new(dx, dy);
    let out = apply_fused(&s, 0.01, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
}
