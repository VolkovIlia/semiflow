//! f32 floor-saturation regime map for `Strang2D` / `Strang3D` slope gates.
//!
//! ADR-0046 (Precision-Policy Bands) Amendment 1; mirrors ADR-0163 / ADR-0120
//! (the f64 floor-recalibration precedents). **Diagnostic only** — these tests
//! assert nothing; they print a per-N self-convergence regime map so the
//! Architect can locate the f32 floor-safe band for the order-2 Strang gates.
//!
//! # Why a separate file?
//!
//! The gate file `generic_float_strang.rs` is at the 500-line suckless hard
//! limit. The diagnostic lives here, self-contained, at identical parameters
//! to the production gate (T=0.2, `n_steps`=200, a=0.1, [-5,5]ᵈ, `WrapDiff`
//! Catmull-Rom path), so it measures the SAME f32 error surface — any flattening
//! is unambiguously the f32 round-off floor (~1.2e-7/step → ~1e-5 accumulated),
//! never a code difference.
//!
//! # The problem (pre-existing, byte-identical at baseline 5069d85)
//!
//! The original `gf1_2d_f32` / `gf2_3d_f32` slope gates fit slope on N∈{64,128}
//! and gated at −1.80. Measured: −1.3459 (2D) / −1.4473 (3D). At N=32 `self_err`
//! is ALREADY ≈4e-5 — the f32 floor DOMINATES the dx² discretization error across
//! the whole {32,64,128} basket, so the measured slope is floor-noise decay, not
//! the true order-2. This regime map confirmed NO floor-safe band exists, so the
//! gates were re-stated as floor-band gates (ADR-0046 Amdt 1 Option B).
//!
//! # What this sweeps
//!
//! A WIDER, mostly-COARSER N range {8,12,16,24,32,48,64}. At coarse N, dx is
//! larger so the dx² discretization error is larger and rises ABOVE the ~1e-5
//! floor — the band where f32 can genuinely show ≈order-2. Watch the very
//! coarsest N for pre-asymptotic behaviour.
//!
//! Run:
//! ```text
//! cargo test -p semiflow-core --features parallel,simd,slow-tests --release \
//!   -- --ignored --nocapture f32_regime
//! ```
//!
//! Gated: `#[cfg(feature = "slow-tests")]`

#![cfg(feature = "slow-tests")]

use semiflow_core::{
    chernoff::{ApplyChernoffExt, ChernoffFunction, Growth},
    diffusion::DiffusionChernoff,
    error::SemiflowError,
    grid::Grid1D,
    grid2d::Grid2D,
    grid3d::Grid3D,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    grid_fn3d::GridFn3D,
    scratch::ScratchPool,
    strang2d::Strang2D,
    strang3d::Strang3D,
    SemiflowFloat,
};

// WrapDiff shim — identical to generic_float_strang.rs (apply_f → ChernoffFunction).
#[derive(Clone)]
struct WrapDiff<F: SemiflowFloat>(DiffusionChernoff<F>);

impl<F: SemiflowFloat> ChernoffFunction<F> for WrapDiff<F> {
    type S = GridFn1D<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &GridFn1D<F>,
        dst: &mut GridFn1D<F>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let result = self.0.apply_f(tau, src)?;
        dst.values.copy_from_slice(&result.values);
        Ok(())
    }

    fn order(&self) -> u32 {
        self.0.order_val()
    }

    fn growth(&self) -> Growth<F> {
        Growth::contraction()
    }
}

// Identical parameters to the production gate.
const N_STEPS: usize = 200;
const T_FINAL: f64 = 0.2;
const DIFFUSION_A: f64 = 0.1;
const X_MIN: f64 = -5.0;
const X_MAX: f64 = 5.0;

/// Wider, mostly-coarser sweep to find the f32 floor-safe band.
const N_SWEEP: [usize; 7] = [8, 12, 16, 24, 32, 48, 64];

fn a_f32(_: f32) -> f32 {
    DIFFUSION_A as f32
}
fn zero_f32(_: f32) -> f32 {
    0.0
}

fn mk_f32(g: Grid1D<f32>) -> WrapDiff<f32> {
    WrapDiff(DiffusionChernoff::<f32>::new(
        a_f32, zero_f32, zero_f32, DIFFUSION_A, g,
    ))
}

// --- 2D f32 runner + self-convergence (mirrors gate file) ---------------------

#[allow(clippy::cast_precision_loss)]
fn run_2d_f32(n: usize) -> GridFn2D<f32> {
    let tau = (T_FINAL / N_STEPS as f64) as f32;
    let (xmin, xmax) = (X_MIN as f32, X_MAX as f32);
    let gx = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D x f32");
    let gy = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D y f32");
    let g2 = Grid2D::<f32>::new(gx, gy);
    let s2 = Strang2D::<_, _, f32>::new(mk_f32(gx), mk_f32(gy));
    let mut u = GridFn2D::<f32>::from_fn_generic(g2, |x, y| (-x * x - y * y).exp());
    for _ in 0..N_STEPS {
        u = s2.apply_chernoff(tau, &u).expect("apply 2D f32 ok");
    }
    u
}

fn self_conv_2d_f32(n: usize) -> f64 {
    let u_coarse = run_2d_f32(n);
    let n2 = 2 * n - 1;
    let u_fine = run_2d_f32(n2);
    let mut sup = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            let vc = f64::from(u_coarse.values[j * n + i]);
            let vf = f64::from(u_fine.values[(j * 2) * n2 + (i * 2)]);
            sup = sup.max((vc - vf).abs());
        }
    }
    sup
}

// --- 3D f32 runner + self-convergence -----------------------------------------

#[allow(clippy::cast_precision_loss)]
fn run_3d_f32(n: usize) -> GridFn3D<f32> {
    let tau = (T_FINAL / N_STEPS as f64) as f32;
    let (xmin, xmax) = (X_MIN as f32, X_MAX as f32);
    let gx = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D x f32");
    let gy = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D y f32");
    let gz = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D z f32");
    let g3 = Grid3D::<f32>::new_generic(gx, gy, gz).expect("Grid3D f32");
    let s3 = Strang3D::<_, _, _, f32>::new(mk_f32(gx), mk_f32(gy), mk_f32(gz));
    let mut u = GridFn3D::<f32>::from_fn_generic(g3, |x, y, z| (-x * x - y * y - z * z).exp());
    for _ in 0..N_STEPS {
        u = s3.apply_chernoff(tau, &u).expect("apply 3D f32 ok");
    }
    u
}

fn self_conv_3d_f32(n: usize) -> f64 {
    let u_coarse = run_3d_f32(n);
    let n2 = 2 * n - 1;
    let u_fine = run_3d_f32(n2);
    let mut sup = 0.0_f64;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let vc = f64::from(u_coarse.values[k * n * n + j * n + i]);
                let vf = f64::from(u_fine.values[(k * 2) * n2 * n2 + (j * 2) * n2 + (i * 2)]);
                sup = sup.max((vc - vf).abs());
            }
        }
    }
    sup
}

// --- shared regime-map printer ------------------------------------------------

/// Print per-N self_err and consecutive log-log segment slopes for `errs`.
#[allow(clippy::cast_precision_loss)]
fn print_regime_map(tag: &str, errs: &[f64]) {
    println!("--- {tag} f32 regime map (T={T_FINAL}, n_steps={N_STEPS}, a={DIFFUSION_A}) ---");
    for (idx, &n) in N_SWEEP.iter().enumerate() {
        let dx = (X_MAX - X_MIN) / (n - 1) as f64;
        if idx == 0 {
            println!("{tag}: N={n:3}  dx={dx:.4}  self_err={:.4e}  seg=—", errs[idx]);
            continue;
        }
        let seg = (errs[idx].ln() - errs[idx - 1].ln())
            / ((n as f64).ln() - (N_SWEEP[idx - 1] as f64).ln());
        println!(
            "{tag}: N={n:3}  dx={dx:.4}  self_err={:.4e}  seg_slope={seg:.3}",
            errs[idx]
        );
    }
}

// ===========================================================================
// Diagnostics (non-asserting; #[ignore]d — zero CI cost)
// ===========================================================================

/// 2D f32 floor-saturation regime map over {8,12,16,24,32,48,64}.
#[test]
#[ignore = "f32 regime-map diagnostic — run with --ignored --nocapture"]
fn gf1_2d_f32_regime_map_diagnostic() {
    let errs: Vec<f64> = N_SWEEP.iter().map(|&n| self_conv_2d_f32(n)).collect();
    print_regime_map("GF1_2D", &errs);
}

/// 3D f32 floor-saturation regime map over {8,12,16,24,32,48,64}.
#[test]
#[ignore = "f32 regime-map diagnostic — run with --ignored --nocapture"]
fn gf2_3d_f32_regime_map_diagnostic() {
    let errs: Vec<f64> = N_SWEEP.iter().map(|&n| self_conv_3d_f32(n)).collect();
    print_regime_map("GF2_3D", &errs);
}
