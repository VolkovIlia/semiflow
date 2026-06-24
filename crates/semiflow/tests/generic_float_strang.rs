//! Generic-over-Float convergence-rate gates for `Strang2D` and `Strang3D`.
//!
//! ADR-0045 §6.3; ADR-0046 (f32 precision band).
//!
//! # Gates defined here
//!
//! | Gate | Float | Operator | Gate kind |
//! |------|-------|----------|-----------|
//! | `GF1_2D_F64` | f64 | `Strang2D<DiffusionChernoff×2>` | slope ≤ −1.95 |
//! | `GF1_2D_F32` | f32 | `Strang2D<DiffusionChernoff×2>` | floor-band (Opt B) |
//! | `GF2_3D_F64` | f64 | `Strang3D<DiffusionChernoff×3>` | slope ≤ −1.95 |
//! | `GF2_3D_F32` | f32 | `Strang3D<DiffusionChernoff×3>` | floor-band (Opt B) |
//!
//! All four tests use **spatial self-convergence** (probe n vs fine 2n−1),
//! which eliminates the need for a high-resolution oracle reference and is
//! insensitive to temporal error (`N_STEPS` is large relative to the spatial
//! error at the finest probe).
//!
//! # f32 floor-band gate (ADR-0046 Amendment 1, ACCEPTED — Option B)
//!
//! The diagnostic regime map (`generic_float_strang_f32_regime.rs`) proved the
//! f32 round-off floor (≈1e-5..4e-5) DOMINATES the dx² error across the whole
//! sane basket: `self_err` is non-monotone, per-segment slopes swing +3.4..−7.0
//! (NOISE, not a convergence law).  No band shows ≈order-2, so a `−1.80` slope
//! was physically unattainable (the latent bug).  Option B instead asserts what
//! f32 CAN guarantee (see [`assert_f32_floor_band`]): finest `self_err` below a
//! documented accuracy CEILING AND refinement still helps overall.  The G3⁶
//! (order-6) gate stays DISABLED on f32 per the parent ADR.
//!
//! # Probe set (N ∈ {32, 64, 128})
//!
//! f64 gates fit slope on `{64, 128}` (skip pre-asymptotic n=32, mirroring
//! `G4_NS2D_aniso`); f32 gates use all three probes for the floor band.
//!
//! # Why concrete types (not generic F)?
//!
//! `Strang2D::apply` under `feature = "parallel"` requires a `pub(crate)` sealed
//! `ParallelPool2D` trait test code cannot name.  Concrete `f64`/`f32` let the
//! compiler monomorphise the right impl.  After Phase 5a, `DiffusionChernoff<f32>`
//! implements `ChernoffFunction<f32>` natively; the `WrapDiff<F>` shim is retired.
//!
//! Gated: `#[cfg(feature = "slow-tests")]`

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_possible_truncation)] // f64→f32 casts are intentional (f32 precision test)

use semiflow::{
    chernoff::ApplyChernoffExt, diffusion::DiffusionChernoff, grid::Grid1D, grid2d::Grid2D,
    grid3d::Grid3D, grid_fn2d::GridFn2D, grid_fn3d::GridFn3D, strang2d::Strang2D,
    strang3d::Strang3D,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Probe spatial node counts; n=32 is diagnostic only; slope on {64, 128}.
const N_SPATIAL: [usize; 3] = [32, 64, 128];

const N_STEPS: usize = 200;
const T_FINAL: f64 = 0.2;
const DIFFUSION_A: f64 = 0.1;
const X_MIN: f64 = -5.0;
const X_MAX: f64 = 5.0;

/// Slope gate for f64 (second-order Chernoff, ADR-0045 §6.3).
const SLOPE_GATE_F64: f64 = -1.95;

/// f32 accuracy CEILING (ADR-0046 Amdt 1, Option B). Observed finest-grid
/// `self_err` ≈ 1.2e-5..1.5e-5; ~30× margin below 5e-4, so a 10× regression
/// (≈1.5e-4) trips it yet benign round-off jitter never does.
const F32_CEILING: f64 = 5.0e-4;

/// f32 floor-of-floors (ADR-0046 Amdt 1, Option B). `self_err` below this means
/// self-convergence collapsed to ~0 (broken probe), not a real solve — the lower
/// wall keeps the gate from being vacuously satisfied by a degenerate zero.
const F32_FLOOR: f64 = 1.0e-7;

// ---------------------------------------------------------------------------
// f64 function pointers (required by DiffusionChernoff::new)
// ---------------------------------------------------------------------------

fn a_f64(_: f64) -> f64 {
    DIFFUSION_A
}
fn zero_f64(_: f64) -> f64 {
    0.0
}

// f32 function pointers
fn a_f32(_: f32) -> f32 {
    DIFFUSION_A as f32
}
fn zero_f32(_: f32) -> f32 {
    0.0
}

// ---------------------------------------------------------------------------
// OLS log-log slope (same as every other slope test in this repo)
// ---------------------------------------------------------------------------

// n ≤ 255 — within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn loglog_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let log_n: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let log_e: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
    let mean_x = log_n.iter().sum::<f64>() / m;
    let mean_y = log_e.iter().sum::<f64>() / m;
    let num: f64 = log_n
        .iter()
        .zip(log_e.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_n.iter().map(|x| (x - mean_x).powi(2)).sum();
    num / den
}

// f32 floor-band gate (ADR-0046 Amdt 1, Option B). f32 is round-off-limited here
// (no asymptotic band → slope unattainable), so assert the regression-sensitive
// invariants f32 CAN guarantee: every probe inside `[floor, ceiling]` (no zero
// tautology, no blow-up — a 10× error regression trips the ceiling) AND finest <
// coarsest (refinement still helps). `errs` is coarse→fine (matches `N_SPATIAL`).
fn assert_f32_floor_band(label: &str, errs: &[f64]) {
    let coarsest = errs[0];
    let finest = *errs.last().expect("non-empty error vec");
    for &e in errs {
        assert!(
            (F32_FLOOR..=F32_CEILING).contains(&e),
            "{label}: self_err {e:.4e} outside f32 floor band [{F32_FLOOR:.0e}, \
             {F32_CEILING:.0e}] — regression or floor lift (ADR-0046 Amdt 1)"
        );
    }
    assert!(
        finest < coarsest,
        "{label}: refinement did not help — finest {finest:.4e} >= coarsest \
         {coarsest:.4e} (ADR-0046 Amdt 1, Option B)"
    );
}

// ===========================================================================
// 2D runners
// ===========================================================================

/// Evolve `Strang2D<DiffusionChernoff<f64>×2>` on an `n × n` grid (n ≤ 255) for `N_STEPS`.
#[allow(clippy::cast_precision_loss)]
fn run_2d_f64(n: usize) -> GridFn2D<f64> {
    let tau = T_FINAL / N_STEPS as f64;
    let gx = Grid1D::new(X_MIN, X_MAX, n).expect("Grid1D x f64");
    let gy = Grid1D::new(X_MIN, X_MAX, n).expect("Grid1D y f64");
    let g2 = Grid2D::<f64>::new(gx, gy);
    let dx = DiffusionChernoff::<f64>::new(a_f64, zero_f64, zero_f64, DIFFUSION_A, gx);
    let dy = DiffusionChernoff::<f64>::new(a_f64, zero_f64, zero_f64, DIFFUSION_A, gy);
    let s2 = Strang2D::<_, _, f64>::new(dx, dy);
    let mut u = GridFn2D::<f64>::from_fn(g2, |x, y| (-x * x - y * y).exp());
    for _ in 0..N_STEPS {
        u = s2.apply_chernoff(tau, &u).expect("apply 2D f64 ok");
    }
    u
}

/// f64 2D spatial self-convergence error for probe size `n` (vs `2n−1`).
fn self_conv_2d_f64(n: usize) -> f64 {
    let u_coarse = run_2d_f64(n);
    let n2 = 2 * n - 1;
    let u_fine = run_2d_f64(n2);
    let mut sup = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            let err = (u_coarse.values[j * n + i] - u_fine.values[(j * 2) * n2 + (i * 2)]).abs();
            if err > sup {
                sup = err;
            }
        }
    }
    sup
}

/// Evolve `Strang2D<DiffusionChernoff<f32>×2, f32>` on an `n × n` grid (n ≤ 255).
#[allow(clippy::cast_precision_loss)]
fn run_2d_f32(n: usize) -> GridFn2D<f32> {
    let tau = (T_FINAL / N_STEPS as f64) as f32;
    let xmin = X_MIN as f32;
    let xmax = X_MAX as f32;
    let gx = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D x f32");
    let gy = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D y f32");
    let g2 = Grid2D::<f32>::new(gx, gy);
    let dx = DiffusionChernoff::<f32>::new(a_f32, zero_f32, zero_f32, DIFFUSION_A, gx);
    let dy = DiffusionChernoff::<f32>::new(a_f32, zero_f32, zero_f32, DIFFUSION_A, gy);
    let s2 = Strang2D::<_, _, f32>::new(dx, dy);
    let mut u = GridFn2D::<f32>::from_fn_generic(g2, |x, y| (-x * x - y * y).exp());
    for _ in 0..N_STEPS {
        u = s2.apply_chernoff(tau, &u).expect("apply 2D f32 ok");
    }
    u
}

/// f32 2D spatial self-convergence error for probe size `n` (vs `2n−1`).
fn self_conv_2d_f32(n: usize) -> f64 {
    let u_coarse = run_2d_f32(n);
    let n2 = 2 * n - 1;
    let u_fine = run_2d_f32(n2);
    let mut sup = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            let vc = f64::from(u_coarse.values[j * n + i]);
            let vf = f64::from(u_fine.values[(j * 2) * n2 + (i * 2)]);
            let err = (vc - vf).abs();
            if err > sup {
                sup = err;
            }
        }
    }
    sup
}

// ===========================================================================
// 3D runners
// ===========================================================================

/// Evolve `Strang3D<DiffusionChernoff<f64>×3>` on an `n × n × n` grid for `N_STEPS` steps.
#[allow(clippy::cast_precision_loss)]
fn run_3d_f64(n: usize) -> GridFn3D<f64> {
    let tau = T_FINAL / N_STEPS as f64;
    let gx = Grid1D::new(X_MIN, X_MAX, n).expect("Grid1D x f64");
    let gy = Grid1D::new(X_MIN, X_MAX, n).expect("Grid1D y f64");
    let gz = Grid1D::new(X_MIN, X_MAX, n).expect("Grid1D z f64");
    let g3 = Grid3D::<f64>::new(gx, gy, gz).expect("Grid3D f64");
    let dx = DiffusionChernoff::<f64>::new(a_f64, zero_f64, zero_f64, DIFFUSION_A, gx);
    let dy = DiffusionChernoff::<f64>::new(a_f64, zero_f64, zero_f64, DIFFUSION_A, gy);
    let dz = DiffusionChernoff::<f64>::new(a_f64, zero_f64, zero_f64, DIFFUSION_A, gz);
    let s3 = Strang3D::<_, _, _, f64>::new(dx, dy, dz);
    let mut u = GridFn3D::<f64>::from_fn(g3, |x, y, z| (-x * x - y * y - z * z).exp());
    for _ in 0..N_STEPS {
        u = s3.apply_chernoff(tau, &u).expect("apply 3D f64 ok");
    }
    u
}

/// f64 3D spatial self-convergence error for probe size `n` (vs `2n−1`).
fn self_conv_3d_f64(n: usize) -> f64 {
    let u_coarse = run_3d_f64(n);
    let n2 = 2 * n - 1;
    let u_fine = run_3d_f64(n2);
    let mut sup = 0.0_f64;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let vc = u_coarse.values[k * n * n + j * n + i];
                let vf = u_fine.values[(k * 2) * n2 * n2 + (j * 2) * n2 + (i * 2)];
                let err = (vc - vf).abs();
                if err > sup {
                    sup = err;
                }
            }
        }
    }
    sup
}

/// Evolve `Strang3D<DiffusionChernoff<f32>×3, f32>` on an `n × n × n` grid (n ≤ 255).
#[allow(clippy::cast_precision_loss)]
fn run_3d_f32(n: usize) -> GridFn3D<f32> {
    let tau = (T_FINAL / N_STEPS as f64) as f32;
    let xmin = X_MIN as f32;
    let xmax = X_MAX as f32;
    let gx = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D x f32");
    let gy = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D y f32");
    let gz = Grid1D::<f32>::new_generic(xmin, xmax, n).expect("Grid1D z f32");
    let g3 = Grid3D::<f32>::new_generic(gx, gy, gz).expect("Grid3D f32");
    let dx = DiffusionChernoff::<f32>::new(a_f32, zero_f32, zero_f32, DIFFUSION_A, gx);
    let dy = DiffusionChernoff::<f32>::new(a_f32, zero_f32, zero_f32, DIFFUSION_A, gy);
    let dz = DiffusionChernoff::<f32>::new(a_f32, zero_f32, zero_f32, DIFFUSION_A, gz);
    let s3 = Strang3D::<_, _, _, f32>::new(dx, dy, dz);
    let mut u = GridFn3D::<f32>::from_fn_generic(g3, |x, y, z| (-x * x - y * y - z * z).exp());
    for _ in 0..N_STEPS {
        u = s3.apply_chernoff(tau, &u).expect("apply 3D f32 ok");
    }
    u
}

/// f32 3D spatial self-convergence error for probe size `n` (vs `2n−1`).
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
                let err = (vc - vf).abs();
                if err > sup {
                    sup = err;
                }
            }
        }
    }
    sup
}

// ===========================================================================
// GF1_2D_F64 — f64 Strang2D slope gate
// ===========================================================================

/// `GF1_2D_F64`: spatial slope ≤ −1.95 for `Strang2D<f64>`. Self-convergence on
/// `N ∈ {32, 64, 128}`, slope on `{64, 128}` (ADR-0045 §6.3).
#[test]
#[allow(clippy::cast_precision_loss)]
fn gf1_2d_f64_slope() {
    let errs: Vec<f64> = N_SPATIAL.iter().map(|&n| self_conv_2d_f64(n)).collect();

    for (&n, &e) in N_SPATIAL.iter().zip(&errs) {
        let dx = (X_MAX - X_MIN) / (n - 1) as f64;
        println!("GF1_2D_F64: N={n:3}, dx={dx:.4}, self_err={e:.4e}");
    }

    // Skip n=32 (pre-asymptotic), slope on {64, 128}.
    let slope = loglog_slope(&N_SPATIAL[1..], &errs[1..]);
    println!("GF1_2D_F64: slope = {slope:.4}  (gate <= {SLOPE_GATE_F64})");

    assert!(
        slope <= SLOPE_GATE_F64,
        "GF1_2D_F64 slope {slope:.4} > gate {SLOPE_GATE_F64} — escalate to Architect"
    );
}

// ===========================================================================
// GF1_2D_F32 — f32 Strang2D floor-band gate (ADR-0046 Amendment 1, Option B)
// ===========================================================================

/// `GF1_2D_F32`: f32 round-off floor-band gate for `Strang2D` with `F = f32`.
///
/// f32 is round-off-limited here (no asymptotic band — see module doc +
/// `generic_float_strang_f32_regime.rs`), so the unattainable `−1.80` slope is
/// REPLACED by [`assert_f32_floor_band`].  Self-convergence on `N ∈ {32, 64, 128}`.
#[test]
#[allow(clippy::cast_precision_loss)]
fn gf1_2d_f32_floor_band() {
    let errs: Vec<f64> = N_SPATIAL.iter().map(|&n| self_conv_2d_f32(n)).collect();

    for (&n, &e) in N_SPATIAL.iter().zip(&errs) {
        let dx = (X_MAX - X_MIN) / (n - 1) as f64;
        println!("GF1_2D_F32: N={n:3}, dx={dx:.4}, self_err={e:.4e}");
    }
    println!("GF1_2D_F32: floor band [{F32_FLOOR:.0e}, {F32_CEILING:.0e}]");

    assert_f32_floor_band("GF1_2D_F32", &errs);
}

// ===========================================================================
// GF2_3D_F64 — f64 Strang3D slope gate
// ===========================================================================

/// `GF2_3D_F64`: spatial slope ≤ −1.95 for `Strang3D<f64>`. Self-convergence on
/// `N ∈ {32, 64, 128}`, slope on `{64, 128}`; complements oracle-based `G5_3D`.
#[test]
#[allow(clippy::cast_precision_loss)]
fn gf2_3d_f64_slope() {
    let errs: Vec<f64> = N_SPATIAL.iter().map(|&n| self_conv_3d_f64(n)).collect();

    for (&n, &e) in N_SPATIAL.iter().zip(&errs) {
        let dx = (X_MAX - X_MIN) / (n - 1) as f64;
        println!("GF2_3D_F64: N={n:3}, dx={dx:.4}, self_err={e:.4e}");
    }

    let slope = loglog_slope(&N_SPATIAL[1..], &errs[1..]);
    println!("GF2_3D_F64: slope = {slope:.4}  (gate <= {SLOPE_GATE_F64})");

    assert!(
        slope <= SLOPE_GATE_F64,
        "GF2_3D_F64 slope {slope:.4} > gate {SLOPE_GATE_F64} — escalate to Architect"
    );
}

// ===========================================================================
// GF2_3D_F32 — f32 Strang3D floor-band gate (ADR-0046 Amendment 1, Option B)
// ===========================================================================

/// `GF2_3D_F32`: f32 round-off floor-band gate for `Strang3D` with `F = f32`.
///
/// Same floor-limited regime as [`gf1_2d_f32_floor_band`]; [`assert_f32_floor_band`]
/// replaces the unattainable `−1.80` slope.  N=128 is the finest probe (255³ ≈
/// 16M f32 values × 2 grids bounds memory).  Self-convergence on `N ∈ {32, 64, 128}`.
#[test]
#[allow(clippy::cast_precision_loss)]
fn gf2_3d_f32_floor_band() {
    let errs: Vec<f64> = N_SPATIAL.iter().map(|&n| self_conv_3d_f32(n)).collect();

    for (&n, &e) in N_SPATIAL.iter().zip(&errs) {
        let dx = (X_MAX - X_MIN) / (n - 1) as f64;
        println!("GF2_3D_F32: N={n:3}, dx={dx:.4}, self_err={e:.4e}");
    }
    println!("GF2_3D_F32: floor band [{F32_FLOOR:.0e}, {F32_CEILING:.0e}]");

    assert_f32_floor_band("GF2_3D_F32", &errs);
}
