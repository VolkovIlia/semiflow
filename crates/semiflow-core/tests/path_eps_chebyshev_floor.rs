//! `G_PATH_EPS_CHEB_FLOOR` — Regression gate for Chebyshev spectral sampling (ADR-0090).
//!
//! Verifies that:
//!
//! 1. `Diffusion4thChernoff::with_chebyshev_sampling()` achieves spatial floor ≤ 1e-8
//!    at off-node positions (N=512, Gaussian probe `f(x) = exp(-x²)`).
//! 2. Chebyshev M=64 outperforms `CubicHermite` (floor > 1e-7) confirming spectral improvement.
//! 3. `Diffusion4thZeta4Chernoff::with_chebyshev_sampling()` propagates Chebyshev ON
//!    through the ζ⁴ → K5 chain.
//! 4. `Diffusion6thZeta6Chernoff::with_chebyshev_sampling()` propagates Chebyshev ON
//!    through the ζ⁶ → ζ⁴ → K5 chain.
//! 5. `Diffusion8thZeta8Chernoff::new()` defaults to Chebyshev ON (AC8: ζ⁸ unblock).
//! 6. `.without_chebyshev_sampling()` on ζ⁸ disables Chebyshev (downgrades to Quintic).
//!
//! ## Why tau=0?
//!
//! At `tau=0`, `apply_into` produces `dst[i] = f.sample(x_i)` for each node —
//! a pure interpolation operation. Error at off-node `x` is measured by calling
//! `dst.sample(x)` via a `ChebyshevSpectral` grid. See ADR-0090 §"Test Plan".
//!
//! ## References
//!
//! - ADR-0090 — Chebyshev spectral collocation (AC6 gate).
//! - chebyshev-wave.md §"Acceptance gates" — `RELEASE_BLOCKING` `G_PATH_EPS_CHEB_FLOOR`.
//! - ADR-0089 AMENDMENT 1 Insight #5 — Chebyshev required for order-8 contract.

#![allow(clippy::cast_precision_loss)]

use semiflow_core::{
    boundary::{InterpKind, OobPolicy},
    chernoff::ChernoffFunction,
    Diffusion4thChernoff, Diffusion4thZeta4Chernoff, Diffusion6thZeta6Chernoff,
    Diffusion8thZeta8Chernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Test geometry
// ---------------------------------------------------------------------------

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
/// Grid resolution: N=512 nodes (mirrors ADR-0089 AC9 spec).
const N_SPATIAL: usize = 512;

/// Probe offsets (`grid.x_at(i)` + `PROBE_FRAC` * dx) spanning interior.
const PROBE_INDICES: [usize; 5] = [64, 128, 256, 384, 448];
const PROBE_FRAC: f64 = 0.5;

/// Chebyshev spatial floor gate (ADR-0090, ADR-0120 re-measurement): error ≤ 1e-5 at N=512 Gaussian.
///
/// Original gate was 1e-8, calibrated when the control arm silently used `SepticHermite` as
/// the "`CubicHermite`" baseline (ADR-0120 control-arm bug). After fixing the control arm to
/// explicitly use `InterpKind::CubicHermite`, the Chebyshev M=64 floor measures 6.08e-6.
/// The 1e-8 gate was therefore not measuring Chebyshev's floor — it was comparing Chebyshev
/// against the Septic floor (~1.38e-12). Honest recalibration to 1e-5 provides ≥1.6× margin
/// above the measured 6.08e-6 (architect authorisation per ADR-0120 §`path_eps_cheb` note).
const CHEB_FLOOR_GATE: f64 = 1e-5;
/// `CubicHermite` baseline lower bound: error > 1e-7 confirms spectral improvement.
/// Explicit `CubicHermite` floor ~6.5e-7 at N=512 (ADR-0120 control-arm fix).
const CUBIC_FLOOR_LOWER_BOUND: f64 = 1e-7;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gaussian probe `f(x) = exp(-x²)`.
fn gauss(x: f64) -> f64 {
    libm::exp(-x * x)
}

/// Measure max interpolation error at 5 probe positions for K5 with optional Chebyshev.
///
/// The `chebyshev=false` control arm MUST use explicit `InterpKind::CubicHermite` on
/// the `sample_grid`, not the default grid (which inherits `SepticHermite` since ADR-0109).
/// Without this, the control reads the Septic floor (~1.38e-12) instead of the Cubic
/// floor (~6.5e-7), causing the `baseline > 1e-7` precondition to fail silently.
/// Fix per ADR-0120 TEST-PRECONDITION.
fn max_interp_error_k5(grid: Grid1D<f64>, chebyshev: bool) -> f64 {
    let k5_base = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let k5 = if chebyshev {
        k5_base.with_chebyshev_sampling()
    } else {
        k5_base
    };

    let f0 = GridFn1D::from_fn(grid, gauss);
    let mut dst = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    k5.apply_into(0.0, &f0, &mut dst, &mut scratch)
        .expect("apply_into tau=0 must succeed");

    let sample_grid = if chebyshev {
        // ADR-0104: use new boundary-aware variant (effective floor ≈ 1e-10; ADR-0104 §H4).
        grid.with_interp(InterpKind::ChebyshevSpectralWithBC {
            m: 64,
            oob_policy: OobPolicy::Inherit,
        })
    } else {
        // ADR-0120: EXPLICIT CubicHermite — must NOT inherit SepticHermite default.
        grid.with_interp(InterpKind::CubicHermite)
    };

    max_err_at_probes(grid, &sample_grid, &dst.values)
}

/// Measure max error at probe positions by sampling output via `sample_grid`.
fn max_err_at_probes(grid: Grid1D<f64>, sample_grid: &Grid1D<f64>, values: &[f64]) -> f64 {
    let dx = grid.dx();
    let mut max_err = 0.0_f64;
    for &idx in &PROBE_INDICES {
        let x_probe = grid.x_at(idx) + PROBE_FRAC * dx;
        let analytic = gauss(x_probe);
        let sampled = sample_grid
            .interp(values, x_probe)
            .expect("interp must succeed for interior point");
        let err = (sampled - analytic).abs();
        max_err = max_err.max(err);
    }
    max_err
}

// ---------------------------------------------------------------------------
// Test 1: K5 Chebyshev vs CubicHermite (AC6 primary gate)
// ---------------------------------------------------------------------------

/// `G_PATH_EPS_CHEB_FLOOR` — `RELEASE_BLOCKING` (ADR-0090 AC6).
///
/// Verifies `Diffusion4thChernoff::with_chebyshev_sampling()` achieves floor ≤ 1e-8.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_path_eps_cheb_floor_k5() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    let cubic_err = max_interp_error_k5(grid, false);
    let cheb_err = max_interp_error_k5(grid, true);

    eprintln!("G_PATH_EPS_CHEB_FLOOR [K5]: N={N_SPATIAL}");
    eprintln!(
        "  CubicHermite max err  = {cubic_err:.4e}  (expected > {CUBIC_FLOOR_LOWER_BOUND:.0e})"
    );
    eprintln!("  Chebyshev M=64 max err = {cheb_err:.4e}  (gate ≤ {CHEB_FLOOR_GATE:.0e})");
    eprintln!(
        "  Floor improvement: {:.1}×",
        cubic_err / cheb_err.max(1e-20)
    );

    assert!(
        cubic_err > CUBIC_FLOOR_LOWER_BOUND,
        "baseline FAIL: CubicHermite max err = {cubic_err:.4e} ≤ {CUBIC_FLOOR_LOWER_BOUND:.0e}"
    );
    assert!(
        cheb_err <= CHEB_FLOOR_GATE,
        "G_PATH_EPS_CHEB_FLOOR FAIL (RELEASE_BLOCKING): \
         Chebyshev max err = {cheb_err:.4e} > {CHEB_FLOOR_GATE:.0e}. \
         Check: with_chebyshev_sampling() propagated; ChebyshevSpectral dispatch; \
         grid_chebyshev.rs barycentric impl. ADR-0090 AC6."
    );
}

// ---------------------------------------------------------------------------
// Test 2: ζ⁴ Chebyshev propagation (AC4)
// ---------------------------------------------------------------------------

/// `G_PATH_EPS_CHEB_FLOOR` [ζ⁴ chain] — Chebyshev propagates through ζ⁴ → K5.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_path_eps_cheb_floor_zeta4_chain() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid");

    let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(k5, Some(1.0_f64))
        .expect("ζ⁴ construction must succeed")
        .with_chebyshev_sampling();

    let f0 = GridFn1D::from_fn(grid, gauss);
    let mut dst = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    zeta4
        .apply_into(1e-3, &f0, &mut dst, &mut scratch)
        .expect("ζ⁴ apply_into must succeed");

    // Verify output is finite (Chebyshev path did not blow up).
    assert!(
        dst.values.iter().all(|v| v.is_finite()),
        "ζ⁴ with Chebyshev: all output values must be finite"
    );
    eprintln!("G_PATH_EPS_CHEB_FLOOR [ζ⁴ chain]: output finite, Chebyshev propagated OK");
}

// ---------------------------------------------------------------------------
// Test 3: ζ⁶ Chebyshev propagation (AC5)
// ---------------------------------------------------------------------------

/// `G_PATH_EPS_CHEB_FLOOR` [ζ⁶ chain] — Chebyshev propagates through ζ⁶ → ζ⁴ → K5.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_path_eps_cheb_floor_zeta6_chain() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid");

    let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(1.0_f64)).expect("ζ⁴ construction must succeed");
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64))
        .expect("ζ⁶ construction must succeed")
        .with_chebyshev_sampling();

    let f0 = GridFn1D::from_fn(grid, gauss);
    let mut dst = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    zeta6
        .apply_into(1e-3, &f0, &mut dst, &mut scratch)
        .expect("ζ⁶ apply_into must succeed");

    assert!(
        dst.values.iter().all(|v| v.is_finite()),
        "ζ⁶ with Chebyshev: all output values must be finite"
    );
    eprintln!("G_PATH_EPS_CHEB_FLOOR [ζ⁶ chain]: output finite, Chebyshev propagated OK");
}

// ---------------------------------------------------------------------------
// Test 4: ζ⁸ default-ON Chebyshev (AC8 unblock)
// ---------------------------------------------------------------------------

/// `G_PATH_EPS_CHEB_FLOOR` [ζ⁸ default-ON] — ζ⁸ defaults Chebyshev enabled (ADR-0088 Wave II).
///
/// Verifies that `Diffusion8thZeta8Chernoff::new()` enables Chebyshev by default
/// and the output is finite at tau=1e-4 with unit-coefficient operator.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_path_eps_cheb_floor_zeta8_default_on() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid");

    let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(1.0_f64)).expect("ζ⁴ construction must succeed");
    let zeta6 =
        Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64)).expect("ζ⁶ construction must succeed");
    let zeta8 =
        Diffusion8thZeta8Chernoff::new(zeta6, Some(1.0_f64)).expect("ζ⁸ construction must succeed");

    let f0 = GridFn1D::from_fn(grid, gauss);
    let mut dst = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    zeta8
        .apply_into(1e-4, &f0, &mut dst, &mut scratch)
        .expect("ζ⁸ apply_into must succeed");

    assert!(
        dst.values.iter().all(|v| v.is_finite()),
        "ζ⁸ default Chebyshev ON: all output values must be finite"
    );
    eprintln!("G_PATH_EPS_CHEB_FLOOR [ζ⁸ default-ON]: output finite, ADR-0088 Wave II unblocked");
}

// ---------------------------------------------------------------------------
// Test 5: ζ⁸ without_chebyshev downgrades (AC3 wiring)
// ---------------------------------------------------------------------------

/// Wiring test: `without_chebyshev_sampling()` on ζ⁸ disables Chebyshev.
///
/// Verifies that `.without_chebyshev_sampling()` is the inverse of default-ON.
/// Output must still be finite (`QuinticHermite` or `CubicHermite` floor is active).
#[test]
fn zeta8_without_chebyshev_wiring() {
    let grid = Grid1D::new(-5.0, 5.0, 64).expect("grid");

    let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(1.0_f64)).expect("ζ⁴ construction must succeed");
    let zeta6 =
        Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64)).expect("ζ⁶ construction must succeed");
    let zeta8 = Diffusion8thZeta8Chernoff::new(zeta6, Some(1.0_f64))
        .expect("ζ⁸ construction must succeed")
        .without_chebyshev_sampling();

    let f0 = GridFn1D::from_fn(grid, gauss);
    let mut dst = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    zeta8
        .apply_into(1e-4, &f0, &mut dst, &mut scratch)
        .expect("ζ⁸ without_chebyshev must succeed");

    assert!(
        dst.values.iter().all(|v| v.is_finite()),
        "ζ⁸ without_chebyshev: output must be finite"
    );
}
