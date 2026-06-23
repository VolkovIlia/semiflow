//! `G_PATH_EPS_FLOOR` — Regression gate for `SepticHermite` vs `CubicHermite` spatial floor (ADR-0089, ADR-0120).
//!
//! Verifies that:
//!
//! 1. `SepticHermite` (v7.0 default) produces per-sample error ≤ 1e-8 at off-node positions
//!    (N=512, Gaussian probe `f(x) = exp(-x²)`).
//! 2. Explicit `CubicHermite` control produces per-sample error > 1e-7, confirming that
//!    `SepticHermite` provides the expected floor improvement.
//!
//! ## Why `CubicHermite` must be EXPLICIT on the control arm (ADR-0120 test-precondition fix)
//!
//! The original test used `quintic=false` as the "`CubicHermite` control", but the `false`
//! branch used the grid with its default `InterpKind` (`SepticHermite` since ADR-0109 v6.0).
//! This caused the control to sample at the Septic floor (~1.38e-12) instead of the Cubic
//! floor (~6.5e-7), so the `baseline > 1e-7` precondition silently failed. The control arm
//! must explicitly set `InterpKind::CubicHermite` to actually measure the Cubic floor.
//!
//! ## Why tau=0?
//!
//! At `tau=0`, `Diffusion4thChernoff::apply_into` produces `dst[i] = f.sample(x_i)` for
//! each node — purely an interpolation operation using the kernel's internal grid.
//! Error at off-node `x` is then sampled by calling `dst.sample(x)` via the explicitly-set
//! interp grid.
//!
//! ## References
//!
//! - ADR-0089 Path ε (AC9) — original `RELEASE_BLOCKING` regression guard.
//! - ADR-0120 — v7.0 honest recalibration; `QuinticHermite` removed, Septic is default.
//! - `crates/semiflow-core/tests/path_eps_spatial_floor.rs` — this file.

#![allow(clippy::cast_precision_loss)]

use semiflow::{
    boundary::InterpKind, chernoff::ChernoffFunction, Diffusion4thChernoff,
    Diffusion4thZeta4Chernoff, Diffusion6thZeta6Chernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Test geometry
// ---------------------------------------------------------------------------

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
/// Grid resolution: N=512 nodes (ADR-0089 AC9 spec).
const N_SPATIAL: usize = 512;

/// Probe offsets (`grid.x_at(i)` + `PROBE_FRAC` * dx) as fractions of one grid cell.
/// Five representative off-node positions spanning the grid interior.
const PROBE_INDICES: [usize; 5] = [64, 128, 256, 384, 448];
const PROBE_FRAC: f64 = 0.5;

/// `SepticHermite` spatial floor gate (ADR-0089/ADR-0120): error ≤ 1e-8 at N=512 Gaussian probe.
/// Measured floor ~1.38e-12 at N=512 under `SepticHermite` (Septic is the v7 default, ADR-0120).
const SEPTIC_FLOOR_GATE: f64 = 1e-8;
/// `CubicHermite` baseline floor (ADR-0089/ADR-0120): error > 1e-7 confirms improvement is real.
/// Measured floor ~6.5e-7 at N=512 under explicit `CubicHermite` (Catmull-Rom, ADR-0089 AMD 1).
const CUBIC_FLOOR_LOWER_BOUND: f64 = 1e-7;

// ---------------------------------------------------------------------------
// Helper: analytic Gaussian and interpolation-error measurement
// ---------------------------------------------------------------------------

/// Gaussian probe `f(x) = exp(-x²)`.
fn gauss(x: f64) -> f64 {
    libm::exp(-x * x)
}

/// Measure max interpolation error at the 5 probe positions.
///
/// `interp_kind` sets the interpolant used for both the K5 grid construction and the
/// `sample_grid`. Explicit kind is MANDATORY to avoid inheriting the `SepticHermite`
/// default on the control (`CubicHermite`) arm — the root cause fixed by ADR-0120.
fn max_interp_error_at_probes(grid: Grid1D<f64>, interp_kind: InterpKind) -> f64 {
    // Build the K5 kernel on a grid with the explicitly-requested interp kind.
    // This ensures the control arm (CubicHermite) truly uses CubicHermite, not the
    // SepticHermite default that was silently inherited in the original test (ADR-0120).
    let kernel_grid = grid.with_interp(interp_kind);
    let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, kernel_grid);

    let f0 = GridFn1D::from_fn(kernel_grid, gauss);
    let mut dst = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    k5.apply_into(0.0, &f0, &mut dst, &mut scratch)
        .expect("apply_into tau=0 must succeed");

    // Sample the output at off-node positions via the same explicitly-set interp kind.
    let sample_grid = grid.with_interp(interp_kind);

    let dx = grid.dx();
    let mut max_err = 0.0_f64;
    for &idx in &PROBE_INDICES {
        let x_probe = grid.x_at(idx) + PROBE_FRAC * dx;
        let analytic = gauss(x_probe);
        let sampled = sample_grid
            .interp(&dst.values, x_probe)
            .expect("interp must succeed for interior point");
        let err = (sampled - analytic).abs();
        max_err = max_err.max(err);
    }
    max_err
}

// ---------------------------------------------------------------------------
// Test 1: K5 SepticHermite (v7 default) vs explicit CubicHermite (ADR-0120 fix)
// ---------------------------------------------------------------------------

/// `G_PATH_EPS_FLOOR` — `RELEASE_BLOCKING` (ADR-0089 AC9, updated ADR-0120).
///
/// Verifies that `SepticHermite` (v7.0 default) achieves spatial floor ≤ 1e-8
/// at N=512 with Gaussian probe, and that explicit `CubicHermite` control produces
/// floor > 1e-7, confirming the Septic improvement is real and measurable.
///
/// ADR-0120 fix: the original test used `quintic=false` to represent `CubicHermite`,
/// but the `false` branch sampled via the default (`SepticHermite`) grid, reading the
/// Septic floor ~1.38e-12 and causing the `baseline > 1e-7` precondition to fail.
/// This version explicitly sets `InterpKind::CubicHermite` on the control arm and
/// `InterpKind::SepticHermite` on the improved arm (default since ADR-0109/v6.0).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_path_eps_floor_septic_improvement() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    // Control arm: EXPLICIT CubicHermite (Catmull-Rom, ADR-0120 precondition fix).
    let cubic_err = max_interp_error_at_probes(grid, InterpKind::CubicHermite);
    // Improved arm: SepticHermite (v7.0 default, ADR-0109/ADR-0120).
    let septic_err = max_interp_error_at_probes(grid, InterpKind::SepticHermite);

    eprintln!(
        "G_PATH_EPS_FLOOR: N={N_SPATIAL}, probe positions idx={PROBE_INDICES:?} + {PROBE_FRAC}·dx"
    );
    eprintln!(
        "  CubicHermite max err   = {cubic_err:.4e}  (expected > {CUBIC_FLOOR_LOWER_BOUND:.0e})"
    );
    eprintln!("  SepticHermite max err  = {septic_err:.4e}  (gate ≤ {SEPTIC_FLOOR_GATE:.0e})");
    eprintln!(
        "  Floor improvement factor: {:.1}×",
        cubic_err / septic_err.max(1e-20)
    );

    // Baseline check: explicit CubicHermite floor must be above 1e-7 (Catmull-Rom O(dx^4)).
    // ADR-0120: this precondition was silently failing because the prior `quintic=false` arm
    // inherited the SepticHermite default, reading ~1.38e-12 instead of ~6.5e-7.
    assert!(
        cubic_err > CUBIC_FLOOR_LOWER_BOUND,
        "G_PATH_EPS_FLOOR baseline FAIL: explicit CubicHermite max err = {cubic_err:.4e} \
         ≤ {CUBIC_FLOOR_LOWER_BOUND:.0e}. Expected Catmull-Rom O(dx^4) floor ≈ 6.5e-7 at N=512. \
         If this fails, InterpKind::CubicHermite is not dispatching correctly. See ADR-0120."
    );

    // Primary gate: SepticHermite floor must be ≤ 1e-8 (O(dx^8) virtual-node interpolant).
    assert!(
        septic_err <= SEPTIC_FLOOR_GATE,
        "G_PATH_EPS_FLOOR FAIL (RELEASE_BLOCKING): SepticHermite max err = {septic_err:.4e} > \
         {SEPTIC_FLOOR_GATE:.0e}. Expected O(dx^8) floor ≈ 1.38e-12 at N=512, Gaussian probe. \
         Check: SepticHermite dispatch in grid.rs::interp; grid_septic.rs impl intact. \
         See ADR-0089 AC9, ADR-0120. RELEASE_BLOCKING."
    );
}

// ---------------------------------------------------------------------------
// Test 2: Constructibility assertions (v7.0: QuinticHermite removed — ADR-0120)
// ---------------------------------------------------------------------------

/// `G_PATH_EPS_FLOOR` constructibility — verifies v7.0 kernel construction works
/// and Chebyshev + `OctonicHermite` opt-ins are functional (ADR-0089, ADR-0120).
///
/// At v7.0: `QuinticHermite` removed; `with_quintic_sampling` / `without_quintic_sampling`
/// builders removed. Default sampling path is `CubicHermite` for K5 and ζ⁴;
/// ζ⁶ no longer does special Quintic direct-wiring (field removed).
#[test]
fn g_path_eps_floor_v7_constructibility() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid");
    let inner = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);

    // K5 base: constructs OK, defaults apply (no quintic_sampling field).
    let k5_default = inner.clone();
    assert!(
        !k5_default.chebyshev_sampling,
        "K5 default: chebyshev_sampling=false"
    );
    assert!(
        !k5_default.octonic_sampling,
        "K5 default: octonic_sampling=false"
    );

    // K5 Chebyshev opt-in works.
    let k5_cheb = inner.clone().with_chebyshev_sampling();
    assert!(
        k5_cheb.chebyshev_sampling,
        "with_chebyshev_sampling sets chebyshev_sampling=true"
    );

    // K5 Octonic opt-in works.
    let k5_oct = inner.clone().with_octonic_sampling();
    assert!(
        k5_oct.octonic_sampling,
        "with_octonic_sampling sets octonic_sampling=true"
    );

    // ζ⁴ constructs OK.
    let zeta4 = Diffusion4thZeta4Chernoff::new(inner.clone(), Some(1.0_f64)).expect("zeta4");
    assert!(
        !zeta4.inner.chebyshev_sampling,
        "ζ⁴ default: chebyshev_sampling=false"
    );

    // ζ⁶ constructs OK (no longer does special quintic wiring).
    let inner2 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let zeta4_for_zeta6 = Diffusion4thZeta4Chernoff::new(inner2, Some(1.0_f64)).expect("zeta4");
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4_for_zeta6, Some(1.0_f64)).expect("zeta6");
    assert!(
        !zeta6.inner.inner.chebyshev_sampling,
        "ζ⁶ default: K5 chebyshev_sampling=false"
    );
}
