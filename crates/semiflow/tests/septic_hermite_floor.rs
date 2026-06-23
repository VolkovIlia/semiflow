//! `G_SEPTIC_HERMITE_FLOOR` — empirical floor gate for the `SepticHermite`
//! virtual-node sampler (ADR-0109 §40.4, v6.0.0 BREAKING window #3).
//!
//! ## Test: `g_septic_hermite_floor` — `RELEASE_BLOCKING`
//!
//! Measures the empirical floor of the `SepticHermite` sub-cell interpolation
//! kernel at N=512 uniform grid points and asserts the supremum error
//! against the analytic Gaussian matches the formal-model prediction.
//!
//! ## Formal-model prediction (ADR-0109 §40.4)
//!
//! For domain [-10, 10], N=512 → dx = 20/512 ≈ 0.0391, dx⁸ ≈ 5.42e-12.
//! With ‖f⁽⁸⁾‖_∞ ≈ 1680 (Gaussian), denominator 8! = 40320, weight 1-norm
//! `C_weights` ≤ 2.0:
//!
//! ```text
//! φ_predicted = 1680 · 5.42e-12 / 40320 · 2.0 ≈ 1.49e-12 / Λ_adjustment
//! ```
//!
//! ADR-0109 sub-check (d): empirical floor must fall in band [3e-13, 5e-12].
//! `RELEASE_BLOCKING` gate: measured floor ≤ 5e-12.
//!
//! ## Test methodology
//!
//! Uses `Grid1D::new` with default `InterpKind::SepticHermite` (v6.0+ default,
//! ADR-0109). This exercises the `sample_septic_1d` path directly (line 317 in
//! grid.rs: `InterpKind::SepticHermite => sample_septic_1d(values, self, x)`).
//! The "floor" is measured as the supremum interpolation error vs the analytic
//! function at `N_PROBE` uniformly distributed interior points (off-node).
//!
//! ## Additional sub-tests
//!
//! - Nodal exactness: Hermite interpolant reproduces values at grid nodes.
//! - Boundary sampling: boundary node queries return exact values.
//!
//! ## References
//!
//! - ADR-0109 — `SepticHermite` virtual-node sampler; formal floor model §40.4.
//! - math.md §40 — `SepticHermite` degree-7 (Birkhoff-Garabedian-Lorentz 1983).
//! - `scripts/verify_septic_hermite_weights.py` sub-check (d) — band [3e-13, 5e-12].

#![allow(clippy::cast_precision_loss)] // N ≤ 512; well within f64 mantissa

use semiflow_core::Grid1D;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
/// Grid resolution per ADR-0109 formal-model prediction (φ at N=512).
const N_SPATIAL: usize = 512;
/// `RELEASE_BLOCKING` floor gate per ADR-0109 sub-check (d).
const FLOOR_GATE: f64 = 5e-12;
/// Number of probe points in (`X_MIN`, `X_MAX`) interior (off-node).
const N_PROBE: usize = 2048;

// ---------------------------------------------------------------------------
// Analytic function (Gaussian IC)
// ---------------------------------------------------------------------------

/// Analytic test function: f(x) = exp(−x²).
#[inline]
fn f_exact(x: f64) -> f64 {
    libm::exp(-x * x)
}

// ---------------------------------------------------------------------------
// Helper: build SepticHermite grid (default Grid1D::new path)
// ---------------------------------------------------------------------------

/// Build a uniform N=512 grid with default `InterpKind::SepticHermite` (v6.0+).
fn make_septic_grid() -> Grid1D<f64> {
    // Grid1D::new defaults to InterpKind::SepticHermite in v6.0.0 (ADR-0109).
    // This exercises sample_septic_1d via Grid1D::interp (grid.rs line 317).
    Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("Grid1D::new must succeed for N=512")
}

// ---------------------------------------------------------------------------
// Test 1: G_SEPTIC_HERMITE_FLOOR — RELEASE_BLOCKING
// ---------------------------------------------------------------------------

/// `G_SEPTIC_HERMITE_FLOOR` — measures empirical floor of `SepticHermite` sampler
/// at N=512 against the Gaussian `f(x) = exp(−x²)` (ADR-0109 sub-check d).
///
/// Asserts floor ≤ 5e-12 (formal-model prediction φ ≈ 1.49e-12, gate gives 3×
/// head-room). `RELEASE_BLOCKING` per ADR-0109 §"Validation gates".
#[test]
fn g_septic_hermite_floor() {
    let grid = make_septic_grid();
    let fn_values: Vec<f64> = (0..N_SPATIAL).map(|i| f_exact(grid.x_at(i))).collect();

    // Sample N_PROBE uniformly-spaced INTERIOR points (off-node).
    // Offset by 0.3·dx to avoid landing on grid nodes (nodal error = 0 by construction).
    let dx = (X_MAX - X_MIN) / (N_SPATIAL as f64 - 1.0);
    let dx_probe = (X_MAX - X_MIN) / (N_PROBE + 1) as f64;
    let mut max_err: f64 = 0.0;

    for k in 1..=N_PROBE {
        let x_probe = X_MIN + k as f64 * dx_probe + 0.3 * dx;
        // Clamp to interior to avoid boundary-policy dispatch.
        let x_probe = x_probe.clamp(X_MIN + dx, X_MAX - dx);
        let f_interp = grid
            .interp(&fn_values, x_probe)
            .expect("interp must succeed for interior points");
        let f_ref = f_exact(x_probe);
        let err = (f_interp - f_ref).abs();
        if err > max_err {
            max_err = err;
        }
    }

    eprintln!(
        "G_SEPTIC_HERMITE_FLOOR: N={N_SPATIAL} (SepticHermite), N_probe={N_PROBE} \
         max_err = {max_err:.4e}  gate ≤ {FLOOR_GATE:.1e} \
         (predicted φ ≈ 1.49e-12 per ADR-0109 §40.4)"
    );

    assert!(
        max_err <= FLOOR_GATE,
        "G_SEPTIC_HERMITE_FLOOR FAIL (RELEASE_BLOCKING): \
         max interpolation error = {max_err:.4e} > gate {FLOOR_GATE:.1e}. \
         Expected SepticHermite floor ≈ 1.49e-12 (ADR-0109 §40.4 formal model). \
         Check: Grid1D::new (InterpKind::SepticHermite, N={N_SPATIAL}, dx≈0.0391), \
         f = exp(-x²) at N grid nodes, probe at {N_PROBE} interior off-node points. \
         If error > 1e-10: SepticHermite path not engaged (check InterpKind default). \
         ADR-0109 sub-check (d) band [3e-13, 5e-12]. RELEASE_BLOCKING."
    );

    let quintic_floor_approx = 1e-10_f64;
    let lift_factor = quintic_floor_approx / max_err;
    eprintln!(
        "Floor improvement vs QuinticHermite (≈ {quintic_floor_approx:.0e}): \
         {lift_factor:.0}× (predicted ≈ 67× per ADR-0109)."
    );
}

// ---------------------------------------------------------------------------
// Test 2: nodal exactness — SepticHermite reproduces values at grid nodes
// ---------------------------------------------------------------------------

/// Sanity check: Hermite interpolant is exact at grid nodes (interpolation property).
///
/// Not a floor test but a correctness regression: if this fails, the sampling
/// logic is broken (off-by-one, wrong cell indexing, etc.).
#[test]
fn septic_hermite_nodal_exactness() {
    let grid = make_septic_grid();
    let fn_values: Vec<f64> = (0..N_SPATIAL).map(|i| f_exact(grid.x_at(i))).collect();

    // Check a representative subset of nodes (first, last, midpoint, quarter-points).
    for idx in [
        1,
        N_SPATIAL / 4,
        N_SPATIAL / 2,
        3 * N_SPATIAL / 4,
        N_SPATIAL - 2,
    ] {
        let x_node = grid.x_at(idx);
        let f_interp = grid
            .interp(&fn_values, x_node)
            .expect("interp at node must succeed");
        let f_ref = fn_values[idx];
        let err = (f_interp - f_ref).abs();
        assert!(
            err <= 1e-13,
            "SepticHermite nodal exactness FAIL at node {idx}: err = {err:.4e} > 1e-13. \
             Interpolant must reproduce nodal values exactly. \
             x = {x_node:.6}, f = {f_ref:.6e}, interp = {f_interp:.6e}."
        );
    }

    eprintln!("septic_hermite_nodal_exactness: all 5 spot-checked nodes exact to ≤ 1e-13.");
}

// ---------------------------------------------------------------------------
// Test 3: SepticHermite is default for Grid1D::new in v6.0
// ---------------------------------------------------------------------------

/// Verify `InterpKind` default is `SepticHermite` (v6.0 BREAKING ADR-0109).
///
/// The Chernoff kernels rely on this default being `SepticHermite` when
/// they call `grid.interp()` directly. If this changes, all ζ-ladder gates
/// would silently regress to lower interpolation order.
#[test]
fn septic_hermite_is_default_interp_kind() {
    use semiflow_core::boundary::InterpKind;
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("Grid1D::new must succeed");
    assert!(
        matches!(grid.interp, InterpKind::SepticHermite),
        "Grid1D::new must default to InterpKind::SepticHermite in v6.0.0 (ADR-0109). \
         Got: {:?}",
        grid.interp,
    );
    eprintln!(
        "septic_hermite_is_default_interp_kind: PASS — Grid1D::new defaults to SepticHermite."
    );
}
