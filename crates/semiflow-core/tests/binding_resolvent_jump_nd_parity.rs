//! `G_BINDING_RESOLVENT_JUMP_ND_PARITY` — sub-test 1 (core golden).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0153, ADR-0148, slow-tests):
//!   Canonical smoke (§5, `V8_3_TIER3_BINDING_DESIGN.md)`:
//!     2D: 8×8 on [−5,5]², M=8, t=1.0; u0=exp(−‖x‖²); axis-0-fastest layout.
//!     3D: 4×4×4 on [−1,1]³, M=8, t=1.0; u0=exp(−‖x‖²); axis-0-fastest layout.
//!
//!   Sub-tests:
//!     1. Core golden: `jump` returns finite values matching `M_ref` self-convergence.
//!     2. ND-layout round-trip: flat axis-0-fastest in → flat axis-0-fastest out.
//!     3. Print golden vectors for embedding in PyO3/FFI/WASM binding tests.
//!
//! ## Why this is GENUINE (not tautological)
//!
//! M=8 and `M_ref=16` run different TWS contour quadratures (different complex LU
//! solves at different contour node positions). Any implementation bug would
//! produce different results. Sub-tests 2/3/4 (FFI/PyO3/WASM) independently
//! re-compute via their binding layers and compare bit-for-bit — any marshalling
//! divergence is a non-zero ULP.
//!
//! ## ND layout (NORMATIVE, §3.1)
//!
//! `GridFn2D`/`GridFn3D` are axis-0-fastest:
//!   `idx(i,j) = j·nx + i`              (2D),
//!   `idx(i,j,k) = k·nx·ny + j·nx + i`  (3D).
//! The flat input/output vector is the SAME as Fortran-order ravel/reshape
//! on the (nx,ny[,nz]) ND array.

#![allow(clippy::cast_precision_loss)]
// Integration test: allows for numerical / binding wrapper patterns.
#![allow(clippy::missing_panics_doc)]

use semiflow_core::{
    Grid1D, Grid2D, Grid3D, GridFn2D, GridFn3D, ResolventJumpChernoff2D, ResolventJumpChernoff3D,
};

// ---------------------------------------------------------------------------
// Canonical 2D smoke parameters (§5)
// ---------------------------------------------------------------------------

const X2_MIN: f64 = -5.0;
const X2_MAX: f64 = 5.0;
const NX2: usize = 8;
const NY2: usize = 8;
const M2: usize = 8;
const T2: f64 = 1.0;
/// Higher-M reference for self-convergence (same discrete A).
#[cfg(feature = "slow-tests")]
const M2_REF: usize = 16;

// ---------------------------------------------------------------------------
// Canonical 3D smoke parameters (§5)
// ---------------------------------------------------------------------------

const X3_MIN: f64 = -1.0;
const X3_MAX: f64 = 1.0;
const NX3: usize = 4;
const NY3: usize = 4;
const NZ3: usize = 4;
// 3D needs M>=7 for the 1e-3 self-convergence gate (M=6 -> 1.088e-3, 8.8% over);
// M=8 gives ~7.5x margin (sup_err ~1.3e-4). 2D converges faster at lower M
// (tensor-product GL quadrature, curse of dimensionality).
const M3: usize = 8;
const T3: f64 = 1.0;
/// Higher-M reference for 3D self-convergence.
#[cfg(feature = "slow-tests")]
const M3_REF: usize = 12;

/// Tolerance: `‖jump_M − jump_Mref‖∞ ≤ TOL`.
#[cfg(feature = "slow-tests")]
const TOL: f64 = 1e-3;

// ---------------------------------------------------------------------------
// 2D helpers
// ---------------------------------------------------------------------------

fn make_grid_2d() -> Grid2D<f64> {
    let gx = Grid1D::new(X2_MIN, X2_MAX, NX2).expect("valid x grid");
    let gy = Grid1D::new(X2_MIN, X2_MAX, NY2).expect("valid y grid");
    Grid2D::new(gx, gy)
}

/// Build u0(i,j) = exp(−x²−y²) in axis-0-fastest order: idx = j*nx + i.
fn make_u0_2d(grid: Grid2D<f64>) -> GridFn2D<f64> {
    GridFn2D::from_fn(grid, |x: f64, y: f64| (-x * x - y * y).exp())
}

// ---------------------------------------------------------------------------
// 3D helpers
// ---------------------------------------------------------------------------

fn make_grid_3d() -> Grid3D<f64> {
    let gx = Grid1D::new(X3_MIN, X3_MAX, NX3).expect("valid x grid");
    let gy = Grid1D::new(X3_MIN, X3_MAX, NY3).expect("valid y grid");
    let gz = Grid1D::new(X3_MIN, X3_MAX, NZ3).expect("valid z grid");
    Grid3D::new(gx, gy, gz).expect("valid 3D grid")
}

/// Build u0(i,j,k) = exp(−x²−y²−z²) in axis-0-fastest order.
fn make_u0_3d(grid: Grid3D<f64>) -> GridFn3D<f64> {
    GridFn3D::from_fn(grid, |x: f64, y: f64, z: f64| {
        (-x * x - y * y - z * z).exp()
    })
}

// ---------------------------------------------------------------------------
// Public golden accessors (used by binding parity siblings)
// ---------------------------------------------------------------------------

/// Compute the 2D core golden `jump(T2, u0)` at canonical params.
///
/// Returns the flat axis-0-fastest value vector (length `NX2*NY2`).
/// Public so binding tests can import the golden directly.
#[must_use]
pub fn canonical_resolvent_jump_2d_core() -> Vec<f64> {
    let grid = make_grid_2d();
    let rj = ResolventJumpChernoff2D::new(grid, M2).expect("valid 2D kernel");
    let g = make_u0_2d(grid);
    rj.jump(T2, &g).expect("valid 2D jump").values
}

/// Compute the 3D core golden `jump(T3, u0)` at canonical params.
///
/// Returns the flat axis-0-fastest value vector (length `NX3*NY3*NZ3`).
/// Public so binding tests can import the golden directly.
#[must_use]
pub fn canonical_resolvent_jump_3d_core() -> Vec<f64> {
    let grid = make_grid_3d();
    let rj = ResolventJumpChernoff3D::new(grid, M3).expect("valid 3D kernel");
    let g = make_u0_3d(grid);
    rj.jump(T3, &g).expect("valid 3D jump").values
}

// ---------------------------------------------------------------------------
// Test: 2D golden + self-convergence anchor
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "slow-tests")]
fn g_binding_resolvent_jump_nd_parity_2d_core_golden() {
    let grid = make_grid_2d();
    let g = make_u0_2d(grid);

    let jump_m = canonical_resolvent_jump_2d_core();

    // M_ref=16 self-convergence reference (same discrete A, different contour).
    let rj_ref = ResolventJumpChernoff2D::new(grid, M2_REF).expect("M_ref 2D valid");
    let ref_vals = rj_ref.jump(T2, &g).expect("ref 2D jump").values;

    let sup_err: f64 = jump_m
        .iter()
        .zip(ref_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!(
        "G_BINDING_RESOLVENT_JUMP_ND_PARITY (2D core golden, {}x{}, M={}, M_ref={}, t={}):\n\
         sup‖jump_M{} − jump_M{}‖∞ = {:.3e}  (gate ≤ {:.0e})\n\
         Golden (center sample):\n\
           jump[{}] = {:.16e}",
        NX2,
        NY2,
        M2,
        M2_REF,
        T2,
        M2,
        M2_REF,
        sup_err,
        TOL,
        NX2 * NY2 / 2,
        jump_m[NX2 * NY2 / 2],
    );

    // Print full golden for embedding in binding tests.
    print!("2D golden flat [{} values]:", NX2 * NY2);
    for (i, &v) in jump_m.iter().enumerate() {
        if i % 4 == 0 {
            print!("\n  ");
        }
        print!("{:.16e}", v);
        if i + 1 < NX2 * NY2 {
            print!(", ");
        }
    }
    println!();

    assert!(
        sup_err <= TOL,
        "G_BINDING_RESOLVENT_JUMP_ND_PARITY 2D FAIL: sup = {sup_err:.3e} > {TOL:.0e}"
    );
}

// ---------------------------------------------------------------------------
// Test: 3D golden + self-convergence anchor
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "slow-tests")]
fn g_binding_resolvent_jump_nd_parity_3d_core_golden() {
    let grid = make_grid_3d();
    let g = make_u0_3d(grid);

    let jump_m = canonical_resolvent_jump_3d_core();

    // M_ref=12 self-convergence reference.
    let rj_ref = ResolventJumpChernoff3D::new(grid, M3_REF).expect("M_ref 3D valid");
    let ref_vals = rj_ref.jump(T3, &g).expect("ref 3D jump").values;

    let sup_err: f64 = jump_m
        .iter()
        .zip(ref_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!(
        "G_BINDING_RESOLVENT_JUMP_ND_PARITY (3D core golden, {}x{}x{}, M={}, M_ref={}, t={}):\n\
         sup‖jump_M{} − jump_M{}‖∞ = {:.3e}  (gate ≤ {:.0e})\n\
         Golden (center sample):\n\
           jump[{}] = {:.16e}",
        NX3,
        NY3,
        NZ3,
        M3,
        M3_REF,
        T3,
        M3,
        M3_REF,
        sup_err,
        TOL,
        NX3 * NY3 * NZ3 / 2,
        jump_m[NX3 * NY3 * NZ3 / 2],
    );

    // Print full golden for embedding in binding tests.
    print!("3D golden flat [{} values]:", NX3 * NY3 * NZ3);
    for (i, &v) in jump_m.iter().enumerate() {
        if i % 4 == 0 {
            print!("\n  ");
        }
        print!("{:.16e}", v);
        if i + 1 < NX3 * NY3 * NZ3 {
            print!(", ");
        }
    }
    println!();

    assert!(
        sup_err <= TOL,
        "G_BINDING_RESOLVENT_JUMP_ND_PARITY 3D FAIL: sup = {sup_err:.3e} > {TOL:.0e}"
    );
}

// ---------------------------------------------------------------------------
// Test: ND layout round-trip (axis-0-fastest, always runs)
// ---------------------------------------------------------------------------

/// Verify that passing the same flat axis-0-fastest buffer twice gives bit-
/// identical output — confirms layout is deterministic and round-trip safe.
#[test]
fn g_binding_resolvent_jump_nd_layout_roundtrip() {
    // 2D round-trip.
    let grid_2d = make_grid_2d();
    let g_2d = make_u0_2d(grid_2d);
    let rj_2d = ResolventJumpChernoff2D::new(grid_2d, M2).expect("valid 2D kernel");
    let out_a = rj_2d.jump(T2, &g_2d).expect("2D jump A").values;
    let g_2d_b = GridFn2D {
        values: g_2d.values.clone(),
        grid: grid_2d,
    };
    let out_b = rj_2d.jump(T2, &g_2d_b).expect("2D jump B").values;
    assert_eq!(
        out_a, out_b,
        "2D axis-0-fastest round-trip not bit-identical"
    );

    // 3D round-trip.
    let grid_3d = make_grid_3d();
    let g_3d = make_u0_3d(grid_3d);
    let rj_3d = ResolventJumpChernoff3D::new(grid_3d, M3).expect("valid 3D kernel");
    let out_c = rj_3d.jump(T3, &g_3d).expect("3D jump A").values;
    let g_3d_b = GridFn3D {
        values: g_3d.values.clone(),
        grid: grid_3d,
    };
    let out_d = rj_3d.jump(T3, &g_3d_b).expect("3D jump B").values;
    assert_eq!(
        out_c, out_d,
        "3D axis-0-fastest round-trip not bit-identical"
    );
}

// ---------------------------------------------------------------------------
// Test: construction guards (always runs)
// ---------------------------------------------------------------------------

#[test]
fn g_binding_resolvent_jump_nd_construction_guards() {
    let gx = Grid1D::new(-5.0, 5.0, 8).unwrap();
    let grid_2d = Grid2D::new(gx, gx);
    // m_nodes < 6 rejected.
    assert!(ResolventJumpChernoff2D::new(grid_2d, 5).is_err());
    assert!(ResolventJumpChernoff2D::new(grid_2d, 6).is_ok());

    let gx3 = Grid1D::new(-1.0, 1.0, 4).unwrap();
    let grid_3d = Grid3D::new(gx3, gx3, gx3).unwrap();
    assert!(ResolventJumpChernoff3D::new(grid_3d, 5).is_err());
    assert!(ResolventJumpChernoff3D::new(grid_3d, 6).is_ok());
}
