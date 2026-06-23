//! Bit-equality regression: `NonSeparable2DAnisotropicChernoff` parallel vs sequential.
//!
//! Closes audit O-2 (v0.9.0). With the `parallel` feature enabled,
//! `NonSeparable2DAnisotropicChernoff` gains `Send + Sync` bounds and its
//! `ChernoffFunction<f64>` impl is used in multi-threaded contexts. This test
//! verifies that:
//!
//! 1. The operator is accepted by Rust's type system as `Send + Sync` (compile-time
//!    check via `_send_sync_check`).
//! 2. Applying the operator from multiple independent threads produces byte-identical
//!    output to the single-thread reference (no shared mutable state leaks).
//! 3. Both constant-β and variable-β(x,y) paths are covered, exercising the full
//!    5-leg Strang composition including `phi_m_beta_step`.
//!
//! Gate: `NS2D_ANISO_PARALLEL_BIT_EQUAL` (RELEASE-BLOCKING).
//! Classification: mirrors `STRANG2D_PARALLEL_BIT_EQUAL` (ADR-0018) for the
//! anisotropic non-separable 2D type.
//!
//! See `docs/audit-findings-v0_9_0.md` §O-2 and `docs/adr/0023-*.md`.

#![cfg(all(feature = "parallel", feature = "slow-tests"))]

use semiflow_core::{
    chernoff::ApplyChernoffExt, BoundaryPolicy, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    NonSeparable2DAnisotropicChernoff,
};

// ---------------------------------------------------------------------------
// Domain constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -5.0;
const X_MAX: f64 = 5.0;
const TAU: f64 = 5e-4;
const N_STEPS: usize = 4;

// CFL-safe β bound: 4·τ·β_norm = 4·5e-4·0.05 = 1e-4; dx(n=64)≈0.079 → dx²≈0.0063. ✓
const BETA_NORM: f64 = 0.05;
const AXX: f64 = 0.1;
const AYY: f64 = 0.1;

// ---------------------------------------------------------------------------
// Send + Sync compile-time check
// ---------------------------------------------------------------------------

/// Verify `NonSeparable2DAnisotropicChernoff<DiffusionChernoff, DiffusionChernoff>`
/// satisfies `Send + Sync` when the `parallel` feature is enabled.
///
/// A field that is not `Send + Sync` will break the build here before causing
/// a runtime data race or a hard-to-diagnose compiler error elsewhere.
fn _send_sync_check() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DiffusionChernoff>();
    assert_send_sync::<NonSeparable2DAnisotropicChernoff<DiffusionChernoff, DiffusionChernoff>>();
}

// ---------------------------------------------------------------------------
// Operator constructors
// ---------------------------------------------------------------------------

fn build_op_constant_beta(
    n: usize,
) -> NonSeparable2DAnisotropicChernoff<DiffusionChernoff, DiffusionChernoff> {
    let gx = Grid1D::new(X_MIN, X_MAX, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic);
    let gy = gx;
    let grid = Grid2D::new(gx, gy);
    let ix = DiffusionChernoff::new(|_| AXX, |_| 0.0, |_| 0.0, AXX, gx);
    let iy = DiffusionChernoff::new(|_| AYY, |_| 0.0, |_| 0.0, AYY, gy);
    NonSeparable2DAnisotropicChernoff::new(ix, iy, |_, _| BETA_NORM, BETA_NORM, grid).unwrap()
}

fn build_op_variable_beta(
    n: usize,
) -> NonSeparable2DAnisotropicChernoff<DiffusionChernoff, DiffusionChernoff> {
    let gx = Grid1D::new(X_MIN, X_MAX, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic);
    let gy = gx;
    let grid = Grid2D::new(gx, gy);
    let ix = DiffusionChernoff::new(|_| AXX, |_| 0.0, |_| 0.0, AXX, gx);
    let iy = DiffusionChernoff::new(|_| AYY, |_| 0.0, |_| 0.0, AYY, gy);
    // β(x,y) = 0.05·exp(-(x²+y²)/4); sup-norm = BETA_NORM = 0.05.
    NonSeparable2DAnisotropicChernoff::new(
        ix,
        iy,
        |x, y| BETA_NORM * (-(x * x + y * y) / 4.0).exp(),
        BETA_NORM,
        grid,
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Runner: evolve N_STEPS steps, return final values
// ---------------------------------------------------------------------------

fn evolve_n_steps(
    op: &NonSeparable2DAnisotropicChernoff<DiffusionChernoff, DiffusionChernoff>,
    f0: &GridFn2D,
) -> Vec<f64> {
    let mut u = f0.clone();
    for _ in 0..N_STEPS {
        u = op.apply_chernoff(TAU, &u).unwrap();
    }
    u.values
}

// ---------------------------------------------------------------------------
// Initial datum: u0(x, y) = exp(-(x² + y²))
// ---------------------------------------------------------------------------

fn make_initial(gx: Grid1D, gy: Grid1D) -> GridFn2D {
    let grid = Grid2D::new(gx, gy);
    GridFn2D::from_fn(grid, |x, y| (-(x * x + y * y)).exp())
}

// ---------------------------------------------------------------------------
// Assertion helper — byte-exact comparison
// ---------------------------------------------------------------------------

fn assert_bit_equal(reference: &[f64], parallel: &[f64], label: &str) {
    assert_eq!(reference.len(), parallel.len(), "length mismatch: {label}");
    let first_bad = reference
        .iter()
        .zip(parallel.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits());
    if let Some(idx) = first_bad {
        panic!(
            "BIT-DIVERGENCE: {label}\n\
             First bad index: {idx}\n\
             reference[{idx}] = {:?} (bits={:064b})\n\
             parallel[{idx}]  = {:?} (bits={:064b})",
            reference[idx],
            reference[idx].to_bits(),
            parallel[idx],
            parallel[idx].to_bits(),
        );
    }
}

// ---------------------------------------------------------------------------
// Test: constant β
// ---------------------------------------------------------------------------

/// Bit-equality gate for `NonSeparable2DAnisotropicChernoff` with constant β.
///
/// Spawns 4 threads each independently applying the shared (immutable) operator
/// to an identical initial state. All results must be byte-identical to the
/// single-thread reference. Exercises the `Send + Sync` carve-out under
/// `#[cfg(feature = "parallel")]`.
#[test]
fn ns2d_aniso_parallel_bit_equal_constant_beta() {
    for &n in &[32_usize, 64, 128] {
        let gx = Grid1D::new(X_MIN, X_MAX, n)
            .unwrap()
            .with_boundary(BoundaryPolicy::Periodic);
        let gy = gx;
        let f0 = make_initial(gx, gy);
        let op = build_op_constant_beta(n);

        // Single-thread reference.
        let reference = evolve_n_steps(&op, &f0);

        // Parallel: 4 threads each independently apply the same op.
        let results: Vec<Vec<f64>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| s.spawn(|| evolve_n_steps(&op, &f0)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (k, parallel) in results.iter().enumerate() {
            assert_bit_equal(
                &reference,
                parallel,
                &format!("constant_beta n={n} thread={k}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test: variable β(x,y)
// ---------------------------------------------------------------------------

/// Bit-equality gate for `NonSeparable2DAnisotropicChernoff` with variable β(x,y).
///
/// Same structure as `ns2d_aniso_parallel_bit_equal_constant_beta` but uses
/// `β(x,y) = 0.05·exp(-(x²+y²)/4)` — a non-constant coupling that exercises
/// the position-dependent `phi_m_beta_step` path (eq. 10.7-ter.7).
#[test]
fn ns2d_aniso_parallel_bit_equal_variable_beta() {
    for &n in &[32_usize, 64, 128] {
        let gx = Grid1D::new(X_MIN, X_MAX, n)
            .unwrap()
            .with_boundary(BoundaryPolicy::Periodic);
        let gy = gx;
        let f0 = make_initial(gx, gy);
        let op = build_op_variable_beta(n);

        // Single-thread reference.
        let reference = evolve_n_steps(&op, &f0);

        // Parallel: 4 threads each independently apply the same op.
        let results: Vec<Vec<f64>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| s.spawn(|| evolve_n_steps(&op, &f0)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (k, parallel) in results.iter().enumerate() {
            assert_bit_equal(
                &reference,
                parallel,
                &format!("variable_beta n={n} thread={k}"),
            );
        }
    }
}
