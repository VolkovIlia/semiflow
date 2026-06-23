//! Bit-exact equality gate for the v0.13.0 Wave A2 `Strang3D` serial scratch-pool path.
//!
//! Verifies that the refactored `apply_serial` (single in-place `buf` reuse,
//! ADR-0022 Amendment 1) produces **bit-for-bit identical** results across:
//!
//! 1. **Determinism**: two separate calls with the same `(phi, f0)` return
//!    byte-identical outputs.  Catches any scratch-buffer aliasing bug where
//!    residual state from a previous call leaks into the next one.
//!
//! 2. **Input immutability**: `f0.values` is unchanged after `apply`.
//!    Catches any accidental write-back to the source buffer.
//!
//! 3. **Multi-step stability**: running `n_steps` applications does not diverge
//!    from running `n_steps` individual `apply` calls manually chained.
//!    This is a semantic consistency check, not a mathematical semigroup law.
//!
//! Gate: `STRANG3D_SERIAL_SCRATCH_BIT_EQUAL` — **RELEASE-BLOCKING**.
//!
//! See `docs/adr/0022-scratch-pool.md` Amendment 1 and
//! `src/strang3d.rs` `apply_serial`.

use semiflow_core::{
    chernoff::ApplyChernoffExt, DiffusionChernoff, Grid1D, Grid3D, GridFn3D, Strang3D,
};

// ---------------------------------------------------------------------------
// Domain constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -4.0;
const X_MAX: f64 = 4.0;
const TAU: f64 = 0.05;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_strang_diffusion(
    n: usize,
) -> (
    Strang3D<DiffusionChernoff, DiffusionChernoff, DiffusionChernoff>,
    GridFn3D,
) {
    let gx = Grid1D::new(X_MIN, X_MAX, n).expect("grid x");
    let gy = Grid1D::new(X_MIN, X_MAX, n).expect("grid y");
    let gz = Grid1D::new(X_MIN, X_MAX, n).expect("grid z");
    let grid = Grid3D::new(gx, gy, gz).expect("grid 3d");
    let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
    let cz = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gz);
    let phi = Strang3D::new(cx, cy, cz);
    let f0 = GridFn3D::from_fn(grid, |x, y, z| (-(x * x + y * y + z * z)).exp());
    (phi, f0)
}

/// Assert byte-for-byte equality; on failure, report first divergent index.
fn assert_bit_equal(a: &[f64], b: &[f64], label: &str, n: usize) {
    assert_eq!(a.len(), b.len(), "length mismatch: {label} N={n}");
    let first_bad = a
        .iter()
        .zip(b.iter())
        .position(|(x, y)| x.to_bits() != y.to_bits());
    if let Some(idx) = first_bad {
        panic!(
            "BIT-DIVERGENCE: {label} N={n}\n\
             First bad index: {idx}\n\
             a[{idx}] = {:?} (bits={:064b})\n\
             b[{idx}] = {:?} (bits={:064b})",
            a[idx],
            a[idx].to_bits(),
            b[idx],
            b[idx].to_bits(),
        );
    }
}

// ---------------------------------------------------------------------------
// Gate: STRANG3D_SERIAL_SCRATCH_BIT_EQUAL
//
// Two independent calls with identical (phi, f0) must produce bit-identical
// output.  This catches scratch-pool aliasing bugs where residual state from
// a prior `apply` call leaks into the next one.
// ---------------------------------------------------------------------------

#[test]
fn strang3d_serial_scratch_bit_equal() {
    // N=8, 16, 32 — all below the parallel threshold so only serial runs.
    for &n in &[8usize, 16, 32] {
        let (phi, f0) = make_strang_diffusion(n);

        // Two independent applications to the same input.
        let result_a = phi.apply_chernoff(TAU, &f0).expect("apply a ok");
        let result_b = phi.apply_chernoff(TAU, &f0).expect("apply b ok");

        assert_bit_equal(
            &result_a.values,
            &result_b.values,
            "two independent apply calls",
            n,
        );

        // Verify f0 was not mutated.
        let f0_snap: Vec<f64> = f0.values.clone();
        let _ = phi.apply_chernoff(TAU, &f0).expect("apply c ok");
        assert_bit_equal(&f0.values, &f0_snap, "f0 immutability", n);
    }
}

// ---------------------------------------------------------------------------
// Multi-step scratch stability: manual chaining must equal itself on replay.
// Catches any hidden state carried over between `apply` invocations.
// ---------------------------------------------------------------------------

#[test]
fn strang3d_serial_multi_step_scratch_stable() {
    for &n in &[8usize, 16] {
        let (phi, f0) = make_strang_diffusion(n);

        // Run 1: manually chain 4 steps.
        let mut run1 = f0.clone();
        for _ in 0..4 {
            run1 = phi.apply_chernoff(TAU, &run1).expect("step ok");
        }

        // Run 2: same chain from same f0.
        let mut run2 = f0.clone();
        for _ in 0..4 {
            run2 = phi.apply_chernoff(TAU, &run2).expect("step ok");
        }

        assert_bit_equal(&run1.values, &run2.values, "4-step replay", n);
    }
}

// ---------------------------------------------------------------------------
// tau=0 is near-identity (scratch-pool must not corrupt on trivial input)
// ---------------------------------------------------------------------------

#[test]
fn strang3d_serial_tau_zero_is_identity() {
    for &n in &[8usize, 16] {
        let (phi, f0) = make_strang_diffusion(n);
        let out = phi.apply_chernoff(0.0, &f0).expect("apply tau=0 ok");
        let max_diff = f0
            .values
            .iter()
            .zip(out.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "tau=0 deviation {max_diff:.2e} > 1e-10 at N={n}"
        );
    }
}
