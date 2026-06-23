//! Bit-exact equality gate for parallel `Strang2D::apply` with `F = f32`.
//!
//! Mirrors `strang2d_parallel_bit_equal.rs` (f64) for the f32 precision path
//! introduced in Wave 5 (ADR-0045 §5.3).
//!
//! Verifies that the multi-thread f32 kernel returns **bit-for-bit identical**
//! results to the single-thread reference for every combination of:
//!
//! - `N ∈ {64, 128}` (grid nodes per axis — smaller than f64 test because
//!   f32 errors at large N are closer to the f32 floor),
//! - `n_steps ∈ {1, 4}` (time steps; multi-step exposes temp-buffer reuse bugs),
//! - `thread_count ∈ {2, 4, 8}` (parallel workers vs single-thread reference).
//! - Inner operator: `DiffusionChernoff<f32>` (via `WrapDiff<f32>` shim).
//!
//! No tolerance, no `approx_eq` — `Vec<f32>` comparison is byte-exact via
//! `f32::to_bits`.
//!
//! # Why a separate `WrapDiff` shim?
//!
//! `DiffusionChernoff<F>` for `F ≠ f64` is NOT `ChernoffFunction<F>` in the
//! production API (the SIMD path must remain f64-only per ADR-0025 §SIMD carve-
//! out). The `WrapDiff<f32>` wrapper bridges `apply_f` into `ChernoffFunction<f32>`
//! for test purposes only.
//!
//! # Bit-equality contract (ADR-0018, extended to f32 in ADR-0045)
//!
//! The parallel f32 path MUST be bit-identical to the serial f32 path.
//! This gate BLOCKS any f32 parallel merge that violates this contract.
//!
//! Gated: `#[cfg(all(feature = "parallel", feature = "slow-tests"))]`.

#![cfg(all(feature = "parallel", feature = "slow-tests"))]

use semiflow::{
    chernoff::{ApplyChernoffExt, ChernoffFunction, Growth},
    diffusion::DiffusionChernoff,
    error::SemiflowError,
    grid::Grid1D,
    grid2d::Grid2D,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    scratch::ScratchPool,
    strang2d::Strang2D,
    strang2d_parallel::FORCE_THREADS,
    SemiflowFloat,
};

// ---------------------------------------------------------------------------
// WrapDiff<f32> — test-only shim (see module-level docs)
// ---------------------------------------------------------------------------

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
        // DiffusionChernoff growth is always contraction (1.0, 0.0).
        Growth::contraction()
    }
}

// ---------------------------------------------------------------------------
// Domain constants
// ---------------------------------------------------------------------------

const X_MIN: f32 = -10.0;
const X_MAX: f32 = 10.0;

/// τ per step for `n_steps = 1` at t = 0.05.
const TAU_BASE: f32 = 0.05;

// ---------------------------------------------------------------------------
// Compile-time Send + Sync sanity check
// ---------------------------------------------------------------------------

fn _send_sync_check() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DiffusionChernoff<f32>>();
    assert_send_sync::<WrapDiff<f32>>();
    assert_send_sync::<Strang2D<WrapDiff<f32>, WrapDiff<f32>, f32>>();
}

// ---------------------------------------------------------------------------
// Generic runner (pins FORCE_THREADS, runs n_steps, returns values)
// ---------------------------------------------------------------------------

/// Run `n_steps` applications of `phi2d.apply_chernoff(tau)` starting from `f0`,
/// with the thread count pinned to `thread_count` via `FORCE_THREADS`.
///
/// Returns the final `GridFn2D::values` as `Vec<f32>`.
// n_steps ≤ 4 — within f32 mantissa range.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::needless_pass_by_value)]
fn run_with_threads(
    phi2d: Strang2D<WrapDiff<f32>, WrapDiff<f32>, f32>,
    f0: &GridFn2D<f32>,
    tau: f32,
    n_steps: usize,
    thread_count: usize,
) -> Vec<f32> {
    // Pin thread count via FORCE_THREADS (same hook as the f64 test).
    FORCE_THREADS.with(|c| c.set(Some(thread_count)));
    let mut u = f0.clone();
    for _ in 0..n_steps {
        u = phi2d.apply_chernoff(tau, &u).expect("apply f32 ok");
    }
    FORCE_THREADS.with(|c| c.set(None)); // restore default
    u.values
}

// ---------------------------------------------------------------------------
// Initial datum: u_0(x, y) = exp(-(x² + y²))
// ---------------------------------------------------------------------------

fn make_initial(grid: Grid2D<f32>) -> GridFn2D<f32> {
    GridFn2D::<f32>::from_fn_generic(grid, |x, y| <f32 as num_traits::Float>::exp(-x * x - y * y))
}

// ---------------------------------------------------------------------------
// DiffusionChernoff<f32> sweep
// ---------------------------------------------------------------------------

/// `GF1_2D_F32_BIT_EQUAL`: parallel `Strang2D<f32>` is bit-identical to serial
/// for `N ∈ {64, 128}`, `n_steps ∈ {1, 4}`, `thread_count ∈ {2, 4, 8}`.
///
/// ADR-0018 bit-equality contract, extended to f32 (ADR-0045 §5.3).
// n_steps ∈ {1, 4} — within f32 mantissa range.
#[allow(clippy::cast_precision_loss)]
#[test]
fn bit_equal_f32_diffusion_chernoff() {
    for &n in &[64usize, 128] {
        let gx = Grid1D::<f32>::new_generic(X_MIN, X_MAX, n).expect("grid x f32");
        let gy = Grid1D::<f32>::new_generic(X_MIN, X_MAX, n).expect("grid y f32");
        let grid = Grid2D::<f32>::new(gx, gy);
        let f0 = make_initial(grid);

        let cx = WrapDiff(DiffusionChernoff::<f32>::new(
            |_| 0.5_f32,
            |_| 0.0_f32,
            |_| 0.0_f32,
            0.5,
            gx,
        ));
        let cy = WrapDiff(DiffusionChernoff::<f32>::new(
            |_| 0.5_f32,
            |_| 0.0_f32,
            |_| 0.0_f32,
            0.5,
            gy,
        ));

        for &n_steps in &[1usize, 4] {
            let tau = TAU_BASE / n_steps as f32;
            let phi = Strang2D::<_, _, f32>::new(cx.clone(), cy.clone());

            // Reference: serial (thread_count = 1 → FORCE_THREADS bypasses parallelism).
            let reference = run_with_threads(phi.clone(), &f0, tau, n_steps, 1);

            // Parallel runs.
            for &k in &[2usize, 4, 8] {
                let parallel = run_with_threads(phi.clone(), &f0, tau, n_steps, k);
                assert_bit_equal_f32(&reference, &parallel, n, n_steps, k);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Assertion helper
// ---------------------------------------------------------------------------

/// Assert byte-for-byte equality between `reference` and `parallel` (f32).
///
/// On mismatch, prints all four coordinates and the first divergent index.
fn assert_bit_equal_f32(
    reference: &[f32],
    parallel: &[f32],
    n: usize,
    n_steps: usize,
    thread_count: usize,
) {
    assert_eq!(
        reference.len(),
        parallel.len(),
        "length mismatch: F=f32 N={n} n_steps={n_steps} k={thread_count}"
    );

    let first_bad = reference
        .iter()
        .zip(parallel.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits());

    if let Some(idx) = first_bad {
        panic!(
            "BIT-DIVERGENCE (F=f32): N={n} n_steps={n_steps} k={thread_count}\n\
             First bad index: {idx}\n\
             serial[{idx}]   = {:?} (bits={:032b})\n\
             parallel[{idx}] = {:?} (bits={:032b})",
            reference[idx],
            reference[idx].to_bits(),
            parallel[idx],
            parallel[idx].to_bits(),
        );
    }
}
