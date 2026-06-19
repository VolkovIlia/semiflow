//! Bit-exact equality gate for parallel `apply` on all 7 stand-alone 1D Chernoff
//! types (ADR-0018 / ADR-0036).
//!
//! Verifies that the multi-thread path returns **bit-for-bit identical** results
//! to the single-thread reference for every combination of:
//!
//! - `N ∈ {1024, 2048, 4096}` (grid nodes; N=1024 is the `MIN_POINTS_PER_THREAD`
//!   threshold — parallel likely falls back to serial, kept as a sanity check),
//! - `n_steps ∈ {1, 4}` (multi-step exposes stale-state bugs),
//! - `thread_count ∈ {1, 2, 4, 8}` (parallel workers vs single-thread reference),
//! - All 7 f64 types: `ShiftChernoff1D`, `DiffusionChernoff`, `Diffusion4thChernoff`,
//!   `Diffusion6thChernoff`, `TruncatedExpDiffusionChernoff`,
//!   `TruncatedExp4thDiffusionChernoff`, `DriftReactionChernoff`.
//!
//! No tolerance — `Vec<f64>` `PartialEq` is byte-exact (compares `u64` bits).
//!
//! Gate: release-blocking (`--features parallel,slow-tests`).
//! See `docs/adr/0036-parallel-1d-chernoff.md`.

#![cfg(all(feature = "parallel", feature = "slow-tests"))]

// Thread-count override hook (exposed under `#[doc(hidden)] pub mod parallel1d`
// when `--features parallel`).
use semiflow_core::{
    chernoff::ApplyChernoffExt, parallel1d::FORCE_THREADS_1D, ChernoffFunction,
    Diffusion4thChernoff, Diffusion6thChernoff, DiffusionChernoff, DriftReactionChernoff, Grid1D,
    GridFn1D, ShiftChernoff1D, TruncatedExp4thDiffusionChernoff, TruncatedExpDiffusionChernoff,
};
use static_assertions::assert_impl_all;

// ---------------------------------------------------------------------------
// Compile-time Send + Sync assertions (ADR-0036 contract)
// ---------------------------------------------------------------------------

assert_impl_all!(ShiftChernoff1D<f64>: Send, Sync);
assert_impl_all!(DiffusionChernoff<f64>: Send, Sync);
assert_impl_all!(Diffusion4thChernoff<f64>: Send, Sync);
assert_impl_all!(Diffusion6thChernoff<f64>: Send, Sync);
assert_impl_all!(TruncatedExpDiffusionChernoff<f64>: Send, Sync);
assert_impl_all!(TruncatedExp4thDiffusionChernoff<f64>: Send, Sync);
assert_impl_all!(DriftReactionChernoff<f64>: Send, Sync);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = 0.0;
const X_MAX: f64 = 1.0;

/// τ for diffusion/shift/drift types — stays within their validity envelopes
/// across N ∈ {1024, 2048, 4096} with the given grid [0, 1].
const TAU: f64 = 1e-3;

/// τ for `TruncatedExpDiffusionChernoff` and `TruncatedExp4thDiffusionChernoff`.
///
/// These types check CFL: for `TruncatedExp` it is `2·τ·a_norm ≤ dx²`; for
/// the 4th-order variant it is `8·τ·a_norm ≤ 3·dx²`.  The binding constraint
/// across N ∈ {1024, 2048, 4096, 8192} is the 4th-order variant at N=8192:
/// dx = 1/8191 ≈ 1.221e-4, dx² ≈ 1.490e-8, need τ ≤ 3·dx²/8 ≈ 5.59e-9.
/// Using τ=5e-9 satisfies both variants at all N in the matrix.
const TAU_TEXP: f64 = 5e-9;

/// `MIN_POINTS_PER_THREAD` from `parallel1d` — hardcoded to avoid pub re-export
/// of a non-hook constant; value is normative in ADR-0036.
const MIN_PPT: usize = 1024;

// ---------------------------------------------------------------------------
// Initial datum: u(x) = sin(π·x) · exp(-x²)
//
// Deterministic; same data is used for both serial and parallel runs.
// ---------------------------------------------------------------------------

fn make_initial(grid: Grid1D<f64>) -> GridFn1D<f64> {
    use std::f64::consts::PI;
    GridFn1D::from_fn(grid, |x| (PI * x).sin() * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Generic bit-equality runner
// ---------------------------------------------------------------------------

/// Run `n_steps` applications of `op.apply_chernoff(tau, &state)` with `FORCE_THREADS_1D`
/// pinned to `thread_count`. Returns the final `values` buffer.
fn run_with_threads<C>(
    op: &C,
    grid: Grid1D<f64>,
    tau: f64,
    n_steps: usize,
    thread_count: usize,
) -> Vec<f64>
where
    C: ChernoffFunction<f64, S = GridFn1D<f64>>,
{
    FORCE_THREADS_1D.with(|c: &std::cell::Cell<Option<usize>>| c.set(Some(thread_count)));
    let mut state = make_initial(grid);
    for _ in 0..n_steps {
        state = op.apply_chernoff(tau, &state).expect("apply ok");
    }
    FORCE_THREADS_1D.with(|c: &std::cell::Cell<Option<usize>>| c.set(None)); // restore
    state.values
}

/// Assert byte-for-byte equality. On mismatch prints all four coordinates and
/// the first divergent index.
fn assert_bit_equal(
    reference: &[f64],
    parallel: &[f64],
    type_name: &str,
    n: usize,
    n_steps: usize,
    thread_count: usize,
) {
    assert_eq!(
        reference.len(),
        parallel.len(),
        "length mismatch: type={type_name} N={n} n_steps={n_steps} k={thread_count}"
    );
    let first_bad = reference
        .iter()
        .zip(parallel.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits());
    if let Some(idx) = first_bad {
        panic!(
            "BIT-DIVERGENCE: type={type_name} N={n} n_steps={n_steps} k={thread_count}\n\
             First bad index: {idx}\n\
             serial[{idx}]   = {} (bits={:064b})\n\
             parallel[{idx}] = {} (bits={:064b})",
            reference[idx],
            reference[idx].to_bits(),
            parallel[idx],
            parallel[idx].to_bits(),
        );
    }
}

/// Run the full (N × n_steps × thread_count) matrix for one operator.
///
/// `tau` is operator-specific: use `TAU` for most types, `TAU_TEXP` for
/// `TruncatedExp*` variants (their CFL condition requires tiny τ at fine grids).
fn run_matrix<C>(op: &C, type_name: &str, tau: f64)
where
    C: ChernoffFunction<f64, S = GridFn1D<f64>>,
{
    // N=1024 is the MIN_POINTS_PER_THREAD threshold (likely serial fallback).
    // N=2048 and N=4096 trigger the parallel path; N=8192 genuinely exercises k=8
    // threads (8192/1024 = 8, so resolve_threads_1d(8192) with FORCE=8 returns 8).
    let _ = MIN_PPT; // documents the source of the 1024 value
    for &n in &[1024usize, 2048, 4096, 8192] {
        let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid ok");
        for &n_steps in &[1usize, 4] {
            let reference = run_with_threads(op, grid, tau, n_steps, 1);
            for &k in &[2usize, 4, 8] {
                let parallel_out = run_with_threads(op, grid, tau, n_steps, k);
                assert_bit_equal(&reference, &parallel_out, type_name, n, n_steps, k);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-type tests — one #[test] per type for clear failure attribution
// ---------------------------------------------------------------------------

#[test]
fn shift_chernoff1d_bit_equal() {
    let n = 4096;
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid ok");
    // ShiftChernoff1D::new(a, b, c, c_norm_bound, grid)
    let op = ShiftChernoff1D::new(
        |_| 1.0_f64, // a(x) = 1
        |_| 0.0_f64, // b(x) = 0
        |_| 0.0_f64, // c(x) = 0
        0.0,
        grid,
    );
    run_matrix(&op, "ShiftChernoff1D", TAU);
}

#[test]
fn diffusion_chernoff_bit_equal() {
    let n = 4096;
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid ok");
    // DiffusionChernoff::new(a, a', a'', a_norm_bound, grid)
    let op = DiffusionChernoff::new(
        |_| 1.0_f64, // a(x) = 1
        |_| 0.0_f64, // a'(x) = 0
        |_| 0.0_f64, // a''(x) = 0
        1.0,
        grid,
    );
    run_matrix(&op, "DiffusionChernoff", TAU);
}

#[test]
fn diffusion4th_chernoff_bit_equal() {
    let n = 4096;
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid ok");
    let op = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    run_matrix(&op, "Diffusion4thChernoff", TAU);
}

#[test]
fn diffusion6th_chernoff_bit_equal() {
    let n = 4096;
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid ok");
    let op = Diffusion6thChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    run_matrix(&op, "Diffusion6thChernoff", TAU);
}

#[test]
fn truncated_exp_diffusion_chernoff_bit_equal() {
    let n = 4096;
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid ok");
    let op = TruncatedExpDiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    run_matrix(&op, "TruncatedExpDiffusionChernoff", TAU_TEXP);
}

#[test]
fn truncated_exp4th_diffusion_chernoff_bit_equal() {
    let n = 4096;
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid ok");
    let op =
        TruncatedExp4thDiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    run_matrix(&op, "TruncatedExp4thDiffusionChernoff", TAU_TEXP);
}

#[test]
fn drift_reaction_chernoff_bit_equal() {
    let n = 4096;
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid ok");
    // DriftReactionChernoff::new(b, c, c_norm_bound, grid)
    let op = DriftReactionChernoff::new(
        |_| 0.0_f64, // b(x) = 0  (pure reaction)
        |_| 0.0_f64, // c(x) = 0
        0.0,
        grid,
    );
    run_matrix(&op, "DriftReactionChernoff", TAU);
}
