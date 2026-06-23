//! Bit-exact equality gate for parallel `Strang2D::apply` (ADR-0018).
//!
//! Verifies that the multi-thread kernel returns **bit-for-bit identical**
//! results to the single-thread reference for every combination of:
//!
//! - `N ∈ {64, 128, 256, 512}` (grid nodes per axis),
//! - `n_steps ∈ {1, 4}` (time steps; multi-step exposes temp-buffer reuse bugs),
//! - `thread_count ∈ {2, 4, 8}` (parallel workers vs single-thread reference),
//! - Inner operator: `DiffusionChernoff` AND `Diffusion4thChernoff`.
//!
//! No tolerance, no `approx_eq` — Vec<f64> `PartialEq` is byte-exact.
//!
//! Failure output includes the full (N, `n_steps`, op, `thread_count`) coordinates
//! and the index of the first byte-divergent element.
//!
//! Gate: `STRANG2D_PARALLEL_BIT_EQUAL` in `contracts/semiflow-core.properties.yaml`.
//! Classification: **RELEASE-BLOCKING**.
//!
//! See `docs/adr/0018-parallel-strang2d.md`.

#![cfg(all(feature = "parallel", feature = "slow-tests"))]

// Thread-count override hook (pub(crate) in strang2d_parallel).
use semiflow::{
    strang2d_parallel::FORCE_THREADS, ChernoffFunction, ChernoffSemigroup, Diffusion4thChernoff,
    DiffusionChernoff, Grid1D, Grid2D, GridFn2D, Strang2D,
};

// ---------------------------------------------------------------------------
// Domain constants — identical to heat_2d_oracle_4th.rs
// ---------------------------------------------------------------------------

const X_MIN: f64 = -15.0;
const X_MAX: f64 = 15.0;

/// τ per step for `n_steps=1` at t=0.1.
const TAU_BASE: f64 = 0.1;

// ---------------------------------------------------------------------------
// Send + Sync sanity check (early compile-time signal)
// ---------------------------------------------------------------------------

/// Compile-time check: `DiffusionChernoff`, `Diffusion4thChernoff`, and
/// `Strang2D<DiffusionChernoff, DiffusionChernoff>` implement `Send + Sync`.
///
/// A future field that is not `Send + Sync` will break the build here before
/// it causes a runtime deadlock or compiler error at the `thread::scope` site.
fn _send_sync_check() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DiffusionChernoff>();
    assert_send_sync::<Diffusion4thChernoff>();
    assert_send_sync::<Strang2D<DiffusionChernoff, DiffusionChernoff>>();
}

// ---------------------------------------------------------------------------
// Generic runner
// ---------------------------------------------------------------------------

/// Run `n_steps` applications of `phi2d.apply_chernoff(tau)` starting from `f0`,
/// with the thread count pinned to `thread_count` via `FORCE_THREADS`.
///
/// Returns the final `GridFn2D::values`.
// n_steps ≤ 4 in all test cases — well within f64 mantissa range.
#[allow(clippy::cast_precision_loss)]
fn run_with_threads<X, Y>(
    phi2d: Strang2D<X, Y>,
    f0: &GridFn2D,
    tau: f64,
    n_steps: usize,
    thread_count: usize,
) -> Vec<f64>
where
    X: ChernoffFunction<S = semiflow::GridFn1D> + Clone + Send + Sync,
    Y: ChernoffFunction<S = semiflow::GridFn1D> + Clone + Send + Sync,
{
    FORCE_THREADS.with(|c| c.set(Some(thread_count)));
    let semi = ChernoffSemigroup::new(phi2d, n_steps).expect("n_steps >= 1");
    let result = semi.evolve(tau * n_steps as f64, f0).expect("evolve ok");
    FORCE_THREADS.with(|c| c.set(None)); // restore default
    result.values
}

// ---------------------------------------------------------------------------
// Initial datum: u0(x, y) = exp(-(x² + y²))
// ---------------------------------------------------------------------------

fn make_initial(grid: Grid2D) -> GridFn2D {
    GridFn2D::from_fn(grid, |x, y| (-(x * x + y * y)).exp())
}

// ---------------------------------------------------------------------------
// DiffusionChernoff sweep
// ---------------------------------------------------------------------------

// n_steps ∈ {1, 4} — well within f64 mantissa range.
#[allow(clippy::cast_precision_loss)]
#[test]
fn bit_equal_diffusion_chernoff() {
    for &n in &[64usize, 128, 256, 512] {
        let gx = Grid1D::new(X_MIN, X_MAX, n).expect("grid x");
        let gy = Grid1D::new(X_MIN, X_MAX, n).expect("grid y");
        let grid = Grid2D::new(gx, gy);
        let f0 = make_initial(grid);

        let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
        let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

        for &n_steps in &[1usize, 4] {
            let tau = TAU_BASE / n_steps as f64;
            let phi = Strang2D::new(cx.clone(), cy.clone());

            // Reference: single thread.
            let reference = run_with_threads(phi.clone(), &f0, tau, n_steps, 1);

            // Parallel runs: thread_count ∈ {2, 4, 8}.
            for &k in &[2usize, 4, 8] {
                let parallel = run_with_threads(phi.clone(), &f0, tau, n_steps, k);
                assert_bit_equal(&reference, &parallel, "DiffusionChernoff", n, n_steps, k);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Diffusion4thChernoff sweep
// ---------------------------------------------------------------------------

// n_steps ∈ {1, 4} — well within f64 mantissa range.
#[allow(clippy::cast_precision_loss)]
#[test]
fn bit_equal_diffusion4th_chernoff() {
    for &n in &[64usize, 128, 256, 512] {
        let gx = Grid1D::new(X_MIN, X_MAX, n).expect("grid x");
        let gy = Grid1D::new(X_MIN, X_MAX, n).expect("grid y");
        let grid = Grid2D::new(gx, gy);
        let f0 = make_initial(grid);

        let cx = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
        let cy = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

        for &n_steps in &[1usize, 4] {
            let tau = TAU_BASE / n_steps as f64;
            let phi = Strang2D::new(cx.clone(), cy.clone());

            // Reference: single thread.
            let reference = run_with_threads(phi.clone(), &f0, tau, n_steps, 1);

            // Parallel runs.
            for &k in &[2usize, 4, 8] {
                let parallel = run_with_threads(phi.clone(), &f0, tau, n_steps, k);
                assert_bit_equal(&reference, &parallel, "Diffusion4thChernoff", n, n_steps, k);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Speedup gate (INFORMATIONAL — warn only, no panic)
// ---------------------------------------------------------------------------

#[test]
fn speedup_gate_informational() {
    // Small N=64 so the test is fast; real speedup only emerges at large N.
    // This test just checks the parallel path runs without error.
    let n = 128usize;
    let gx = Grid1D::new(X_MIN, X_MAX, n).expect("grid x");
    let gy = Grid1D::new(X_MIN, X_MAX, n).expect("grid y");
    let grid = Grid2D::new(gx, gy);
    let f0 = make_initial(grid);

    let cx = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
    let phi = Strang2D::new(cx, cy);

    let t_serial = {
        let start = std::time::Instant::now();
        let _ = run_with_threads(phi.clone(), &f0, TAU_BASE, 4, 1);
        start.elapsed()
    };
    let t_parallel = {
        let start = std::time::Instant::now();
        let _ = run_with_threads(phi, &f0, TAU_BASE, 4, 4);
        start.elapsed()
    };

    let ratio = t_serial.as_secs_f64() / t_parallel.as_secs_f64();
    eprintln!(
        "[speedup_gate] N={n}: serial={:.3}s parallel(4t)={:.3}s ratio={:.2}x",
        t_serial.as_secs_f64(),
        t_parallel.as_secs_f64(),
        ratio
    );
    // INFORMATIONAL — no panic. Production speedup verification requires N≥1600.
}

// ---------------------------------------------------------------------------
// Assertion helper
// ---------------------------------------------------------------------------

/// Assert byte-for-byte equality between `reference` and `parallel`.
///
/// On mismatch, prints all four coordinates and the first divergent index.
fn assert_bit_equal(
    reference: &[f64],
    parallel: &[f64],
    op_name: &str,
    n: usize,
    n_steps: usize,
    thread_count: usize,
) {
    assert_eq!(
        reference.len(),
        parallel.len(),
        "length mismatch: op={op_name} N={n} n_steps={n_steps} k={thread_count}"
    );

    let first_bad = reference
        .iter()
        .zip(parallel.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits());

    if let Some(idx) = first_bad {
        panic!(
            "BIT-DIVERGENCE: op={op_name} N={n} n_steps={n_steps} k={thread_count}\n\
             First bad index: {idx}\n\
             serial[{idx}]   = {:?} (bits={:064b})\n\
             parallel[{idx}] = {:?} (bits={:064b})",
            reference[idx],
            reference[idx].to_bits(),
            parallel[idx],
            parallel[idx].to_bits(),
        );
    }
}
