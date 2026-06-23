//! Bit-exact equality gate for parallel `Strang3D::apply`.
//!
//! Verifies that the multi-thread kernel returns **bit-for-bit identical**
//! results to the single-thread reference for every combination of:
//!
//! - `N ∈ {16, 32, 64}` (grid nodes per axis — chosen so the test runs fast;
//!   N=16 and N=32 force the small-grid serial fallback to be exercised too),
//! - `n_steps ∈ {1, 4}` (time steps; multi-step exposes temp-buffer reuse bugs),
//! - `thread_count ∈ {2, 4, 8}` (parallel workers vs single-thread reference),
//! - Inner operator: `DiffusionChernoff`.
//!
//! No tolerance, no `approx_eq` — `Vec<f64>` comparison is byte-exact via
//! `f64::to_bits`.
//!
//! Failure output includes full `(N, n_steps, op, thread_count)` coordinates
//! and the first byte-divergent element index.
//!
//! Gate: `STRANG3D_PARALLEL_BIT_EQUAL` (mirrors `STRANG2D_PARALLEL_BIT_EQUAL`).
//! Classification: **RELEASE-BLOCKING**.
//!
//! See `docs/adr/0018-parallel-strang2d.md` and the 3D parallel implementation
//! in `src/strang3d_parallel.rs`.

#![cfg(all(feature = "parallel", feature = "slow-tests"))]

// Thread-count override hook (pub in strang3d_parallel).
use semiflow::{
    strang3d_parallel::FORCE_THREADS_3D, ChernoffFunction, ChernoffSemigroup, DiffusionChernoff,
    Grid1D, Grid3D, GridFn3D, Strang3D,
};

// ---------------------------------------------------------------------------
// Domain constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -8.0;
const X_MAX: f64 = 8.0;

/// τ per step for `n_steps=1` at t=0.1.
const TAU_BASE: f64 = 0.1;

// ---------------------------------------------------------------------------
// Send + Sync compile-time sanity check
// ---------------------------------------------------------------------------

/// Verifies that `DiffusionChernoff` and `Strang3D<...>` satisfy the
/// `Send + Sync` bounds required by the parallel `ChernoffFunction` impl.
fn _send_sync_check() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DiffusionChernoff>();
    assert_send_sync::<Strang3D<DiffusionChernoff, DiffusionChernoff, DiffusionChernoff>>();
}

// ---------------------------------------------------------------------------
// Generic runner
// ---------------------------------------------------------------------------

/// Run `n_steps` applications of `phi3d.apply_chernoff(tau)` starting from `f0`,
/// with the thread count pinned to `thread_count` via `FORCE_THREADS_3D`.
///
/// Returns the final `GridFn3D::values`.
#[allow(clippy::cast_precision_loss)]
fn run_with_threads<X, Y, Z>(
    phi3d: Strang3D<X, Y, Z>,
    f0: &GridFn3D,
    tau: f64,
    n_steps: usize,
    thread_count: usize,
) -> Vec<f64>
where
    X: ChernoffFunction<S = semiflow::GridFn1D> + Clone + Send + Sync,
    Y: ChernoffFunction<S = semiflow::GridFn1D> + Clone + Send + Sync,
    Z: ChernoffFunction<S = semiflow::GridFn1D> + Clone + Send + Sync,
{
    FORCE_THREADS_3D.with(|c| c.set(Some(thread_count)));
    let semi = ChernoffSemigroup::new(phi3d, n_steps).expect("n_steps >= 1");
    let result = semi.evolve(tau * n_steps as f64, f0).expect("evolve ok");
    FORCE_THREADS_3D.with(|c| c.set(None)); // restore default
    result.values
}

// ---------------------------------------------------------------------------
// Initial datum: u0(x, y, z) = exp(-(x² + y² + z²))
// ---------------------------------------------------------------------------

fn make_initial(grid: Grid3D) -> GridFn3D {
    GridFn3D::from_fn(grid, |x, y, z| (-(x * x + y * y + z * z)).exp())
}

// ---------------------------------------------------------------------------
// DiffusionChernoff sweep
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
#[test]
fn bit_equal_diffusion_chernoff_3d() {
    for &n in &[16usize, 32, 64] {
        let gx = Grid1D::new(X_MIN, X_MAX, n).expect("grid x");
        let gy = Grid1D::new(X_MIN, X_MAX, n).expect("grid y");
        let gz = Grid1D::new(X_MIN, X_MAX, n).expect("grid z");
        let grid = Grid3D::new(gx, gy, gz).expect("grid 3d");
        let f0 = make_initial(grid);

        let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
        let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
        let cz = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gz);

        for &n_steps in &[1usize, 4] {
            let tau = TAU_BASE / n_steps as f64;
            let phi = Strang3D::new(cx.clone(), cy.clone(), cz.clone());

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
// Speedup gate (INFORMATIONAL — warn only, no panic)
// ---------------------------------------------------------------------------

#[test]
fn speedup_gate_informational_3d() {
    let n = 32usize;
    let gx = Grid1D::new(X_MIN, X_MAX, n).expect("grid x");
    let gy = Grid1D::new(X_MIN, X_MAX, n).expect("grid y");
    let gz = Grid1D::new(X_MIN, X_MAX, n).expect("grid z");
    let grid = Grid3D::new(gx, gy, gz).expect("grid 3d");
    let f0 = make_initial(grid);

    let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
    let cz = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gz);
    let phi = Strang3D::new(cx, cy, cz);

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
        "[speedup_gate_3d] N={n}: serial={:.3}s parallel(4t)={:.3}s ratio={:.2}x",
        t_serial.as_secs_f64(),
        t_parallel.as_secs_f64(),
        ratio,
    );
    // INFORMATIONAL — no panic.
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
