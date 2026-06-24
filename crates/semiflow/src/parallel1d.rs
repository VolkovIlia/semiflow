//! Pointwise-parallel evaluator for 1D Chernoff types.
//!
//! Bit-equality invariants (mirrors ADR-0018 contract):
//! - Deterministic ceiling-division chunking (`n.div_ceil(n_threads)`).
//! - Disjoint `chunks_mut` writes (no overlapping output slots).
//! - No cross-thread floating-point reductions.
//! - No atomics in the hot path (only `std::sync::Mutex` for first-error capture).
//! - f64-only (the `parallel_eval` signature takes `Vec<f64>`).
//!
//! Cutoff controlled by `min_points_per_thread()` (default 2048 per ADR-0036
//! Amendment 1). Override via `REMIZOV_PARALLEL_THRESHOLD` env var at process
//! startup (clamped to `[64, 1_000_000]`) for bench/test purposes.

use alloc::{vec, vec::Vec};

use crate::error::SemiflowError;

/// Minimum total points to engage parallel evaluation in the 1D Chernoff hot loop.
///
/// Default 2048 mirrors ADR-0036 Amendment 1 production setting. Override via
/// `REMIZOV_PARALLEL_THRESHOLD=<usize>` env var for bench/test (clamped to
/// `[64, 1_000_000]`). The value is read once at first call and cached for the
/// process lifetime.
///
/// ## Implementation note (ADR-0041 AC-4)
///
/// Uses `AtomicUsize` with a sentinel value of `usize::MAX` rather than
/// `OnceLock<usize>` to guarantee allocation-free reads after the first call.
/// `OnceLock` internally calls `std::env::var` which allocates a `CString` key
/// on the first lookup; the `AtomicUsize` path avoids that allocation entirely.
///
/// # Note
/// Setting this value too low (< 256) causes thread-scope overhead to dominate
/// for short 1D vectors; bench-gate any configuration change.
// called from chernoff.rs under #[cfg(feature = "parallel")] and from diffusion/shift1d callers
#[allow(dead_code)]
#[cfg(feature = "std")]
pub(crate) fn min_points_per_thread() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    // Sentinel: usize::MAX means "not yet initialised".
    static THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);
    let v = THRESHOLD.load(Ordering::Relaxed);
    if v != usize::MAX {
        return v;
    }
    // First call (rare): read env var, clamp, store.  Races are benign — at
    // worst two threads read the env var simultaneously and both store the
    // same value.
    let parsed = read_threshold_env();
    THRESHOLD.store(parsed, Ordering::Relaxed);
    parsed
}

/// Read `REMIZOV_PARALLEL_THRESHOLD` from the environment.
///
/// Separated into its own `#[cold]` function so the hot `min_points_per_thread()`
/// path remains a single `AtomicUsize::load` in steady state.
///
/// This function may allocate (via `std::env::var`'s `CString` key conversion),
/// but it is only called ONCE per process lifetime — during the first invocation
/// of `min_points_per_thread()`, which `ChernoffSemigroup::new` pre-triggers
/// at construction time (outside any `allocation_counter::measure()` scope).
#[cfg(feature = "std")]
#[cold]
fn read_threshold_env() -> usize {
    std::env::var("REMIZOV_PARALLEL_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map_or(2048, |v| v.clamp(64, 1_000_000))
}

/// `no_std` fallback: always returns the compile-time default (2048).
/// Environment variable override is a `std`-only feature.
// used when feature = "std" is absent; rustc dead_code lint fires in std builds
#[allow(dead_code)]
#[cfg(not(feature = "std"))]
pub(crate) fn min_points_per_thread() -> usize {
    MIN_POINTS_PER_THREAD
}

/// Sentinel constant kept for backward compatibility with `strang2d_parallel.rs`
/// and `strang3d_parallel.rs` which reference `MIN_POINTS_PER_THREAD` via their
/// own `MIN_ROWS_PER_THREAD` / `MIN_PENCILS_PER_THREAD` constants.
// referenced by tests and indirectly via strang*_parallel constants
#[allow(dead_code)]
pub(crate) const MIN_POINTS_PER_THREAD: usize = 2048;

#[cfg(feature = "parallel")]
use std::cell::Cell;

#[cfg(feature = "parallel")]
thread_local! {
    /// Test-only override for thread count. Set via `with(|c| c.set(Some(k)))`
    /// in the bit-equality test harness. `None` → use `available_parallelism()`.
    pub static FORCE_THREADS_1D: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(feature = "parallel")]
pub(crate) fn available_parallelism_cached() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    // Atomic sentinel: 0 = not yet read, >0 = cached value.
    // `available_parallelism()` always returns >= 1, so 0 is a safe sentinel.
    // Atomic load/store is allocation-free; used instead of OnceLock to avoid
    // any internal allocations during the init path (ADR-0041 AC-4 gate).
    static CACHED: AtomicUsize = AtomicUsize::new(0);
    let v = CACHED.load(Ordering::Relaxed);
    if v != 0 {
        return v;
    }
    let nproc = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    // Relaxed store: races are benign (worst case: read nproc twice on first call).
    CACHED.store(nproc, Ordering::Relaxed);
    nproc
}

#[cfg(feature = "parallel")]
fn resolve_threads_1d(n: usize) -> usize {
    let raw = FORCE_THREADS_1D
        .with(Cell::get)
        .unwrap_or_else(available_parallelism_cached);
    raw.min(n / min_points_per_thread()).max(1)
}

/// Fill `out[i] = eval(i)` for all `i in 0..out.len()` in-place.
///
/// This is the allocation-free inner primitive used by the Wave 1 scratch-pool
/// path (`apply_into` overrides). Bit-equality contract is identical to
/// [`parallel_eval`]: a single `eval(i)` per slot, no FP reordering.
///
/// Under `--features parallel`: multi-threaded when `n >= 2 * min_points_per_thread()`.
// called from diffusion.rs, diffusion4.rs, diffusion6.rs, drift_reaction.rs, shift1d.rs
#[allow(dead_code)]
pub(crate) fn parallel_eval_into<E>(out: &mut [f64], eval: E) -> Result<(), SemiflowError>
where
    E: Fn(usize) -> Result<f64, SemiflowError> + Sync,
{
    #[cfg(feature = "parallel")]
    {
        let n = out.len();
        let n_threads = resolve_threads_1d(n);
        if n_threads > 1 {
            return parallel_eval_into_threaded(out, n_threads, eval);
        }
    }
    serial_eval_into(out, eval)
}

/// Evaluate `eval(i)` for `i in 0..n` and collect into a fresh `Vec<f64>`.
///
/// Under `--features parallel`: chunks the index range across
/// `available_parallelism()` threads via `std::thread::scope` when
/// `n >= 2 * min_points_per_thread()`; otherwise serial. The threshold is
/// controlled by `min_points_per_thread()` (default 2048, overridable via
/// `REMIZOV_PARALLEL_THRESHOLD` env var — see ADR-0036 Amendment 1).
///
/// Bit-equality: the f64 written to `out[i]` is the result of a single
/// `eval(i)` call on a single thread; no FP rearrangement.
// called from diffusion.rs, shift1d.rs; lint fires when feature = "parallel" is off
#[allow(dead_code)]
pub(crate) fn parallel_eval<E>(n: usize, eval: E) -> Result<Vec<f64>, SemiflowError>
where
    E: Fn(usize) -> Result<f64, SemiflowError> + Sync,
{
    let mut out = vec![0.0_f64; n];
    parallel_eval_into(&mut out, eval)?;
    Ok(out)
}

fn serial_eval_into<E>(out: &mut [f64], eval: E) -> Result<(), SemiflowError>
where
    E: Fn(usize) -> Result<f64, SemiflowError>,
{
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = eval(i)?;
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn parallel_eval_into_threaded<E>(
    out: &mut [f64],
    n_threads: usize,
    eval: E,
) -> Result<(), SemiflowError>
where
    E: Fn(usize) -> Result<f64, SemiflowError> + Sync,
{
    let n = out.len();
    let chunk_size = n.div_ceil(n_threads);
    let error: std::sync::Mutex<Option<SemiflowError>> = std::sync::Mutex::new(None);
    let eval_ref = &eval;
    let error_ref = &error;
    std::thread::scope(|s| {
        for (k, slot_chunk) in out.chunks_mut(chunk_size).enumerate() {
            let start = k * chunk_size;
            s.spawn(move || {
                for (j, slot) in slot_chunk.iter_mut().enumerate() {
                    match eval_ref(start + j) {
                        Ok(v) => *slot = v,
                        Err(e) => {
                            let mut g = error_ref.lock().unwrap();
                            if g.is_none() {
                                *g = Some(e);
                            }
                            return;
                        }
                    }
                }
            });
        }
    });
    match error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn serial_passthrough() {
        let n = 32;
        let result = parallel_eval(n, |i| Ok(i as f64 * 2.0)).unwrap();
        let expected: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn error_propagation() {
        let result = parallel_eval(16, |i| {
            if i == 5 {
                Err(SemiflowError::DomainViolation {
                    what: "synthetic-test-error",
                    value: i as f64,
                })
            } else {
                Ok(i as f64)
            }
        });
        assert!(result.is_err());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_error_propagation() {
        // Force the multi-thread path: N >= 2 * MIN_POINTS_PER_THREAD and
        // FORCE_THREADS_1D = 4. Without the force, resolve_threads_1d(4096)
        // would cap at available_parallelism() which may be 1 in CI.
        FORCE_THREADS_1D.with(|c| c.set(Some(4)));
        let n = 4096;
        let result = parallel_eval(n, |i| {
            if i == 3000 {
                Err(SemiflowError::DomainViolation {
                    what: "synthetic-parallel-test-error",
                    value: i as f64,
                })
            } else {
                Ok(i as f64)
            }
        });
        FORCE_THREADS_1D.with(|c| c.set(None));
        assert!(
            result.is_err(),
            "parallel_eval did not propagate error from multi-thread path"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_bit_equal_to_serial() {
        let n = 4096;
        FORCE_THREADS_1D.with(|c| c.set(Some(1)));
        let serial = parallel_eval(n, |i| Ok((i as f64).sin())).unwrap();
        for threads in [2, 4, 8] {
            FORCE_THREADS_1D.with(|c| c.set(Some(threads)));
            let parallel = parallel_eval(n, |i| Ok((i as f64).sin())).unwrap();
            assert_eq!(
                serial, parallel,
                "byte-identity broken at threads={threads}"
            );
        }
        FORCE_THREADS_1D.with(|c| c.set(None));
    }

    /// Verify the threshold parsing logic used by `min_points_per_thread`.
    ///
    /// Directly tests the clamp-and-parse contract (ADR-0036 Amendment 1) without
    /// relying on the `OnceLock` singleton (which cannot be reset per-test).
    #[test]
    fn parallel_threshold_parse_contract() {
        // Helper that mirrors the OnceLock init logic but does not use the singleton.
        fn parse_threshold(raw: Option<&str>) -> usize {
            raw.and_then(|s| s.parse::<usize>().ok())
                .map_or(2048, |v| v.clamp(64, 1_000_000))
        }

        // Default when env var absent.
        assert_eq!(parse_threshold(None), 2048, "default must be 2048");

        // Below minimum clamps to 64.
        assert_eq!(parse_threshold(Some("10")), 64);

        // Above maximum clamps to 1_000_000.
        assert_eq!(parse_threshold(Some("9999999")), 1_000_000);

        // In-range value passes through unchanged.
        assert_eq!(parse_threshold(Some("256")), 256);
        assert_eq!(parse_threshold(Some("2048")), 2048);

        // Non-numeric value falls back to default.
        assert_eq!(parse_threshold(Some("not_a_number")), 2048);

        // Actual singleton is in [64, 1_000_000].
        let t = min_points_per_thread();
        assert!(
            (64..=1_000_000).contains(&t),
            "min_points_per_thread() = {t} out of range [64, 1_000_000]"
        );
    }
}
