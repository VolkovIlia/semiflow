//! HDR percentile snapshot for L-gate latency harness.
//!
//! Implements NIST nearest-rank percentile (ASTM E29-13 §6):
//! `rank = ceil((pct/100) * N)`, 1-indexed, so `sorted[rank - 1]`.
//!
//! Named "HDR" by convention (every benchmarking literature reader expects
//! "HDR" for tail-latency snapshots); the inner implementation is array-backed
//! nearest-rank, not log-bucket histograms. Using a 4th direct dep
//! (`hdrhistogram` crate) would violate the ≤3 dep budget (ADR-0068 §Rationale).
//!
//! ## Usage
//!
//! ```rust
//! use semiflow::HdrSnapshot;
//! let mut snap = HdrSnapshot::new(16);
//! for ns in [10, 20, 30, 40, 50] { snap.record(ns); }
//! assert_eq!(snap.percentile(50.0), 30); // ceil(2.5) = 3, sorted[2] = 30
//! assert_eq!(snap.count(), 5);
//! ```

// pct_clamped ∈ [0,100] and n >= 1 ensure raw.ceil() >= 0; cast to usize is safe.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

extern crate alloc;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use num_traits::Float;

/// Array-backed nearest-rank percentile snapshot (NIST ASTM E29-13 §6).
///
/// Lazily sorts on first `percentile` call after any `record`.
/// `record` is O(1); `percentile` is O(N log N) on first call after
/// new records, O(1) on subsequent calls without new records.
#[derive(Debug, Clone)]
pub struct HdrSnapshot {
    samples: Vec<i64>,
    is_sorted: bool,
}

impl HdrSnapshot {
    /// Create a new, empty snapshot with `capacity` pre-allocated slots.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            is_sorted: true,
        }
    }

    /// Record one latency sample in nanoseconds.
    pub fn record(&mut self, ns: i64) {
        self.samples.push(ns);
        self.is_sorted = false;
    }

    /// Compute the NIST nearest-rank percentile.
    ///
    /// - `pct` is clamped to `[0.0, 100.0]`.
    /// - Returns `0` for an empty snapshot.
    /// - Formula: `rank = ceil((pct / 100) * N)`, 1-indexed → `sorted[rank - 1]`.
    pub fn percentile(&mut self, pct: f64) -> i64 {
        let n = self.samples.len();
        if n == 0 {
            return 0;
        }
        self.ensure_sorted();
        let pct_clamped = pct.clamp(0.0, 100.0);
        let rank = compute_nist_rank(pct_clamped, n);
        // rank is in [1, n] by invariant; subtract 1 for 0-based index.
        self.samples[rank - 1]
    }

    /// Number of samples recorded.
    #[must_use]
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Clear all samples. After this call `count() == 0`.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.is_sorted = true;
    }

    /// Lazily sort the sample buffer.
    fn ensure_sorted(&mut self) {
        if !self.is_sorted {
            self.samples.sort_unstable();
            self.is_sorted = true;
        }
    }
}

/// NIST nearest-rank: `rank = ceil((pct / 100) * n)`, clamped to `[1, n]`.
///
/// Pure function; extracted to stay under the 50-line function limit.
fn compute_nist_rank(pct_clamped: f64, n: usize) -> usize {
    // pct_clamped ∈ [0, 100], n >= 1 (caller pre-checked).
    let raw = (pct_clamped / 100.0) * (n as f64);
    // ceil(0.0) = 0 for pct = 0 — clamp to 1 so we always return a sample.
    let rank = raw.ceil() as usize;
    rank.clamp(1, n)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::HdrSnapshot;

    #[test]
    fn empty_returns_zero() {
        let mut snap = HdrSnapshot::new(0);
        assert_eq!(snap.percentile(50.0), 0);
        assert_eq!(snap.count(), 0);
    }

    #[test]
    fn single_sample_all_percentiles() {
        let mut snap = HdrSnapshot::new(1);
        snap.record(42);
        assert_eq!(snap.percentile(50.0), 42);
        assert_eq!(snap.percentile(99.9), 42);
        assert_eq!(snap.percentile(0.0), 42);
        assert_eq!(snap.percentile(100.0), 42);
    }

    #[test]
    fn five_samples_p50_and_p99() {
        // sorted: [10, 20, 30, 40, 50]
        // p50: ceil(0.5 * 5) = ceil(2.5) = 3 → sorted[2] = 30
        // p99: ceil(0.99 * 5) = ceil(4.95) = 5 → sorted[4] = 50
        let mut snap = HdrSnapshot::new(8);
        for v in [10, 20, 30, 40, 50] {
            snap.record(v);
        }
        assert_eq!(snap.percentile(50.0), 30);
        assert_eq!(snap.percentile(99.0), 50);
    }

    #[test]
    fn unsorted_input_sorted_lazily() {
        // record(5), record(1), record(3) → sorted: [1, 3, 5]
        // p50: ceil(0.5 * 3) = ceil(1.5) = 2 → sorted[1] = 3
        let mut snap = HdrSnapshot::new(4);
        snap.record(5);
        snap.record(1);
        snap.record(3);
        assert_eq!(snap.percentile(50.0), 3);
    }

    #[test]
    fn clear_resets_count() {
        let mut snap = HdrSnapshot::new(4);
        snap.record(10);
        snap.record(20);
        assert_eq!(snap.count(), 2);
        snap.clear();
        assert_eq!(snap.count(), 0);
        assert_eq!(snap.percentile(50.0), 0);
    }

    #[test]
    fn p0_clamp_returns_first_sample() {
        let mut snap = HdrSnapshot::new(4);
        for v in [1, 2, 3, 4] {
            snap.record(v);
        }
        // pct=0 → rank=ceil(0)=0 → clamped to 1 → sorted[0]=1
        assert_eq!(snap.percentile(0.0), 1);
    }
}
