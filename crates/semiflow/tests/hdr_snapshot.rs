//! v2.6 — `HdrSnapshot` API contract tests.
//!
//! Validates the NIST nearest-rank percentile semantics (ASTM E29-13 §6)
//! through the public `HdrSnapshot` surface. Complements the unit tests
//! embedded in `hdr.rs` with proptest coverage of the rank formula.
//!
//! Gate: no `slow-tests` gate — all cases are fast (≤ 200 samples each).

// Integration test/bench: allows for numerical patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use proptest::prelude::*;
use semiflow_core::HdrSnapshot;

// ---------------------------------------------------------------------------
// Deterministic unit tests
// ---------------------------------------------------------------------------

#[test]
fn percentile_on_empty_is_zero() {
    let mut s = HdrSnapshot::new(0);
    assert_eq!(s.percentile(50.0), 0);
    assert_eq!(s.percentile(0.0), 0);
    assert_eq!(s.percentile(100.0), 0);
    assert_eq!(s.count(), 0);
}

/// NIST nearest-rank on 5 elements: `ceil(0.5 * 5) = ceil(2.5) = 3` → `sorted[2] = 30`.
#[test]
fn percentile_50_on_sorted_5_elements_is_median_by_nist_rank() {
    let mut s = HdrSnapshot::new(5);
    for v in [10, 20, 30, 40, 50] {
        s.record(v);
    }
    // NIST: rank = ceil(0.5 * 5) = ceil(2.5) = 3 → sorted[2] = 30
    assert_eq!(s.percentile(50.0), 30);
}

/// NIST rank at p99.9 on 1000 elements.
///
/// Due to IEEE 754 float arithmetic: `99.9 / 100.0 * 1000.0 = 999.0000000000001`,
/// so `ceil(...) = 1000` → `sorted[999] = 1000`.
/// This is the correct NIST nearest-rank result per the implementation spec.
#[test]
fn percentile_99_9_on_1000_samples_returns_last() {
    let mut s = HdrSnapshot::new(1000);
    for v in 1_i64..=1000 {
        s.record(v);
    }
    // IEEE 754: 99.9/100.0 * 1000.0 = 999.0000000000001 → ceil = 1000 → sorted[999] = 1000
    assert_eq!(s.percentile(99.9), 1000);
}

/// `pct < 0` must clamp to `pct = 0` → rank = ceil(0) = 0 → clamped to 1 → smallest.
#[test]
fn percentile_clamps_negative_to_smallest() {
    let mut s = HdrSnapshot::new(3);
    s.record(10);
    s.record(20);
    s.record(30);
    // pct < 0 → clamp to 0 → rank = ceil(0 * 3) = 0 → clamped to 1 → sorted[0] = 10
    assert_eq!(s.percentile(-5.0), 10);
    assert_eq!(s.percentile(-100.0), 10);
}

/// `pct > 100` must clamp to `pct = 100` → rank = n → sorted[n-1] = max.
#[test]
fn percentile_over_100_clamps_to_max() {
    let mut s = HdrSnapshot::new(3);
    s.record(10);
    s.record(20);
    s.record(30);
    assert_eq!(s.percentile(150.0), 30);
    assert_eq!(s.percentile(1000.0), 30);
}

#[test]
fn clear_resets_count_and_returns_zero() {
    let mut s = HdrSnapshot::new(4);
    s.record(10);
    s.record(20);
    assert_eq!(s.count(), 2);
    s.clear();
    assert_eq!(s.count(), 0);
    assert_eq!(s.percentile(50.0), 0);
}

/// Unsorted input is lazily sorted on first `percentile` call.
#[test]
fn unsorted_input_sorts_lazily() {
    let mut s = HdrSnapshot::new(3);
    s.record(5);
    s.record(1);
    s.record(3);
    // sorted: [1, 3, 5]
    // p50: ceil(0.5 * 3) = ceil(1.5) = 2 → sorted[1] = 3
    assert_eq!(s.percentile(50.0), 3);
    // p99.9: ceil(0.999 * 3) = ceil(2.997) = 3 → sorted[2] = 5
    assert_eq!(s.percentile(99.9), 5);
}

// ---------------------------------------------------------------------------
// proptest: matches naive NIST nearest-rank reference (200 cases)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// For any non-empty sample vector and any `pct ∈ [0, 100]`, `HdrSnapshot::percentile`
    /// must equal the naive NIST nearest-rank formula applied to the sorted samples.
    ///
    /// Naive reference: `rank = ceil(pct/100 * N)`, clamped to `[1, N]`, 1-indexed
    /// → `sorted[rank - 1]`.
    #[test]
    fn percentile_matches_naive_nist_nearest_rank(
        samples in proptest::collection::vec(0i64..1_000_000, 1..=200),
        pct in 0.0f64..=100.0f64,
    ) {
        let mut s = HdrSnapshot::new(samples.len());
        for &v in &samples {
            s.record(v);
        }
        let got = s.percentile(pct);

        // Naive NIST nearest-rank reference.
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        let n = sorted.len() as f64;
        let rank_raw = (pct / 100.0 * n).ceil() as usize;
        let rank = rank_raw.max(1).min(sorted.len());
        let expected = sorted[rank - 1];

        prop_assert_eq!(
            got,
            expected,
            "NIST mismatch: pct={}, samples={:?}, got={}, expected={}",
            pct,
            samples,
            got,
            expected
        );
    }
}
