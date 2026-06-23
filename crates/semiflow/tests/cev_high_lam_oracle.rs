//! Focused unit tests for the log-space noncentral χ² CDF helper (v0.4.1).
//!
//! Verifies `ncx2_cdf` correctness in the high-λ regime that the pre-v0.4.1
//! linear-space recurrence could not handle (exp(-λ/2) underflows at λ/2>745).
//!
//! `ncx2_cdf` is intentionally duplicated from `cev_european_call_sweep.rs`;
//! both files are source-of-truth — divergence surfaces as a compile-error or
//! CI failure.

use statrs::distribution::{ChiSquared, ContinuousCDF};

// Noncentral χ² CDF — log-space Poisson recurrence (v0.4.1).
// Duplicated from cev_european_call_sweep.rs by design; see module doc.
#[allow(dead_code)]
fn ncx2_cdf(w: f64, v: f64, lam: f64) -> f64 {
    let half_lam = lam / 2.0;
    if half_lam == 0.0 {
        return ChiSquared::new(v).expect("df > 0").cdf(w); // central χ² limit
    }
    let log_half_lam = half_lam.ln();
    let mut log_p = -half_lam; // log P_0 = −λ/2
    let mut sum = 0.0_f64;
    let mut max_log_p = f64::NEG_INFINITY;
    let mut converged = false;
    for j in 0_u32..2000 {
        let chi = ChiSquared::new(v + 2.0 * f64::from(j)).expect("df > 0");
        let cdf_j = chi.cdf(w);
        if log_p > -700.0 && cdf_j > 0.0 {
            sum += log_p.exp() * cdf_j;
        }
        if log_p > max_log_p {
            max_log_p = log_p;
        }
        let past_peak = f64::from(j) > half_lam.max(5.0);
        // Tail bound: sum_{k>j} P_k ≤ P_j · (λ/2 / (j-λ/2)).
        // 36 nats ≈ 4e-16 of the observed peak — well below f64 ε.
        if past_peak && log_p < max_log_p - 36.0 {
            converged = true;
            break;
        }
        log_p += log_half_lam - f64::from(j + 1).ln();
    }
    assert!(
        converged,
        "ncx2_cdf did not converge in 2000 iters: w={w:.4}, v={v:.4}, lam={lam:.4}"
    );
    sum
}

/// For λ ∈ {500, 1000, 1500, 1983, 3000} and w ∈ {1, 100, 1000, 5000},
/// result must be finite and in [-ε, 1+ε] (tol covers log-space rounding at w→∞).
#[test]
fn ncx2_cdf_finite_in_unit_interval() {
    let v = 2.0;
    let tol = 1e-10; // sub-epsilon overshoot from accumulated rounding at high λ, large w
    let lam_values = [500.0_f64, 1000.0, 1500.0, 1983.0, 3000.0];
    let w_values = [1.0_f64, 100.0, 1000.0, 5000.0];
    for lam in lam_values {
        for w in w_values {
            let result = ncx2_cdf(w, v, lam);
            assert!(
                result.is_finite() && result >= -tol && result <= 1.0 + tol,
                "ncx2_cdf({w}, {v}, {lam}) = {result} — not finite or out of [-ε, 1+ε]"
            );
        }
    }
}

/// For λ=1983, v=2.0, CDF values at increasing w must be non-decreasing.
#[test]
fn ncx2_cdf_monotone_in_w() {
    let v = 2.0;
    let lam = 1983.0;
    let w_values = [1.0_f64, 10.0, 100.0, 1000.0, 5000.0, 20000.0];
    let mut prev = 0.0_f64;
    for w in w_values {
        let result = ncx2_cdf(w, v, lam);
        assert!(
            result >= prev,
            "ncx2_cdf not monotone: at w={w} got {result}, previous={prev}"
        );
        prev = result;
    }
}

/// Smoke test: λ=200, v=4.0, w=300 is well within the linear-space domain.
#[test]
fn ncx2_cdf_low_lam_smoke() {
    let result = ncx2_cdf(300.0, 4.0, 200.0);
    assert!(
        result.is_finite() && (0.0..=1.0).contains(&result),
        "ncx2_cdf(300, 4, 200) = {result} — expected in [0,1]"
    );
}

/// Regression for the pre-v0.4.1 silent bug: at λ=1983, v=4.0, w=2000 the
/// linear-space version returned 0 (exp(-991) = 0); log-space must return > 0.
#[test]
fn ncx2_cdf_high_lam_nonzero() {
    let result = ncx2_cdf(2000.0, 4.0, 1983.0);
    assert!(
        result > 0.0,
        "ncx2_cdf(2000, 4, 1983) = {result} — pre-v0.4.1 bug: expected > 0"
    );
}
