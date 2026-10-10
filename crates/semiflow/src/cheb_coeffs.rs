//! Chebyshev coefficients of `e^{−z(1+x)}` on `[−1, 1]` for every `z ≥ 0` (ADR-0205).
//!
//! `e^{−z(1+x)} = Σ_{k≥0}' aₖ Tₖ(x)` with `a₀ = cₖ`, `aₖ = 2(−1)ᵏ cₖ` and
//! `cₖ = e^{−z} Iₖ(z)` (exponentially scaled modified Bessel functions; Abramowitz
//! & Stegun 9.6.34). The scaled values satisfy `c₀ + 2Σ_{k≥1} cₖ = 1` and lie in
//! `[0, 1]`, so they never overflow — unlike `Iₖ(z)` itself, whose overflow at
//! `z ≈ 700` forced the old kernel to split every action into substeps of
//! `z ≤ 200`, making its cost `∝ z` instead of `∝ √z`.
//!
//! * `z ≤ 1`: the convergent power series `Iₖ(z) = Σ (z/2)^{2j+k}/(j!(j+k)!)`.
//! * `z > 1`: Miller's backward recurrence `I_{k−1} = (2k/z) Iₖ + I_{k+1}` from a
//!   start index where `cₖ < e^{−138}`, normalised by the identity above. `Iₖ` is
//!   the minimal solution of the recurrence as `k → ∞`, so the backward sweep is
//!   stable; powers of two rescale it before overflow, exactly.
//!
//! The degree is the smallest `m` whose truncated tail `2Σ_{k>m} cₖ` is at most
//! `tol/4`: since `‖Tₖ(B)‖₂ ≤ 1` for a symmetric `B` with spectrum in `[−1, 1]`,
//! that bounds the truncation error of the matrix series by `tol/4·‖v‖`.

use alloc::vec::Vec;

/// Smallest degree ever returned (safety floor; also the old kernel's).
pub(crate) const MIN_CHEB_DEGREE: usize = 3;

/// Largest single-expansion degree; actions needing more are split into equal
/// substeps (memory cap of the coefficient table: 8 MiB). Reached only for
/// `z ≳ 1.5·10¹⁰`.
pub(crate) const MAX_CHEB_DEGREE: usize = 1 << 20;

/// `−ln` of the relative size at which the backward recurrence starts.
const START_DECAY: f64 = 138.0;

/// Exponentially scaled modified Bessel values `cₖ = e^{−z} Iₖ(z)`, `k = 0..out.len()`.
pub(crate) fn scaled_bessel_into(z: f64, out: &mut [f64]) {
    if out.is_empty() {
        return;
    }
    if z <= 0.0 {
        out.fill(0.0);
        out[0] = 1.0;
    } else if z <= 1.0 {
        series_into(z, out);
    } else {
        miller_into(z, out);
    }
}

/// Power series, `z ∈ (0, 1]`: `(z/2)^k/k! · Σ_j (z²/4)^j / (j!·(k+1)…(k+j))`.
#[allow(clippy::float_cmp)] // exact fixed-point test: the term no longer changes the sum
fn series_into(z: f64, out: &mut [f64]) {
    let em_z = crate::float::SemiflowFloat::libm_exp(-z);
    let hz = 0.5 * z;
    let hz2 = hz * hz;
    let mut lead = 1.0_f64; // (z/2)^k / k!
    #[allow(clippy::cast_precision_loss)] // k, j are small series indices
    for (k, c) in out.iter_mut().enumerate() {
        if k > 0 {
            lead *= hz / k as f64;
        }
        let (mut term, mut sum) = (lead, lead);
        for j in 1..200_u32 {
            term *= hz2 / (f64::from(j) * (f64::from(j) + k as f64));
            let next = sum + term;
            if next == sum {
                break;
            }
            sum = next;
        }
        *c = em_z * sum;
    }
}

/// Backward recurrence from `k_start`, normalised by `c₀ + 2Σ cₖ = 1`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn miller_into(z: f64, out: &mut [f64]) {
    let k_max = out.len() - 1;
    let gauss = libm::sqrt(2.0 * START_DECAY * z);
    let k_start = k_max.max(gauss as usize) + 64;
    // 2^-512: rescale factor applied (exactly) whenever the iterate exceeds 2^512.
    let (big, shrink) = (
        f64::from_bits(0x5FF0_0000_0000_0000),
        f64::from_bits(0x1FF0_0000_0000_0000),
    );
    let (mut j_next, mut j_k) = (0.0_f64, f64::MIN_POSITIVE * 1e10);
    let mut sum = 0.0_f64; // Σ_{k ≥ 1} J_k
    for k in (1..=k_start).rev() {
        if k <= k_max {
            out[k] = j_k;
        }
        sum += j_k;
        let j_prev = (2.0 * k as f64 / z) * j_k + j_next;
        j_next = j_k;
        j_k = j_prev;
        if j_k > big {
            j_k *= shrink;
            j_next *= shrink;
            sum *= shrink;
            for c in out.iter_mut().take(k_max + 1).skip(k.max(1).min(k_max + 1)) {
                *c *= shrink;
            }
        }
    }
    out[0] = j_k;
    let norm = 1.0 / (j_k + 2.0 * sum);
    for c in out.iter_mut() {
        *c *= norm;
    }
}

/// Number of stored coefficients that certainly covers the selected degree.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn table_len(z: f64, tol: f64) -> usize {
    let decay = -libm::log(tol.max(f64::MIN_POSITIVE)) + 8.0;
    let gauss = libm::sqrt(2.0 * decay * z.max(0.0)) + 2.0 * decay + 16.0;
    (gauss as usize).clamp(MIN_CHEB_DEGREE + 2, MAX_CHEB_DEGREE + 2)
}

/// Degree `m` and signed coefficients `a₀..a_m` of `e^{−z(1+x)}` into `coeffs`.
///
/// `m` is the smallest degree `≥ MIN_CHEB_DEGREE` with `2Σ_{k>m} cₖ ≤ tol/4`
/// (capped at `MAX_CHEB_DEGREE`; callers split larger actions into substeps).
pub(crate) fn exp_chebyshev_coefficients(z: f64, tol: f64, coeffs: &mut Vec<f64>) -> usize {
    let len = table_len(z, tol);
    coeffs.clear();
    coeffs.resize(len, 0.0);
    scaled_bessel_into(z, coeffs);
    let target = 0.25 * tol;
    // Tail from the top: the first k (going down) where the tail exceeds the
    // target fixes m = k (terms 0..=k kept, tail Σ_{j>k} within the target).
    let mut tail = 0.0_f64;
    let mut m = MIN_CHEB_DEGREE;
    for k in (MIN_CHEB_DEGREE + 1..len).rev() {
        tail += 2.0 * coeffs[k];
        if tail > target {
            m = k;
            break;
        }
    }
    let m = m.min(MAX_CHEB_DEGREE).min(len - 1);
    coeffs.truncate(m + 1);
    for (k, c) in coeffs.iter_mut().enumerate().skip(1) {
        *c *= if k % 2 == 0 { 2.0 } else { -2.0 };
    }
    m
}

/// Degree [`exp_chebyshev_coefficients`] selects (cost probes; allocates).
pub(crate) fn exp_chebyshev_degree(z: f64, tol: f64) -> usize {
    let mut tmp = Vec::new();
    exp_chebyshev_coefficients(z, tol, &mut tmp)
}

#[cfg(test)]
#[path = "cheb_coeffs_tests.rs"]
mod tests;
