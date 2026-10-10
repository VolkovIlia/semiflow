//! Per-M matrix-exponential helpers for `matrix_system` (closed form + Padé-13).
//!
//! Provides `matrix_exp_dispatch`, `mat_vec_mul`, and the
//! per-size backends `matrix_exp_m{1,2,3,4}`.
//!
//! All functions are `pub(super)` — visible only to `matrix_system.rs`.

// `f64` has inherent math methods only when `std` is linked; otherwise they come
// from `num_traits::Float` (libm). Test builds link `std` even without the
// feature (harness, dev-dependencies), hence the `allow`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{error::SemiflowError, float::SemiflowFloat, matrix_pade::mat_exp_pade13};

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Dispatch matrix exponential: closed form for M ∈ {1, 2};
/// Padé[13/13] (Higham 2005, ADR-0125) for M ≥ 3.
pub(super) fn matrix_exp_dispatch<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    match M {
        0 => Ok([[F::zero(); M]; M]),
        1 => Ok(matrix_exp_m1(a)),
        2 => Ok(matrix_exp_m2(a)),
        3 => matrix_exp_m3(a),
        4 => matrix_exp_m4(a),
        _ => mat_exp_pade13(a),
    }
}

// ---------------------------------------------------------------------------
// Matrix-vector multiply
// ---------------------------------------------------------------------------

/// M×M matrix-vector multiply: out = A·v.
#[inline]
pub(super) fn mat_vec_mul<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M], v: &[F; M]) -> [F; M] {
    let mut out = [F::zero(); M];
    for i in 0..M {
        for j in 0..M {
            out[i] += a[i][j] * v[j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Per-M closed-form backends
// ---------------------------------------------------------------------------

/// M=1: scalar exponential.
fn matrix_exp_m1<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M]) -> [[F; M]; M] {
    let mut out = [[F::zero(); M]; M];
    out[0][0] = a[0][0].libm_exp();
    out
}

/// M=2: closed form `e^A = e^μ·[C(δ)·I + S(δ)·(A − μI)]`, `μ = tr/2`, `δ = (tr² − 4 det)/4`.
///
/// `(A − μI)² = δ·I` (Cayley–Hamilton), so `C = cosh √δ`, `S = sinh(√δ)/√δ` for
/// `δ > 0`, and `C = cos √−δ`, `S = sin(√−δ)/√−δ` for complex eigenvalues
/// (`δ < 0`). For `|δ| ≤ 1/16` both are evaluated from their common even power
/// series, which stays accurate as the eigenvalues coalesce.
///
/// Replaces a Putzer formula that (a) took `√|disc|` when the discriminant was
/// NEGATIVE, i.e. used `cosh/sinh` where `cos/sin` belong — wrong for any coupling
/// with complex eigenvalues (e.g. a rotation `[[0, 1], [−1, 0]]` gave `cosh 1`
/// instead of `cos 1`); and (b) divided `(e^λ₁ − e^λ₂)/(λ₁ − λ₂)`, which loses
/// `≈ u/|λ₁−λ₂|` relative accuracy for nearly repeated eigenvalues.
fn matrix_exp_m2<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M]) -> [[F; M]; M] {
    let half = F::from(0.5).unwrap_or(F::one());
    let mu = half * (a[0][0] + a[1][1]);
    // δ = ((a00 − a11)/2)² + a01·a10 — same value as (tr² − 4 det)/4, no cancellation of tr².
    let d = half * (a[0][0] - a[1][1]);
    let delta = d * d + a[0][1] * a[1][0];
    let (c, s) = cosh_sinhc_of_sqrt(delta);
    let e_mu = mu.libm_exp();
    let mut out = [[F::zero(); M]; M];
    out[0][0] = e_mu * (c + s * (a[0][0] - mu));
    out[0][1] = e_mu * (s * a[0][1]);
    out[1][0] = e_mu * (s * a[1][0]);
    out[1][1] = e_mu * (c + s * (a[1][1] - mu));
    out
}

/// `(cosh √δ, sinh(√δ)/√δ)` for real `δ` of either sign (`cos`/`sin` branch for `δ < 0`).
fn cosh_sinhc_of_sqrt<F: SemiflowFloat>(delta: F) -> (F, F) {
    let small = F::from(0.0625).unwrap_or(F::zero());
    if delta.abs() <= small {
        // Σ δᵏ/(2k)! and Σ δᵏ/(2k+1)! to 10 terms: remainder ≤ (1/16)¹⁰/20! < 1e-30.
        let (mut c, mut s) = (F::zero(), F::zero());
        let (mut tc, mut ts) = (F::one(), F::one());
        for k in 1_u32..=10 {
            c += tc;
            s += ts;
            let two_k = F::from(f64::from(2 * k)).unwrap_or(F::one());
            tc = tc * delta / ((two_k - F::one()) * two_k);
            ts = ts * delta / (two_k * (two_k + F::one()));
        }
        (c, s)
    } else if delta > F::zero() {
        let r = delta.sqrt();
        (r.libm_cosh(), r.libm_sinh() / r)
    } else {
        let r = (-delta).sqrt();
        let (sn, cs) = r.libm_sin_cos();
        (cs, sn / r)
    }
}

/// M=3: Padé[13/13] scaling-and-squaring (Higham 2005), shared with `M ≥ 5`.
///
/// The former degree-12 Taylor helper accumulated `Bᵈ/d` instead of `Bᵈ/d!`
/// (the running term was never divided by `d`), so `MatrixDiffusionChernoff<F, 3|4>`
/// exponentiated the wrong series: `e^{0.5}` came out as `1.6931` instead of `1.6487`.
fn matrix_exp_m3<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    mat_exp_pade13(a)
}

/// M=4: same Padé[13/13] backend as M=3.
fn matrix_exp_m4<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    mat_exp_pade13(a)
}
