//! Augmented-matrix Horner helpers for the φ-action (ADR-0189 §58.2).
//!
//! The augmented operator is:
//! ```text
//! Ã = [[τA,  v·e₁ᵀ],   dim = (n+p) × (n+p)
//!       [0,   J_p  ]]
//! ```
//! where `e₁` is the first unit vector in R^p and `J_p` is the unit
//! super-diagonal p×p nilpotent: `(J_p·c)[i] = c[i+1]` for `i < p−1`, else 0.
//!
//! One outer Horner iteration computes `T_m((1/s)·Ã) · y_aug` in-place.

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    generator_action::GeneratorAction,
};

// ---------------------------------------------------------------------------
// Augmented matvec
// ---------------------------------------------------------------------------

/// Compute `(1/(j·s)) · Ã · [w_u; w_c]` in-place.
///
/// - `w_u[0..n]` and `w_c[0..p]` are updated (overwritten) with the result.
/// - `av_buf[0..n]` is scratch for the A-matvec output.
///
/// # Invariant
/// `av_buf` must not alias `w_u` or `w_c`.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn aug_matvec_inplace<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    v: &[F],
    w_u: &mut [F],
    w_c: &mut [F],
    tau_over_js: F,
    one_over_js: F,
    av_buf: &mut [F],
) {
    let n = op.dim();
    let p = w_c.len();

    // av_buf ← A · w_u (reads w_u, writes av_buf)
    op.apply_generator(&w_u[..n], &mut av_buf[..n]);

    // Cache w_c[0] before modifying anything
    let c0 = if p > 0 { w_c[0] } else { F::zero() };

    // w_u ← (τ/(j·s)) · av_buf[i]  +  (1/(j·s)) · v[i] · c0
    for i in 0..n {
        w_u[i] = tau_over_js * av_buf[i] + one_over_js * v[i] * c0;
    }

    // w_c ← (1/(j·s)) · J_p · w_c
    // (J_p·c)[i] = c[i+1] for i < p−1, 0 at the bottom
    for i in 0..p.saturating_sub(1) {
        w_c[i] = one_over_js * w_c[i + 1];
    }
    if p > 0 {
        w_c[p - 1] = F::zero();
    }
}

// ---------------------------------------------------------------------------
// One outer Horner iteration
// ---------------------------------------------------------------------------

/// Apply `T_m((1/s)·Ã)` to `y_aug` in-place (one outer Horner sweep).
///
/// After `s` calls the accumulated state approximates `exp(Ã)·z_init`.
///
/// # Arguments
/// - `y_aug` size `n + p`.  Updated in-place.
/// - `w_aug` size `n + p`.  Working buffer; value on entry is irrelevant.
/// - `av_buf` size `n`.  Scratch for each A-matvec.
///
/// # Errors
/// Returns `Ok(())` always; `Result` is kept for forward compatibility
/// (e.g., generators that can fail).
#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::unnecessary_wraps
)]
pub(crate) fn aug_horner_outer<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    v: &[F],
    tau: F,
    s: u32,
    m: u32,
    y_aug: &mut [F],
    w_aug: &mut [F],
    av_buf: &mut [F],
) -> Result<(), SemiflowError> {
    let n = op.dim();
    let p = y_aug.len() - n;
    let s_f64 = f64::from(s);

    // w ← y (start of Horner inner loop: w^(0) = y)
    w_aug.copy_from_slice(y_aug);

    for j in 1..=m {
        let j_f64 = f64::from(j);
        let tau_over_js: F = from_f64(tau.to_f64().unwrap_or(1.0) / (j_f64 * s_f64));
        let one_over_js: F = from_f64(1.0 / (j_f64 * s_f64));

        // split w_aug into [w_u | w_c]
        let (w_u, w_c) = w_aug.split_at_mut(n);
        aug_matvec_inplace(op, v, w_u, &mut w_c[..p], tau_over_js, one_over_js, av_buf);

        // y ← y + w
        for (yi, &wi) in y_aug.iter_mut().zip(w_aug.iter()) {
            *yi += wi;
        }
    }
    Ok(())
}
