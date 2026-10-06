//! Augmented-matrix Horner helpers for the φ-action (ADR-0189 §58.2).
//!
//! The augmented operator is (ADR-0202 §62.4, η-scaled):
//! ```text
//! B̃_η = [[τG,  η·W],   dim = (n+p) × (n+p)
//!         [0,   J_p ]]
//! ```
//! where `W` has `q ≤ p` non-zero columns (column `j` = `cols[j]·coef[j]`, the
//! remaining columns are zero) and `J_p` is the unit super-diagonal p×p
//! nilpotent: `(J_p·c)[i] = c[i+1]` for `i < p−1`, else 0.  The coupling
//! magnitude is normalised by the power of two `η` (see [`eta_scaling`]) so it
//! never drives the substep count.
//!
//! One outer Horner iteration computes `T_m((1/s)·B̃_η) · y_aug` in-place.

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    generator_action::GeneratorAction,
};

/// Maximum number of coupling columns (`PHI_MAX`).
const MAX_COUPLING_COLS: usize = 3;

/// Clamp for the η exponent (§62.4): keeps `2^{±e}` a normal `f64`.
const ETA_EXP_CLAMP: i32 = 1000;

/// Coupling block of the augmented matrix: column `j` is `coef[j]·cols[j]`;
/// columns `≥ cols.len()` are zero.  `coef` carries `η` and any `τ^k` weight.
pub(crate) struct Coupling<'a, F> {
    /// Coupling columns (each of length `n`).
    pub cols: &'a [&'a [F]],
    /// Per-column scalar factor (same length as `cols`).
    pub coef: &'a [F],
}

/// `⌈log₂ x⌉` for finite `x > 0`, from the IEEE exponent field (exact for powers of two).
fn ceil_log2(x: f64) -> i32 {
    let bits = x.to_bits();
    let field = i32::try_from((bits >> 52) & 0x7ff).unwrap_or(0);
    if field == 0 {
        return -ETA_EXP_CLAMP; // zero / subnormal: clamped anyway
    }
    let exp = field - 1023;
    if bits & ((1_u64 << 52) - 1) == 0 {
        exp
    } else {
        exp + 1
    }
}

/// Exact `2^e` as `F`, or `None` if it is not a finite non-zero `F` (e.g. `f32` overflow).
fn exact_pow2<F: SemiflowFloat>(e: i32) -> Option<F> {
    let biased = u64::try_from(1023 + e).ok()?;
    let value: F = from_f64(f64::from_bits(biased << 52));
    (value.is_finite() && value != F::zero()).then_some(value)
}

/// Power-of-two coupling scale (§62.4): returns `(η, 1/η)` with `η = 2^{−e}`,
/// `e = ⌈log₂ max_mag⌉` clamped to `[−1000, 1000]`; `(1, 1)` if `max_mag` is
/// zero or non-finite.  Powers of two are exact, so `φ(2^j v) = 2^j φ(v)` bit for bit.
pub(crate) fn eta_scaling<F: SemiflowFloat>(max_mag: f64) -> (F, F) {
    if !(max_mag > 0.0 && max_mag.is_finite()) {
        return (F::one(), F::one());
    }
    let e = ceil_log2(max_mag).clamp(-ETA_EXP_CLAMP, ETA_EXP_CLAMP);
    match (exact_pow2::<F>(-e), exact_pow2::<F>(e)) {
        (Some(eta), Some(inv)) => (eta, inv),
        _ => (F::one(), F::one()),
    }
}

// ---------------------------------------------------------------------------
// Augmented matvec
// ---------------------------------------------------------------------------

/// Compute `(1/(j·s)) · B̃_η · [w_u; w_c]` in-place.
///
/// - `w_u[0..n]` and `w_c[0..p]` are updated (overwritten) with the result.
/// - `av_buf[0..n]` is scratch for the G-matvec output.
///
/// # Invariant
/// `av_buf` must not alias `w_u` or `w_c`.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn aug_matvec_inplace<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    coupling: &Coupling<'_, F>,
    w_u: &mut [F],
    w_c: &mut [F],
    tau_over_js: F,
    one_over_js: F,
    av_buf: &mut [F],
) {
    let n = op.dim();
    let p = w_c.len();

    // av_buf ← G · w_u (reads w_u, writes av_buf)
    op.apply_generator(&w_u[..n], &mut av_buf[..n]);

    // Per-column scalar cj = (coef_j·w_c[j])/(j·s), hoisted out of the i-loop.
    // coef carries η and w_c carries 1/η, so that pair is multiplied first
    // (exact power-of-two scalings keep φ(2^j v) = 2^j φ(v) bitwise).
    debug_assert!(coupling.cols.len() <= MAX_COUPLING_COLS);
    let mut scalars = [F::zero(); MAX_COUPLING_COLS];
    for (j, scalar) in scalars.iter_mut().enumerate().take(coupling.cols.len()) {
        *scalar = coupling.coef[j] * w_c[j] * one_over_js;
    }

    // w_u ← (τ/(j·s))·av_buf + Σ_j cj·cols_j   (w_c is only overwritten below.)
    for i in 0..n {
        let mut acc = tau_over_js * av_buf[i];
        for (col, &scalar) in coupling.cols.iter().zip(&scalars) {
            acc += scalar * col[i];
        }
        w_u[i] = acc;
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

/// Apply `T_m((1/s)·B̃_η)` to `y_aug` in-place (one outer Horner sweep).
///
/// After `s` calls the accumulated state approximates `exp(B̃_η)·z_init`.
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
    coupling: &Coupling<'_, F>,
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
        aug_matvec_inplace(
            op,
            coupling,
            w_u,
            &mut w_c[..p],
            tau_over_js,
            one_over_js,
            av_buf,
        );

        // y ← y + w
        for (yi, &wi) in y_aug.iter_mut().zip(w_aug.iter()) {
            *yi += wi;
        }
    }
    Ok(())
}

/// Run `s` outer Horner sweeps `y_aug ← T_m(B̃_η/s)·y_aug`, returning early on error.
///
/// # Errors
/// Propagates the first error from [`aug_horner_outer`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sweeps<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    coupling: &Coupling<'_, F>,
    tau: F,
    s: u32,
    m: u32,
    y_aug: &mut [F],
    w_aug: &mut [F],
    av_buf: &mut [F],
) -> Result<(), SemiflowError> {
    for _ in 0..s {
        aug_horner_outer(op, coupling, tau, s, m, y_aug, w_aug, av_buf)?;
    }
    Ok(())
}
