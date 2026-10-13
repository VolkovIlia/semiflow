//! Diagonal shifts of a CSR matrix: the trace shift of the Taylor `expmv` kernel
//! (Al-Mohy & Higham 2011, §3.1; ADR-0205) and the Gershgorin-midpoint shift of
//! the augmented φ operator (ADR-0210).

use crate::float::SemiflowFloat;

/// Trace shift `(μ, ‖A − μI‖_∞)`, `μ = tr(A)/n` (Al-Mohy & Higham 2011, §3.1):
/// `e^{−τA} = e^{−τμ}·e^{−τ(A−μI)}`, and the Taylor cost scales with the norm of
/// the shifted matrix — half of `‖A‖_∞` for a diffusion stencil. Kept only when
/// it lowers the bound; otherwise `(0, ‖A‖_∞)` (and the kernel's bits are those
/// of the unshifted one).
#[allow(clippy::cast_precision_loss)] // n < 2^53
pub(crate) fn trace_shift<F: SemiflowFloat>(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[F],
    norm_inf: f64,
) -> (f64, f64) {
    if n == 0 || !norm_inf.is_finite() {
        return (0.0, norm_inf);
    }
    let to64 = |v: &F| v.to_f64().unwrap_or(f64::NAN);
    let row = |i: usize| (row_ptr[i]..row_ptr[i + 1]).map(move |k| (col_idx[k] as usize, &vals[k]));
    let diag = |i: usize| -> f64 { row(i).filter(|&(j, _)| j == i).map(|(_, v)| to64(v)).sum() };
    let mu = (0..n).map(diag).sum::<f64>() / n as f64;
    let mut norm = 0.0_f64;
    for i in 0..n {
        let off: f64 = row(i)
            .filter(|&(j, _)| j != i)
            .map(|(_, v)| libm::fabs(to64(v)))
            .sum();
        let r = off + libm::fabs(diag(i) - mu);
        if r.is_nan() || r > norm {
            norm = r; // NaN propagates and fails the comparison below
        }
    }
    if mu.is_finite() && norm < norm_inf {
        (mu, norm)
    } else {
        (0.0, norm_inf)
    }
}

/// Gershgorin-midpoint shift `(μ, ‖A − μI‖_∞)` from per-row `(dᵢ, rᵢ)` (diagonal,
/// off-diagonal absolute sum): `μ = (maxᵢ(dᵢ + rᵢ) + minᵢ(dᵢ − rᵢ))/2` minimises
/// `maxᵢ(|dᵢ − μ| + rᵢ)` over all scalars (the centre of the Gershgorin hull).
/// Unlike the trace mean it stays centred when the diagonal varies (a lumped mass
/// of contrast 50 scales the rows by 50). `None` when `μ = 0`, a value is not
/// finite, or the shifted norm is not below `norm_inf`.
pub(crate) fn midpoint_shift(
    rows: impl Iterator<Item = (f64, f64)> + Clone,
    norm_inf: f64,
) -> Option<(f64, f64)> {
    // `f64::max`/`min` would skip a NaN row: reject it, as documented.
    if rows.clone().any(|(d, r)| d.is_nan() || r.is_nan()) {
        return None;
    }
    let (hi, lo) = rows
        .clone()
        .fold((f64::NEG_INFINITY, f64::INFINITY), |(hi, lo), (d, r)| {
            (hi.max(d + r), lo.min(d - r))
        });
    let mu = 0.5 * hi + 0.5 * lo;
    let norm = rows.fold(0.0_f64, |acc, (d, r)| acc.max(libm::fabs(d - mu) + r));
    (mu != 0.0 && mu.is_finite() && norm.is_finite() && norm < norm_inf).then_some((mu, norm))
}

/// `(dᵢ, rᵢ)` of row `i` of a CSR matrix (duplicate diagonal entries summed).
pub(crate) fn csr_row<F: SemiflowFloat>(
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[F],
    i: usize,
) -> (f64, f64) {
    let (mut d, mut r) = (0.0_f64, 0.0_f64);
    for k in row_ptr[i]..row_ptr[i + 1] {
        let v = vals[k].to_f64().unwrap_or(f64::NAN);
        if col_idx[k] as usize == i {
            d += v;
        } else {
            r += libm::fabs(v);
        }
    }
    (d, r)
}
