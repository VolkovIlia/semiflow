//! Trace shift of the augmented φ operator (ADR-0210, math §67).
//!
//! The φ-actions evaluate `e^{B̃_η}` for the augmented operator
//! `B̃_η = [[τG, ηW], [0, J_p]]` by `s` Taylor sweeps whose cost is driven by a
//! norm bound (`τ‖G‖ + 2`, ADR-0198/0202). Since `μI` commutes with every block,
//! `e^{B̃_η} = e^{μ}·e^{B̃_η − μI}` for any scalar `μ`; with `μ = τ·ḡ` and `ḡ` the
//! centre of `G`'s Gershgorin hull, the shifted operator is bounded by
//! `max(τ‖G − ḡI‖_∞, |μ|) + 2` (top rows: shifted generator plus the η-scaled
//! coupling `≤ 1`; bottom rows: `J_p − μI`) — about half of `τ‖G‖ + 2` for a
//! diffusion generator, whose Gershgorin hull is `[−‖G‖, 0]`. The shift is used only when the
//! generator reports one ([`GeneratorAction::diagonal_shift`]) AND it lowers the
//! number of generator applications `s·m`; otherwise the unshifted arithmetic
//! runs verbatim.

use crate::{
    error::SemiflowError,
    expmv::select_s_m,
    float::{from_f64, SemiflowFloat},
    generator_action::GeneratorAction,
    phi_action::{phi_schedule, PHI_NORM_TIGHTEN},
    phi_action_helpers::{Coupling, SweepPlan, MAX_COUPLING_COLS},
};

/// Per-term scalars of one shifted Horner step `j`, each divided by `j·s`:
/// `τ/(js)`, `1/(js)`, `μ/(js)`.
pub(crate) struct Scales<F> {
    pub(crate) tau: F,
    pub(crate) one: F,
    pub(crate) mu: F,
}

/// `(1/(j·s))·(B̃_η − μI)·[w_u; w_c]` in place (the shifted twin of
/// `aug_matvec_inplace`):
/// `w_u ← (τ/(js))·G w_u − (μ/(js))·w_u + Σ_j c_j·col_j`,
/// `w_c[i] ← (1/(js))·w_c[i+1] − (μ/(js))·w_c[i]` for `i < p − 1`,
/// `w_c[p−1] ← −(μ/(js))·w_c[p−1]`.
pub(crate) fn aug_matvec_shifted<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    coupling: &Coupling<'_, F>,
    w_u: &mut [F],
    w_c: &mut [F],
    k: &Scales<F>,
    av_buf: &mut [F],
) {
    let n = op.dim();
    let p = w_c.len();
    op.apply_generator(&w_u[..n], &mut av_buf[..n]);
    // Coupling scalars from the OLD w_c (coef·w_c first: η·(1/η) stays exact).
    let mut scalars = [F::zero(); MAX_COUPLING_COLS];
    for (j, scalar) in scalars.iter_mut().enumerate().take(coupling.cols.len()) {
        *scalar = coupling.coef[j] * w_c[j] * k.one;
    }
    for i in 0..n {
        let mut acc = k.tau * av_buf[i] - k.mu * w_u[i];
        for (col, &scalar) in coupling.cols.iter().zip(&scalars) {
            acc += scalar * col[i];
        }
        w_u[i] = acc;
    }
    // Ascending i reads the old w_c[i + 1] before it is overwritten.
    for i in 0..p.saturating_sub(1) {
        w_c[i] = k.one * w_c[i + 1] - k.mu * w_c[i];
    }
    if p > 0 {
        w_c[p - 1] = -(k.mu * w_c[p - 1]);
    }
}

/// The sweep schedule for `op` and `tau`: the unshifted `(s, m)` of
/// [`phi_schedule`], or the shifted one when it needs strictly fewer generator
/// applications.
///
/// # Errors
/// As [`phi_schedule`] (non-finite or negative `τ` / norm bound).
pub(crate) fn sweep_plan<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    tau: F,
) -> Result<SweepPlan<F>, SemiflowError> {
    let norm = op.norm_bound().to_f64().unwrap_or(f64::NAN);
    let tau64 = tau.to_f64().unwrap_or(f64::NAN);
    let (s, m) = phi_schedule(norm, tau64)?;
    let unshifted = SweepPlan {
        s,
        m,
        mu: F::zero(),
    };
    let Some((gbar, norm_shifted)) = op.diagonal_shift() else {
        return Ok(unshifted);
    };
    let mu = tau * gbar;
    let mu64 = mu.to_f64().unwrap_or(f64::NAN);
    let bound = (tau64 * norm_shifted.to_f64().unwrap_or(f64::NAN)).max(libm::fabs(mu64)) + 2.0;
    if mu64 == 0.0 || !mu64.is_finite() || !bound.is_finite() {
        return Ok(unshifted);
    }
    match select_s_m(bound * PHI_NORM_TIGHTEN, 1.0) {
        Ok((s2, m2)) if u64::from(s2) * u64::from(m2) < u64::from(s) * u64::from(m) => {
            Ok(SweepPlan { s: s2, m: m2, mu })
        }
        _ => Ok(unshifted),
    }
}

/// `(s, m)` the φ-actions actually use for `op` and `tau` — shift-aware, unlike
/// [`crate::phi_action::phi_cost_probe`], which takes only a norm. Each φ sweep
/// costs `s·m` generator applications. `(0, 0)` for an invalid `τ` or bound.
#[must_use]
pub fn phi_sweep_probe<F: SemiflowFloat, Op: GeneratorAction<F>>(op: &Op, tau: F) -> (u32, u32) {
    sweep_plan(op, tau).map_or((0, 0), |p| (p.s, p.m))
}

/// `(μ, ‖G − μI‖_∞)` for `G = −Q` from the Gershgorin-midpoint shift of the CSR
/// matrix `Q` (`G`'s rows are `(−dᵢ, rᵢ)`), or `None` when it gives no reduction
/// below `norm_inf` (ADR-0210).
pub(crate) fn negated_csr_shift<F: SemiflowFloat>(
    n: usize,
    rows: (&[usize], &[u32], &[F]),
    norm_inf: F,
) -> Option<(F, F)> {
    let (rp, ci, vals) = rows;
    let g_rows = (0..n).map(move |i| {
        let (d, r) = crate::csr_shift::csr_row(rp, ci, vals, i);
        (-d, r)
    });
    let (mu, norm) = crate::csr_shift::midpoint_shift(g_rows, norm_inf.to_f64()?)?;
    Some((from_f64(mu), from_f64(norm)))
}
