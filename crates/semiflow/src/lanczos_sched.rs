//! Lanczos schedule and tridiagonal exponential for `graph_krylov` (ADR-0205).
//!
//! **Schedule.** For a symmetric `L ⪰ 0` with spectrum in `[0, λ]`, Hochbruck &
//! Lubich (SIAM J. Numer. Anal. 34, 1997, Thm. 2) bound the error of the `m`-step
//! Lanczos approximation of `e^{−hL}v` (`‖v‖ = 1`, `ρ = λ/4`, `x = ρh`) by
//!
//! * `10·e^{−m²/(5x)}`               for `√(4x) ≤ m ≤ 2x`,
//! * `(10/x)·e^{−x}·(e·x/m)^m`        for `m ≥ 2x`.
//!
//! Both bounds come from Saad's lemma `err ≤ 2·E_{m−1}(h)`, `E_d(h)` the best
//! degree-`d` uniform approximation error of `e^{−hλ}` on `[0, λ]`, which is
//! nondecreasing in `h`; a bound proven at a step `h` therefore holds for every
//! shorter step. The same lemma with Chebyshev interpolation (`|e^{−μ}⁽ᵐ⁾| ≤ 1`
//! on `[0, 4x]`) gives the small-step bound
//!
//! * `4·xᵐ/m!`                         for every `x`,
//!
//! which is sharper for short steps and the only one available for `m ≤ 2`.
//! The kernel splits `τ` into `s` equal steps and picks `(s, m ≤ m_max)` minimising
//! `s·m` such that the bound of one step is `≤ tol/s` (so `s` steps stay within
//! `tol·‖v‖`; `e^{−hL}` is a contraction). The old kernel took the steps from the
//! Taylor backward-error radii `θ_m` (`λh ≤ 1.09` at `m = 18`) — a property of the
//! truncated Taylor series, not of Lanczos — and paid a dense Padé-13 per step;
//! the bound above admits `λh ≈ 7.4` at `m = 18`, `≈ 24` at `m = 30` (tol 1e-10).
//!
//! **Tridiagonal exponential.** `e^{−hT_m}e₁` from the eigen-decomposition of the
//! symmetric tridiagonal `T_m` (implicit QL with Wilkinson shifts, EISPACK `tql2`),
//! computed in `f64` whatever `F` is.

use alloc::vec;
use alloc::vec::Vec;

/// Largest supported Krylov dimension (basis memory `m·n`).
pub(crate) const MAX_LANCZOS_M: usize = 512;

/// `ln` of the Hochbruck–Lubich bound for `m` steps at `x = ρh > 0`, or `+∞`
/// where neither regime of the theorem applies (`m < √(4x)`).
fn ln_hl_bound(m: usize, x: f64) -> f64 {
    #[allow(clippy::cast_precision_loss)] // m ≤ MAX_LANCZOS_M
    let mf = m as f64;
    if mf >= 2.0 * x {
        libm::log(10.0 / x) - x + mf * (1.0 + libm::log(x / mf))
    } else if mf * mf >= 4.0 * x {
        libm::log(10.0) - mf * mf / (5.0 * x)
    } else {
        f64::INFINITY
    }
}

/// Largest `x = ρh` with `4xᵐ/m! ≤ delta` (`ln_fact = ln m!`).
fn max_rho_h_interp(m: usize, ln_fact: f64, delta: f64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let mf = m as f64;
    libm::exp((libm::log(delta) + ln_fact - libm::log(4.0)) / mf)
}

/// Largest `x = ρh` admissible for `m` steps at per-step tolerance `delta`: the
/// larger of the two bounds' admissible steps (each bound covers all shorter
/// steps, see the module notes).
fn max_rho_h(m: usize, ln_fact: f64, delta: f64) -> f64 {
    let hl = max_rho_h_hl(m, delta);
    let interp = max_rho_h_interp(m, ln_fact, delta);
    if hl.is_nan() || interp.is_nan() {
        return f64::NAN;
    }
    hl.max(interp)
}

/// Largest `x = ρh` with Hochbruck–Lubich bound `≤ delta` for `m` steps.
fn max_rho_h_hl(m: usize, delta: f64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let mf = m as f64;
    let ln_d = libm::log(delta);
    // Regime m ≤ 2x: closed form x_B = m²/(5 ln(10/δ)), valid on [m/2, m²/4].
    let x_b = mf * mf / (5.0 * (libm::log(10.0) - ln_d));
    if x_b >= 0.5 * mf {
        return x_b;
    }
    // Regime m ≥ 2x: the bound increases on (0, m/2]; bisect ln f = ln δ.
    let (mut lo, mut hi) = (0.0_f64, 0.5 * mf);
    if ln_hl_bound(m, hi) <= ln_d {
        return hi;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if ln_hl_bound(m, mid) <= ln_d {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Substeps `s ≥ 1` such that each step's bound is `≤ tol/s` for this `m`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn substeps_for(m: usize, ln_fact: f64, rho_tau: f64, tol: f64) -> Option<u64> {
    let mut s = 1_u64;
    for _ in 0..64 {
        let x = max_rho_h(m, ln_fact, tol / s as f64);
        if x.is_nan() || x <= 0.0 {
            return None;
        }
        let need = libm::ceil(rho_tau / x).max(1.0);
        if need > 1e15 {
            return None;
        }
        let need = need as u64;
        if need <= s {
            return Some(s);
        }
        s = need;
    }
    Some(s)
}

/// `(s, m)` minimising `s·m` over `1 ≤ m ≤ min(m_max, n)` (rigorous a-priori schedule).
pub(crate) fn lanczos_schedule(
    lambda_max: f64,
    tau: f64,
    tol: f64,
    m_max: usize,
    n: usize,
) -> (u64, usize) {
    let m_cap = m_max.clamp(1, MAX_LANCZOS_M).min(n.max(1));
    let rho_tau = 0.25 * lambda_max * tau;
    if rho_tau.is_nan() || rho_tau <= 0.0 {
        return (1, 1);
    }
    let mut best: Option<(u64, usize, u128)> = None;
    let mut ln_fact = 0.0_f64; // ln m!
    for m in 1..=m_cap {
        #[allow(clippy::cast_precision_loss)]
        {
            ln_fact += libm::log(m as f64);
        }
        let Some(s) = substeps_for(m, ln_fact, rho_tau, tol) else {
            continue;
        };
        #[allow(clippy::cast_possible_truncation)]
        let cost = u128::from(s) * m as u128;
        if best.map_or(true, |(_, _, c)| cost < c) {
            best = Some((s, m, cost));
        }
    }
    best.map_or((u64::MAX, m_cap), |(s, m, _)| (s, m))
}

/// `y = e^{−h·T}e₁` for the symmetric tridiagonal `T` (diagonal `alpha`, off-diagonal
/// `beta[k]` between rows `k` and `k+1`), via its eigen-decomposition `T = ZΘZᵀ`.
///
/// Returns `None` if the QL iteration fails to converge (never observed; callers
/// report it as `ConvergenceFailed`).
#[allow(clippy::many_single_char_names)] // d, e, z, w, h, m: standard tql2 notation
pub(crate) fn tridiag_exp_e1(alpha: &[f64], beta: &[f64], h: f64) -> Option<Vec<f64>> {
    let m = alpha.len();
    let mut d = alpha.to_vec();
    let mut e = vec![0.0; m];
    e[..m.saturating_sub(1)].copy_from_slice(&beta[..m.saturating_sub(1)]);
    let mut z = vec![0.0; m * m];
    for i in 0..m {
        z[i * m + i] = 1.0;
    }
    tql2(&mut d, &mut e, &mut z)?;
    // y = Z · diag(e^{−hθ}) · Zᵀ e₁ ;  (Zᵀ e₁)_i = Z[0][i]
    let w: Vec<f64> = (0..m).map(|i| libm::exp(-h * d[i]) * z[i]).collect();
    Some(
        (0..m)
            .map(|j| (0..m).map(|i| z[j * m + i] * w[i]).sum())
            .collect(),
    )
}

/// Implicit QL with shifts for a symmetric tridiagonal matrix (EISPACK `tql2`,
/// Numerical Recipes `tqli`). On exit `d` holds the eigenvalues and column `i` of
/// the row-major `z` the eigenvector of `d[i]`.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn tql2(d: &mut [f64], e: &mut [f64], z: &mut [f64]) -> Option<()> {
    let n = d.len();
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = libm::fabs(d[m]) + libm::fabs(d[m + 1]);
                if libm::fabs(e[m]) <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 64 {
                return None;
            }
            ql_sweep(d, e, z, l, m);
        }
    }
    Some(())
}

/// One implicit QL sweep on the unreduced block `l..=m`.
#[allow(clippy::many_single_char_names)]
fn ql_sweep(d: &mut [f64], e: &mut [f64], z: &mut [f64], l: usize, m: usize) {
    let n = d.len();
    let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
    let mut r = libm::hypot(g, 1.0);
    g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r } else { -r });
    let (mut s, mut c, mut p) = (1.0_f64, 1.0_f64, 0.0_f64);
    let mut i = m;
    while i > l {
        i -= 1;
        let f = s * e[i];
        let b = c * e[i];
        r = libm::hypot(f, g);
        e[i + 1] = r;
        if r == 0.0 {
            d[i + 1] -= p;
            e[m] = 0.0;
            return;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2.0 * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b;
        for k in 0..n {
            let zf = z[k * n + i + 1];
            z[k * n + i + 1] = s * z[k * n + i] + c * zf;
            z[k * n + i] = c * z[k * n + i] - s * zf;
        }
    }
    d[l] -= p;
    e[l] = g;
    e[m] = 0.0;
}
