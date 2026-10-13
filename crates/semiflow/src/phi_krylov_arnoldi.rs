//! Arnoldi and Expokit-style substep control for [`crate::phi_krylov`] (ADR-0211).
//!
//! `expv` advances `y ← e^{t·B}y` over `t ∈ [0, 1]` for the augmented operator `B`
//! (Sidje, ACM TOMS 24 (1998), `expv`): per substep an `m`-step Arnoldi basis of
//! `B` from the current `y`, the first column of the exponential of the
//! `(m+2)`-augmented Hessenberg (whose extra rows carry the `φ₁`/`φ₂` corrector
//! terms used as the local error estimate), step rejection with the same basis, and
//! the step-size heuristic `Δ ← γ·Δ·(Δ·tol/err)^{1/m}` rounded to two significant
//! digits. Only `e^{tH}e₁` is ever used, so it is computed as a vector action
//! (scaled truncated Taylor with the Al-Mohy–Higham `(s, m)` of the generator-level
//! `expmv`): `O(s·m·(m+2)²)` per trial step instead of a dense matrix exponential.

use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError,
    expmv::select_s_m,
    float::{from_f64, SemiflowFloat},
    generator_action::GeneratorAction,
    phi_action_helpers::{aug_matvec_inplace, Coupling},
    phi_krylov::{axpy, dot, norm2, KrylovPhiOptions, KrylovPhiReport},
    scratch::ScratchPool,
};

/// Safety factors of `expv` (Sidje 1998).
const GAMMA: f64 = 0.9;
const DELTA: f64 = 1.2;
/// Happy-breakdown threshold on `h_{j+1,j}` (relative to `‖B‖` and `β`).
const BREAKDOWN: f64 = 1e-12;
/// Rejections allowed per substep.
const MAX_REJECT: usize = 10;

/// `B = [[τG, W_η], [0, J_p]]` as an operator on `n + p` vectors.
pub(crate) struct Augmented<'a, F: SemiflowFloat, Op> {
    pub(crate) op: &'a Op,
    pub(crate) coupling: &'a Coupling<'a, F>,
    pub(crate) tau: F,
    pub(crate) n: usize,
}

impl<F: SemiflowFloat, Op: GeneratorAction<F>> Augmented<'_, F, Op> {
    /// `out ← B·x` (`av` scratch of length `n`).
    fn apply(&self, x: &[F], out: &mut [F], av: &mut [F]) {
        out.copy_from_slice(x);
        let (w_u, w_c) = out.split_at_mut(self.n);
        aug_matvec_inplace(self.op, self.coupling, w_u, w_c, self.tau, F::one(), av);
    }
}

/// Round `x > 0` up to two significant digits (Expokit's step rounding).
fn round2(x: f64) -> f64 {
    let s = libm::pow(10.0, libm::floor(libm::log10(x)) - 1.0);
    libm::ceil(x / s) * s
}

/// Arnoldi basis and the `(m+2)×(m+2)` augmented Hessenberg of one substep.
struct Basis<F> {
    v: Vec<Vec<F>>,
    h: Vec<f64>,
    /// Krylov dimension actually built (`< m` on a happy breakdown).
    mb: usize,
    /// `0` on a happy breakdown (exact), `2` otherwise (corrector rows present).
    k1: usize,
    /// `‖B·v_{m+1}‖₂` for the `φ₂` error term.
    avnorm: f64,
}

/// `e^{t·H}e₁` for the leading `mx × mx` block `H` of the stored `dim × dim`
/// augmented Hessenberg (row `r` has entries in columns `≥ r − 1`), by `s` scaled
/// Taylor steps of degree `deg` on the vector, `(s, deg)` from [`select_s_m`] on the
/// 1-norm of `tH`.
///
/// # Errors
/// `DomainViolation` for a non-finite `t·‖H‖₁` (from [`select_s_m`]).
#[allow(clippy::many_single_char_names)] // h, t, s, y, w, k, f: Taylor notation
pub(crate) fn hess_exp_e1(
    h: &[f64],
    dim: usize,
    mx: usize,
    t: f64,
) -> Result<Vec<f64>, SemiflowError> {
    let norm = (0..mx)
        .map(|c| (0..mx).map(|r| libm::fabs(h[r * dim + c])).sum::<f64>())
        .fold(0.0, f64::max);
    let (s, deg) = select_s_m(norm, t)?;
    let ts = t / f64::from(s);
    let (mut y, mut w, mut hw) = (vec![0.0; mx], vec![0.0; mx], vec![0.0; mx]);
    y[0] = 1.0;
    for _ in 0..s {
        w.copy_from_slice(&y);
        for k in 1..=deg {
            for (r, out) in hw.iter_mut().enumerate() {
                let row = &h[r * dim..r * dim + mx];
                let c0 = r.saturating_sub(1);
                *out = row[c0..].iter().zip(&w[c0..]).map(|(a, b)| a * b).sum();
            }
            let f = ts / f64::from(k);
            for (wi, &x) in w.iter_mut().zip(&hw) {
                *wi = f * x;
            }
            for (yi, &wi) in y.iter_mut().zip(&w) {
                *yi += wi;
            }
        }
    }
    Ok(y)
}

/// Build an `m`-step Arnoldi basis of `B` from `y/β` (one modified Gram–Schmidt
/// pass, as Expokit, phipm and KIOPS: the orthogonalisation, `O(m²n)` per substep,
/// dominates a sparse generator's `O(m·nnz)` applications).
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)] // Arnoldi notation
fn arnoldi<F: SemiflowFloat, Op: GeneratorAction<F>>(
    aug: &Augmented<'_, F, Op>,
    y: &[F],
    beta: f64,
    m: usize,
    anorm: f64,
    av: &mut [F],
    rep: &mut KrylovPhiReport,
) -> Basis<F> {
    let dim = m + 2;
    let mut h = vec![0.0; dim * dim];
    let mut v: Vec<Vec<F>> = Vec::with_capacity(m + 1);
    v.push(y.iter().map(|&x| x * from_f64::<F>(1.0 / beta)).collect());
    for j in 0..m {
        let mut z = vec![F::zero(); y.len()];
        aug.apply(&v[j], &mut z, av);
        rep.matvecs += 1;
        for (i, vi) in v.iter().enumerate() {
            let c = dot(&z, vi);
            h[i * dim + j] = c;
            axpy(&mut z, -c, vi);
        }
        let hn = norm2(&z);
        if hn <= BREAKDOWN * anorm.max(1.0) {
            return Basis {
                v,
                h,
                mb: j + 1,
                k1: 0,
                avnorm: 0.0,
            };
        }
        h[(j + 1) * dim + j] = hn;
        v.push(z.iter().map(|&x| x * from_f64::<F>(1.0 / hn)).collect());
    }
    h[(m + 1) * dim + m] = 1.0;
    let mut z = vec![F::zero(); y.len()];
    aug.apply(&v[m], &mut z, av);
    rep.matvecs += 1;
    Basis {
        v,
        h,
        mb: m,
        k1: 2,
        avnorm: norm2(&z),
    }
}

/// Local error estimate and exponent of one trial step (Sidje 1998) from
/// `f = e^{tH}e₁` of the `(m+2)`-augmented Hessenberg.
fn local_error(f: &[f64], m: usize, beta: f64, avnorm: f64) -> (f64, f64) {
    let phi1 = libm::fabs(beta * f[m]);
    let phi2 = libm::fabs(beta * f[m + 1] * avnorm);
    #[allow(clippy::cast_precision_loss)] // m ≤ n + 3
    let (one_m, one_m1) = (1.0 / m as f64, 1.0 / (m as f64 - 1.0).max(1.0));
    if phi1 > 10.0 * phi2 {
        (phi2, one_m)
    } else if phi1 > phi2 {
        ((phi1 * phi2) / (phi1 - phi2), one_m)
    } else {
        (phi1, one_m1)
    }
}

/// `y ← e^{B}y` by adaptive Krylov substeps over unit time.
///
/// # Errors
/// `ConvergenceFailed` on the substep or rejection cap; `DomainViolation` for a
/// non-finite norm bound.
pub(crate) fn expv<F: SemiflowFloat, Op: GeneratorAction<F>>(
    aug: &Augmented<'_, F, Op>,
    anorm: f64,
    y: &mut [F],
    opts: KrylovPhiOptions,
    scratch: &mut ScratchPool<F>,
) -> Result<KrylovPhiReport, SemiflowError> {
    if !anorm.is_finite() {
        return Err(SemiflowError::DomainViolation {
            what: "phi_combination_krylov: non-finite norm bound",
            value: anorm,
        });
    }
    let mut rep = KrylovPhiReport::default();
    let beta0 = norm2(y);
    if beta0 == 0.0 {
        return Ok(rep);
    }
    let tol = opts.tol * beta0;
    let m = opts.m.clamp(2, y.len().max(2));
    let mut av = scratch.take_vec(aug.n);
    let result = substeps(aug, anorm, y, tol, m, opts.max_substeps, &mut av, &mut rep);
    scratch.return_vec(av);
    rep.est_error /= beta0;
    result.map(|()| rep)
}

/// The substep loop of [`expv`].
#[allow(clippy::too_many_arguments)]
fn substeps<F: SemiflowFloat, Op: GeneratorAction<F>>(
    aug: &Augmented<'_, F, Op>,
    anorm: f64,
    y: &mut [F],
    tol: f64,
    m: usize,
    max_substeps: usize,
    av: &mut [F],
    rep: &mut KrylovPhiReport,
) -> Result<(), SemiflowError> {
    #[allow(clippy::cast_precision_loss)]
    let mf = m as f64;
    let fact = libm::pow((mf + 1.0) / core::f64::consts::E, mf + 1.0)
        * libm::sqrt(2.0 * core::f64::consts::PI * (mf + 1.0));
    let mut beta = norm2(y);
    let mut elapsed = 0.0_f64;
    let mut next_step = round2(
        (1.0 / anorm.max(f64::MIN_POSITIVE))
            * libm::pow(fact * tol / (4.0 * beta * anorm.max(1.0)), 1.0 / mf),
    );
    while elapsed < 1.0 {
        if rep.substeps >= max_substeps {
            return Err(SemiflowError::ConvergenceFailed {
                last_residual: rep.est_error,
                max_iter: max_substeps,
            });
        }
        let basis = arnoldi(aug, y, beta, m, anorm, av, rep);
        let remaining = 1.0 - elapsed;
        let (t_step, coeff, err) = trial_steps(&basis, m, beta, tol, (next_step, remaining), rep)?;
        y.fill(F::zero());
        for (i, &c) in coeff.iter().enumerate() {
            axpy(y, beta * c, &basis.v[i]);
        }
        elapsed += t_step;
        rep.substeps += 1;
        rep.est_error += err.max(anorm * f64::EPSILON * beta);
        let xm = 1.0 / mf;
        next_step = if err > 0.0 {
            round2(GAMMA * t_step * libm::pow(t_step * tol / err, xm))
        } else {
            1.0 - elapsed
        };
        beta = norm2(y);
        if beta == 0.0 {
            return Ok(());
        }
    }
    Ok(())
}

/// Try `min(t_new, remaining)`, shrinking it until the local error passes; returns
/// the accepted step, the coefficients `F[:mx, 0]` (one per basis vector) and the
/// error. On a happy breakdown the projection is exact: the whole remaining time.
fn trial_steps<F: SemiflowFloat>(
    b: &Basis<F>,
    m: usize,
    beta: f64,
    tol: f64,
    (t_new, remaining): (f64, f64),
    rep: &mut KrylovPhiReport,
) -> Result<(f64, Vec<f64>, f64), SemiflowError> {
    let dim = m + 2;
    if b.k1 == 0 {
        return Ok((remaining, hess_exp_e1(&b.h, dim, b.mb, remaining)?, 0.0));
    }
    let mut t_step = t_new.min(remaining);
    for _ in 0..=MAX_REJECT {
        let mut f = hess_exp_e1(&b.h, dim, m + 2, t_step)?;
        let (err, xm) = local_error(&f, m, beta, b.avnorm);
        if err <= DELTA * t_step * tol {
            f.truncate(m + 1);
            return Ok((t_step, f, err));
        }
        rep.rejected += 1;
        t_step = round2(GAMMA * t_step * libm::pow(t_step * tol / err, xm));
    }
    Err(SemiflowError::ConvergenceFailed {
        last_residual: tol,
        max_iter: MAX_REJECT,
    })
}
