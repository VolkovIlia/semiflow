//! Inner solver and a-priori Krylov dimension of [`super::RationalKrylov`]
//! (ADR-0206, math §66.3–§66.4).
//!
//! **Excess factor.** For a tridiagonal `A` with non-positive off-diagonals (every
//! path-graph Laplacian), `S = λM + A` has row excesses `sᵢ = λmᵢ + Σⱼ aᵢⱼ ≥ 0`
//! wherever `A`'s row sums are not more negative than `−λmᵢ`. Writing the `LDLᵀ`
//! pivots as `dᵢ = |βᵢ| + eᵢ` gives the recurrence `e₀ = s₀`,
//! `eᵢ = sᵢ + |βᵢ₋₁|·eᵢ₋₁/dᵢ₋₁` — algebraically the textbook
//! `dᵢ = δᵢ − βᵢ₋₁²/dᵢ₋₁`, but with only non-negative terms, so every pivot excess
//! carries a relative error `O(iε)`. The textbook form subtracts two `O(‖A‖)`
//! numbers to get an excess of size `O(λ)`, losing `log₁₀(‖A‖/λ) ≈ log₁₀(λ_max t)`
//! digits of the slow eigenvalues. The row sums are formed with error-free
//! transformations, so the factor is accurate for the operator AS STORED.
//!
//! **A-priori dimension.** For self-adjoint `Z` with spectrum in `[0, γ]`,
//! `‖f(Z)v − ‖v‖·Q_m f(T_m)e₁‖ ≤ 2·E_{m−1}(f)·‖v‖` with `E_{m−1}` the best uniform
//! polynomial approximation error of `f` on `[0, γ]` (Saad 1992; Hochbruck &
//! Lubich 1997), and `E_{m−1} ≤ Σ_{k≥m}|c_k|` for the Chebyshev coefficients
//! `c_k` of `f(θ) = e^{r(1 − γ/θ)}`, `r = t/γ`. The smallest `m` with
//! `2·Σ_{k≥m}|c_k| ≤ tol` is the stopping dimension: it cannot stop before a small
//! slow component has entered the Krylov space (an a-posteriori difference test
//! can).

use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    scratch::ScratchPool,
    spd_resolvent::{SpdResolvent, SpdSolver},
    symmetric_operator::SymmetricOperator,
};

/// `a + b = s + e` exactly (Knuth's `TwoSum`).
fn two_sum<F: SemiflowFloat>(a: F, b: F) -> (F, F) {
    let s = a + b;
    let bp = s - a;
    (s, (a - (s - bp)) + (b - bp))
}

/// Cancellation-free `LDLᵀ` of `λM + A` (module docs).
#[derive(Clone)]
pub(super) struct ExcessFactor<F> {
    d: Vec<F>,
    l: Vec<F>,
}

impl<F: SemiflowFloat> ExcessFactor<F> {
    /// `None` when `A` is not tridiagonal, has a positive off-diagonal entry, or
    /// some row excess is negative or non-finite (the caller then uses
    /// [`SpdResolvent`]). `mass` is validated by the caller.
    #[allow(clippy::many_single_char_names)] // n, i, j, k, d, l, e, s: LDLᵀ notation
    pub(super) fn new(op: &SymmetricOperator<F>, lambda: F, mass: Option<&[F]>) -> Option<Self> {
        let n = op.n();
        let (rp, ci, va) = op.csr();
        // Super-diagonal βᵢ (duplicates summed, as `SpdResolvent` does).
        let mut beta = vec![F::zero(); n];
        for i in 0..n {
            for k in rp[i]..rp[i + 1] {
                let j = ci[k] as usize;
                // A NaN off-diagonal is not ≤ 0 either: rejected.
                let positive = va[k] > F::zero() || va[k].is_nan();
                if j.abs_diff(i) > 1 || (j != i && positive) {
                    return None;
                }
                if j == i + 1 {
                    beta[i] += va[k];
                }
            }
        }
        let mut d = Vec::with_capacity(n);
        let mut l = vec![F::zero(); n];
        let mut e = F::zero();
        for i in 0..n {
            let s = row_excess(
                lambda * mass.map_or(F::one(), |m| m[i]),
                i,
                (rp, ci, va),
                &beta,
            );
            if !(s >= F::zero() && s.is_finite()) {
                return None;
            }
            e = if i == 0 {
                s
            } else {
                l[i] = beta[i - 1] / d[i - 1];
                s + (-beta[i - 1]) * e / d[i - 1]
            };
            let pivot = -beta[i] + e;
            if !(pivot > F::zero() && pivot.is_finite()) {
                return None;
            }
            d.push(pivot);
        }
        Some(Self { d, l })
    }

    /// `x ← (λM + A)⁻¹ b` (forward/backward substitution, as `SpdResolvent`).
    fn solve(&self, b: &[F], x: &mut [F]) {
        let n = self.d.len();
        x.copy_from_slice(b);
        for i in 1..n {
            x[i] -= self.l[i] * x[i - 1];
        }
        x[n - 1] /= self.d[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = x[i] / self.d[i] - self.l[i + 1] * x[i + 1];
        }
    }
}

/// `sᵢ = λmᵢ + aᵢᵢ + βᵢ₋₁ + βᵢ` (row `i` of the upper triangle plus the sub-diagonal
/// as the factor sees it), compensated: relative error `≤ 2ε` plus `O(ε²)·Σ|·|`.
fn row_excess<F: SemiflowFloat>(
    lambda_m: F,
    i: usize,
    (rp, ci, va): (&[usize], &[u32], &[F]),
    beta: &[F],
) -> F {
    let (mut sum, mut err) = (lambda_m, F::zero());
    let mut add = |x: F| {
        let (s, e) = two_sum(sum, x);
        sum = s;
        err += e;
    };
    for k in rp[i]..rp[i + 1] {
        if ci[k] as usize >= i {
            add(va[k]);
        }
    }
    if i > 0 {
        add(beta[i - 1]);
    }
    sum + err
}

/// The solver of `S = M/γ + A` used by the Lanczos iteration.
#[derive(Clone)]
pub(super) enum Inner<F: SemiflowFloat> {
    /// `t·‖M⁻¹A‖ < ε`: the action is the identity to rounding; no factor.
    Identity,
    /// Tridiagonal M-matrix pattern: the excess factor.
    Excess(ExcessFactor<F>),
    /// Every other SPD pattern (or a forced PCG): the shared resolvent.
    Resolvent(SpdResolvent<F>),
}

impl<F: SemiflowFloat> Inner<F> {
    /// Excess factor when `solver` allows a direct tridiagonal factor and the
    /// pattern qualifies; otherwise [`SpdResolvent`] with `solver`.
    ///
    /// # Errors
    /// As [`SpdResolvent::new`].
    pub(super) fn build(
        op: &SymmetricOperator<F>,
        lambda: F,
        mass: Option<&[F]>,
        solver: SpdSolver,
        inner_tol: F,
    ) -> Result<Self, SemiflowError> {
        if matches!(solver, SpdSolver::Auto | SpdSolver::Tridiagonal) {
            if let Some(f) = ExcessFactor::new(op, lambda, mass) {
                return Ok(Self::Excess(f));
            }
        }
        Ok(Self::Resolvent(
            op.resolvent(lambda, mass, solver, inner_tol)?,
        ))
    }

    /// `x ← S⁻¹b`; returns the CG iteration count (0 on a direct path).
    ///
    /// # Errors
    /// As [`SpdResolvent::solve_into`]; a non-finite solution (`DomainViolation`).
    pub(super) fn solve(
        &self,
        b: &[F],
        x: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<usize, SemiflowError> {
        let iterations = match self {
            Self::Identity => {
                x.copy_from_slice(b);
                0
            }
            Self::Excess(f) => {
                f.solve(b, x);
                0
            }
            Self::Resolvent(r) => r.solve_into(b, x, scratch)?.iterations,
        };
        if let Some(&bad) = x.iter().find(|v| !v.is_finite()) {
            return Err(SemiflowError::DomainViolation {
                what: "rational_krylov: non-finite resolvent solution",
                value: bad.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(iterations)
    }
}

/// Chebyshev coefficients below this are rounding noise of the discrete transform
/// (`≈ ε·√N` for the point counts used here) and are not summed into the tail.
const COEFF_NOISE: f64 = 64.0 * f64::EPSILON;

/// Signed Chebyshev coefficients `c_0 … c_{N−1}` of `f(θ) = e^{r(1 − γ/θ)}` on
/// `[0, γ]` from the interpolant at `N = pts` first-kind Chebyshev points (phases
/// `(2k + 1)jπ/(2N)` reduced exactly mod `4N`).
fn cheb_coeffs(r: f64, pts: usize) -> Vec<f64> {
    #[allow(clippy::cast_precision_loss)] // pts < 2^20
    let pts_f = pts as f64;
    #[allow(clippy::cast_precision_loss)]
    let table: Vec<f64> = (0..4 * pts)
        .map(|k| libm::cos(k as f64 * core::f64::consts::PI / (2.0 * pts_f)))
        .collect();
    let fx: Vec<f64> = (0..pts)
        .map(|k| {
            let theta = 0.5 * (table[2 * k + 1] + 1.0);
            if theta > 0.0 {
                libm::exp(r * (1.0 - 1.0 / theta))
            } else {
                0.0
            }
        })
        .collect();
    (0..pts)
        .map(|j| {
            let s: f64 = (0..pts)
                .map(|k| fx[k] * table[(j * (2 * k + 1)) % (4 * pts)])
                .sum();
            if j == 0 {
                s / pts_f
            } else {
                2.0 * s / pts_f
            }
        })
        .collect()
}

/// Smallest `m ∈ [1, m_cap]` with `2·Σ_{k≥m, |c_k| > COEFF_NOISE}|c_k| ≤ tol`.
fn tail_dim(c: &[f64], tol: f64, m_cap: usize) -> Option<usize> {
    let mut tail = 0.0;
    let mut best = None;
    for m in (1..c.len()).rev() {
        if libm::fabs(c[m]) > COEFF_NOISE {
            tail += libm::fabs(c[m]);
        }
        if 2.0 * tail > tol {
            break;
        }
        if m <= m_cap {
            best = Some(m);
        }
    }
    best
}

/// `p_{m−1} = Σ_{k<m} c_k T_k` must reproduce the exactly known `f(γ) = 1` and
/// `f(0) = 0` (`x = ±1`) to `tol/2`: necessary for `E_{m−1} ≤ tol/2`, and violated
/// whenever the grid misses the boundary layer at `θ ≈ γ` (`r ≫ 1`) or the step at
/// `θ ≈ rγ` (`r ≪ 1`) — every sample then sits on one plateau of `f`.
fn endpoints_resolved(c: &[f64], m: usize, tol: f64) -> bool {
    let (mut hi, mut lo) = (0.0, 0.0);
    for (k, &ck) in c[..m].iter().enumerate() {
        hi += ck;
        lo += if k % 2 == 0 { ck } else { -ck };
    }
    libm::fabs(hi - 1.0) <= 0.5 * tol && libm::fabs(lo) <= 0.5 * tol
}

/// Smallest `m ∈ [1, m_cap]` with `2·Σ_{k≥m}|c_k| ≤ tol` for the Chebyshev
/// coefficients of `f(θ) = e^{r(1 − γ/θ)}` on `[0, γ]` (module docs). The point
/// count doubles from 65 until the last quarter of the coefficients sits at the
/// rounding plateau (or reaches `2·m_cap + 65`), so the cost is `O(m_min²)`, not
/// `O(m_cap²)`. Coefficients below [`COEFF_NOISE`] are dropped, so a `tol` below
/// `≈ 1e−13` is met only down to that plateau. `None`: no `m ≤ m_cap` qualifies, or
/// the interpolant does not reproduce `f(0) = 0` and `f(γ) = 1` to `tol/2` (an
/// extreme ratio `r = t/γ` the grid cannot resolve) — the action then reports
/// `ConvergenceFailed` rather than a silently wrong result.
pub(super) fn a_priori_dim(r: f64, tol: f64, m_cap: usize) -> Option<usize> {
    let max_pts = 2 * m_cap + 65;
    let mut pts = max_pts.min(65);
    loop {
        let c = cheb_coeffs(r, pts);
        let plateau = c[3 * pts / 4..]
            .iter()
            .all(|x| libm::fabs(*x) <= COEFF_NOISE);
        if plateau || pts == max_pts {
            return tail_dim(&c, tol, m_cap).filter(|&m| endpoints_resolved(&c, m, tol));
        }
        pts = (2 * pts - 1).min(max_pts);
    }
}
