//! SPD resolvent / steady solve `x = (λ·M + A)⁻¹ b` (ADR-0202 D1, math §62.1–§62.2).
//!
//! Factor once, solve many. Dispatch by structure ([`SpdSolver::Auto`]): symmetric
//! tridiagonal `LDLᵀ` (exact, `O(n)`, positive pivots certify SPD) or PCG with `IC(0)`
//! / Jacobi. `λ = 0` with a singular `A` is rejected (a-priori row-sum test and the
//! pivot certificate). Both paths use only `+ − × ÷` and `sqrt` in a fixed order, so
//! results are bit-identical between `std` and `no_std` builds (ADR-0200).

use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    pcg::{pcg_shifted, Jacobi, Preconditioner, Shift},
    pcg_ic0::Ic0,
    pow2::pow2_scale,
    scratch::ScratchPool,
    symmetric_operator::{SymmetricLinearOp, SymmetricOperator},
};

/// Which algorithm [`SpdResolvent::new`] uses (§62.2).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SpdSolver {
    /// Tridiagonal `LDLᵀ` if the CSR pattern is tridiagonal, else PCG with `IC(0)`.
    #[default]
    Auto,
    /// Force the tridiagonal `LDLᵀ` path; `Unsupported` if the pattern is not tridiagonal.
    Tridiagonal,
    /// Force preconditioned CG. `Ok` from `solve_into` on this path means the
    /// *recursive* CG residual met `tol`; the true residual is recomputed and
    /// reported in [`SolveReport`] and may exceed `tol` near the attainable-accuracy
    /// floor `ε·κ(S)`. A gross failure (true residual above `max(1e3·tol, 1e-3)`)
    /// is returned as `ConvergenceFailed`.
    Pcg {
        /// Preconditioner.
        precond: Precond,
        /// CG iteration cap. `None` → `2·n + 16` (§62.2.b).
        max_iter: Option<usize>,
    },
}

/// PCG preconditioner choice.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Precond {
    /// `IC(0)`, zero fill-in; on a non-positive pivot it falls back to Jacobi,
    /// visible in [`SpdResolvent::method`].
    #[default]
    Ic0,
    /// Diagonal preconditioner.
    Jacobi,
}

/// Method actually in use (after `Auto` dispatch and any `IC(0)` → Jacobi fallback).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResolventMethod {
    /// Symmetric tridiagonal `LDLᵀ`.
    Tridiagonal,
    /// PCG with `IC(0)`.
    PcgIc0,
    /// PCG with Jacobi.
    PcgJacobi,
}

/// Per-solve diagnostics. Direct path: `iterations = 0`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolveReport {
    /// CG iterations (0 on the direct path).
    pub iterations: usize,
    /// `‖b − (λM+A)x‖₂ / ‖b‖₂` (0 when `b = 0`), recomputed after the solve.
    pub rel_residual: f64,
}

#[derive(Clone)]
enum Kernel<F> {
    Tridiag { d: Vec<F>, l: Vec<F> },
    Ic0(Ic0<F>),
    Jacobi(Jacobi<F>),
}

/// Factor-once / solve-many resolvent of an SPD sparse operator:
/// `x = (λ·M + A)⁻¹ b`, `λ ≥ 0`, `M = diag(mass) > 0` (default `I`).
#[derive(Clone)]
pub struct SpdResolvent<F: SemiflowFloat = f64> {
    op: SymmetricOperator<F>,
    lambda: F,
    mass: Vec<F>,
    tol: F,
    max_iter: usize,
    kernel: Kernel<F>,
}

fn domain(what: &'static str, value: f64) -> SemiflowError {
    SemiflowError::DomainViolation { what, value }
}

/// Validate scalars and return the mass vector (ones when `None`).
fn validate_args<F: SemiflowFloat>(
    n: usize,
    lambda: F,
    mass: Option<&[F]>,
    tol: F,
) -> Result<Vec<F>, SemiflowError> {
    if !lambda.is_finite() || lambda < F::zero() {
        return Err(domain(
            "spd_resolvent: lambda must be finite and >= 0",
            lambda.to_f64().unwrap_or(f64::NAN),
        ));
    }
    if !(tol > F::zero() && tol < F::one()) {
        return Err(domain(
            "spd_resolvent: tol must be in (0, 1)",
            tol.to_f64().unwrap_or(f64::NAN),
        ));
    }
    let Some(m) = mass else {
        return Ok(alloc::vec![F::one(); n]);
    };
    if m.len() != n {
        #[allow(clippy::cast_precision_loss)]
        return Err(domain("spd_resolvent: mass.len() != n", m.len() as f64));
    }
    if let Some(&bad) = m.iter().find(|&&v| !v.is_finite() || v <= F::zero()) {
        return Err(domain(
            "spd_resolvent: mass must be finite and > 0",
            bad.to_f64().unwrap_or(f64::NAN),
        ));
    }
    Ok(m.to_vec())
}

/// `λ = 0` and every row sum of `A` vanishes (pure-Neumann null space `A·1 = 0`) ⇒ singular.
fn reject_null_space<F: SemiflowFloat>(
    op: &SymmetricOperator<F>,
    lambda: F,
) -> Result<(), SemiflowError> {
    if lambda != F::zero() {
        return Ok(());
    }
    let (rp, _, va) = op.csr();
    let slack = F::from(8.0_f64).unwrap_or(F::one()) * F::epsilon();
    let all_zero = (0..op.n()).all(|i| {
        let row = &va[rp[i]..rp[i + 1]];
        let sum = row.iter().fold(F::zero(), |s, &v| s + v);
        let mag = row.iter().fold(F::zero(), |s, &v| s + v.abs());
        sum.abs() <= slack * mag
    });
    if all_zero {
        return Err(domain(
            "spd_resolvent: numerically singular: all row sums <= 8*eps*row magnitude (constant vector in the null space)",
            0.0,
        ));
    }
    Ok(())
}

/// Structural test (§62.2): every stored entry has `|i − j| ≤ 1`.
fn is_tridiagonal<F: SemiflowFloat>(op: &SymmetricOperator<F>) -> bool {
    let (rp, ci, _) = op.csr();
    (0..op.n()).all(|i| {
        ci[rp[i]..rp[i + 1]]
            .iter()
            .all(|&j| (j as usize).abs_diff(i) <= 1)
    })
}

/// Diagonal `δᵢ = λmᵢ + aᵢᵢ` and super-diagonal `βᵢ = a_{i,i+1}` (0 when not stored).
fn diag_and_super<F: SemiflowFloat>(
    op: &SymmetricOperator<F>,
    lambda: F,
    mass: &[F],
) -> (Vec<F>, Vec<F>) {
    let (rp, ci, va) = op.csr();
    let n = op.n();
    let mut delta: Vec<F> = mass.iter().map(|&m| lambda * m).collect();
    let mut beta = alloc::vec![F::zero(); n];
    for i in 0..n {
        for k in rp[i]..rp[i + 1] {
            let j = ci[k] as usize;
            if j == i {
                delta[i] += va[k];
            } else if j == i + 1 {
                // Duplicate CSR entries are summed (same as the matvec does).
                beta[i] += va[k];
            }
        }
    }
    (delta, beta)
}

/// `LDLᵀ` factor of the SPD tridiagonal `S` with the positive-pivot certificate (§62.2.a).
fn factor_tridiag<F: SemiflowFloat>(
    delta: &[F],
    beta: &[F],
) -> Result<(Vec<F>, Vec<F>), SemiflowError> {
    let n = delta.len();
    #[allow(clippy::cast_precision_loss)]
    let n_f = F::from(n as f64).unwrap_or(F::one());
    let dmax = delta.iter().fold(F::zero(), |m, &v| m.max(v.abs()));
    let threshold = n_f * F::epsilon() * dmax;
    let (mut d, mut l) = (Vec::with_capacity(n), alloc::vec![F::zero(); n]);
    for i in 0..n {
        let pivot = if i == 0 {
            delta[0]
        } else {
            l[i] = beta[i - 1] / d[i - 1];
            delta[i] - l[i] * beta[i - 1]
        };
        if pivot.is_nan() || pivot <= threshold {
            return Err(domain(
                "spd_resolvent: tridiagonal pivot <= n*eps*max|d| (not positive definite)",
                pivot.to_f64().unwrap_or(f64::NAN),
            ));
        }
        d.push(pivot);
    }
    Ok((d, l))
}

/// Forward/backward substitution on the `LDLᵀ` factor; `x` receives the solution.
fn solve_tridiag<F: SemiflowFloat>(diag: &[F], lower: &[F], rhs: &[F], sol: &mut [F]) {
    let len = diag.len();
    sol.copy_from_slice(rhs);
    for i in 1..len {
        sol[i] -= lower[i] * sol[i - 1];
    }
    sol[len - 1] /= diag[len - 1];
    for i in (0..len - 1).rev() {
        sol[i] = sol[i] / diag[i] - lower[i + 1] * sol[i + 1];
    }
}

fn norm2<F: SemiflowFloat>(v: &[F]) -> F {
    v.iter().fold(F::zero(), |s, &x| s + x * x).sqrt()
}

impl<F: SemiflowFloat> SpdResolvent<F> {
    /// Build a resolvent of `(λ·M + A)` for `A = op`.
    ///
    /// # Errors
    /// `DomainViolation`: `λ < 0` or non-finite; `mass.len() != n`; mass `≤ 0` or
    /// non-finite; `tol ∉ (0, 1)`; `λ = 0` and every row sum of `A` is 0 (pure-Neumann
    /// null space); tridiagonal `LDLᵀ` pivot `≤ n·ε·max|aᵢᵢ + λmᵢ|` (not positive
    /// definite); a non-positive diagonal on the PCG path.
    /// `Unsupported`: [`SpdSolver::Tridiagonal`] on a non-tridiagonal pattern.
    pub fn new(
        op: &SymmetricOperator<F>,
        lambda: F,
        mass: Option<&[F]>,
        solver: SpdSolver,
        tol: F,
    ) -> Result<Self, SemiflowError> {
        let n = op.n();
        if n == 0 {
            return Err(domain(
                "spd_resolvent: operator dimension n must be >= 1",
                0.0,
            ));
        }
        let mass = validate_args(n, lambda, mass, tol)?;
        reject_null_space(op, lambda)?;
        let tri = is_tridiagonal(op);
        let (kernel, max_iter) = match solver {
            SpdSolver::Tridiagonal if !tri => {
                return Err(SemiflowError::Unsupported {
                    feature: "spd_resolvent: tridiagonal solver on non-tridiagonal operator",
                });
            }
            SpdSolver::Tridiagonal => (build_tridiag(op, lambda, &mass)?, 0),
            SpdSolver::Auto if tri => (build_tridiag(op, lambda, &mass)?, 0),
            SpdSolver::Auto => (build_pcg(op, lambda, &mass, Precond::Ic0)?, default_cap(n)),
            SpdSolver::Pcg { precond, max_iter } => (
                build_pcg(op, lambda, &mass, precond)?,
                max_iter.unwrap_or_else(|| default_cap(n)),
            ),
        };
        Ok(Self {
            op: op.clone(),
            lambda,
            mass,
            tol,
            max_iter,
            kernel,
        })
    }

    /// Operator dimension.
    #[must_use]
    pub fn n(&self) -> usize {
        self.op.n()
    }

    /// Method actually in use.
    #[must_use]
    pub fn method(&self) -> ResolventMethod {
        match self.kernel {
            Kernel::Tridiag { .. } => ResolventMethod::Tridiagonal,
            Kernel::Ic0(_) => ResolventMethod::PcgIc0,
            Kernel::Jacobi(_) => ResolventMethod::PcgJacobi,
        }
    }

    /// `x ← (λM+A)⁻¹ b`. `x` is overwritten (not used as a warm start).
    ///
    /// The right-hand side is prescaled by an exact power of two so that
    /// `‖b‖∞ ∈ [1, 2)`: the solve and the reported residual are scale-invariant and
    /// free of under/overflow. On the PCG path `Ok` means the recursive CG residual
    /// met `tol`; the true residual is recomputed and reported in
    /// [`SolveReport::rel_residual`] and may exceed `tol` near the attainable-accuracy
    /// floor `ε·κ(S)`.
    ///
    /// # Errors
    /// `DomainViolation`: `b.len()` / `x.len() != n`, non-finite `b`.
    /// `DomainViolation` also if the solution overflows the float type after unscaling.
    /// `ConvergenceFailed { last_residual, max_iter }`: PCG only. Two triggers: the
    /// iteration cap `max_iter` was reached, or the recomputed residual exceeds
    /// `max(1e3·tol, 1e-3)` (breakdown on a singular system; `max_iter` is then the
    /// configured cap, not the iteration count). In both cases `last_residual` is the
    /// relative true residual `‖b − (λM+A)x‖₂/‖b‖₂` of the last iterate.
    pub fn solve_into(
        &self,
        b: &[F],
        x: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<SolveReport, SemiflowError> {
        let n = self.n();
        if b.len() != n || x.len() != n {
            #[allow(clippy::cast_precision_loss)]
            return Err(domain(
                "spd_resolvent: b.len() or x.len() != n",
                b.len().min(x.len()) as f64,
            ));
        }
        if let Some(&bad) = b.iter().find(|v| !v.is_finite()) {
            return Err(domain(
                "spd_resolvent: non-finite right-hand side",
                bad.to_f64().unwrap_or(f64::NAN),
            ));
        }
        let Some((scale, inv_scale)) = pow2_scale(b) else {
            x.fill(F::zero());
            return Ok(SolveReport {
                iterations: 0,
                rel_residual: 0.0,
            });
        };
        let mut bs = scratch.take_vec(n);
        let mut xs = scratch.take_vec(n);
        for (o, &v) in bs.iter_mut().zip(b) {
            *o = v * scale;
        }
        let result = self.solve_scaled(&bs, &mut xs, scratch);
        for (o, &v) in x.iter_mut().zip(xs.iter()) {
            *o = v * inv_scale;
        }
        scratch.return_vec(bs);
        scratch.return_vec(xs);
        if result.is_ok() && x.iter().any(|v| !v.is_finite()) {
            return Err(domain(
                "spd_resolvent: solution not representable (overflow)",
                f64::NAN,
            ));
        }
        result
    }

    /// Solve and verify on the O(1)-scaled system.
    fn solve_scaled(
        &self,
        bs: &[F],
        xs: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<SolveReport, SemiflowError> {
        let run = match &self.kernel {
            Kernel::Tridiag { d, l } => {
                solve_tridiag(d, l, bs, xs);
                Ok(0)
            }
            Kernel::Ic0(p) => self.run_pcg(p, bs, xs, scratch),
            Kernel::Jacobi(p) => self.run_pcg(p, bs, xs, scratch),
        };
        let iterations = match run {
            Ok(iterations) => iterations,
            // Cap reached: report the relative TRUE residual of the last iterate,
            // the same quantity as on the gross-residual guard below.
            Err(SemiflowError::ConvergenceFailed { max_iter, .. }) => {
                let last_residual = self.rel_residual(bs, xs, scratch);
                return Err(SemiflowError::ConvergenceFailed {
                    last_residual,
                    max_iter,
                });
            }
            Err(other) => return Err(other),
        };
        let rel_residual = self.rel_residual(bs, xs, scratch);
        let pcg = !matches!(self.kernel, Kernel::Tridiag { .. });
        let limit = (1e3 * self.tol.to_f64().unwrap_or(1.0)).max(GROSS_RESIDUAL);
        if pcg && (rel_residual.is_nan() || rel_residual > limit) {
            return Err(SemiflowError::ConvergenceFailed {
                last_residual: rel_residual,
                max_iter: self.max_iter,
            });
        }
        Ok(SolveReport {
            iterations,
            rel_residual,
        })
    }

    fn run_pcg(
        &self,
        precond: &dyn Preconditioner<F>,
        b: &[F],
        x: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<usize, SemiflowError> {
        x.fill(F::zero());
        let shift = Shift::Mass {
            lambda: self.lambda,
            mass: &self.mass,
        };
        pcg_shifted(
            &self.op,
            &shift,
            b,
            x,
            precond,
            self.tol,
            self.max_iter,
            scratch,
        )
    }

    /// `‖b − (λM+A)x‖₂ / ‖b‖₂`, recomputed from scratch (0 when `b = 0`).
    fn rel_residual(&self, b: &[F], x: &[F], scratch: &mut ScratchPool<F>) -> f64 {
        let nb = norm2(b);
        if nb == F::zero() {
            return 0.0;
        }
        let mut ax = scratch.take_vec(self.n());
        self.op.apply_into_slice(x, &mut ax);
        for i in 0..ax.len() {
            ax[i] = b[i] - (self.lambda * self.mass[i] * x[i] + ax[i]);
        }
        let rel = norm2(&ax) / nb;
        scratch.return_vec(ax);
        rel.to_f64().unwrap_or(f64::NAN)
    }
}

/// Gross-failure floor for the recomputed PCG residual (see [`SpdSolver::Pcg`]).
const GROSS_RESIDUAL: f64 = 1e-3;

/// Default CG cap `2n + 16` (§62.2.b).
fn default_cap(n: usize) -> usize {
    2 * n + 16
}

fn build_tridiag<F: SemiflowFloat>(
    op: &SymmetricOperator<F>,
    lambda: F,
    mass: &[F],
) -> Result<Kernel<F>, SemiflowError> {
    let (delta, beta) = diag_and_super(op, lambda, mass);
    let (d, l) = factor_tridiag(&delta, &beta)?;
    Ok(Kernel::Tridiag { d, l })
}

fn build_pcg<F: SemiflowFloat>(
    op: &SymmetricOperator<F>,
    lambda: F,
    mass: &[F],
    precond: Precond,
) -> Result<Kernel<F>, SemiflowError> {
    let (delta, _) = diag_and_super(op, lambda, mass);
    if let Some(&bad) = delta.iter().find(|&&v| v.is_nan() || v <= F::zero()) {
        return Err(domain(
            "spd_resolvent: non-positive diagonal of lambda*M + A",
            bad.to_f64().unwrap_or(f64::NAN),
        ));
    }
    if precond == Precond::Ic0 {
        let (rp, ci, va) = op.csr();
        let shift: Vec<F> = mass.iter().map(|&m| lambda * m).collect();
        if let Some(ic) = Ic0::build(op.n(), rp, ci, va, &shift) {
            return Ok(Kernel::Ic0(ic));
        }
    }
    Ok(Kernel::Jacobi(Jacobi::from_diag(&delta)))
}
