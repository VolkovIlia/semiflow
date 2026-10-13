//! The shift-and-invert Lanczos iteration behind [`super::RationalKrylov`]
//! (ADR-0206): `M`-orthonormal basis, full reorthogonalisation, the projected
//! coefficients `f(T)e₁` at the a-priori dimension (or an invariant subspace).

use alloc::vec::Vec;

use super::{RationalKrylov, RationalReport};
use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    lanczos_sched::tridiag_fn_e1,
    scratch::ScratchPool,
};

/// Lanczos state of one action: `M`-orthonormal basis and the tridiagonal `T`.
pub(super) struct Lanczos<F: SemiflowFloat> {
    pub(super) basis: Vec<Vec<F>>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    n: usize,
}

impl<F: SemiflowFloat> Lanczos<F> {
    pub(super) fn new(n: usize) -> Self {
        Self {
            basis: Vec::new(),
            alpha: Vec::new(),
            beta: Vec::new(),
            n,
        }
    }

    pub(super) fn basis_push(&mut self, scratch: &mut ScratchPool<F>) -> &mut Vec<F> {
        self.basis.push(scratch.take_vec(self.n));
        let last = self.basis.len() - 1;
        &mut self.basis[last]
    }

    /// `q₀ = v·down / ‖v·down‖_M`; returns `‖v·down‖_M` (`down` a power of two).
    pub(super) fn start(
        &mut self,
        rk: &RationalKrylov<F>,
        v: &[F],
        down: F,
        scratch: &mut ScratchPool<F>,
    ) -> F {
        let first = self.basis_push(scratch);
        for (q, &x) in first.iter_mut().zip(v) {
            *q = x * down;
        }
        let norm = rk.norm_m(first);
        for q in first.iter_mut() {
            *q /= norm;
        }
        norm
    }

    /// `out ← up · Σ_k (‖v‖_M·c_k)·q_k`.
    pub(super) fn combine(&self, coeffs: &[f64], norm_v: F, up: F, out: &mut [F]) {
        out.fill(F::zero());
        for (q, &c) in self.basis.iter().zip(coeffs) {
            let c = norm_v * from_f64::<F>(c);
            for (o, &x) in out.iter_mut().zip(q) {
                *o += c * x;
            }
        }
        for o in out.iter_mut() {
            *o *= up;
        }
    }

    pub(super) fn release(self, scratch: &mut ScratchPool<F>) {
        for q in self.basis {
            scratch.return_vec(q);
        }
    }

    /// Run to the a-priori dimension (or an invariant subspace); returns the
    /// coefficients `f(T)e₁`.
    pub(super) fn run(
        &mut self,
        rk: &RationalKrylov<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(Vec<f64>, RationalReport), SemiflowError> {
        let target = rk.m_min.unwrap_or(usize::MAX);
        let mut inner = 0;
        let mut w = scratch.take_vec(self.n);
        let mut rhs = scratch.take_vec(self.n);
        let outcome = loop {
            let j = self.basis.len() - 1;
            match self.step(rk, j, &mut w, &mut rhs, scratch) {
                Ok(it) => inner += it,
                Err(e) => break Err(e),
            }
            // Invariant subspace (or the whole space): the projection is exact.
            let breakdown =
                self.beta[j] <= f64::EPSILON * rk.gamma.to_f64().unwrap_or(1.0) || j + 1 == self.n;
            if j + 1 >= target || breakdown || j + 1 >= rk.m_max {
                let reached = j + 1 >= target || breakdown;
                break self.finish(rk, j, inner, reached);
            }
            self.push_normalised(&w, j, scratch);
        };
        scratch.return_vec(w);
        scratch.return_vec(rhs);
        outcome
    }

    /// Coefficients at dimension `j + 1` and the a-posteriori difference to `j`;
    /// `ConvergenceFailed` when the a-priori dimension was not `reached`.
    fn finish(
        &self,
        rk: &RationalKrylov<F>,
        j: usize,
        inner: usize,
        reached: bool,
    ) -> Result<(Vec<f64>, RationalReport), SemiflowError> {
        let f = |th| rk.f(th);
        let failed = SemiflowError::ConvergenceFailed {
            last_residual: f64::NAN,
            max_iter: rk.m_max,
        };
        let Some(c) = tridiag_fn_e1(&self.alpha, &self.beta[..j], f) else {
            return Err(failed);
        };
        let prev = if j == 0 {
            Some(Vec::new())
        } else {
            tridiag_fn_e1(&self.alpha[..j], &self.beta[..j - 1], f)
        };
        let diff = prev.map_or(f64::NAN, |p| coeff_diff(&c, &p));
        if !reached {
            return Err(SemiflowError::ConvergenceFailed {
                last_residual: diff,
                max_iter: rk.m_max,
            });
        }
        let report = RationalReport {
            solves: j + 1,
            krylov_dim: j + 1,
            est_error: diff,
            inner_iterations: inner,
        };
        Ok((c, report))
    }

    /// `q_{j+1} = w / β_j`.
    fn push_normalised(&mut self, w: &[F], j: usize, scratch: &mut ScratchPool<F>) {
        let inv_beta = from_f64::<F>(1.0 / self.beta[j]);
        let next = self.basis_push(scratch);
        for (q, &x) in next.iter_mut().zip(w) {
            *q = x * inv_beta;
        }
    }

    /// One Lanczos step on `Z = (M/γ + A)⁻¹M` with full `M`-reorthogonalisation:
    /// appends `α_j`, `β_j`; leaves the unnormalised next vector in `w`.
    fn step(
        &mut self,
        rk: &RationalKrylov<F>,
        j: usize,
        w: &mut [F],
        rhs: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<usize, SemiflowError> {
        for (i, (r, &q)) in rhs.iter_mut().zip(&self.basis[j]).enumerate() {
            *r = rk.weight(i) * q;
        }
        let iterations = rk.inner.solve(rhs, w, scratch)?;
        let alpha = rk.dot_m(w, &self.basis[j]);
        axpy(w, -alpha, &self.basis[j]);
        if j > 0 {
            axpy(w, -from_f64::<F>(self.beta[j - 1]), &self.basis[j - 1]);
        }
        for q in &self.basis {
            let c = rk.dot_m(w, q);
            axpy(w, -c, q);
        }
        self.alpha.push(alpha.to_f64().unwrap_or(f64::NAN));
        self.beta.push(rk.norm_m(w).to_f64().unwrap_or(f64::NAN));
        Ok(iterations)
    }
}

/// `y ← y + a·x`.
fn axpy<F: SemiflowFloat>(y: &mut [F], a: F, x: &[F]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// `‖c − [prev; 0]‖₂`.
fn coeff_diff(c: &[f64], prev: &[f64]) -> f64 {
    libm::sqrt(
        c.iter()
            .enumerate()
            .map(|(i, &x)| {
                let d = x - prev.get(i).copied().unwrap_or(0.0);
                d * d
            })
            .sum::<f64>(),
    )
}
