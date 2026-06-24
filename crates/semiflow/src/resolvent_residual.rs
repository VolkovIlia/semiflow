//! `LaplaceChernoffResolventResidual` gate-wrapper and `Sampleable` impl for
//! [`GridFn1D`] (split from `resolvent.rs` for suckless line-cap compliance).

// Grid size n cast to f64 for error reporting; n ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

use alloc::vec::Vec;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid_fn::GridFn1D,
    resolvent::{LaplaceChernoffResolvent, Sampleable},
};

// G_RES_RES wrapper (v4.0 Wave E, ADR-0083, math.md §34).

/// Gate-wrapper: computes ‖(λI − A) R̃(λ) f − f‖_∞ via 3-pt FD Laplacian.
/// `G_RES_RES` (`RELEASE_BLOCKING)`: residual ≤ `budget` at λ=1.0, N=512, Gaussian f.
/// Test harness only; NOT a [`ChernoffFunction`] impl.
pub struct LaplaceChernoffResolventResidual<C, F = f64>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    /// Inner resolvent.
    pub inner: LaplaceChernoffResolvent<C, F>,
    residual_budget: F,
}
impl<C, F> LaplaceChernoffResolventResidual<C, F>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    /// Construct with the given residual budget (e.g. `1e-3` for `G_RES_RES`).
    pub fn new(inner: LaplaceChernoffResolvent<C, F>, budget: F) -> Self {
        Self {
            inner,
            residual_budget: budget,
        }
    }
    /// The configured residual budget.
    pub fn budget(&self) -> F {
        self.residual_budget
    }
    /// Compute ‖(λI − A) R̃(λ) f − f‖_∞ on interior nodes (3-pt FD, A = ∂_xx).
    ///
    /// # Errors
    ///
    /// `DomainViolation` if `lambda ≤ 0`, not finite, or grid has < 3 nodes.
    /// Propagates errors from `inner.eval`.
    pub fn verify_residual(
        &self,
        lambda: F,
        f: &<C as ChernoffFunction<F>>::S,
    ) -> Result<F, SemiflowError>
    where
        C::S: Clone,
        C: ChernoffFunction<F, S = GridFn1D<F>>,
    {
        if !lambda.is_finite() || lambda <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "lambda must be finite and > 0 (real-λ contract; complex λ deferred v4.0)",
                value: lambda.to_f64().unwrap_or(f64::NAN),
            });
        }
        let r = self.inner.eval(lambda, f)?;
        let n = r.values.len();
        if n < 3 {
            return Err(SemiflowError::DomainViolation {
                what: "verify_residual: grid must have >= 3 nodes for central differences",
                value: n as f64,
            });
        }
        let dx2 = f.grid.dx() * f.grid.dx();
        let mut max_err = F::zero();
        for i in 1..n - 1 {
            let lap_r =
                (r.values[i + 1] - from_f64::<F>(2.0) * r.values[i] + r.values[i - 1]) / dx2;
            let err = (lambda * r.values[i] - lap_r - f.values[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        Ok(max_err)
    }
}

impl<F: SemiflowFloat> Sampleable<F> for GridFn1D<F> {
    fn sample_at(&self, x: &[F]) -> Result<F, SemiflowError> {
        if x.is_empty() {
            return Err(SemiflowError::DomainViolation {
                what: "sample_at: x0 must have at least one coordinate",
                value: 0.0,
            });
        }
        self.sample_generic(x[0])
    }

    fn fresh_from_fn(&self, f: &dyn Fn(&[F]) -> F) -> Result<Self, SemiflowError> {
        let grid = self.grid;
        let values: Vec<F> = (0..grid.n).map(|i| f(&[grid.x_at(i)])).collect();
        Ok(Self { values, grid })
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    include!("resolvent_residual_tests_mod.rs");
}
