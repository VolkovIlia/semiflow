//! [`Nonlinearity<F>`] and [`NonlinearityDiff<F>`] traits + concrete impls
//! (ADR-0189 §D3).
//!
//! `N(u)` is supplied as a **declarative spec evaluated natively in the loop**,
//! never as a per-step host callback. Phase-1 ships [`AllenCahn<F>`] for
//! `N(u) = u − u³`; further menu entries are deferred to subsequent ADRs.
//!
//! ## Convention
//! `eval` always overwrites the output slice. `vjp` **accumulates**
//! (`out += J_N(u)ᵀ · w`) to compose cleanly with adjoint accumulators.

use crate::{error::SemiflowError, float::SemiflowFloat};

// ---------------------------------------------------------------------------
// Nonlinearity trait
// ---------------------------------------------------------------------------

/// Pointwise nonlinear right-hand side `N(u)`.
///
/// Implement this for concrete semilinear problems. The ETDRK4 driver
/// calls `eval` once per stage; no per-step host crossing occurs.
///
/// # Contract
/// `u` and `n_out` must both have length `n` (problem dimension). Implementors
/// MUST write every element of `n_out` before returning `Ok`.
pub trait Nonlinearity<F: SemiflowFloat>: Send + Sync {
    /// `n_out ← N(u)`.
    ///
    /// # Errors
    /// Returns [`SemiflowError`] if the nonlinearity evaluation fails.
    fn eval(&self, u: &[F], n_out: &mut [F]) -> Result<(), SemiflowError>;
}

// ---------------------------------------------------------------------------
// NonlinearityDiff trait (differentiable extension for adjoint — D4)
// ---------------------------------------------------------------------------

/// Extension of [`Nonlinearity`] providing Jacobian-vector products.
///
/// Required for end-to-end differentiation through an ETDRK4 step (D4).
/// Both `jvp` and `vjp` are **element-wise** for pointwise nonlinearities.
pub trait NonlinearityDiff<F: SemiflowFloat>: Nonlinearity<F> {
    /// `out ← J_N(u) · du`  (forward / tangent mode, overwrites `out`).
    ///
    /// # Errors
    /// Returns [`SemiflowError`] if the Jacobian-vector product fails.
    fn jvp(&self, u: &[F], du: &[F], out: &mut [F]) -> Result<(), SemiflowError>;

    /// `out += J_N(u)ᵀ · w`  (reverse / adjoint mode; **accumulates** into `out`).
    ///
    /// For pointwise nonlinearities `J_N(u)ᵀ = J_N(u)` (diagonal Jacobian).
    ///
    /// # Errors
    /// Returns [`SemiflowError`] if the vector-Jacobian product fails.
    fn vjp(&self, u: &[F], w: &[F], out: &mut [F]) -> Result<(), SemiflowError>;
}

// ---------------------------------------------------------------------------
// AllenCahn — N(u) = u − u³  (element-wise)
// ---------------------------------------------------------------------------

/// Allen–Cahn nonlinearity `N(u) = u − u³` (element-wise).
///
/// Used with the ETDRK4 driver for `∂ₜu = ε·u_xx + N(u)`. The linear
/// part `ε·u_xx` is handled by the generator `L`; `AllenCahn` provides
/// only the nonlinear reaction term `N(u) = u − u³`.
///
/// Jacobian: `J_N(u)_i = 1 − 3uᵢ²` (diagonal, symmetric).
#[derive(Clone, Copy, Debug, Default)]
pub struct AllenCahn<F: SemiflowFloat> {
    _marker: core::marker::PhantomData<F>,
}

impl<F: SemiflowFloat> AllenCahn<F> {
    /// Construct the Allen–Cahn nonlinearity.
    #[must_use]
    pub fn new() -> Self {
        Self { _marker: core::marker::PhantomData }
    }
}

impl<F: SemiflowFloat> Nonlinearity<F> for AllenCahn<F> {
    /// `n_out[i] ← u[i] − u[i]³`.
    fn eval(&self, u: &[F], n_out: &mut [F]) -> Result<(), SemiflowError> {
        for (ui, out_i) in u.iter().zip(n_out.iter_mut()) {
            *out_i = *ui - *ui * *ui * *ui;
        }
        Ok(())
    }
}

impl<F: SemiflowFloat> NonlinearityDiff<F> for AllenCahn<F> {
    /// `out[i] ← (1 − 3uᵢ²) · du[i]`.
    fn jvp(&self, u: &[F], du: &[F], out: &mut [F]) -> Result<(), SemiflowError> {
        let three = F::one() + F::one() + F::one();
        for ((ui, dui), out_i) in u.iter().zip(du.iter()).zip(out.iter_mut()) {
            *out_i = (F::one() - three * *ui * *ui) * *dui;
        }
        Ok(())
    }

    /// `out[i] += (1 − 3uᵢ²) · w[i]`.  Symmetric Jacobian ⇒ same formula as JVP.
    fn vjp(&self, u: &[F], w: &[F], out: &mut [F]) -> Result<(), SemiflowError> {
        let three = F::one() + F::one() + F::one();
        for ((ui, wi), out_i) in u.iter().zip(w.iter()).zip(out.iter_mut()) {
            *out_i += (F::one() - three * *ui * *ui) * *wi;
        }
        Ok(())
    }
}
