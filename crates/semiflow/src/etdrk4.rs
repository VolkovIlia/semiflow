//! ETDRK4 exponential time-differencing driver (ADR-0189 §D2).
//!
//! Solves the semilinear initial-value problem
//! ```text
//! ∂_t u = L u + N(u),   u(0) = u₀
//! ```
//! where `L` is a linear generator ([`GeneratorAction`]) and `N` is a pointwise
//! nonlinearity ([`Nonlinearity`]).
//!
//! ## Algorithm
//!
//! Cox–Matthews (2002) fourth-order exponential time differencing (ETDRK4).
//! The linear part is handled exactly via φ-functions; the nonlinear part is
//! evaluated at four explicit stages.  No implicit solve is required.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use semiflow::{Etdrk4, AllenCahn, scratch::ScratchPool};
//!
//! // Build driver (validates dim > 0)
//! let driver = Etdrk4::new(my_op, AllenCahn::<f64>::new(), h)?;
//!
//! // Single step
//! driver.step(&u, &mut u_next, &mut scratch)?;
//!
//! // Multiple steps; final state written to `out`
//! driver.integrate(&u0, 100, &mut out, &mut scratch)?;
//! ```
//!
//! ## References
//!
//! - Cox & Matthews (2002) *J. Comput. Phys.* **176**, 430–455.
//! - ADR-0189; `contracts/semiflow-core.math.md` §58.

use crate::{
    error::SemiflowError,
    etdrk4_helpers::etdrk4_step as step_impl,
    float::SemiflowFloat,
    generator_action::GeneratorAction,
    nonlinearity::Nonlinearity,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Etdrk4 driver struct
// ---------------------------------------------------------------------------

/// ETDRK4 semilinear time-stepping driver.
///
/// Wraps a linear generator `Op` and a nonlinearity `Nl` together with a
/// fixed step size `h`.  After construction via [`Etdrk4::new`], call
/// [`step`](Etdrk4::step) for single steps or [`integrate`](Etdrk4::integrate)
/// for multiple steps.
///
/// Generic parameters:
/// - `F`: scalar float type (`f32` or `f64`).
/// - `Op`: linear generator implementing [`GeneratorAction<F>`].
/// - `Nl`: nonlinearity implementing [`Nonlinearity<F>`].
pub struct Etdrk4<F: SemiflowFloat, Op: GeneratorAction<F>, Nl: Nonlinearity<F>> {
    op: Op,
    nl: Nl,
    h: F,
    n: usize,
}

impl<F, Op, Nl> Etdrk4<F, Op, Nl>
where
    F: SemiflowFloat,
    Op: GeneratorAction<F>,
    Nl: Nonlinearity<F>,
{
    /// Construct the ETDRK4 driver.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `op.dim() == 0`.
    pub fn new(op: Op, nl: Nl, h: F) -> Result<Self, SemiflowError> {
        let n = op.dim();
        if n == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "Etdrk4::new: op.dim() must be > 0",
                value: 0.0,
            });
        }
        Ok(Self { op, nl, h, n })
    }

    /// Advance one step: `u_next ← ETDRK4(u, h)`.
    ///
    /// Borrows temporary buffers from `scratch`; all are returned before
    /// this call exits.
    ///
    /// # Errors
    /// Propagates any error from the generator or nonlinearity.
    pub fn step(
        &self,
        u: &[F],
        u_next: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        step_impl(&self.op, &self.nl, self.h, u, u_next, scratch)
    }

    /// Integrate `n_steps` steps from `u0`; final state written to `out[..n]`.
    ///
    /// Uses one additional scratch buffer of size `n` for the ping-pong swap.
    ///
    /// # Errors
    /// Propagates any error from [`step`](Etdrk4::step).
    pub fn integrate(
        &self,
        u0: &[F],
        n_steps: usize,
        out: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let n = self.n;
        out[..n].copy_from_slice(&u0[..n]);
        let mut tmp = scratch.take_vec(n);
        for _ in 0..n_steps {
            self.step(out, &mut tmp, scratch)?;
            out[..n].copy_from_slice(&tmp[..n]);
        }
        scratch.return_vec(tmp);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests (fast)
// ---------------------------------------------------------------------------

#[cfg(test)]
include!("etdrk4_tests_mod.rs");
