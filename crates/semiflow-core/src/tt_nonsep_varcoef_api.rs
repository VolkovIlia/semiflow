//! Public wrapper type for `tt_nonsep_varcoef` (ADR-0169, v9.2.0).
//!
//! Sibling module — contains [`S3NonSepVarCoefEvolver`] which is the
//! boundary-as-type public surface for the low-CP-rank non-separable
//! variable-coefficient S³ evolver.  Raw `pub(crate)` free functions stay in
//! `tt_nonsep_varcoef`.
//!
//! Feature-gated: `s3-poc`.
#![cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]
#![allow(clippy::cast_possible_truncation)]

extern crate alloc;
use alloc::vec::Vec;

use crate::{
    float::{from_f64, SemiflowFloat},
    tt_nonsep_varcoef::{CpCoef, CoefRole, nonsep_evolve},
};

/// Order-2 low-CP-rank non-separable variable-coefficient evolver (S³ POC).
///
/// Converges at order 2 for operators whose coefficient `a(x)` has fixed CP-rank
/// `m`: `a(x) = a₀ + Σ_{r<m} ∏ⱼ a_{r,j}(xⱼ)`.  The [`CpCoef::terms`] field is a
/// fixed-length Vec — generic full-rank `a(x)` (requiring exponentially many terms)
/// is the CP-rank wall and is UNREPRESENTABLE.  Parabolicity is checked fail-loud
/// at construction.
///
/// ## Proven boundary
/// Order-2 ONLY for fixed CP-rank `m` coefficients `a(x)=a₀+Σ_{r<m}∏ⱼ a_{r,j}(xⱼ)`.
/// [`crate::CpCoef::terms`] is a fixed-`m` Vec — generic full-CP-rank `a(x)` is the
/// CP-RANK WALL, UNREPRESENTABLE by type. Parabolicity `c(x)>0` is checked
/// fail-loud at construction. Proof: `g_s3_nonsep_varcoef` (RELEASE-BLOCKING,
/// `slow-tests`), slope ≤ −1.9 on `cos(x)sin(y)·∂²ₓ` (the §53.3 floor case). See
/// ADR-0167, math.md §53.4.
pub struct S3NonSepVarCoefEvolver<F: SemiflowFloat> {
    /// Grid size (same on every axis).
    n: usize,
    /// Number of spatial dimensions.
    d: usize,
    /// Grid spacing.
    dx: F,
    /// Constant leading diffusion (`a₀`).
    a0: F,
    /// CP-rank coefficient list (enforces the rank wall).
    coefs: Vec<CpCoef<F>>,
}

impl<F: SemiflowFloat> S3NonSepVarCoefEvolver<F> {
    /// Construct the evolver; validates shapes and parabolicity fail-loud.
    ///
    /// For every `CoefRole::Diffusion` coef, reconstructs `c(x)` on the full
    /// `n^d` grid and fails if any point has `c(x) ≤ 0`.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] on shape mismatch, `n < 2`,
    /// `dx ≤ 0`, `a0 ≤ 0`, or parabolicity violation on the grid.
    pub fn new(
        n: usize,
        d: usize,
        dx: F,
        a0: F,
        coefs: Vec<CpCoef<F>>,
    ) -> Result<Self, crate::SemiflowError> {
        validate_nonsep(n, d, dx, a0, &coefs)?;
        Ok(Self { n, d, dx, a0, coefs })
    }

    /// Evolve state `u0` (flat `n^d`) for `nsteps` steps of size `tau`.
    ///
    /// Returns `(evolved_state, max_imag_residue)`.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if `u0.len() != n^d`.
    pub fn evolve(
        &self,
        u0: &[F],
        tau: F,
        nsteps: usize,
    ) -> Result<(Vec<F>, F), crate::SemiflowError> {
        let nd = self.n.pow(self.d as u32);
        if u0.len() != nd {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "u0 length must equal n^d",
            });
        }
        Ok(nonsep_evolve(u0, self.n, self.d, self.dx, self.a0, &self.coefs, tau, nsteps))
    }
}

/// Validate inputs for `S3NonSepVarCoefEvolver::new` (kept separate for line-cap).
fn validate_nonsep<F: SemiflowFloat>(
    n: usize,
    d: usize,
    dx: F,
    a0: F,
    coefs: &[CpCoef<F>],
) -> Result<(), crate::SemiflowError> {
    if n < 2 {
        return Err(crate::SemiflowError::S3OutOfClass { detail: "n must be >= 2" });
    }
    if dx <= F::zero() {
        return Err(crate::SemiflowError::S3OutOfClass { detail: "dx must be > 0" });
    }
    if a0 <= F::zero() {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "a0 must be > 0 (constant leading diffusion)",
        });
    }
    check_coef_shapes(n, d, coefs)?;
    check_parabolicity(n, d, coefs)
}

/// Verify that all CP-terms in all coefs have correct shapes (d × n).
fn check_coef_shapes<F: SemiflowFloat>(
    n: usize,
    d: usize,
    coefs: &[CpCoef<F>],
) -> Result<(), crate::SemiflowError> {
    for coef in coefs {
        for term in &coef.terms {
            if term.factor.len() != d {
                return Err(crate::SemiflowError::S3OutOfClass {
                    detail: "CpTerm factor must have length d",
                });
            }
            for fj in &term.factor {
                if fj.len() != n {
                    return Err(crate::SemiflowError::S3OutOfClass {
                        detail: "CpTerm factor[j] must have length n",
                    });
                }
            }
        }
    }
    Ok(())
}

/// Release-mode parabolicity check: c(x) > 0 on full n^d grid for Diffusion coefs.
fn check_parabolicity<F: SemiflowFloat>(
    n: usize,
    d: usize,
    coefs: &[CpCoef<F>],
) -> Result<(), crate::SemiflowError> {
    for coef in coefs {
        if coef.role != CoefRole::Diffusion {
            continue;
        }
        let nd = n.pow(d as u32);
        for flat in 0..nd {
            let mut cx = coef.c0;
            for term in &coef.terms {
                let mut prod = from_f64::<F>(1.0);
                let mut idx = flat;
                for ax in (0..d).rev() {
                    let coord = idx % n;
                    idx /= n;
                    prod *= term.factor[ax][coord];
                }
                cx += prod;
            }
            if cx <= F::zero() {
                return Err(crate::SemiflowError::S3OutOfClass {
                    detail: "reconstructed c(x) <= 0: not parabolic",
                });
            }
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "s3-poc"))]
mod tests {
    use super::*;
    use crate::tt_nonsep_varcoef::{CoefRole, CpCoef, CpTerm};

    #[test]
    fn rejects_n_less_than_2() {
        assert!(S3NonSepVarCoefEvolver::<f64>::new(1, 2, 0.1, 1.0, vec![]).is_err());
    }

    #[test]
    fn rejects_dx_zero() {
        assert!(S3NonSepVarCoefEvolver::<f64>::new(4, 2, 0.0, 1.0, vec![]).is_err());
    }

    #[test]
    fn rejects_nonpositive_a0() {
        assert!(S3NonSepVarCoefEvolver::<f64>::new(4, 2, 0.1, 0.0, vec![]).is_err());
    }

    #[test]
    fn rejects_non_parabolic_coef() {
        // c0 = 0, no terms → reconstructed c(x) = 0 ≤ 0 everywhere
        let coef = CpCoef { c0: 0.0, terms: vec![], role: CoefRole::Diffusion };
        assert!(S3NonSepVarCoefEvolver::<f64>::new(4, 2, 0.1, 1.0, vec![coef]).is_err());
    }

    #[test]
    fn rejects_bad_factor_shape() {
        // factor has length 1 (d=1), but d=2
        let factor: Vec<Vec<f64>> = vec![vec![1.0f64; 4]];
        let t = CpTerm { factor };
        let coef = CpCoef { c0: 1.0, terms: vec![t], role: CoefRole::Diffusion };
        assert!(S3NonSepVarCoefEvolver::<f64>::new(4, 2, 0.1, 1.0, vec![coef]).is_err());
    }

    #[test]
    fn accepts_valid_inputs() {
        assert!(S3NonSepVarCoefEvolver::<f64>::new(4, 2, 0.1, 1.0, vec![]).is_ok());
    }
}
