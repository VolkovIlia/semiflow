//! Public wrapper type for `tt_dense_coupling` (ADR-0169, v9.2.0).
//!
//! Sibling module — contains [`S3DenseCouplingEvolver`] which is the
//! boundary-as-type public surface for the rank-1-dense all-pairs coupling
//! S³ evolver.  Raw `pub(crate)` free functions stay in `tt_dense_coupling`.
//!
//! Feature-gated: `s3-poc`.
#![cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]
#![allow(clippy::cast_possible_truncation)]

extern crate alloc;
use alloc::vec::Vec;

use crate::{
    float::SemiflowFloat,
    tt_dense_coupling::{dense_coupling_evolve, rank1_dense_matrix},
};

/// Solver-free rank-1-dense all-pairs coupling evolver (S³ POC).
///
/// Exact-in-time for operators with diffusion matrix `D = diag(a) + λ·g·gᵀ`
/// (rank-1-dense: every off-diagonal non-zero, off-diagonal block rank 1).
/// The `rank1_dense` constructor is the ONLY way to create this type; an
/// arbitrary full-rank `D` matrix is unconstructible (the info-theoretic wall
/// is enforced by the absence of a public raw-`D` constructor).
///
/// ## Proven boundary
/// Bounded-rank ONLY for rank-1-dense `D = diag(a) + λ·g·gᵀ`. Generic full-rank
/// coupling is an INFO-THEORETIC WALL: no TT method compresses a generic full-rank
/// tensor (proven negative, `g_s3_dense_coupling` non-vacuous rank-2-contrast). No
/// public constructor accepts an arbitrary `D` — the wall is enforced by type. See
/// ADR-0165, math.md §53.2.
pub struct S3DenseCouplingEvolver<F: SemiflowFloat> {
    /// Grid size (same on every axis).
    n: usize,
    /// Number of spatial dimensions.
    d: usize,
    /// Grid spacing.
    dx: F,
    /// Row-major `d×d` coupling matrix `D` (built from rank-1-dense decomposition).
    d_mat: Vec<F>,
    /// Per-axis drift coefficients `bⱼ` (length `d`).
    b: Vec<F>,
}

impl<F: SemiflowFloat> S3DenseCouplingEvolver<F> {
    /// Construct from a rank-1-dense decomposition `D = diag(a) + λ·g·gᵀ`.
    ///
    /// This is the ONLY public constructor — passing a raw `d×d` matrix is
    /// impossible, which is the info-theoretic-wall type enforcement.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if length checks fail,
    /// `n < 2`, `dx ≤ 0`, or any diagonal `a[j] ≤ 0`.
    #[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
    pub fn rank1_dense(
        n: usize,
        d: usize,
        dx: F,
        a: &[F],
        g: &[F],
        lambda: F,
        b: Vec<F>,
    ) -> Result<Self, crate::SemiflowError> {
        validate_dense_coupling(n, d, dx, a, g, &b)?;
        let d_mat = rank1_dense_matrix(a, g, lambda);
        Ok(Self { n, d, dx, d_mat, b })
    }

    /// Evolve state `u0` (flat `n^d`) by one time step `tau`.
    ///
    /// Returns `(evolved_state, max_imag_residue)`.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if `u0.len() != n^d`.
    pub fn evolve(&self, u0: &[F], tau: F) -> Result<(Vec<F>, F), crate::SemiflowError> {
        let nd = self.n.pow(self.d as u32);
        if u0.len() != nd {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "u0 length must equal n^d",
            });
        }
        let result = dense_coupling_evolve(u0, self.n, self.d, self.dx, &self.d_mat, &self.b, tau);
        Ok(result)
    }
}

/// Validation helper — keeps the constructor ≤ 50 lines.
#[allow(clippy::many_single_char_names)]
fn validate_dense_coupling<F: SemiflowFloat>(
    n: usize,
    d: usize,
    dx: F,
    a: &[F],
    g: &[F],
    b: &[F],
) -> Result<(), crate::SemiflowError> {
    if a.len() != d || g.len() != d || b.len() != d {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "a, g, b must each have length d",
        });
    }
    if n < 2 {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "n must be >= 2",
        });
    }
    if dx <= F::zero() {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "dx must be > 0",
        });
    }
    for &aj in a {
        if aj <= F::zero() {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "diagonal a[j] must be > 0",
            });
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "s3-poc"))]
mod tests {
    use super::*;

    #[test]
    fn rejects_n_less_than_2() {
        let r = S3DenseCouplingEvolver::<f64>::rank1_dense(
            1,
            2,
            0.1,
            &[1.0; 2],
            &[1.0; 2],
            0.1,
            vec![0.0; 2],
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_dx_zero() {
        let r = S3DenseCouplingEvolver::<f64>::rank1_dense(
            4,
            2,
            0.0,
            &[1.0; 2],
            &[1.0; 2],
            0.1,
            vec![0.0; 2],
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_nonpositive_a() {
        let r = S3DenseCouplingEvolver::<f64>::rank1_dense(
            4,
            2,
            0.1,
            &[0.0, 1.0],
            &[1.0; 2],
            0.1,
            vec![0.0; 2],
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_length_mismatch() {
        let r = S3DenseCouplingEvolver::<f64>::rank1_dense(
            4,
            3,
            0.1,
            &[1.0; 2],
            &[1.0; 2],
            0.1,
            vec![0.0; 2],
        );
        assert!(r.is_err());
    }

    #[test]
    fn accepts_valid_inputs() {
        let r = S3DenseCouplingEvolver::<f64>::rank1_dense(
            4,
            2,
            0.1,
            &[1.0; 2],
            &[1.0; 2],
            0.1,
            vec![0.0; 2],
        );
        assert!(r.is_ok());
    }
}
