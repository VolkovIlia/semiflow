//! Public wrapper type for `tt_drift_spectral` (ADR-0169, v9.2.0).
//!
//! Sibling module — contains [`S3DriftSpectralEvolver`] which is the
//! boundary-as-type public surface for the constant-coefficient drift+diffusion
//! S³ evolver.  Raw `pub(crate)` free functions stay in `tt_drift_spectral`.
//!
//! Feature-gated: `s3-poc`.
#![cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]
#![allow(clippy::cast_possible_truncation)]

extern crate alloc;
use alloc::vec::Vec;

use crate::{float::SemiflowFloat, tt_drift_spectral::apply_drift_spectral_axis};

/// Solver-free constant-coefficient diffusion + drift evolver (S³ POC).
///
/// Exact-in-time for operators `L = Σⱼ aⱼ∂²ₓⱼ + bⱼ∂ₓⱼ` via the complex
/// Fourier symbol.  The constructor enforces the class boundary — variable
/// or non-parabolic diffusion is unconstructible.
///
/// ## Proven boundary
/// Exact ONLY for the constant-coefficient diffusion+drift class. Variable
/// coefficients are out of class (use [`crate::S3VarCoefEvolver`]). The proof is
/// `g_s3_drift_spectral` (RELEASE-BLOCKING, `slow-tests`): exactness ≤1e-12 vs
/// independent dense Padé `expm`; Δrank-preservation under drift; operational
/// cost-scaling at d∈{8,10}. No absolute generic-input curse-escape is claimed
/// (info-theoretically false). See ADR-0164, math.md §53.1.
pub struct S3DriftSpectralEvolver<F: SemiflowFloat> {
    /// Grid size (same on every axis).
    n: usize,
    /// Number of spatial dimensions.
    d: usize,
    /// Grid spacing.
    dx: F,
    /// Per-axis diffusion coefficients `aⱼ > 0` (length `d`).
    a: Vec<F>,
    /// Per-axis drift coefficients `bⱼ` (length `d`).
    b: Vec<F>,
}

impl<F: SemiflowFloat> S3DriftSpectralEvolver<F> {
    /// Construct the evolver; fails if input is out of class.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if `a.len() != d`,
    /// `b.len() != d`, `n < 2`, `dx ≤ 0`, or any `a[j] ≤ 0` (parabolicity).
    pub fn new(
        n: usize,
        d: usize,
        dx: F,
        a: Vec<F>,
        b: Vec<F>,
    ) -> Result<Self, crate::SemiflowError> {
        validate_drift_spectral(n, d, dx, &a, &b)?;
        Ok(Self { n, d, dx, a, b })
    }

    /// Evolve state `u0` (flat `n^d`) by one time step `tau`.
    ///
    /// Returns `(evolved_state, max_imag_residue)`.  Residue < 1e-12 in class.
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
        let mut u = u0.to_vec();
        let mut max_imag = F::zero();
        for j in 0..self.d {
            let imag =
                apply_drift_spectral_axis(&mut u, self.n, self.dx, self.a[j], self.b[j], tau);
            if imag > max_imag {
                max_imag = imag;
            }
        }
        Ok((u, max_imag))
    }
}

/// Validation helper — extracted so the constructor stays ≤ 50 lines.
fn validate_drift_spectral<F: SemiflowFloat>(
    n: usize,
    d: usize,
    dx: F,
    a: &[F],
    b: &[F],
) -> Result<(), crate::SemiflowError> {
    if a.len() != d || b.len() != d {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "diffusion a and drift b must each have length d",
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
                detail: "diffusion a[j] must be > 0 (parabolicity)",
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
        let r = S3DriftSpectralEvolver::<f64>::new(1, 2, 0.1, vec![1.0; 2], vec![0.0; 2]);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_dx_zero() {
        let r = S3DriftSpectralEvolver::<f64>::new(4, 2, 0.0, vec![1.0; 2], vec![0.0; 2]);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_nonpositive_a() {
        let r = S3DriftSpectralEvolver::<f64>::new(4, 2, 0.1, vec![0.0, 1.0], vec![0.0; 2]);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_length_mismatch() {
        let r = S3DriftSpectralEvolver::<f64>::new(4, 3, 0.1, vec![1.0; 2], vec![0.0; 2]);
        assert!(r.is_err());
    }

    #[test]
    fn accepts_valid_inputs() {
        let r = S3DriftSpectralEvolver::<f64>::new(4, 2, 0.1, vec![1.0; 2], vec![0.0; 2]);
        assert!(r.is_ok());
    }
}
