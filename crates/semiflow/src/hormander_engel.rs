//! Engel step-3 filiform Carnot group Chernoff approximation (ADR-0095, math.md §28.bis).
//!
//! # Background
//!
//! The **Engel group** E is the unique (up to isomorphism) smallest non-trivial
//! step-3 Carnot group (Bonfiglioli-Lanconelli-Uguzzoni 2007 Prop. 4.3.8, dim = 4).
//!
//! Stratification: g = g₁ ⊕ g₂ ⊕ g₃
//! - g₁ = span{X₁, X₂}  (2 horizontal generators — Bonfiglioli "two-generator rule")
//! - g₂ = span{X₃ = [X₁, X₂]}
//! - g₃ = span{X₄ = [X₁, X₃] = [X₁, [X₁, X₂]]}
//!
//! Left-invariant fields on ℝ⁴ coords (x₁, x₂, x₃, x₄):
//! ```text
//! X₁ = ∂_{x₁}
//! X₂ = ∂_{x₂} + x₁ ∂_{x₃} + (x₁²/2) ∂_{x₄}
//! X₃ = ∂_{x₃} + x₁ ∂_{x₄}   (= [X₁, X₂])
//! X₄ = ∂_{x₄}                 (= [X₁, X₃])
//! ```
//!
//! Sub-Laplacian: `L_E` = X₁² + X₂²
//!
//! # Palindromic Strang-Hörmander
//!
//! `F_Engel(τ)` = exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)
//!
//! Each exp(σ·Xₖ²) is a 1D Gaussian convolution along the integral curves of Xₖ,
//! evaluated via 32-pt Gauss-Hermite quadrature (mirror `hormander_heisenberg.rs`).
//!
//! # X₁ flow (trivial ambient coupling)
//!
//! Integral curve of X₁ from (x₁, x₂, x₃, x₄): `(x₁ + s, x₂, x₃, x₄)` (no coupling).
//! exp(σ·X₁²) f(x) = π^{-1/2} Σ wₖ f(x₁ - √(2σ)·ξₖ, x₂, x₃, x₄)
//!
//! # X₂ flow (polynomial coupling in x₃, x₄)
//!
//! Integral curve of X₂ from point p = (x₁, x₂, x₃, x₄) with parameter s:
//!   (x₁, x₂ + s, x₃ + s·x₁, x₄ + s²·x₁/2 + s·x₃)
//!
//! This is a polynomial parametric flow — fully `no_std` + libm-evaluable (no complex
//! arithmetic, no special functions beyond exp and sqrt).
//!
//! exp(σ·X₂²) f(x) = π^{-1/2} Σ wₖ f(x₁, x₂ − √(2σ)·ξₖ, `x₂_src_x3`, `x₂_src_x4`)
//!
//! # Self-convergence validation
//!
//! No closed-form heat kernel oracle exists for Engel (Bonfiglioli 2007 §18.3;
//! Folland-Kaplan restricted to H-type groups; Engel NOT H-type).
//! Gate `G_HORM_ENGEL` uses probe-vs-2N self-convergence (mirror v2.2 `G_NS2D_aniso`).
//!
//! # References
//!
//! - Bonfiglioli-Lanconelli-Uguzzoni 2007 §4.3.6 (filiform Engel)
//! - Bratzlavsky 1974, cited via Bonfiglioli Theorem 4.3.6 (basis)
//! - Galkin-Remizov 2025 IJM Theorem 3.1 (K=2 tangency, conditionally extended)
//! - ADR-0095 + math.md §28.bis

extern crate alloc;
use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid_nd::GridFnND,
    hormander::{bracket_central_diff, HypoellipticChernoff, VectorField},
    hormander_engel_helpers::{sample_4d, sample_axis0, GH32_NODES_ENGEL, GH32_WEIGHTS_ENGEL},
    scratch::ScratchPool,
};

// ─── Engel left-invariant vector fields ──────────────────────────────────────

/// Left-invariant field X₁ = ∂_{x₁} on Engel group ℝ⁴.
///
/// Returns (1, 0, 0, 0) at any point — constant unit vector in x₁-direction.
///
/// Reference: Bonfiglioli 2007 §4.3.6, Bratzlavsky 1974 basis; math.md §28.bis.1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngelX1<F: SemiflowFloat = f64> {
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat> EngelX1<F> {
    /// Construct X₁ = ∂_{x₁} (zero-sized, infallible).
    #[must_use]
    pub fn new() -> Self {
        Self { _f: PhantomData }
    }
}

impl<F: SemiflowFloat> Default for EngelX1<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SemiflowFloat> VectorField<F, 4> for EngelX1<F> {
    /// X₁(x₁, x₂, x₃, x₄) = (1, 0, 0, 0).
    fn evaluate(&self, _x: &[F; 4], out: &mut [F; 4]) -> Result<(), SemiflowError> {
        out[0] = F::one();
        out[1] = F::zero();
        out[2] = F::zero();
        out[3] = F::zero();
        Ok(())
    }
}

/// Left-invariant field X₂ = ∂_{x₂} + x₁·∂_{x₃} + (x₁²/2)·∂_{x₄} on Engel group ℝ⁴.
///
/// Returns (0, 1, x₁, x₁²/2) at point (x₁, x₂, x₃, x₄).
///
/// Reference: Bonfiglioli 2007 §4.3.6, Bratzlavsky 1974 basis; math.md §28.bis.1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngelX2<F: SemiflowFloat = f64> {
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat> EngelX2<F> {
    /// Construct X₂ (zero-sized, infallible).
    #[must_use]
    pub fn new() -> Self {
        Self { _f: PhantomData }
    }
}

impl<F: SemiflowFloat> Default for EngelX2<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SemiflowFloat> VectorField<F, 4> for EngelX2<F> {
    /// X₂(x₁, x₂, x₃, x₄) = (0, 1, x₁, x₁²/2).
    fn evaluate(&self, x: &[F; 4], out: &mut [F; 4]) -> Result<(), SemiflowError> {
        let half = crate::float::from_f64::<F>(0.5_f64);
        out[0] = F::zero();
        out[1] = F::one();
        out[2] = x[0]; // x₁
        out[3] = half * x[0] * x[0]; // x₁²/2
        Ok(())
    }
}

// ─── Zero field on ℝ⁴ (X₀ = 0 for sub-Laplacian case) ───────────────────────

struct ZeroField4;

impl VectorField<f64, 4> for ZeroField4 {
    fn evaluate(&self, _x: &[f64; 4], out: &mut [f64; 4]) -> Result<(), SemiflowError> {
        out[0] = 0.0;
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = 0.0;
        Ok(())
    }
}

// ─── HypoellipticChernoff<f64, 4, 2> constructor ─────────────────────────────

impl HypoellipticChernoff<f64, 4, 2> {
    /// Construct the Engel step-3 Carnot group Chernoff approximation.
    ///
    /// Sets X₁ = `EngelX1`, X₂ = `EngelX2` and verifies the step-3 Carnot
    /// bracket structure at the origin via `bracket_central_diff`.
    ///
    /// Verification: [X₁, X₂] ≈ (0, 0, 1, 0) = X₃ (step-2 bracket).
    /// The step-3 structure [X₁, X₃] = (0, 0, 0, 1) = X₄ is verified by
    /// `T_HORM_ENGEL_BRACKETS` sympy (`scripts/verify_engel_brackets.py`, 5/5 PASS).
    ///
    /// # Errors
    /// - `DomainViolation` if bracket check fails.
    pub fn new_engel() -> Result<Self, SemiflowError> {
        let x1 = EngelX1::<f64>::new();
        let x2 = EngelX2::<f64>::new();
        // Verify [X₁, X₂] ≈ X₃ = (0, 0, 1, x₁) at origin → (0, 0, 1, 0).
        let origin = [0.0_f64; 4];
        let mut bracket_12 = [0.0_f64; 4];
        bracket_central_diff(&x1, &x2, &origin, &mut bracket_12)?;
        let eps = 1e-6_f64;
        let expected = [0.0_f64, 0.0_f64, 1.0_f64, 0.0_f64];
        for (&got, &exp) in bracket_12.iter().zip(expected.iter()) {
            if (got - exp).abs() > eps {
                return Err(SemiflowError::DomainViolation {
                    what: "Engel step-3 check: [X₁,X₂] deviates from X₃ at origin",
                    value: (got - exp).abs(),
                });
            }
        }
        let x0: alloc::boxed::Box<dyn VectorField<f64, 4>> = alloc::boxed::Box::new(ZeroField4);
        let diff: alloc::vec::Vec<alloc::boxed::Box<dyn VectorField<f64, 4>>> = alloc::vec![
            alloc::boxed::Box::new(EngelX1::<f64>::new()),
            alloc::boxed::Box::new(EngelX2::<f64>::new()),
        ];
        Ok(Self {
            x0_drift: x0,
            x_diff: diff,
            _f: PhantomData,
        })
    }
}

// ─── ChernoffFunction impl for Engel ─────────────────────────────────────────

/// State type for Engel: `GridFnND`<f64, 4> (4D tensor-product grid).
///
/// Grid layout: axes [x₁, x₂, x₃, x₄], axis 0 fastest.
/// Size 32⁴ = 1,048,576 points × 8 bytes = 8 MB per state buffer.
impl ChernoffFunction<f64> for HypoellipticChernoff<f64, 4, 2> {
    type S = GridFnND<f64, 4>;

    /// Palindromic Strang-Hörmander for Engel:
    /// `exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)`.
    ///
    /// Each sub-step uses 32-pt Gauss-Hermite quadrature along the integral
    /// curve of Xₖ, with multilinear interpolation into the 4D grid.
    ///
    /// Reference: math.md §28.bis.2, ADR-0095.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFnND<f64, 4>,
        dst: &mut GridFnND<f64, 4>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "tau must be finite and non-negative",
                value: tau,
            });
        }
        let n = src.values.len();
        let mut mid = GridFnND {
            values: alloc::vec![0.0_f64; n],
            grid: src.grid.clone(),
        };
        // Leg 1: exp(τ/4 · X₁²)
        engel_diffuse_x1(src, &mut mid, tau * 0.25)?;
        // Leg 2: exp(τ/2 · X₂²)
        let mut mid2 = GridFnND {
            values: alloc::vec![0.0_f64; n],
            grid: src.grid.clone(),
        };
        engel_diffuse_x2(&mid, &mut mid2, tau * 0.5)?;
        // Leg 3: exp(τ/4 · X₁²)
        engel_diffuse_x1(&mid2, dst, tau * 0.25)?;
        Ok(())
    }

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

// ─── Engel sub-step helpers ───────────────────────────────────────────────────

/// Apply `exp(sigma · X₁²)` where X₁ = ∂_{x₁} (no coupling in ambient ℝ⁴).
///
/// Integral curve of X₁ from p = (x₁, x₂, x₃, x₄) with step s:
///   p + s·X₁(p) = (x₁ + s, x₂, x₃, x₄)   — trivially linear, no coupling.
///
/// (exp(σ·X₁²) f)(x) = π^{-1/2} Σ wₖ f(x₁ − √(2σ)·ξₖ, x₂, x₃, x₄)
// Result kept for API symmetry with coupled-axis variants that use `?`.
#[allow(clippy::unnecessary_wraps)]
fn engel_diffuse_x1(
    src: &GridFnND<f64, 4>,
    dst: &mut GridFnND<f64, 4>,
    sigma: f64,
) -> Result<(), SemiflowError> {
    if sigma <= 0.0 {
        dst.values.copy_from_slice(&src.values);
        return Ok(());
    }
    let grid = &src.grid;
    let [n0, n1, n2, n3] = [
        grid.axes[0].n,
        grid.axes[1].n,
        grid.axes[2].n,
        grid.axes[3].n,
    ];
    let scale = libm::sqrt(2.0 * sigma);
    let pi_inv_sqrt = 1.0 / libm::sqrt(core::f64::consts::PI);
    // X₁ diffuses only along axis 0; axes 1,2,3 are frozen for each pencil.
    for i3 in 0..n3 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                for i0 in 0..n0 {
                    let x0 = grid.axes[0].x_at(i0);
                    let mut val = 0.0_f64;
                    for (&xi, &wi) in GH32_NODES_ENGEL.iter().zip(GH32_WEIGHTS_ENGEL.iter()) {
                        let s = scale * xi;
                        let x0_src = x0 - s;
                        // Evaluate f at (x0_src, x1_fixed, x2_fixed, x3_fixed).
                        let f_val = sample_axis0(src, x0_src, i1, i2, i3);
                        val += wi * f_val;
                    }
                    let flat = grid.flat_idx(&[i0, i1, i2, i3]);
                    dst.values[flat] = pi_inv_sqrt * val;
                }
            }
        }
    }
    Ok(())
}

/// Apply `exp(sigma · X₂²)` where X₂ = ∂_{x₂} + x₁·∂_{x₃} + (x₁²/2)·∂_{x₄}.
///
/// Integral curve of X₂ from p = (x₁, x₂, x₃, x₄) with parameter s:
///   (x₁, x₂ + s, x₃ + s·x₁, x₄ + s²·x₁/2 + s·x₃)
///
/// This is a degree-2 polynomial parametric flow — fully evaluable in `no_std`.
///
/// (exp(σ·X₂²) f)(x) = π^{-1/2} Σ wₖ f(x₁, x₂ − s, x₃ − s·x₁, x₄ − s²·x₁/2 − s·x₃)
/// where s = √(2σ)·ξₖ.
// Result kept for API symmetry with coupled-axis variants that use `?`.
#[allow(clippy::unnecessary_wraps)]
fn engel_diffuse_x2(
    src: &GridFnND<f64, 4>,
    dst: &mut GridFnND<f64, 4>,
    sigma: f64,
) -> Result<(), SemiflowError> {
    if sigma <= 0.0 {
        dst.values.copy_from_slice(&src.values);
        return Ok(());
    }
    let grid = &src.grid;
    let [n0, n1, n2, n3] = [
        grid.axes[0].n,
        grid.axes[1].n,
        grid.axes[2].n,
        grid.axes[3].n,
    ];
    let scale = libm::sqrt(2.0 * sigma);
    let pi_inv_sqrt = 1.0 / libm::sqrt(core::f64::consts::PI);
    // X₂ diffuses along axis 1 (x₂), with coupling into axes 2 (x₃) and 3 (x₄).
    // x₁ (axis 0) is frozen for each pencil.
    for i3 in 0..n3 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                let x1_fixed = grid.axes[1].x_at(i1); // x₂ at node i1
                let x2_coord = grid.axes[2].x_at(i2); // x₃ at node i2
                let x3_coord = grid.axes[3].x_at(i3); // x₄ at node i3
                for i0 in 0..n0 {
                    let x0_fixed = grid.axes[0].x_at(i0); // x₁ at node i0 (frozen)
                    let mut val = 0.0_f64;
                    for (&xi, &wi) in GH32_NODES_ENGEL.iter().zip(GH32_WEIGHTS_ENGEL.iter()) {
                        let s = scale * xi;
                        // Source coordinates along X₂ integral curve (backwards by s):
                        let x1_src = x1_fixed - s; // x₂ source
                        let x2_src = x2_coord - s * x0_fixed; // x₃ source
                        let x3_src = x3_coord - s * s * x0_fixed * 0.5 - s * x2_coord; // x₄ src
                        let f_val = sample_4d(src, x0_fixed, x1_src, x2_src, x3_src);
                        val += wi * f_val;
                    }
                    let flat = grid.flat_idx(&[i0, i1, i2, i3]);
                    dst.values[flat] = pi_inv_sqrt * val;
                }
            }
        }
    }
    Ok(())
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::grid_nd::GridND;

    #[test]
    fn engel_x1_is_constant_unit() {
        let x1 = EngelX1::<f64>::new();
        let mut out = [0.0_f64; 4];
        x1.evaluate(&[1.0, 2.0, 3.0, 4.0], &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn engel_x1_at_origin() {
        let x1 = EngelX1::<f64>::new();
        let mut out = [0.0_f64; 4];
        x1.evaluate(&[0.0, 0.0, 0.0, 0.0], &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn engel_x2_at_origin() {
        let x2 = EngelX2::<f64>::new();
        let mut out = [0.0_f64; 4];
        x2.evaluate(&[0.0, 0.0, 0.0, 0.0], &mut out).unwrap();
        // X₂(0,0,0,0) = (0, 1, 0, 0) since x₁=0
        assert_eq!(out, [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn engel_x2_off_axis() {
        let x2 = EngelX2::<f64>::new();
        let mut out = [0.0_f64; 4];
        x2.evaluate(&[2.0, 0.0, 0.0, 0.0], &mut out).unwrap();
        // X₂(2,0,0,0) = (0, 1, 2, 2) since x₁=2: x₁=2, x₁²/2=2
        assert!((out[0] - 0.0).abs() < 1e-15);
        assert!((out[1] - 1.0).abs() < 1e-15);
        assert!((out[2] - 2.0).abs() < 1e-15);
        assert!((out[3] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn new_engel_bracket_check_passes() {
        let result = HypoellipticChernoff::<f64, 4, 2>::new_engel();
        assert!(result.is_ok(), "Engel bracket check should pass");
    }

    #[test]
    fn engel_apply_into_smoke() {
        // Minimal smoke test: apply one Chernoff step to a 4D Gaussian.
        use crate::grid::Grid1D;
        let ax = Grid1D::new(-2.0_f64, 2.0, 8).unwrap();
        let grid = GridND::<f64, 4>::new([ax, ax, ax, ax]).unwrap();
        let u0 = GridFnND::from_fn(grid.clone(), |x: &[f64; 4]| {
            (-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) * 0.5).exp()
        });
        let chernoff = HypoellipticChernoff::<f64, 4, 2>::new_engel().unwrap();
        let mut dst = GridFnND {
            values: alloc::vec![0.0_f64; u0.values.len()],
            grid: grid.clone(),
        };
        let mut scratch = crate::scratch::ScratchPool::new();
        chernoff
            .apply_into(0.1, &u0, &mut dst, &mut scratch)
            .unwrap();
        // After one Chernoff step the maximum value should be positive and finite.
        let max_val = dst.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.0, "post-step max value must be positive");
        assert!(max_val.is_finite(), "post-step max value must be finite");
    }
}
