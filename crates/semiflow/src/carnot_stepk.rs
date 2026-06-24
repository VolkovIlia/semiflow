//! Step-4 filiform Carnot group Chernoff approximation (ADR-0136 Amendment 1,
//! math.md §28.bis.7, v8.0.0 F4 order-2 real baseline).
//!
//! # Background
//!
//! The **filiform N=5 step-4 Carnot group** is the free step-4 nilpotent Lie
//! group on 2 generators in dimension 5 (Bonfiglioli 2007 §4.3.6, N=5 sibling
//! of the Engel N=4 group). Stratification: g₁⊕g₂⊕g₃⊕g₄ (step 4).
//!
//! Left-invariant fields on ℝ⁵ coords (x₁, x₂, x₃, x₄, x₅):
//! ```text
//! X₁ = ∂_{x₁}
//! X₂ = ∂_{x₂} + x₁·∂_{x₃} + (x₁²/2)·∂_{x₄} + (x₁³/6)·∂_{x₅}
//! X₃ = [X₁,X₂] = ∂_{x₃} + x₁·∂_{x₄} + (x₁²/2)·∂_{x₅}
//! X₄ = [X₁,X₃] = ∂_{x₄} + x₁·∂_{x₅}
//! X₅ = [X₁,X₄] = ∂_{x₅}
//! ```
//! Sub-Laplacian: L₅ = X₁² + X₂²
//!
//! # Palindromic Strang (order-2 real)
//!
//! S(τ) = exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)
//!
//! Each exp(σ·Xₖ²) is a 1D Gaussian convolution along the integral curve of Xₖ,
//! via 32-pt Gauss-Hermite quadrature (mirror Engel implementation).
//!
//! Order-2 tangency is an algebraic identity (Lemma 28.bis.7, palindromic
//! symmetry cancels the τ² term for any A,B; the τ³ residual is non-zero —
//! confirmed on a generic degree-10 probe). This is the inner map S used by
//! `ComplexTripleJump` in `carnot_complex.rs` to achieve order-4.
//!
//! # X₁ flow (trivial, no coupling)
//!
//! Φˢ_{X₁}(x₁,x₂,x₃,x₄,x₅) = (x₁+s, x₂, x₃, x₄, x₅)
//!
//! # X₂ flow (polynomial coupling, 5-coordinate)
//!
//! Φˢ_{X₂}(x₁,x₂,x₃,x₄,x₅) = (x₁, x₂+s, x₃+s·x₁, x₄+s²·x₁/2+s·x₃,
//!                                x₅+s³·x₁/6+s²·x₃/2+s·x₄)
//!
//! # References
//!
//! - Bonfiglioli-Lanconelli-Uguzzoni 2007 §4.3.6 (filiform N=5)
//! - Galkin-Remizov 2025 IJM Theorem 3.1 (K=2 tangency, conditionally extended)
//! - ADR-0136 Amendment 1 + math.md §28.bis.7
//! - `hormander_engel.rs` (structural mirror for D=4 Engel case)

extern crate alloc;
use core::marker::PhantomData;

use crate::{
    carnot_stepk_helpers::{lerp_idx_1d_f64, sample_5d},
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid_nd::GridFnND,
    hormander::{bracket_central_diff, HypoellipticChernoff, VectorField},
    hormander_engel_helpers::{GH32_NODES_ENGEL, GH32_WEIGHTS_ENGEL},
    scratch::ScratchPool,
};

// ─── Filiform N=5 left-invariant vector fields ───────────────────────────────

/// Left-invariant field X₁ = ∂_{x₁} on the filiform N=5 step-4 Carnot group.
///
/// Returns (1,0,0,0,0) at any point — constant unit vector in x₁-direction.
///
/// Reference: math.md §28.bis.7a; Bonfiglioli 2007 §4.3.6 N=5 sibling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Filiform5X1<F: SemiflowFloat = f64> {
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat> Filiform5X1<F> {
    /// Construct X₁ = ∂_{x₁} (zero-sized, infallible).
    #[must_use]
    pub fn new() -> Self {
        Self { _f: PhantomData }
    }
}

impl<F: SemiflowFloat> Default for Filiform5X1<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SemiflowFloat> VectorField<F, 5> for Filiform5X1<F> {
    /// X₁(x) = (1, 0, 0, 0, 0) — trivial, no coupling.
    fn evaluate(&self, _x: &[F; 5], out: &mut [F; 5]) -> Result<(), SemiflowError> {
        out[0] = F::one();
        out[1] = F::zero();
        out[2] = F::zero();
        out[3] = F::zero();
        out[4] = F::zero();
        Ok(())
    }
}

/// Left-invariant field X₂ on the filiform N=5 step-4 Carnot group.
///
/// X₂ = ∂_{x₂} + x₁·∂_{x₃} + (x₁²/2)·∂_{x₄} + (x₁³/6)·∂_{x₅}
///
/// Returns (0, 1, x₁, x₁²/2, x₁³/6) at point (x₁,x₂,x₃,x₄,x₅).
///
/// Reference: math.md §28.bis.7a.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Filiform5X2<F: SemiflowFloat = f64> {
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat> Filiform5X2<F> {
    /// Construct X₂ (zero-sized, infallible).
    #[must_use]
    pub fn new() -> Self {
        Self { _f: PhantomData }
    }
}

impl<F: SemiflowFloat> Default for Filiform5X2<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SemiflowFloat> VectorField<F, 5> for Filiform5X2<F> {
    /// X₂(x₁,x₂,x₃,x₄,x₅) = (0, 1, x₁, x₁²/2, x₁³/6).
    fn evaluate(&self, x: &[F; 5], out: &mut [F; 5]) -> Result<(), SemiflowError> {
        let half = from_f64::<F>(0.5_f64);
        let sixth = from_f64::<F>(1.0 / 6.0);
        let x1 = x[0];
        out[0] = F::zero();
        out[1] = F::one();
        out[2] = x1;
        out[3] = half * x1 * x1;
        out[4] = sixth * x1 * x1 * x1;
        Ok(())
    }
}

// ─── Zero drift field on ℝ⁵ ──────────────────────────────────────────────────

struct ZeroField5;

impl VectorField<f64, 5> for ZeroField5 {
    fn evaluate(&self, _x: &[f64; 5], out: &mut [f64; 5]) -> Result<(), SemiflowError> {
        *out = [0.0; 5];
        Ok(())
    }
}

// ─── HypoellipticChernoff<f64, 5, 2> constructor (filiform N=5) ──────────────

impl HypoellipticChernoff<f64, 5, 2> {
    /// Construct the filiform N=5 step-4 Carnot group Chernoff approximation.
    ///
    /// Sets X₁ = `Filiform5X1`, X₂ = `Filiform5X2` and verifies the step-2
    /// bracket structure at the origin: `[X₁,X₂]` ≈ (0,0,1,0,0) = X₃.
    ///
    /// Full step-4 verification (X₅=`[X₁,X₄]`) is covered by `T_CARNOT_STEP4`
    /// sympy oracle (`scripts/carnot_step4_kit.py`, PASS at architect time).
    ///
    /// # Errors
    /// - `DomainViolation` if `[X₁,X₂]` bracket check fails.
    pub fn new_filiform5() -> Result<Self, SemiflowError> {
        let x1 = Filiform5X1::<f64>::new();
        let x2 = Filiform5X2::<f64>::new();
        let origin = [0.0_f64; 5];
        let mut bracket_12 = [0.0_f64; 5];
        bracket_central_diff(&x1, &x2, &origin, &mut bracket_12)?;
        let eps = 1e-6_f64;
        let expected = [0.0, 0.0, 1.0, 0.0, 0.0];
        for (&got, &exp) in bracket_12.iter().zip(expected.iter()) {
            if (got - exp).abs() > eps {
                return Err(SemiflowError::DomainViolation {
                    what: "Filiform5 step-4 check: [X₁,X₂] deviates from X₃ at origin",
                    value: (got - exp).abs(),
                });
            }
        }
        let x0: alloc::boxed::Box<dyn VectorField<f64, 5>> = alloc::boxed::Box::new(ZeroField5);
        let diff: alloc::vec::Vec<alloc::boxed::Box<dyn VectorField<f64, 5>>> = alloc::vec![
            alloc::boxed::Box::new(Filiform5X1::<f64>::new()),
            alloc::boxed::Box::new(Filiform5X2::<f64>::new()),
        ];
        Ok(Self {
            x0_drift: x0,
            x_diff: diff,
            _f: PhantomData,
        })
    }
}

// ─── ChernoffFunction impl (palindromic Strang, order 2) ─────────────────────

/// State type: `GridFnND<f64, 5>` (5D tensor-product grid, N per axis).
///
/// Palindromic Strang-Hörmander:
/// `S(τ) = exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)`
///
/// Inner map for `ComplexTripleJump` in `carnot_complex.rs`.
impl ChernoffFunction<f64> for HypoellipticChernoff<f64, 5, 2> {
    type S = GridFnND<f64, 5>;

    /// Apply one palindromic Strang step to a 5D grid function.
    ///
    /// Legs: exp(τ/4·X₁²), exp(τ/2·X₂²), exp(τ/4·X₁²).
    /// Each leg uses 32-pt Gauss-Hermite quadrature (`GH32_NODES_ENGEL` constants).
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFnND<f64, 5>,
        dst: &mut GridFnND<f64, 5>,
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
        // Leg 1: exp(τ/4·X₁²)
        filiform5_diffuse_x1(src, &mut mid, tau * 0.25)?;
        let mut mid2 = GridFnND {
            values: alloc::vec![0.0_f64; n],
            grid: src.grid.clone(),
        };
        // Leg 2: exp(τ/2·X₂²)
        filiform5_diffuse_x2(&mid, &mut mid2, tau * 0.5)?;
        // Leg 3: exp(τ/4·X₁²)
        filiform5_diffuse_x1(&mid2, dst, tau * 0.25)?;
        Ok(())
    }

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

// ─── Sub-step helpers ─────────────────────────────────────────────────────────

/// Apply `exp(sigma·X₁²)` where X₁ = ∂_{x₁} (trivial, no coupling).
///
/// Φˢ_{X₁}(x₁,…,x₅) = (x₁+s, x₂, x₃, x₄, x₅)
/// (exp(σ·X₁²) f)(x) = π^{-1/2} Σ wₖ f(x₁−√(2σ)·ξₖ, x₂, x₃, x₄, x₅)
// Result kept for API symmetry with the Chernoff kernel call-sites that use `?`.
#[allow(clippy::unnecessary_wraps)]
fn filiform5_diffuse_x1(
    src: &GridFnND<f64, 5>,
    dst: &mut GridFnND<f64, 5>,
    sigma: f64,
) -> Result<(), SemiflowError> {
    if sigma <= 0.0 {
        dst.values.copy_from_slice(&src.values);
        return Ok(());
    }
    let g = &src.grid;
    let [n0, n1, n2, n3, n4] = [
        g.axes[0].n,
        g.axes[1].n,
        g.axes[2].n,
        g.axes[3].n,
        g.axes[4].n,
    ];
    let scale = libm::sqrt(2.0 * sigma);
    let pi_inv_sqrt = 1.0 / libm::sqrt(core::f64::consts::PI);
    for i4 in 0..n4 {
        for i3 in 0..n3 {
            for i2 in 0..n2 {
                for i1 in 0..n1 {
                    for i0 in 0..n0 {
                        let x0 = g.axes[0].x_at(i0);
                        let mut val = 0.0_f64;
                        for (&xi, &wi) in GH32_NODES_ENGEL.iter().zip(GH32_WEIGHTS_ENGEL.iter()) {
                            let x0_src = x0 - scale * xi;
                            let fv = sample_axis0_5d(src, x0_src, i1, i2, i3, i4);
                            val += wi * fv;
                        }
                        let flat = g.flat_idx(&[i0, i1, i2, i3, i4]);
                        dst.values[flat] = pi_inv_sqrt * val;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Apply `exp(sigma·X₂²)` using the X₂ integral curve.
///
/// Φˢ_{X₂}(x₁,x₂,x₃,x₄,x₅) = (x₁, x₂+s, x₃+s·x₁, x₄+s²·x₁/2+s·x₃,
///                               x₅+s³·x₁/6+s²·x₃/2+s·x₄)
///
/// Source (backwards-by-s): subtract s from the above.
// Result kept for API symmetry with the Chernoff kernel call-sites that use `?`.
#[allow(clippy::unnecessary_wraps)]
fn filiform5_diffuse_x2(
    src: &GridFnND<f64, 5>,
    dst: &mut GridFnND<f64, 5>,
    sigma: f64,
) -> Result<(), SemiflowError> {
    if sigma <= 0.0 {
        dst.values.copy_from_slice(&src.values);
        return Ok(());
    }
    let g = &src.grid;
    let [n0, n1, n2, n3, n4] = [
        g.axes[0].n,
        g.axes[1].n,
        g.axes[2].n,
        g.axes[3].n,
        g.axes[4].n,
    ];
    let scale = libm::sqrt(2.0 * sigma);
    let pi_inv_sqrt = 1.0 / libm::sqrt(core::f64::consts::PI);
    for i4 in 0..n4 {
        for i3 in 0..n3 {
            for i2 in 0..n2 {
                let x3c = g.axes[2].x_at(i2);
                let x4c = g.axes[3].x_at(i3);
                let x5c = g.axes[4].x_at(i4);
                for i1 in 0..n1 {
                    let x2n = g.axes[1].x_at(i1);
                    for i0 in 0..n0 {
                        let x1f = g.axes[0].x_at(i0);
                        let val = filiform5_x2_gh_sum(src, scale, x1f, x2n, x3c, x4c, x5c);
                        let flat = g.flat_idx(&[i0, i1, i2, i3, i4]);
                        dst.values[flat] = pi_inv_sqrt * val;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Accumulate the 32-point GH quadrature sum for one (x₁,x₂,x₃,x₄,x₅) point
/// in the real X₂ diffusion step.
///
/// Returns `Σ wₖ · f(Φ^{−s}(x))` before π^{-1/2} scaling. Extracted from
/// `filiform5_diffuse_x2` to satisfy the 50-line function cap.
// 5 coordinate arguments required for the 5D filiform flow; struct would add indirection.
#[allow(clippy::too_many_arguments)]
fn filiform5_x2_gh_sum(
    src: &GridFnND<f64, 5>,
    scale: f64,
    x1f: f64,
    x2n: f64,
    x3c: f64,
    x4c: f64,
    x5c: f64,
) -> f64 {
    let mut val = 0.0_f64;
    for (&xi, &wi) in GH32_NODES_ENGEL.iter().zip(GH32_WEIGHTS_ENGEL.iter()) {
        let s = scale * xi;
        // Source coords along X₂ integral curve (backwards by s):
        let x2_src = x2n - s;
        let x3_src = x3c - s * x1f;
        let x4_src = x4c - s * s * x1f * 0.5 - s * x3c;
        let x5_src = x5c - s * s * s * x1f / 6.0 - s * s * x3c * 0.5 - s * x4c;
        let fv = sample_5d(src, x1f, x2_src, x3_src, x4_src, x5_src);
        val += wi * fv;
    }
    val
}

/// Sample src at (`x0_src`, i1..i4 fixed) via clamped linear interp on axis 0.
///
/// X₁ diffuses only along axis 0; other axes are index-exact.
#[inline]
fn sample_axis0_5d(
    src: &GridFnND<f64, 5>,
    x0_src: f64,
    i1: usize,
    i2: usize,
    i3: usize,
    i4: usize,
) -> f64 {
    let (k0, k1, alpha) = lerp_idx_1d_f64(x0_src, &src.grid.axes[0]);
    let f0 = src.values[src.grid.flat_idx(&[k0, i1, i2, i3, i4])];
    let f1 = src.values[src.grid.flat_idx(&[k1, i1, i2, i3, i4])];
    f0 * (1.0 - alpha) + f1 * alpha
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::{grid::Grid1D, grid_nd::GridND};

    #[test]
    fn filiform5_x1_is_constant_unit() {
        let x1 = Filiform5X1::<f64>::new();
        let mut out = [0.0_f64; 5];
        x1.evaluate(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn filiform5_x2_at_origin() {
        let x2 = Filiform5X2::<f64>::new();
        let mut out = [0.0_f64; 5];
        x2.evaluate(&[0.0; 5], &mut out).unwrap();
        assert_eq!(out, [0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn filiform5_x2_off_axis() {
        let x2 = Filiform5X2::<f64>::new();
        let mut out = [0.0_f64; 5];
        x2.evaluate(&[2.0, 0.0, 0.0, 0.0, 0.0], &mut out).unwrap();
        // X₂(2,...) = (0, 1, 2, 2, 4/3)
        assert!((out[0] - 0.0).abs() < 1e-15);
        assert!((out[1] - 1.0).abs() < 1e-15);
        assert!((out[2] - 2.0).abs() < 1e-15);
        assert!((out[3] - 2.0).abs() < 1e-15); // x₁²/2 = 4/2
        assert!((out[4] - 4.0 / 3.0).abs() < 1e-14); // x₁³/6 = 8/6
    }

    #[test]
    fn new_filiform5_bracket_check_passes() {
        let result = HypoellipticChernoff::<f64, 5, 2>::new_filiform5();
        assert!(result.is_ok(), "Filiform5 bracket check failed");
    }

    #[test]
    fn filiform5_apply_into_smoke() {
        let ax = Grid1D::new(-2.0_f64, 2.0, 6).unwrap();
        let grid = GridND::<f64, 5>::new([ax; 5]).unwrap();
        let u0 = GridFnND::from_fn(grid.clone(), |x: &[f64; 5]| {
            libm::exp(-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) * 0.5)
        });
        let chernoff = HypoellipticChernoff::<f64, 5, 2>::new_filiform5().unwrap();
        let mut dst = GridFnND {
            values: alloc::vec![0.0_f64; u0.values.len()],
            grid: grid.clone(),
        };
        let mut scratch = crate::scratch::ScratchPool::new();
        chernoff
            .apply_into(0.05, &u0, &mut dst, &mut scratch)
            .unwrap();
        let max_val = dst.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.0, "post-step max must be positive");
        assert!(max_val.is_finite(), "post-step max must be finite");
    }
}
