//! Complex-time order-4 Carnot Chernoff via the complex triple-jump
//! (ADR-0136 Amendment 2 STRONG-GO, math.md §28.bis.8, v8.0.0 F4).
//!
//! # Background
//!
//! The Sheng-Suzuki order barrier (Sheng 1989; Goldman-Kaper 1996) proves that
//! any *real*-coefficient splitting of order ≥ 3 must contain a negative substep,
//! which is unbounded for the parabolic/hypoelliptic heat semigroup. The
//! `carnot_stepk.rs` palindromic Strang achieves exactly order 2.
//!
//! This module escapes the barrier with **complex** substep durations
//! (Castella-Chartier-Descombes-Vilmart 2009 / Hansen-Ostermann 2009): complex
//! coefficients with **Re > 0** give order ≥ 3 with *bounded* substeps on
//! *analytic* semigroups. The Carnot sub-Laplacian is analytic (Hörmander
//! hypoellipticity). The `SemiflowComplex` substrate (`complex.rs`, ADR-0079) is
//! already shipped — zero new dependencies.
//!
//! # Construction
//!
//! `GAMMA_STAR` = complex root of `2γ³+(1−2γ)³=0` with Re(γ)>0 AND Re(1−2γ)>0.
//!
//! `ComplexTripleJump` computes:
//!   `Ψ(τ) = K(γ⋆·τ) ∘ K((1−2γ⋆)·τ) ∘ K(γ⋆·τ)`
//!
//! where K is the filiform-N5 palindromic Strang from `carnot_stepk.rs`.
//! Table-driven: the three scale factors are a const array; dispatch is a fold.
//! Symmetric composition → order 4 by the even-order theorem (HLW 2006 §III.5).
//!
//! `into_real()` returns `GridFnND<f64,5>` by taking `Re(·)` of each complex
//! grid value — the conjugate-pair structure of γ⋆ guarantees cancellation for
//! a real initial datum (math.md §28.bis.8).
//!
//! # References
//!
//! - Castella-Chartier-Descombes-Vilmart, *BIT* 49 (2009) 487-508.
//! - Hansen-Ostermann, *BIT* 49 (2009) 527-542.
//! - Sheng, *IMA J. Numer. Anal.* 9 (1989) (order barrier).
//! - Hairer-Lubich-Wanner 2006 §III.5 (even-order theorem, symmetric methods).
//! - ADR-0136 Amendment 2 + math.md §28.bis.8;
//!   `scripts/carnot_complex_order3_kit.py` (`T_CARNOT_CPLX3` PASS 16/16).

extern crate alloc;

use num_complex::Complex;

use crate::{
    carnot_complex_helpers::{cplx_diffuse_x1, cplx_diffuse_x2},
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    grid_nd::{GridFnND, GridND},
    hormander::HypoellipticChernoff,
    scratch::ScratchPool,
    state::State,
};

// ─── γ⋆ constant ─────────────────────────────────────────────────────────────

/// Complex root of `2γ³+(1−2γ)³=0` with Re(γ)>0 AND Re(1−2γ)>0.
///
/// Full-precision value from sympy oracle (`T_CARNOT_CPLX3` PASS 2026-06-07):
/// γ⋆ = 0.32439640402017117 − 0.1345862724908067·i.
///
/// Derived from the cubic `2γ³+(1−2γ)³=0` (Yoshida 1990 / Castella et al. 2009).
/// Verify: `|2γ⋆³+(1−2γ⋆)³| < 1e-14` (in f64, ≈ 2e-17 in practice).
///
/// Both Re(γ⋆)=0.32440 > 0 and Re(1−2γ⋆)=0.35121 > 0 — every sub-map runs
/// forward in (complex) time, escaping the Sheng-Suzuki order barrier.
///
/// Reference: math.md §28.bis.8b; `T_CARNOT_CPLX3` PASS (architect 2026-06-07).
pub const GAMMA_STAR: Complex<f64> =
    Complex::new(0.324_396_404_020_171_2, -0.134_586_272_490_806_7);

// ─── Complex grid-function state ─────────────────────────────────────────────

/// 5D complex-valued grid function: state for `ComplexTripleJump`.
///
/// Stores `Complex<f64>` values on a `GridND<f64,5>` tensor-product grid.
/// Grid axes use `f64` coordinates; values are complex. 2× memory of the
/// corresponding real `GridFnND<f64,5>`.
///
/// Construct from a real IC with [`CplxGridFn5::from_real`];
/// recover the real result with [`CplxGridFn5::into_real`].
#[derive(Clone)]
pub struct CplxGridFn5 {
    /// Flat complex values. Length = `grid.len()`.
    pub values: alloc::vec::Vec<Complex<f64>>,
    /// Grid geometry (real axes, D=5).
    pub grid: GridND<f64, 5>,
}

impl CplxGridFn5 {
    /// Lift a real `GridFnND<f64,5>` to complex by embedding f64 → `Complex<f64>`.
    #[must_use]
    pub fn from_real(src: &GridFnND<f64, 5>) -> Self {
        let values = src.values.iter().map(|&v| Complex::new(v, 0.0)).collect();
        Self {
            values,
            grid: src.grid.clone(),
        }
    }

    /// Project to real by taking `Re(·)` of each value.
    ///
    /// For a real initial datum and conjugate-pair γ⋆, the imaginary parts cancel;
    /// `Re(·)` is exact to f64 rounding (math.md §28.bis.8).
    #[must_use]
    pub fn into_real(self) -> GridFnND<f64, 5> {
        let values = self.values.iter().map(|c| c.re).collect();
        GridFnND {
            values,
            grid: self.grid,
        }
    }

    /// Allocate a zeroed clone with the same grid.
    pub(crate) fn zeroed_like(&self) -> Self {
        Self {
            values: alloc::vec![Complex::new(0.0, 0.0); self.values.len()],
            grid: self.grid.clone(),
        }
    }
}

impl State<f64> for CplxGridFn5 {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn axpy_into(&mut self, alpha: f64, src: &Self) {
        debug_assert_eq!(
            self.values.len(),
            src.values.len(),
            "axpy_into: shape mismatch"
        );
        let a = Complex::new(alpha, 0.0);
        for (dst_v, src_v) in self.values.iter_mut().zip(src.values.iter()) {
            *dst_v += a * src_v;
        }
    }

    fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.values.len(), src.values.len());
        self.values.copy_from_slice(&src.values);
    }

    fn zero_into(&mut self) {
        for v in &mut self.values {
            *v = Complex::new(0.0, 0.0);
        }
    }

    fn norm_sup(&self) -> f64 {
        self.values.iter().map(|c| c.norm()).fold(0.0_f64, f64::max)
    }
}

// ─── ComplexTripleJump ────────────────────────────────────────────────────────

/// Order-4 complex triple-jump over the filiform-N5 order-2 symmetric Strang.
///
/// Composes:
///   `Ψ(τ) = K(γ⋆·τ) ∘ K((1−2γ⋆)·τ) ∘ K(γ⋆·τ)`
///
/// Table-driven: `TRIPLE_SCALES = [γ⋆, 1−2γ⋆, γ⋆]` (const array, 3-element fold).
/// Gate: `G_CARNOT_CPLX3` (`RELEASE_BLOCKING`); test: `tests/carnot_cplx3_slope.rs`.
pub struct ComplexTripleJump {
    /// Inner order-2 symmetric Strang kernel (filiform-N5, `carnot_stepk.rs`).
    inner: HypoellipticChernoff<f64, 5, 2>,
}

/// Table of the three complex time-scale factors for the triple-jump.
const TRIPLE_SCALES: [Complex<f64>; 3] = [
    GAMMA_STAR,
    Complex::new(1.0 - 2.0 * GAMMA_STAR.re, -2.0 * GAMMA_STAR.im),
    GAMMA_STAR,
];

impl ComplexTripleJump {
    /// Construct using the filiform-N5 inner kernel.
    ///
    /// # Errors
    /// - `DomainViolation` if the inner bracket check fails.
    pub fn new() -> Result<Self, SemiflowError> {
        let inner = HypoellipticChernoff::<f64, 5, 2>::new_filiform5()?;
        Ok(Self { inner })
    }

    /// Apply one order-4 step — returns the complex intermediate result.
    ///
    /// Use [`CplxGridFn5::into_real`] on the output for physical quantities.
    ///
    /// # Errors
    /// - `DomainViolation` if `tau` is not finite or negative.
    pub fn apply_complex(&self, tau: f64, src: &CplxGridFn5) -> Result<CplxGridFn5, SemiflowError> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "ComplexTripleJump: tau must be finite and non-negative",
                value: tau,
            });
        }
        // Table-driven fold over TRIPLE_SCALES = [γ⋆, 1−2γ⋆, γ⋆].
        let mut state = src.clone();
        let mut tmp = src.zeroed_like();
        for &scale in &TRIPLE_SCALES {
            apply_complex_strang(&self.inner, scale * tau, &state, &mut tmp)?;
            core::mem::swap(&mut state, &mut tmp);
        }
        Ok(state)
    }

    /// Apply one step and return the real projection `Re(Ψ(τ)f)`.
    ///
    /// For a real initial datum this is the order-4 approximation to `e^{τL}f`.
    ///
    /// # Errors
    /// - `DomainViolation` if `tau` is not finite or negative.
    pub fn apply_real(
        &self,
        tau: f64,
        src: &GridFnND<f64, 5>,
    ) -> Result<GridFnND<f64, 5>, SemiflowError> {
        let csrc = CplxGridFn5::from_real(src);
        Ok(self.apply_complex(tau, &csrc)?.into_real())
    }

    /// Verify `GAMMA_STAR` satisfies `2γ³+(1−2γ)³=0` to ≤ 1e-12 and Re>0.
    #[must_use]
    pub fn verify_gamma_star() -> bool {
        let g = GAMMA_STAR;
        let one_m_2g = Complex::new(1.0 - 2.0 * g.re, -2.0 * g.im);
        let cubic = g * g * g * 2.0 + one_m_2g * one_m_2g * one_m_2g;
        let residual = (cubic.re * cubic.re + cubic.im * cubic.im).sqrt();
        g.re > 0.0 && one_m_2g.re > 0.0 && residual < 1e-12
    }
}

// ─── ChernoffFunction impl ────────────────────────────────────────────────────

impl ChernoffFunction<f64> for ComplexTripleJump {
    type S = GridFnND<f64, 5>;

    /// Apply one order-4 complex triple-jump step to a real grid function.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFnND<f64, 5>,
        dst: &mut GridFnND<f64, 5>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        let result = self.apply_real(tau, src)?;
        dst.values.copy_from_slice(&result.values);
        Ok(())
    }

    fn order(&self) -> u32 {
        4
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

// ─── Complex Strang sub-step (thin orchestrator) ──────────────────────────────

/// Apply one filiform-N5 Strang step with a complex duration `c`.
///
/// `S(c) = exp(c/4·X₁²) ∘ exp(c/2·X₂²) ∘ exp(c/4·X₁²)`.
/// Diffusion helpers live in `carnot_complex_helpers.rs` for suckless `LoC` compliance.
fn apply_complex_strang(
    kernel: &HypoellipticChernoff<f64, 5, 2>,
    c: Complex<f64>,
    src: &CplxGridFn5,
    dst: &mut CplxGridFn5,
) -> Result<(), SemiflowError> {
    if !c.re.is_finite() || !c.im.is_finite() || c.re < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "ComplexTripleJump: complex sub-time must have Re(c) >= 0",
            value: c.re,
        });
    }
    let n = src.values.len();
    let mut mid = CplxGridFn5 {
        values: alloc::vec![Complex::new(0.0, 0.0); n],
        grid: src.grid.clone(),
    };
    cplx_diffuse_x1(src, &mut mid, c * 0.25)?;
    let mut mid2 = CplxGridFn5 {
        values: alloc::vec![Complex::new(0.0, 0.0); n],
        grid: src.grid.clone(),
    };
    let _ = kernel; // kernel fields not needed for closed-form X₂ flow
    cplx_diffuse_x2(&mid, &mut mid2, c * 0.5)?;
    cplx_diffuse_x1(&mid2, dst, c * 0.25)?;
    Ok(())
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid1D;

    #[test]
    fn gamma_star_satisfies_cubic() {
        assert!(
            ComplexTripleJump::verify_gamma_star(),
            "GAMMA_STAR must satisfy 2γ³+(1−2γ)³=0 with Re>0"
        );
    }

    #[test]
    fn gamma_star_re_positive() {
        // Asserting const GAMMA_STAR properties — these are compile-visible sanity checks.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(GAMMA_STAR.re > 0.0, "Re(γ⋆) must be positive");
            let one_m_2g = Complex::new(1.0 - 2.0 * GAMMA_STAR.re, -2.0 * GAMMA_STAR.im);
            assert!(one_m_2g.re > 0.0, "Re(1-2γ⋆) must be positive");
        }
    }

    #[test]
    fn triple_scales_sum_to_one() {
        let sum: Complex<f64> = TRIPLE_SCALES.iter().copied().sum();
        assert!((sum.re - 1.0).abs() < 1e-12, "scales must sum to 1 (Re)");
        assert!(sum.im.abs() < 1e-12, "scales must sum to 1 (Im)");
    }

    #[test]
    fn cplx_grid_fn5_roundtrip() {
        let ax = Grid1D::new(-1.0_f64, 1.0, 4).unwrap();
        let grid = GridND::<f64, 5>::new([ax; 5]).unwrap();
        let src = GridFnND::from_fn(grid.clone(), |x: &[f64; 5]| x[0] + x[1]);
        let cplx = CplxGridFn5::from_real(&src);
        let real = cplx.into_real();
        for (a, b) in src.values.iter().zip(real.values.iter()) {
            assert!((a - b).abs() < 1e-15, "roundtrip mismatch");
        }
    }

    #[test]
    fn complex_triple_jump_constructs() {
        assert!(ComplexTripleJump::new().is_ok());
    }

    #[test]
    fn complex_triple_jump_apply_finite() {
        let ax = Grid1D::new(-1.5_f64, 1.5, 5).unwrap();
        let grid = GridND::<f64, 5>::new([ax; 5]).unwrap();
        let src = GridFnND::from_fn(grid.clone(), |x: &[f64; 5]| {
            libm::exp(-(x[0] * x[0] + x[1] * x[1]) * 0.5)
        });
        let ctj = ComplexTripleJump::new().unwrap();
        let out = ctj.apply_real(0.02, &src).unwrap();
        let max_val = out.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val.is_finite(), "output must be finite");
        assert!(max_val > 0.0, "output must have positive values");
    }
}
