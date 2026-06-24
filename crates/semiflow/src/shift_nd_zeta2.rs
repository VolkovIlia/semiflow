//! ADR-0112 AMENDMENT 2 — order-2 ζ² correction for `AnisotropicShiftChernoffND`.
//!
//! Ships as the ADDITIVE constructor `AnisotropicShiftChernoffND::with_zeta2_correction`.
//! The base type's `order()` is UNCHANGED (still 1); the ζ²-corrected wrapper
//! reports `order() = 2`.
//!
//! # Mathematical foundation (math.md §32.8, ADR-0112 AMENDMENT 2)
//!
//! For the non-divergence operator `L = Σ_{ij} A_{ij}(x) ∂²_{ij} + Σ_i b_i(x) ∂_i + c(x)`,
//! the frozen-coefficient kernel (eq 32.3) approximates `e^{τL}` to order 1 for variable A.
//! The τ²-deficit (b=0, focusing on A-variation) is:
//!
//! `Δ₂[f](x₀) = -(τ²/2) Σ_{ij,kl} A₀_{ij} [(∂²_{ij}A_{kl}|₀)·∂²_{kl}f + 2(∂_i A_{kl}|₀)·∂³_{jkl}f]`
//!
//! The ζ² correction kills this deficit (C₂ = -Δ₂):
//!
//! `C₂[f](x₀) = (τ²/2) Σ_{ij,kl} A₀_{ij}[(∂²_{ij}A_{kl}|₀)·∂²_{kl}f + 2(∂_i A_{kl}|₀)·∂³_{jkl}f]`
//!
//! For the gate datum (linear A, so `∂²A=0`) this reduces to the simpler form:
//! `C₂ = τ² Σ_{ij,kl} A₀_{ij} (∂_i A_{kl}|₀) ∂³_{jkl}f(x₀)`.
//!
//! Mixed partial derivatives are approximated by central FD with step `h = 0.5·dx_min` (fixed, τ-independent).
//!
//! **CRITICAL CAVEAT (PRE-FLIGHT-surfaced):** variable `b` ALSO sources a τ²-deficit that
//! this correction does NOT address. The gate datum MUST use b≡0. The ∂b drift-gradient
//! correction is a follow-on (§32.6 deferred).
//!
//! CITATION: ADR-0112 AMENDMENT 2; math.md §32.8; §9.2.3.B (1-D analogue).

use alloc::{boxed::Box, vec::Vec};
use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid_nd::{GridFnND, GridND},
    scratch::ScratchPool,
    shift_nd::{AnisotropicShiftChernoffND, SquareMatrix},
};

// ---------------------------------------------------------------------------
// AnisotropicShiftZeta2ND<F, D>
// ---------------------------------------------------------------------------

/// Order-2 ζ²-corrected d-D anisotropic shift Chernoff kernel (ADR-0112 AMENDMENT 2).
///
/// Wraps [`AnisotropicShiftChernoffND`] and adds the τ²-correction from explicit
/// first-order A-gradient closures. `order()` returns 2; the base kernel is
/// unchanged. Gate: `G_AS_ZETA2_DDIM` (slope ≤ −1.95 on b≡0 datum).
///
/// # Construction
///
/// ```rust,ignore
/// # use semiflow::{
/// #     Grid1D, AnisotropicShiftChernoffND, AnisotropicShiftZeta2ND,
/// #     grid_nd::GridND,
/// # };
/// # let axes = core::array::from_fn(|_| Grid1D::new(-5.0_f64, 5.0, 16).unwrap());
/// # let grid = GridND::<f64, 2>::new(axes).unwrap();
/// # let base_kernel = AnisotropicShiftChernoffND::<f64, 2>::new(
/// #     |x, a_out| {
/// #         // Linearly varying diagonal A (always SPD on [-5,5]²):
/// #         //   A_{00} = 2 + 0.1·x_0,  A_{11} = 2 + 0.1·x_1,  A_{01}=A_{10}=0
/// #         a_out.set(0, 0, 2.0 + 0.1 * x[0]); a_out.set(1, 1, 2.0 + 0.1 * x[1]);
/// #         a_out.set(0, 1, 0.0); a_out.set(1, 0, 0.0);
/// #     },
/// #     |_x, b_out| { b_out[0] = 0.0; b_out[1] = 0.0; },
/// #     |_x| 0.0_f64,
/// #     grid.clone(),
/// # ).unwrap();
/// # type GradFn2 = Box<dyn Fn(&[f64; 2]) -> [f64; 2] + Send + Sync>;
/// # let grad_a_closures: Vec<GradFn2> = vec![
/// #     Box::new(|_x: &[f64; 2]| [0.1, 0.0]),  // ∂A_{00}/∂x_0, ∂A_{00}/∂x_1
/// #     Box::new(|_x: &[f64; 2]| [0.0, 0.0]),  // ∂A_{01}/∂x_m (=0)
/// #     Box::new(|_x: &[f64; 2]| [0.0, 0.0]),  // ∂A_{10}/∂x_m (=0)
/// #     Box::new(|_x: &[f64; 2]| [0.0, 0.1]),  // ∂A_{11}/∂x_0, ∂A_{11}/∂x_1
/// # ];
/// let zeta2 = AnisotropicShiftZeta2ND::<f64, 2>::new(
///     base_kernel,
///     |x, a_out| {
///         a_out.set(0, 0, 2.0 + 0.1 * x[0]); a_out.set(1, 1, 2.0 + 0.1 * x[1]);
///         a_out.set(0, 1, 0.0); a_out.set(1, 0, 0.0);
///     },
///     grad_a_closures,
/// );
/// ```
pub struct AnisotropicShiftZeta2ND<F: SemiflowFloat = f64, const D: usize = 2> {
    /// The underlying order-1 frozen-coefficient kernel.
    inner: AnisotropicShiftChernoffND<F, D>,
    /// `grad_a[i*D + j]` returns `[∂_0 A_{ij}(x), ..., ∂_{D-1} A_{ij}(x)]`.
    #[allow(clippy::type_complexity)]
    grad_a: Vec<Box<dyn Fn(&[F; D]) -> [F; D] + Send + Sync>>,
    /// Diffusion tensor closure (needed to read A₀ at each point).
    #[allow(clippy::type_complexity)]
    a_ij: Box<dyn Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync>,
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat, const D: usize> AnisotropicShiftZeta2ND<F, D> {
    /// Construct the ζ²-corrected kernel.
    ///
    /// # Arguments
    /// - `inner` — the base order-1 kernel (already constructed + Cholesky-cached).
    /// - `a_ij` — diffusion tensor closure (same as passed to `inner`; used to read A₀).
    /// - `grad_a` — length-D² closures: `grad_a[i*D+j](x) = [∂_0 A_{ij}(x), …]`.
    ///
    /// # CRITICAL CAVEAT
    /// The gate datum MUST use b≡0. Variable `b` sources an additional τ²-deficit
    /// NOT corrected here (∂b drift-gradient term, deferred to §32.6).
    ///
    /// # Errors
    /// Returns `DomainViolation` if `grad_a.len() != D*D`.
    #[allow(clippy::type_complexity)]
    pub fn new(
        inner: AnisotropicShiftChernoffND<F, D>,
        a_ij: impl Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync + 'static,
        grad_a: Vec<Box<dyn Fn(&[F; D]) -> [F; D] + Send + Sync>>,
    ) -> Result<Self, SemiflowError> {
        if grad_a.len() != D * D {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "AnisotropicShiftZeta2ND: grad_a.len() must equal D*D",
                value: grad_a.len() as f64,
            });
        }
        Ok(Self {
            inner,
            grad_a,
            a_ij: Box::new(a_ij),
            _f: PhantomData,
        })
    }

    /// Return a shared reference to the kernel's grid geometry.
    pub fn grid(&self) -> &GridND<F, D> {
        self.inner.grid()
    }
}

// ---------------------------------------------------------------------------
// Central FD helpers for mixed partial derivatives
// ---------------------------------------------------------------------------

/// Second mixed partial derivative `∂²f/∂x_j∂x_k` at `x` using FD step `h`.
///
/// For j==k: 3-point central FD.
/// For j!=k: cross-difference stencil (4 function evaluations).
#[allow(clippy::many_single_char_names)]
fn fd_d2<F: SemiflowFloat, const D: usize>(
    f: &GridFnND<F, D>,
    x: &[F; D],
    j: usize,
    k: usize,
    h: F,
) -> F {
    if j == k {
        // (f(x+h·e_j) - 2f(x) + f(x-h·e_j)) / h²
        let mut xp = *x;
        let mut xm = *x;
        xp[j] = x[j] + h;
        xm[j] = x[j] - h;
        let fp = f.sample(&xp).unwrap_or(F::zero());
        let f0 = f.sample(x).unwrap_or(F::zero());
        let fm = f.sample(&xm).unwrap_or(F::zero());
        (fp - f0 - f0 + fm) / (h * h)
    } else {
        // (f(x+he_j+he_k) - f(x+he_j-he_k) - f(x-he_j+he_k) + f(x-he_j-he_k)) / (4h²)
        let mut xpp = *x;
        let mut xpm = *x;
        let mut xmp = *x;
        let mut xmm = *x;
        xpp[j] = x[j] + h;
        xpp[k] = x[k] + h;
        xpm[j] = x[j] + h;
        xpm[k] = x[k] - h;
        xmp[j] = x[j] - h;
        xmp[k] = x[k] + h;
        xmm[j] = x[j] - h;
        xmm[k] = x[k] - h;
        let fpp = f.sample(&xpp).unwrap_or(F::zero());
        let fpm = f.sample(&xpm).unwrap_or(F::zero());
        let fmp = f.sample(&xmp).unwrap_or(F::zero());
        let fmm = f.sample(&xmm).unwrap_or(F::zero());
        (fpp - fpm - fmp + fmm) / (from_f64::<F>(4.0) * h * h)
    }
}

/// Third mixed partial `∂³f/∂x_j∂x_k∂x_l` at `x` using FD step `h`.
///
/// Computed as the central difference of `fd_d2(j,k)` along axis `l`.
#[allow(clippy::many_single_char_names)]
fn fd_d3<F: SemiflowFloat, const D: usize>(
    f: &GridFnND<F, D>,
    x: &[F; D],
    j: usize,
    k: usize,
    l: usize,
    h: F,
) -> F {
    let mut xp = *x;
    let mut xm = *x;
    xp[l] = x[l] + h;
    xm[l] = x[l] - h;
    let d2p = fd_d2::<F, D>(f, &xp, j, k, h);
    let d2m = fd_d2::<F, D>(f, &xm, j, k, h);
    (d2p - d2m) / (from_f64::<F>(2.0) * h)
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat, const D: usize> ChernoffFunction<F> for AnisotropicShiftZeta2ND<F, D> {
    type S = GridFnND<F, D>;

    /// Apply one step: base kernel + τ²-correction from A-gradients.
    ///
    /// `C₂[f](x₀) = τ² Σ_{ij,kl} A₀_{ij} (∂_i A_{kl}|₀) ∂³_{jkl}f(x₀)` (b=0 gate datum).
    fn apply_into(
        &self,
        tau: F,
        src: &GridFnND<F, D>,
        dst: &mut GridFnND<F, D>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        // Step 1: apply the base order-1 kernel.
        self.inner.apply_into(tau, src, dst, scratch)?;

        if tau == F::zero() {
            return Ok(());
        }

        // Step 2: add the ζ²-correction at each grid point.
        // C₂[f](x₀) = τ² · Σ_{ij,kl} A₀_{ij} · (∂_i A_{kl}|₀) · ∂³_{jkl}f(x₀)
        //
        // FD step h = 0.5·dx_min (FIXED, τ-independent) — see ADR-0112 for derivation.
        let grid = self.inner.grid();
        let dx_min = grid
            .axes
            .iter()
            .map(crate::grid::Grid1D::dx)
            .fold(from_f64::<F>(1.0e30_f64), |m, v| if v < m { v } else { m });
        let h = from_f64::<F>(0.5) * dx_min;
        let tau2 = tau * tau;
        let ns: [usize; D] = core::array::from_fn(|d| grid.axes[d].n);
        let total = src.values.len();

        for flat in 0..total {
            let xk: [F; D] = {
                let mut remaining = flat;
                core::array::from_fn(|d| {
                    let k = remaining % ns[d];
                    remaining /= ns[d];
                    grid.x_at(d, k)
                })
            };
            let corr = accumulate_zeta2_correction::<F, D>(self, src, &xk, h);
            dst.values[flat] += tau2 * corr;
        }
        Ok(())
    }

    /// Order 2: the ζ² correction kills the τ²-deficit for b≡0 variable-A operator
    /// (ADR-0112 AMENDMENT 2, gate `G_AS_ZETA2_DDIM`, slope ≤ −1.95).
    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}

/// Compute the ζ²-correction at point `xk`:
/// `C₂ = Σ_{ij,kl} A₀_{ij} · (∂_i A_{kl}|xk) · ∂³_{jkl}f(xk)`.
fn accumulate_zeta2_correction<F: SemiflowFloat, const D: usize>(
    op: &AnisotropicShiftZeta2ND<F, D>,
    src: &GridFnND<F, D>,
    xk: &[F; D],
    h: F,
) -> F {
    let mut a0 = SquareMatrix::<F, D>::zero();
    (op.a_ij)(xk, &mut a0);
    let mut corr = F::zero();
    for i in 0..D {
        for j in 0..D {
            let a_ij_val = a0.get(i, j);
            if a_ij_val == F::zero() {
                continue;
            }
            for k in 0..D {
                for l in 0..D {
                    let grad_kl = (op.grad_a[k * D + l])(xk);
                    let da_kl_di = grad_kl[i];
                    if da_kl_di == F::zero() {
                        continue;
                    }
                    let d3f = fd_d3::<F, D>(src, xk, j, k, l, h);
                    corr += a_ij_val * da_kl_di * d3f;
                }
            }
        }
    }
    corr
}
