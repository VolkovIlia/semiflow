//! Complex-λ Laplace-Chernoff resolvent (math.md §22.9, ADR-0127).
//!
//! Extends the shipped real-λ resolvent to **complex λ with `Re λ > ω`**
//! (Hille-Yosida, Pazy 1983 §1.5; Engel-Nagel 2000 §II.1).
//!
//! ## Formula (GL32 contour, real substitution `s = Re(λ) t`)
//!
//! ```text
//! R̃(λ) g = (1/Re λ) Σ_k w_k · exp(-i Im(λ)/Re(λ) · s_k) · (C(s_k/(Re(λ)·n)))^n g
//! ```
//!
//! The imaginary part of λ enters as a complex phase scalar per GL node;
//! the inner Chernoff steps remain real (`τ_k = s_k/(Re(λ)·n)` is real).
//!
//! ## SPEC TRAP (from ADR-0127 pre-flight, sub-check 3)
//!
//! The validity condition is **`Re λ > ω`**, NOT `|λ| > ω`.
//! λ = -0.5 + 5i has |λ| ≈ 5 but Re λ < 0: the integral DIVERGES (residual ~1e+92).
//! The guard `lambda.re() <= omega` is MANDATORY and fail-closed.
//!
//! ## Acceptance gate
//!
//! `G_CPLX_RES` (`RELEASE_BLOCKING`, ADR-0127): residual
//! `‖(λI-A) R̃(λ) g - g‖_∞ ≤ 1e-3` at λ=1+1i, A=∂²ₓ (N=64, reflecting BC, ω=0),
//! Gaussian g(x)=e^{-x²}. Expected ≈ 3e-5 (~30× margin).

use num_traits::ToPrimitive;

use crate::{
    chernoff::ChernoffFunction,
    complex::SemiflowComplex,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
    resolvent_quad::{GL32_NODES, GL32_WEIGHTS},
    schrodinger_complex_state::GridFnComplex1D,
    scratch::ScratchPool,
};

/// Number of trapezoid nodes — mirrors `TRAPEZOID_N` from `resolvent.rs`.
const TRAP_N: usize = 256;

impl<C, F> LaplaceChernoffResolvent<C, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    /// Complex-λ resolvent `R̃(λ) g` for `Re λ > ω` (math.md §22.9, ADR-0127).
    ///
    /// Uses the GL32 or [`TrapezoidWithTail`](crate::LaplaceQuadrature::TrapezoidWithTail) quadrature with a REAL substitution
    /// `s = Re(λ) t`; the imaginary part of λ contributes a per-node phase
    /// `exp(-i Im(λ)/Re(λ) · s_k)`. Inner Chernoff steps remain real.
    ///
    /// # Errors
    ///
    /// - `DomainViolation` if `lambda.re() <= omega` — **the spec trap**: large
    ///   `|λ|` with `Re λ ≤ ω` makes the Laplace integral DIVERGE (residual ~1e+92).
    ///   Always guards on `Re λ`, never on `|λ|`.
    /// - `DomainViolation` if `lambda` non-finite.
    /// - Propagates errors from inner `apply_into`.
    pub fn eval_complex<Cx>(&self, lambda: Cx, omega: F) -> EvalComplex<'_, C, F, Cx>
    where
        Cx: SemiflowComplex<Real = F>,
    {
        EvalComplex {
            resolvent: self,
            lambda,
            omega,
        }
    }
}

/// Builder returned by [`LaplaceChernoffResolvent::eval_complex`].
///
/// Call `.apply(g)` to evaluate the resolvent on a real initial datum `g`.
pub struct EvalComplex<'a, C, F, Cx>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
    Cx: SemiflowComplex<Real = F>,
{
    resolvent: &'a LaplaceChernoffResolvent<C, F>,
    lambda: Cx,
    omega: F,
}

impl<C, F, Cx> EvalComplex<'_, C, F, Cx>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
    Cx: SemiflowComplex<Real = F>,
{
    /// Evaluate `R̃(λ) g`, returning a complex grid function.
    ///
    /// # Errors
    ///
    /// `DomainViolation` if `Re λ ≤ ω` or `λ` non-finite.
    /// Propagates inner `apply_into` errors.
    pub fn apply(&self, g: &GridFn1D<F>) -> Result<GridFnComplex1D<Cx>, SemiflowError> {
        validate_complex_lambda(self.lambda, self.omega)?;
        match self.resolvent.quadrature {
            LaplaceQuadrature::GaussLaguerre32 => gl32_complex(self.resolvent, self.lambda, g),
            LaplaceQuadrature::TrapezoidWithTail { t_max } => {
                trap_complex(self.resolvent, self.lambda, g, t_max)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Guard: `Re λ > ω` (NOT `|λ| > ω` — see §22.9 SPEC TRAP).
fn validate_complex_lambda<Cx: SemiflowComplex>(
    lambda: Cx,
    omega: Cx::Real,
) -> Result<(), SemiflowError> {
    if !lambda.is_finite() {
        return Err(SemiflowError::DomainViolation {
            what: "complex lambda must be finite",
            value: f64::NAN,
        });
    }
    if lambda.re() <= omega {
        return Err(SemiflowError::DomainViolation {
            what: "Re(lambda) must be > omega (growth bound); NOT |lambda| > omega — see §22.9 SPEC TRAP",
            value: lambda.re().to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GL32 complex accumulation
// ---------------------------------------------------------------------------

fn gl32_complex<C, F, Cx>(
    res: &LaplaceChernoffResolvent<C, F>,
    lambda: Cx,
    g: &GridFn1D<F>,
) -> Result<GridFnComplex1D<Cx>, SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
    Cx: SemiflowComplex<Real = F>,
{
    use crate::state::State as _;
    let lam_re = lambda.re();
    let lam_im = lambda.im();
    let n_f = n_to_f::<F>(res.n);
    let inv_lam_re = F::one() / lam_re;
    let mut acc = zero_complex_like::<Cx, F>(g.grid);
    let mut scratch: ScratchPool<F> = ScratchPool::new();
    let mut buf_a = g.clone();
    let mut buf_b = g.clone();

    for k in 0..32 {
        let s_k = from_f64::<F>(GL32_NODES[k]);
        let w_k = from_f64::<F>(GL32_WEIGHTS[k]);
        let tau = s_k * inv_lam_re / n_f;
        buf_a.copy_from(g);
        run_n_steps_real(&res.inner, tau, &mut buf_a, &mut buf_b, res.n, &mut scratch)?;
        // phase = exp(-i * lam_im * s_k / lam_re) = cos(...) - i*sin(...)
        let phase_arg = -lam_im * s_k * inv_lam_re;
        let phase: Cx = Cx::from_polar(F::one(), phase_arg);
        let weight: Cx = Cx::from_real(w_k * inv_lam_re);
        axpy_complex_real(&mut acc, weight * phase, &buf_a);
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// TrapezoidWithTail complex accumulation
// ---------------------------------------------------------------------------

fn trap_complex<C, F, Cx>(
    res: &LaplaceChernoffResolvent<C, F>,
    lambda: Cx,
    g: &GridFn1D<F>,
    t_max: F,
) -> Result<GridFnComplex1D<Cx>, SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
    Cx: SemiflowComplex<Real = F>,
{
    use crate::state::State as _;
    let lam_re = lambda.re();
    let lam_im = lambda.im();
    let m = TRAP_N;
    let dt = t_max / n_to_f::<F>(m);
    let n_f = n_to_f::<F>(res.n);
    let mut acc = zero_complex_like::<Cx, F>(g.grid);
    let mut scratch: ScratchPool<F> = ScratchPool::new();
    let mut buf_a = g.clone();
    let mut buf_b = g.clone();

    for j in 0..=m {
        let t_j = n_to_f::<F>(j) * dt;
        let tau = t_j / n_f;
        // complex weight: e^{-(lam_re + i*lam_im)*t_j} * trapezoid_weight
        let damp = (-lam_re * t_j).exp();
        let phase_arg = -lam_im * t_j;
        let phase: Cx = Cx::from_polar(F::one(), phase_arg);
        let tw = trap_weight(j, m, dt);
        let weight: Cx = Cx::from_real(damp * tw) * phase;
        buf_a.copy_from(g);
        run_n_steps_real(&res.inner, tau, &mut buf_a, &mut buf_b, res.n, &mut scratch)?;
        axpy_complex_real(&mut acc, weight, &buf_a);
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert usize to F (counts ≤ 2^53, cast is exact).
#[inline]
#[allow(clippy::cast_precision_loss)]
fn n_to_f<F: SemiflowFloat>(n: usize) -> F {
    from_f64::<F>(n as f64)
}

/// Trapezoid weight: half at endpoints, full elsewhere.
#[inline]
fn trap_weight<F: SemiflowFloat>(j: usize, m: usize, dt: F) -> F {
    if j == 0 || j == m {
        from_f64::<F>(0.5) * dt
    } else {
        dt
    }
}

/// Zero-initialised complex grid function on `grid`.
fn zero_complex_like<Cx: SemiflowComplex<Real = F>, F: SemiflowFloat>(
    grid: Grid1D<F>,
) -> GridFnComplex1D<Cx> {
    GridFnComplex1D::from_fn(grid, |_| Cx::zero())
}

/// `acc[i] += alpha * src[i]` for complex `alpha`, real `src`.
fn axpy_complex_real<Cx, F>(acc: &mut GridFnComplex1D<Cx>, alpha: Cx, src: &GridFn1D<F>)
where
    Cx: SemiflowComplex<Real = F>,
    F: SemiflowFloat,
{
    for (a, &r) in acc.values.iter_mut().zip(src.values.iter()) {
        *a += alpha * Cx::from_real(r);
    }
}

/// Apply `n` real Chernoff steps in-place: `buf_a ↦ (C(tau))^n buf_a`.
///
/// On entry `buf_a` holds the initial state; `buf_b` is scratch.
fn run_n_steps_real<C, F>(
    inner: &C,
    tau: F,
    buf_a: &mut GridFn1D<F>,
    buf_b: &mut GridFn1D<F>,
    n: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    for _ in 0..n {
        inner.apply_into(tau, buf_a, buf_b, scratch)?;
        core::mem::swap(buf_a, buf_b);
    }
    Ok(())
}
