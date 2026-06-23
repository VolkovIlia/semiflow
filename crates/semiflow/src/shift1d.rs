//! [`ShiftChernoff1D`] — Chernoff function for `L = a(x)∂² + b(x)∂ + c(x)`.
//!
//! Implements formula (6) of Theorem 6, Remizov 2025
//! (Vladikavkaz Math. J. 27(4), DOI 10.46698/a3908-1212-5385-q):
//!
//! ```text
//! (S(τ) f)(x) = ¼ f(x + 2√(a(x)·τ))
//!             + ¼ f(x − 2√(a(x)·τ))   ← MINUS sign (not plus)
//!             + ½ f(x + 2·b(x)·τ)     ← coefficient ½ (not ¼)
//!             + τ·c(x)·f(x)            ← linear in τ (not exp)
//! ```
//!
//! All four coefficients and signs are verified verbatim against the published
//! PDF in `.dev-docs/verification/theorem-6-correspondence.md`.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 1)
//!
//! `ShiftChernoff1D<F: SemiflowFloat = f64>` — the `= f64` default keeps all
//! existing call-sites compiling unchanged. `ShiftChernoff1D<f64>` implements
//! the `ChernoffFunction` trait (f64-monomorphic interface). Other `F` types
//! use `apply_f` directly (scalar path).

use alloc::sync::Arc;

use num_traits::Float;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion_storage::Storage3,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
};

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Chernoff function for the 1-D operator `L = a(x)∂²_x + b(x)∂_x + c(x)`.
///
/// Implements formula (6) of Theorem 6, Remizov 2025 — the four-term shift
/// approximation verifying `S'(0)f = Lf` (consistency order 1).
///
/// Uses `fn` pointers (no-alloc, no `Box<dyn Fn>`) so the struct is
/// `no_std`-friendly. For variable (pre-sampled-array) coefficients, use
/// [`ShiftChernoff1D::with_closure`] (ADR-0034 ext).
///
/// ## Generic-over-Float (ADR-0025)
///
/// `ShiftChernoff1D<F: SemiflowFloat = f64>` — `= f64` keeps all existing
/// call-sites unchanged. `ShiftChernoff1D<f64>` implements [`ChernoffFunction`]
/// (f64 path with SIMD `catmull_rom`). Non-f64 types use `apply_f`.
///
/// # Theorem 6 preconditions (caller's responsibility)
/// - `a(x) > 0` for all `x` in the domain (strict ellipticity).
/// - `a, b, c ∈ UC_b(ℝ)` with bounded derivatives up to order 3.
///
/// # Example
///
/// ```rust
/// use semiflow::{Grid1D, GridFn1D, ShiftChernoff1D};
/// let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
/// // Pure diffusion: a=0.5, b=0, c=0
/// let s = ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.0, grid);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let u1 = s.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
#[derive(Clone)]
pub struct ShiftChernoff1D<F: SemiflowFloat = f64> {
    /// Diffusion coefficient `a(x)`. Must satisfy `a(x) > 0` everywhere.
    pub a: fn(F) -> F,
    /// Drift coefficient `b(x)`.
    pub b: fn(F) -> F,
    /// Reaction coefficient `c(x)`.
    pub c: fn(F) -> F,
    /// Caller-supplied upper bound `‖c‖_∞` for the growth estimate.
    pub c_norm_bound: f64,
    /// Reference grid. Needed for `zeroed_like` / shape creation.
    pub grid: Grid1D<F>,
    /// Optional closure storage (set by `with_closure`; overrides fn-ptr fields).
    storage: Option<Storage3<F>>,
}

// ---------------------------------------------------------------------------
// impl ShiftChernoff1D<f64> — concrete f64 path (backwards-compatible)
// ---------------------------------------------------------------------------

impl ShiftChernoff1D<f64> {
    /// Construct a `ShiftChernoff1D` (f64, backwards-compatible).
    #[must_use]
    pub fn new(
        a: fn(f64) -> f64,
        b: fn(f64) -> f64,
        c: fn(f64) -> f64,
        c_norm_bound: f64,
        grid: Grid1D<f64>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            c_norm_bound,
            grid,
            storage: None,
        }
    }

    /// Construct a `ShiftChernoff1D` from owned closures (ADR-0034 ext).
    ///
    /// Enables variable `a(x)`, `b(x)`, `c(x)` via pre-sampled-array closures.
    /// Models `∂_t u = a(x)∂²u + b(x)∂u + c(x)u` (formula 6, Theorem 6).
    ///
    /// # Example
    ///
    /// ```rust
    /// use semiflow::{Grid1D, GridFn1D, ShiftChernoff1D};
    /// let grid = Grid1D::new(-1.0, 1.0, 32).unwrap();
    /// let s = ShiftChernoff1D::with_closure(
    ///     |_| 0.5_f64, |_| 0.0, |_| 0.0, 0.0, grid,
    /// );
    /// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    /// let u1 = s.apply_chernoff(0.01, &u0).unwrap();
    /// assert_eq!(u1.values.len(), 32);
    /// ```
    #[must_use]
    pub fn with_closure<A, B, C>(a: A, b: B, c: C, c_norm_bound: f64, grid: Grid1D<f64>) -> Self
    where
        A: Fn(f64) -> f64 + Send + Sync + 'static,
        B: Fn(f64) -> f64 + Send + Sync + 'static,
        C: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        fn _zero(_: f64) -> f64 {
            0.0
        }
        Self {
            a: _zero,
            b: _zero,
            c: _zero,
            c_norm_bound,
            grid,
            storage: Some(Storage3::Closure {
                f0: Arc::new(a),
                f1: Arc::new(b),
                f2: Arc::new(c),
            }),
        }
    }

    /// Evaluate `a(x)` — dispatches between fn-ptr and closure storage.
    #[inline]
    pub(crate) fn eval_a(&self, x: f64) -> f64 {
        match &self.storage {
            Some(s) => s.eval0(x),
            None => (self.a)(x),
        }
    }

    /// Evaluate `b(x)` — dispatches between fn-ptr and closure storage.
    #[inline]
    pub(crate) fn eval_b(&self, x: f64) -> f64 {
        match &self.storage {
            Some(s) => s.eval1(x),
            None => (self.b)(x),
        }
    }

    /// Evaluate `c(x)` — dispatches between fn-ptr and closure storage.
    #[inline]
    pub(crate) fn eval_c(&self, x: f64) -> f64 {
        match &self.storage {
            Some(s) => s.eval2(x),
            None => (self.c)(x),
        }
    }
}

// ---------------------------------------------------------------------------
// impl<F> ShiftChernoff1D<F> — generic path for non-f64 types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> ShiftChernoff1D<F> {
    /// Construct a `ShiftChernoff1D<F>` (generic version for non-f64 floats).
    #[must_use]
    pub fn new_generic(
        a: fn(F) -> F,
        b: fn(F) -> F,
        c: fn(F) -> F,
        c_norm_bound: f64,
        grid: Grid1D<F>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            c_norm_bound,
            grid,
            storage: None,
        }
    }

    /// Apply formula (6) — generic scalar path for non-f64 types.
    ///
    /// Uses `sample_generic` (scalar Catmull-Rom). For `F = f64`, use
    /// `ChernoffFunction::apply` to preserve the SIMD `catmull_rom` path.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0` or non-finite.
    /// - [`SemiflowError::DomainViolation`] if `a(x) < 0` or non-finite at any node.
    /// - [`SemiflowError::Unsupported`] from `f.sample_generic()`.
    pub fn apply_f(&self, tau: F, f: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError> {
        validate_tau_generic(tau)?;
        let mut out = f.zeroed_like();
        for i in 0..f.values.len() {
            out.values[i] = apply_at_node_generic(self, tau, f, i)?;
        }
        Ok(out)
    }

    /// Consistency order: 1.
    pub fn order_val(&self) -> u32 {
        1
    }

    /// Growth bound `(M, ω) = (1, c_norm_bound)`.
    pub fn growth_val(&self) -> (f64, f64) {
        (1.0, self.c_norm_bound)
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl for ShiftChernoff1D<f64>
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for ShiftChernoff1D<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order: `S'(0) f = a f'' + b f' + c f = L f` (order 1).
    fn order(&self) -> u32 {
        1
    }

    /// Growth bound `(1, ‖c‖_∞)`.
    fn growth(&self) -> Growth<f64> {
        Growth::new(1.0, self.c_norm_bound)
    }

    /// Allocation-free apply. See inherent `apply_chernoff` for allocating variant.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut crate::scratch::ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        let _ = scratch;
        validate_tau_f64(tau)?;
        let n = src.values.len();
        dst.values.resize(n, 0.0);
        crate::parallel1d::parallel_eval_into(&mut dst.values, |i| {
            apply_at_node_f64(self, tau, src, i)
        })
    }
}

// Phase 5a: additive impl — delegates to generic scalar apply_f path.
impl ChernoffFunction<f32> for ShiftChernoff1D<f32> {
    type S = GridFn1D<f32>;

    /// Consistency order 1 (mirrors f64 impl).
    fn order(&self) -> u32 {
        1
    }

    /// Growth bound `(1, ‖c‖_∞)`.
    fn growth(&self) -> Growth<f32> {
        Growth::new(1.0, self.c_norm_bound as f32)
    }

    /// Scalar apply: delegates to `apply_f` (generic scalar path; no SIMD in 5a).
    fn apply_into(
        &self,
        tau: f32,
        src: &GridFn1D<f32>,
        dst: &mut GridFn1D<f32>,
        _scratch: &mut crate::scratch::ScratchPool<f32>,
    ) -> Result<(), SemiflowError> {
        let result = self.apply_f(tau, src)?;
        dst.values.resize(result.values.len(), 0.0);
        dst.values.copy_from_slice(&result.values);
        dst.grid = result.grid;
        Ok(())
    }
}

impl ShiftChernoff1D<f64> {
    /// Allocating single-step apply (v3.0 replacement for v2.x `apply`).
    ///
    /// # Errors
    /// Same conditions as `apply_into`.
    pub fn apply_chernoff(
        &self,
        tau: f64,
        f: &GridFn1D<f64>,
    ) -> Result<GridFn1D<f64>, SemiflowError> {
        validate_tau_f64(tau)?;
        let n = f.values.len();
        let values = crate::parallel1d::parallel_eval(n, |i| apply_at_node_f64(self, tau, f, i))?;
        Ok(GridFn1D {
            values,
            grid: f.grid,
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers — f64 path
// ---------------------------------------------------------------------------

#[inline]
fn validate_tau_f64(tau: f64) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau,
        });
    }
    Ok(())
}

#[inline]
fn validate_a_f64(a_x: f64, x: f64) -> Result<(), SemiflowError> {
    if !a_x.is_finite() || a_x < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "a(x) must be finite and >= 0 (Theorem 6: inf a > 0)",
            value: x,
        });
    }
    Ok(())
}

/// Compute formula (6) at a single grid node `i` (f64, uses `f.sample` = SIMD `catmull_rom`).
#[inline]
fn apply_at_node_f64(
    s: &ShiftChernoff1D<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    i: usize,
) -> Result<f64, SemiflowError> {
    let x = s.grid.x_at(i);
    let a_x = s.eval_a(x);
    let b_x = s.eval_b(x);
    let c_x = s.eval_c(x);

    validate_a_f64(a_x, x)?;

    let s_diff = 2.0 * libm::sqrt(a_x * tau);
    let s_drift = 2.0 * b_x * tau;

    let term1 = 0.25 * f.sample(x + s_diff)?;
    let term2 = 0.25 * f.sample(x - s_diff)?;
    let term3 = 0.50 * f.sample(x + s_drift)?;
    let term4 = tau * c_x * f.values[i];

    Ok(term1 + term2 + term3 + term4)
}

// ---------------------------------------------------------------------------
// Private helpers — generic path
// ---------------------------------------------------------------------------

#[inline]
fn validate_tau_generic<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

#[inline]
fn validate_a_generic<F: SemiflowFloat>(a_x: F, x: F) -> Result<(), SemiflowError> {
    if !a_x.is_finite() || a_x < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "a(x) must be finite and >= 0 (Theorem 6: inf a > 0)",
            value: x.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// Compute formula (6) at a single grid node `i` (generic, uses `sample_generic`).
#[inline]
fn apply_at_node_generic<F: SemiflowFloat>(
    s: &ShiftChernoff1D<F>,
    tau: F,
    f: &GridFn1D<F>,
    i: usize,
) -> Result<F, SemiflowError> {
    let x = s.grid.x_at(i);
    let a_x = (s.a)(x);
    let b_x = (s.b)(x);
    let c_x = (s.c)(x);

    validate_a_generic(a_x, x)?;

    let two = from_f64::<F>(2.0);
    let quarter = from_f64::<F>(0.25);
    let half_v = from_f64::<F>(0.5);

    let s_diff = two * Float::sqrt(a_x * tau);
    let s_drift = two * b_x * tau;

    let term1 = quarter * f.sample_generic(x + s_diff)?;
    let term2 = quarter * f.sample_generic(x - s_diff)?;
    let term3 = half_v * f.sample_generic(x + s_drift)?;
    let term4 = tau * c_x * f.values[i];

    Ok(term1 + term2 + term3 + term4)
}
