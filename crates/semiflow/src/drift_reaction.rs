//! [`DriftReactionChernoff`] — RK2 characteristic-flow Chernoff for
//! `B = b(x)∂_x + c(x)` (v0.2.2, ADR-0006 Amendment 5, math.md §9.3).
//!
//! Implements a Runge–Kutta-2 (midpoint + trapezoidal) characteristic step:
//!
//! ```text
//! X(τ, x)    := x + τ · b(x + (τ/2) · b(x))            // midpoint foot-point
//! (R(τ) f)(x) = exp((τ/2) · (c(x) + c(X(τ,x)))) · f(X(τ,x)) // trapezoidal reaction
//! ```
//!
//! Local O(τ³) / **global O(τ²)** for `b ∈ C²(ℝ)`, `c ∈ C¹(ℝ)`.
//!
//! Sign convention: shift is **PLUS** (`X = x + τ·b(...)`).
//!
//! ## Adjoint (ADR-0114)
//!
//! The adjoint of `B = b·∂_x` in `L²([a,b])` is `Bᵀ = −b·∂_x` (integration
//! by parts). Therefore `Aᵀ = (−Δ + b·∂_x)ᵀ = −Δ − b·∂_x`. The transpose
//! semigroup `exp(τ Aᵀ)` is computed by the same RK2 formula with the drift
//! sign negated. `DriftReactionChernoff` implements `AdjointApply<f64>` and
//! overrides `ChernoffFunction::apply_adjoint_into` accordingly.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 1)
//!
//! `DriftReactionChernoff<F: SemiflowFloat = f64>` — the `= f64` default keeps
//! all existing call-sites compiling unchanged. `DriftReactionChernoff<f64>`
//! implements the `ChernoffFunction` trait (f64-monomorphic interface, preserving
//! `libm::exp` path). Other `F` types use `apply_f` (scalar path).

use alloc::sync::Arc;

use num_traits::Float;

use crate::{
    adjoint::AdjointApply,
    chernoff::{ChernoffFunction, Growth},
    diffusion_storage::Storage2,
    error::SemiflowError,
    float::{half, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Chernoff function for the drift+reaction operator `B f = b(x)f'(x) + c(x)f(x)`.
///
/// Implements an RK2 characteristic-flow step (midpoint + trapezoidal reaction),
/// achieving global order 2. Primary role: compose with a diffusion type via
/// [`crate::StrangSplit`] for convection-diffusion-reaction equations.
///
/// Uses `fn` pointers (no `Box<dyn Fn>`, `no_std`-friendly). `Copy`-able.
///
/// ## Generic-over-Float (ADR-0025)
///
/// `DriftReactionChernoff<F: SemiflowFloat = f64>` — `= f64` keeps all existing
/// call-sites unchanged. `DriftReactionChernoff<f64>` implements [`ChernoffFunction`].
/// Non-f64 types use `apply_f`.
///
/// # Caller invariants
/// - `b, c ∈ UC_b(ℝ)` with bounded derivatives to order 3.
/// - `c_norm_bound` is a valid upper bound for `‖c‖_∞`.
///
/// # Example
///
/// ```rust
/// use semiflow::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, DriftReactionChernoff};
/// let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
/// // Pure advection: b=0.5, c=0
/// let drift = DriftReactionChernoff::new(|_| 0.5, |_| 0.0, 0.0, grid);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let u1 = drift.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct DriftReactionChernoff<F: SemiflowFloat = f64> {
    /// Drift coefficient `b(x)`.
    pub b: fn(F) -> F,
    /// Reaction coefficient `c(x)`.
    pub c: fn(F) -> F,
    /// Caller-supplied upper bound for `‖c‖_∞`, used by `growth()`.
    pub c_norm_bound: f64,
    /// Reference grid geometry (used to iterate nodes and create output).
    pub grid: Grid1D<F>,
    /// Optional closure storage (set by `with_closure`; overrides fn-ptr fields).
    storage: Option<Storage2<F>>,
}

// ---------------------------------------------------------------------------
// impl DriftReactionChernoff<f64> — concrete f64 path (backwards-compatible)
// ---------------------------------------------------------------------------

impl DriftReactionChernoff<f64> {
    /// Construct a `DriftReactionChernoff` (f64, backwards-compatible).
    #[must_use]
    pub fn new(b: fn(f64) -> f64, c: fn(f64) -> f64, c_norm_bound: f64, grid: Grid1D<f64>) -> Self {
        Self {
            b,
            c,
            c_norm_bound,
            grid,
            storage: None,
        }
    }

    /// Construct a `DriftReactionChernoff` from owned closures (v2.3, ADR-0034 ext).
    ///
    /// Enables variable `b(x)` and `c(x)` via pre-sampled-array closures.
    /// Models `∂_t u = b(x)∂_x u + c(x)u` (math.md §9.3).
    ///
    /// # Example
    ///
    /// ```rust
    /// use semiflow::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, DriftReactionChernoff};
    /// let grid = Grid1D::new(-1.0, 1.0, 32).unwrap();
    /// let dr = DriftReactionChernoff::with_closure(
    ///     |_| 0.5_f64, |_| -0.1, 0.1, grid,
    /// );
    /// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    /// let u1 = dr.apply_chernoff(0.01, &u0).unwrap();
    /// assert_eq!(u1.values.len(), 32);
    /// ```
    #[must_use]
    pub fn with_closure<B, C>(b: B, c: C, c_norm_bound: f64, grid: Grid1D<f64>) -> Self
    where
        B: Fn(f64) -> f64 + Send + Sync + 'static,
        C: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        fn _zero(_: f64) -> f64 {
            0.0
        }
        Self {
            b: _zero,
            c: _zero,
            c_norm_bound,
            grid,
            storage: Some(Storage2::Closure {
                f0: Arc::new(b),
                f1: Arc::new(c),
            }),
        }
    }

    /// Evaluate `b(x)` — dispatches between fn-ptr and closure storage.
    #[inline]
    pub(crate) fn eval_b(&self, x: f64) -> f64 {
        match &self.storage {
            Some(s) => s.eval0(x),
            None => (self.b)(x),
        }
    }

    /// Evaluate `c(x)` — dispatches between fn-ptr and closure storage.
    #[inline]
    pub(crate) fn eval_c(&self, x: f64) -> f64 {
        match &self.storage {
            Some(s) => s.eval1(x),
            None => (self.c)(x),
        }
    }
}

// ---------------------------------------------------------------------------
// impl<F> DriftReactionChernoff<F> — generic path for non-f64 types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> DriftReactionChernoff<F> {
    /// Construct a `DriftReactionChernoff<F>` (generic version for non-f64 floats).
    #[must_use]
    pub fn new_generic(b: fn(F) -> F, c: fn(F) -> F, c_norm_bound: f64, grid: Grid1D<F>) -> Self {
        Self {
            b,
            c,
            c_norm_bound,
            grid,
            storage: None,
        }
    }

    /// Apply the RK2 characteristic formula — generic scalar path for non-f64 types.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0` or non-finite.
    /// - [`SemiflowError::Unsupported`] propagated from `f.sample_generic()`.
    pub fn apply_f(&self, tau: F, f: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError> {
        validate_tau_generic(tau)?;
        let mut out = f.zeroed_like();
        for i in 0..f.values.len() {
            out.values[i] = apply_at_node_generic(self, tau, f, i)?;
        }
        Ok(out)
    }

    /// Consistency order: 2.
    pub fn order_val(&self) -> u32 {
        2
    }

    /// Growth bound `(M, ω) = (1.0, c_norm_bound)`.
    pub fn growth_val(&self) -> (f64, f64) {
        (1.0, self.c_norm_bound)
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl for DriftReactionChernoff<f64>
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for DriftReactionChernoff<f64> {
    type S = GridFn1D<f64>;

    /// Apply the RK2 characteristic formula into `dst` (allocation-free).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0` or non-finite.
    /// - [`SemiflowError::Unsupported`] propagated from `f.sample()`.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        validate_tau_f64(tau)?;
        let n = src.values.len();
        dst.values.resize(n, 0.0);
        crate::parallel1d::parallel_eval_into(&mut dst.values, |i| {
            apply_at_node_f64(self, tau, src, i)
        })?;
        dst.grid = src.grid;
        Ok(())
    }

    /// Consistency order: 2.
    fn order(&self) -> u32 {
        2
    }

    /// Growth bound `(M, ω) = (1.0, c_norm_bound)`.
    fn growth(&self) -> Growth<f64> {
        Growth {
            multiplier: 1.0,
            omega: self.c_norm_bound,
        }
    }

    /// Transpose (adjoint) apply: `exp(τ Aᵀ) src` into `dst` (ADR-0114).
    ///
    /// `Aᵀ = (−Δ + b·∂_x)ᵀ = −Δ − b·∂_x` — same RK2 formula, drift negated.
    ///
    /// This override satisfies `|⟨S(τ)u,g⟩ − ⟨u,S*(τ)g⟩| ≤ C·τ³`
    /// (order-2 wrapper, p=2) for seeded-random `u`, `g`.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0` or non-finite.
    fn apply_adjoint_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        validate_tau_f64(tau)?;
        let n = src.values.len();
        dst.values.resize(n, 0.0);
        let grid = self.grid;
        // Capture b and c via closures to avoid a helper struct.
        // eval_b_negated: -b(x); eval_c: c(x) unchanged.
        let eval_b_neg = |x: f64| -self.eval_b(x);
        let eval_c = |x: f64| self.eval_c(x);
        for i in 0..n {
            dst.values[i] = apply_rk2_at_node(grid, tau, src, i, eval_b_neg, eval_c)?;
        }
        dst.grid = src.grid;
        Ok(())
    }
}

// Phase 5a: additive impl — delegates to generic scalar apply_f path.
impl ChernoffFunction<f32> for DriftReactionChernoff<f32> {
    type S = GridFn1D<f32>;

    /// Consistency order 2 (mirrors f64 impl).
    fn order(&self) -> u32 {
        2
    }

    /// Growth bound `(M, ω) = (1.0, c_norm_bound)`.
    fn growth(&self) -> Growth<f32> {
        Growth::new(1.0, self.c_norm_bound as f32)
    }

    /// Scalar apply: delegates to `apply_f` (generic scalar path).
    fn apply_into(
        &self,
        tau: f32,
        src: &GridFn1D<f32>,
        dst: &mut GridFn1D<f32>,
        _scratch: &mut ScratchPool<f32>,
    ) -> Result<(), SemiflowError> {
        let result = self.apply_f(tau, src)?;
        dst.values.resize(result.values.len(), 0.0);
        dst.values.copy_from_slice(&result.values);
        dst.grid = result.grid;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AdjointApply marker for DriftReactionChernoff<f64>
// ---------------------------------------------------------------------------

/// `AdjointApply` marker for `DriftReactionChernoff<f64>`.
///
/// Signals that `ChernoffFunction::apply_adjoint_into` is correctly overridden
/// (negated-drift transpose kernel). See ADR-0114.
impl AdjointApply<f64> for DriftReactionChernoff<f64> {}

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

/// RK2 characteristic step at a single grid node `i` (f64, uses `f.sample` = SIMD).
#[inline]
fn apply_at_node_f64(
    r: &DriftReactionChernoff<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    i: usize,
) -> Result<f64, SemiflowError> {
    apply_rk2_at_node(r.grid, tau, f, i, |xx| r.eval_b(xx), |xx| r.eval_c(xx))
}

/// Core RK2 step: generic over drift/reaction via function arguments.
///
/// Used by both `apply_into` (forward) and `apply_adjoint_into` (transpose,
/// with negated drift). Zero extra allocation.
#[inline]
fn apply_rk2_at_node(
    grid: Grid1D<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    i: usize,
    eval_b: impl Fn(f64) -> f64,
    eval_c: impl Fn(f64) -> f64,
) -> Result<f64, SemiflowError> {
    let x = grid.x_at(i);

    let b1 = eval_b(x);
    let x_mid = x + 0.5 * tau * b1;
    let b_mid = eval_b(x_mid);
    let x_foot = x + tau * b_mid;

    let c0 = eval_c(x);
    let c1 = eval_c(x_foot);
    let factor = libm::exp(0.5 * tau * (c0 + c1));

    let shifted = f.sample(x_foot)?;
    Ok(factor * shifted)
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

/// RK2 characteristic step at a single grid node `i` (generic, uses `sample_generic`).
#[inline]
fn apply_at_node_generic<F: SemiflowFloat>(
    r: &DriftReactionChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    i: usize,
) -> Result<F, SemiflowError> {
    let x = r.grid.x_at(i);
    let half_v = half::<F>();

    let b1 = (r.b)(x);
    let x_mid = x + half_v * tau * b1;
    let b_mid = (r.b)(x_mid);
    let x_foot = x + tau * b_mid;

    let c0 = (r.c)(x);
    let c1 = (r.c)(x_foot);
    let factor = Float::exp(half_v * tau * (c0 + c1));

    let shifted = f.sample_generic(x_foot)?;
    Ok(factor * shifted)
}
