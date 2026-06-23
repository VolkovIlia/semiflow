//! [`Diffusion6thChernoff`] — ζ⁶ Chernoff for `A_self = ∂_x(a(x)·∂_x)` (v0.7.0, ADR-0015).
//!
//! ζ⁶ formula (math.md §9.2.6, NORMATIVE):
//!
//! ```text
//! D_ζ⁶(τ) f(x) = D_γ⁶(τ) f(x)
//!              + τ² · [ a·a'·f'''_FD9 + ½·a·a''·f''_FD9 + ¼·a'·a''·f'_FD9 ]
//! ```
//!
//! γ⁶-A baseline: `D_γ⁶ = S(τ/2) ∘ K7(τ;a) ∘ S(τ/2)` — 7-point K-kernel.
//!
//! 7-point K-kernel weights (P=5, Fourier-symbol ξ⁶-matched, NORMATIVE):
//!
//! ```text
//! K7_W0 = 67/120,  K7_W1 = 27/128,  K7_W2 = 1/192,  K7_W3 = 3/640
//! Shifts: h = 2√(a·τ),  H = 2√(3·a·τ),  J = 2√(5·a·τ)
//! K7 f|_x = W0·f(x) + W1·[f(x+h)+f(x-h)] + W2·[f(x+H)+f(x-H)] + W3·[f(x+J)+f(x-J)]
//! ```
//!
//! 9-point Fornberg (1988) central FD stencils (offsets k = -4..+4):
//!
//! ```text
//! f'  : [+1/280, -4/105, +1/5, -4/5, 0, +4/5, -1/5, +4/105, -1/280] / Δ    [O(Δ⁸)]
//! f'' : [-1/560, +8/315, -1/5, +8/5, -205/72, +8/5, -1/5, +8/315, -1/560] / Δ² [O(Δ⁸)]
//! f''': [-7/240, +3/10, -169/120, +61/30, 0, -61/30, +169/120, -3/10, +7/240] / Δ³ [O(Δ⁶)]
//! ```
//!
//! Stencil step: `Δ = max(4·dx, τ^{1/2})` (NORMATIVE, math.md §9.2.6).
//!
//! Sympy gates: `Z6_tau0` ✓  `Z6_tau1` ✓  `Z6_tau2` ✓  `Z6_const_a` ✓  `Z6_spatial_order` ✓.
//!
//! Caller invariant: `a ∈ C⁷(ℝ)` (was C⁵ for ζ⁴).
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 1)
//!
//! `Diffusion6thChernoff<F: SemiflowFloat = f64>` — the `= f64` default keeps all
//! existing call-sites compiling unchanged. `Diffusion6thChernoff<f64>` implements
//! the `ChernoffFunction` trait (f64-monomorphic interface, preserving SIMD
//! bit-equality). Other `F` types use `apply_f` directly (scalar path).

use alloc::sync::Arc;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion_storage::Storage,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// 7-point K-kernel weights (P=5, NORMATIVE — DO NOT CHANGE).
// ---------------------------------------------------------------------------
const K7_W0: f64 = 67.0 / 120.0;
const K7_W1: f64 = 27.0 / 128.0;
const K7_W2: f64 = 1.0 / 192.0;
const K7_W3: f64 = 3.0 / 640.0;
const K7_P: f64 = 5.0; // third pair: J = 2·sqrt(K7_P · a · τ)

// ---------------------------------------------------------------------------
// 9-point Fornberg (1988) FD weights (offsets k = -4..+4).
// ---------------------------------------------------------------------------

// f' stencil — O(Δ⁸).
const C1_9: [f64; 9] = [
    1.0 / 280.0,
    -4.0 / 105.0,
    1.0 / 5.0,
    -4.0 / 5.0,
    0.0,
    4.0 / 5.0,
    -1.0 / 5.0,
    4.0 / 105.0,
    -1.0 / 280.0,
];

// f'' stencil — O(Δ⁸).
const C2_9: [f64; 9] = [
    -1.0 / 560.0,
    8.0 / 315.0,
    -1.0 / 5.0,
    8.0 / 5.0,
    -205.0 / 72.0,
    8.0 / 5.0,
    -1.0 / 5.0,
    8.0 / 315.0,
    -1.0 / 560.0,
];

// f''' stencil — O(Δ⁶).
const C3_9: [f64; 9] = [
    -7.0 / 240.0,
    3.0 / 10.0,
    -169.0 / 120.0,
    61.0 / 30.0,
    0.0,
    -61.0 / 30.0,
    169.0 / 120.0,
    -3.0 / 10.0,
    7.0 / 240.0,
];

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Chernoff function for `A_self f = ∂_x(a(x)·∂_x f)` (v0.7.0 ζ⁶, ADR-0015).
///
/// Implements the ζ⁶ formula (math.md §9.2.6): 7-point K-kernel baseline with
/// 9-point O(Δ⁸) Fornberg FD stencils, achieving consistency order 2 with the
/// highest spatial accuracy available in the diffusion family (`a ∈ C⁷`).
///
/// ADDITIVE sibling to [`crate::Diffusion4thChernoff`] (ζ⁴) — identical
/// 5-arg constructor. Callers switch from ζ⁴ to ζ⁶ by changing one type name.
///
/// ## Generic-over-Float (ADR-0025)
///
/// `Diffusion6thChernoff<F: SemiflowFloat = f64>` — the `= f64` default keeps all
/// existing call-sites compiling unchanged. `Diffusion6thChernoff<f64>` implements
/// [`ChernoffFunction`] (SIMD path, bit-equal to v0.8.x). Non-f64 types use `apply_f`.
///
/// # Caller invariants
/// - `a(x) > 0` everywhere (strict ellipticity).
/// - `a ∈ C⁷(ℝ)` with bounded derivatives through order 8.
/// - Constant-`a`: pass `|_| F::zero()` for both derivatives.
///
/// # Example
///
/// ```rust
/// use semiflow::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, Diffusion6thChernoff};
/// let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
/// let diff6 = Diffusion6thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let u1 = diff6.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct Diffusion6thChernoff<F: SemiflowFloat = f64> {
    /// Diffusion coefficient `a(x)`. Caller MUST guarantee `a(x) > 0`.
    pub a: fn(F) -> F,
    /// First derivative `a'(x)`. Pass `|_| F::zero()` for constant `a`.
    pub a_prime: fn(F) -> F,
    /// Second derivative `a''(x)`. Pass `|_| F::zero()` for constant `a`.
    pub a_double_prime: fn(F) -> F,
    /// Upper bound for `‖a‖_∞` (diagnostics only; not used in compute).
    pub a_norm_bound: f64,
    /// Reference grid geometry (node iteration and output allocation).
    pub grid: Grid1D<F>,
    /// Optional closure storage (set by `with_closure`; overrides fn-ptr fields).
    storage: Option<Storage<F>>,
}

// ---------------------------------------------------------------------------
// impl Diffusion6thChernoff<f64> — concrete f64 path (backwards-compatible)
// ---------------------------------------------------------------------------

impl Diffusion6thChernoff<f64> {
    /// Construct a `Diffusion6thChernoff` (v0.7.0 ζ⁶, 5-arg constructor).
    ///
    /// Drop-in replacement for `Diffusion4thChernoff::new` — same argument order.
    #[must_use]
    pub fn new(
        a: fn(f64) -> f64,
        a_prime: fn(f64) -> f64,
        a_double_prime: fn(f64) -> f64,
        a_norm_bound: f64,
        grid: Grid1D<f64>,
    ) -> Self {
        Self {
            a,
            a_prime,
            a_double_prime,
            a_norm_bound,
            grid,
            storage: None,
        }
    }

    /// Construct a `Diffusion6thChernoff` from owned closures (v2.3, ADR-0034 ext).
    ///
    /// Enables variable `a(x)` via Python/FFI pre-sampled-array callbacks.
    /// Math ref: math.md §9.2.6 — same ζ⁶ formula; only the coefficient source changes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use semiflow::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, Diffusion6thChernoff};
    /// let grid = Grid1D::new(0.0, 1.0, 32).unwrap();
    /// let diff6 = Diffusion6thChernoff::with_closure(
    ///     |_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid,
    /// );
    /// let u0 = GridFn1D::from_fn(grid, |x| x * (1.0 - x));
    /// let u1 = diff6.apply_chernoff(0.01, &u0).unwrap();
    /// assert_eq!(u1.values.len(), 32);
    /// ```
    #[must_use]
    pub fn with_closure<A, P, D>(
        a: A,
        a_prime: P,
        a_double_prime: D,
        a_norm_bound: f64,
        grid: Grid1D<f64>,
    ) -> Self
    where
        A: Fn(f64) -> f64 + Send + Sync + 'static,
        P: Fn(f64) -> f64 + Send + Sync + 'static,
        D: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        fn _zero(_: f64) -> f64 {
            0.0
        }
        Self {
            a: _zero,
            a_prime: _zero,
            a_double_prime: _zero,
            a_norm_bound,
            grid,
            storage: Some(Storage::Closure {
                a: Arc::new(a),
                a_prime: Arc::new(a_prime),
                a_double_prime: Arc::new(a_double_prime),
            }),
        }
    }

    /// Evaluate `a(x)` — dispatches between fn-ptr and closure storage.
    #[inline]
    pub(crate) fn eval_a(&self, x: f64) -> f64 {
        match &self.storage {
            Some(s) => s.eval_a(x),
            None => (self.a)(x),
        }
    }

    /// Evaluate `a'(x)` — dispatches between fn-ptr and closure storage.
    #[inline]
    pub(crate) fn eval_ap(&self, x: f64) -> f64 {
        match &self.storage {
            Some(s) => s.eval_ap(x),
            None => (self.a_prime)(x),
        }
    }

    /// Evaluate `a''(x)` — dispatches between fn-ptr and closure storage.
    #[inline]
    pub(crate) fn eval_app(&self, x: f64) -> f64 {
        match &self.storage {
            Some(s) => s.eval_app(x),
            None => (self.a_double_prime)(x),
        }
    }
}

// ---------------------------------------------------------------------------
// impl<F> Diffusion6thChernoff<F> — generic path for non-f64 types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> Diffusion6thChernoff<F> {
    /// Construct a `Diffusion6thChernoff<F>` (generic version for non-f64 floats).
    #[must_use]
    pub fn new_generic(
        a: fn(F) -> F,
        a_prime: fn(F) -> F,
        a_double_prime: fn(F) -> F,
        a_norm_bound: f64,
        grid: Grid1D<F>,
    ) -> Self {
        Self {
            a,
            a_prime,
            a_double_prime,
            a_norm_bound,
            grid,
            storage: None,
        }
    }

    /// Apply `D_ζ⁶(τ)` to `f` — generic scalar path for non-f64 types.
    ///
    /// Uses `sample_generic` (scalar interpolation). For `F = f64`, use
    /// `ChernoffFunction::apply` to preserve SIMD path.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0`, non-finite, or
    ///   `a(x_pre) ≤ 0` / non-finite at `x_pre`.
    /// - [`SemiflowError::Unsupported`] propagated from `f.sample_generic()`.
    pub fn apply_f(&self, tau: F, f: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError> {
        validate_tau_generic(tau)?;
        let mut out = f.zeroed_like();
        for i in 0..f.values.len() {
            out.values[i] = apply_at_node_generic(self, tau, f, i)?;
        }
        Ok(out)
    }

    /// Consistency order: 2 (ζ⁶, variable `a ∈ C⁷`).
    pub fn order_val(&self) -> u32 {
        2
    }

    /// Growth bound `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    pub fn growth_val(&self) -> (f64, f64) {
        (1.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl for Diffusion6thChernoff<f64>
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for Diffusion6thChernoff<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order **2** (τ-axis: `S(τ)f = e^{τA}f + O(τ²)` for variable
    /// `a ∈ C⁷`). Spatial accuracy O(dx⁶) is **independent** of this and is
    /// verified by gate G3⁶ (convergence slope ≥ 5.85 on the heat oracle), not by
    /// `order()`. See math.md §11.1.bis (v0.6.1 NORMATIVE clarification) and
    /// audit-findings-v0_6_0.md D1.
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction (inherited from γ-A).
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }

    /// Allocation-free override: writes directly into `dst.values` via `parallel_eval_into`.
    ///
    /// Bit-identical to [`apply`] by construction: same `apply_at_node_f64` per node.
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
        })
    }
}

// Phase 5b: f32 SIMD path — uses apply_at_node_f32 (catmull_rom_f32 + fd9_f32).
impl ChernoffFunction<f32> for Diffusion6thChernoff<f32> {
    type S = GridFn1D<f32>;

    /// Consistency order 2 (mirrors f64 impl; see math.md §11.1.bis).
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    fn growth(&self) -> Growth<f32> {
        Growth::contraction()
    }

    /// f32 SIMD apply (Phase 5b): uses `apply_at_node_f32` with `catmull_rom_f32`
    /// + `fd9_f32` SIMD dispatchers. Bit-identical to scalar via `FORCE_SCALAR`.
    fn apply_into(
        &self,
        tau: f32,
        src: &GridFn1D<f32>,
        dst: &mut GridFn1D<f32>,
        _scratch: &mut ScratchPool<f32>,
    ) -> Result<(), SemiflowError> {
        helpers_f32::validate_tau_f32(tau)?;
        let n = src.values.len();
        dst.values.resize(n, 0.0);
        for i in 0..n {
            dst.values[i] = helpers_f32::apply_at_node_f32(self, tau, src, i)?;
        }
        dst.grid = src.grid;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private helpers — extracted to child modules to keep file under 500 lines
// ---------------------------------------------------------------------------

#[path = "diffusion6_helpers.rs"]
mod helpers_f64;
use helpers_f64::{apply_at_node_f64, validate_tau_f64};

#[path = "diffusion6_generic.rs"]
mod helpers_generic;
use helpers_generic::{apply_at_node_generic, validate_tau_generic};

#[path = "diffusion6_helpers_f32.rs"]
mod helpers_f32;
