//! [`Diffusion4thChernoff`] — ζ⁴ Chernoff for `A_self = ∂_x(a(x)·∂_x)` (v0.6.0, ADR-0013).
//!
//! ζ⁴ formula (math.md §9.2.4, NORMATIVE):
//!
//! ```text
//! D_ζ⁴(τ) f(x) = D_γ(τ) f(x)
//!              + τ² · [ a·a'·f'''_FD⁴ + ½·a·a''·f''_FD⁶ + ¼·a'·a''·f'_FD⁶ ]
//! ```
//!
//! γ-A baseline (§9.2.3.A): `D_γ = S(τ/2) ∘ K(τ;a) ∘ S(τ/2)` — BIT-EQUAL to v0.5.0.
//!
//! 7-point Fornberg (1988) central FD stencils:
//!
//! ```text
//! f'  : [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] / Δ        [O(Δ⁶)]
//! f'' : [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] / Δ²   [O(Δ⁶)]
//! f''': [1/8, -1, 13/8, 0, -13/8, 1, -1/8] / Δ³             [O(Δ⁴)]
//! ```
//!
//! Stencil step: `Δ = max(3·dx, τ^{3/4})` (NORMATIVE, math.md §9.2.4).
//!
//! Sympy gates (`verify_v0_6_0_zeta4.py`): `Z⁴_τ⁰` ✓  `Z⁴_τ¹` ✓  `Z⁴_τ²` ✓
//!   Z⁴_const-a ✓  Z⁴_spatial-order ✓.
//!
//! Caller invariant: `a ∈ C⁵(ℝ)` (was C³ for v0.5.0 ζ-A).
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 1)
//!
//! `Diffusion4thChernoff<F: SemiflowFloat = f64>` — the `= f64` default keeps all
//! existing call-sites compiling unchanged. `Diffusion4thChernoff<f64>` implements
//! the `ChernoffFunction` trait (the f64-monomorphic interface, preserving SIMD
//! bit-equality). Other `F` types use `apply_f` directly (scalar path).

use alloc::sync::Arc;

use crate::{
    boundary::InterpKind,
    chernoff::{ChernoffFunction, Growth},
    diffusion_storage::Storage,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// Fourier-symbol weights — DO NOT CHANGE (§9.2.1 derivation, unique solution).
// Same as diffusion.rs; copied to avoid cross-module private dependency.
const W0: f64 = 7.0 / 12.0;
const W1: f64 = 3.0 / 16.0;
const W2: f64 = 1.0 / 48.0;

// 7-point Fornberg (1988) f' weights (offsets k = -3..+3).
const C1: [f64; 7] = [
    -1.0 / 60.0,
    3.0 / 20.0,
    -3.0 / 4.0,
    0.0,
    3.0 / 4.0,
    -3.0 / 20.0,
    1.0 / 60.0,
];

// 7-point Fornberg (1988) f'' weights (offsets k = -3..+3).
const C2: [f64; 7] = [
    1.0 / 90.0,
    -3.0 / 20.0,
    3.0 / 2.0,
    -49.0 / 18.0,
    3.0 / 2.0,
    -3.0 / 20.0,
    1.0 / 90.0,
];

// 7-point Fornberg (1988) f''' weights (offsets k = -3..+3).
const C3: [f64; 7] = [
    1.0 / 8.0,
    -1.0,
    13.0 / 8.0,
    0.0,
    -13.0 / 8.0,
    1.0,
    -1.0 / 8.0,
];

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Chernoff function for `A_self f = ∂_x(a(x)·∂_x f)` (v0.6.0 ζ⁴, ADR-0013).
///
/// ζ⁴ formula (math.md §9.2.4): γ-A baseline with 7-point O(Δ⁶) Fornberg FD
/// stencils. Additive sibling of [`crate::DiffusionChernoff`] — same 5-arg
/// constructor. `F = f64` default (ADR-0025); non-f64 types use `apply_f`.
///
/// # Caller invariants: `a(x) > 0`; `a ∈ C⁵(ℝ)`.
///
/// ```rust
/// use semiflow_core::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, Diffusion4thChernoff};
/// let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
/// let diff4 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let u1 = diff4.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct Diffusion4thChernoff<F: SemiflowFloat = f64> {
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
    ///
    /// `None` for the legacy `new` path (zero-heap fn-pointers).
    /// `Some` for the `with_closure` path (v2.3, ADR-0034 extension).
    storage: Option<Storage<F>>,
    /// ADR-0090: opt-in Chebyshev spectral sampling. Default `false`.
    ///
    /// When `true`, uses `ChebyshevSpectralWithBC` for virtual-node sampling.
    /// Set via `.with_chebyshev_sampling()`.
    /// f64-only; non-f64 callers receive `SemiflowError::Unsupported` at apply time.
    pub chebyshev_sampling: bool,
    /// Number of Chebyshev-Lobatto virtual nodes M when `chebyshev_sampling = true`.
    /// Default 64 when set via `.with_chebyshev_sampling()`.
    pub chebyshev_m: usize,
    /// ADR-0117: opt-in `OctonicHermite` (degree-9) spatial sampling for ζ⁶/ζ⁸ gates.
    ///
    /// Default `false`. Set via `.with_octonic_sampling()`. Overrides Chebyshev/Quintic
    /// when active; provides O(dx¹⁰) virtual-node floor (≈ 9.1e-16 at N=512).
    /// f64-only; non-f64 callers get `SemiflowError::Unsupported`.
    pub octonic_sampling: bool,
}

// ---------------------------------------------------------------------------
// impl Diffusion4thChernoff<f64> — concrete f64 path (backwards-compatible)
// ---------------------------------------------------------------------------

impl Diffusion4thChernoff<f64> {
    /// Construct a `Diffusion4thChernoff` (v0.6.0 ζ⁴, 5-arg constructor).
    ///
    /// Drop-in replacement for `DiffusionChernoff::new` — same argument order.
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
            chebyshev_sampling: false,
            chebyshev_m: 64,
            octonic_sampling: false,
        }
    }

    /// Construct a `Diffusion4thChernoff` from owned closures (v2.3, ADR-0034 ext).
    ///
    /// Enables variable `a(x)` via Python/FFI pre-sampled-array callbacks that
    /// cannot be expressed as bare `fn` pointers.  Closures are heap-owned via
    /// `Arc`; `Clone` is cheap (increments reference counts).
    ///
    /// Math ref: math.md §9.2.4 — same ζ⁴ formula; only the coefficient
    /// source changes from fn-ptrs to `Arc<dyn Fn>`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use semiflow_core::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, Diffusion4thChernoff};
    /// let grid = Grid1D::new(0.0, 1.0, 32).unwrap();
    /// let diff4 = Diffusion4thChernoff::with_closure(
    ///     |_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid,
    /// );
    /// let u0 = GridFn1D::from_fn(grid, |x| x * (1.0 - x));
    /// let u1 = diff4.apply_chernoff(0.01, &u0).unwrap();
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
        // Dummy fn-ptrs satisfy the public field types; compute uses `storage`.
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
            chebyshev_sampling: false,
            chebyshev_m: 64,
            octonic_sampling: false,
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
// impl<F> Diffusion4thChernoff<F> — generic path for non-f64 types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> Diffusion4thChernoff<F> {
    /// Construct a `Diffusion4thChernoff<F>` (generic version for non-f64 floats).
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
            chebyshev_sampling: false,
            chebyshev_m: 64,
            octonic_sampling: false,
        }
    }

    /// Opt-in to Chebyshev spectral sampling with default M=64 (ADR-0090).
    ///
    /// Exponential-accuracy spatial floor for smooth `f ∈ C^∞`. The Chebyshev M=64
    /// barycentric floor itself is ≤ 1e-15; intermediate `SepticHermite` virtual-node
    /// evaluations dominate at ≈ 1.49e-12 (ADR-0109). For sub-1.49e-12 precision,
    /// use `OctonicHermite` (ADR-0117) or `Grid1D::cheb_m` with a finer grid.
    ///
    /// Chebyshev wins over `octonic_sampling` when both are set. f64-only; non-f64
    /// callers will get a compile error from the ladder guard or a runtime `Unsupported`
    /// on `apply_into`.
    #[must_use]
    pub fn with_chebyshev_sampling(mut self) -> Self {
        self.chebyshev_sampling = true;
        self.chebyshev_m = 64;
        self
    }

    /// Opt-in to Chebyshev spectral sampling with explicit M (ADR-0090).
    ///
    /// M ∈ {8, 16, 32, 64, 128, 256, 512}. Higher M gives a tighter floor
    /// at ~2× cost per M doubling. Default is 64 via `.with_chebyshev_sampling()`.
    #[must_use]
    pub fn with_chebyshev_sampling_m(mut self, m: usize) -> Self {
        self.chebyshev_sampling = true;
        self.chebyshev_m = m;
        self
    }

    /// Remove Chebyshev spectral sampling (debugging / comparison only).
    ///
    /// Reverts to the default `CubicHermite` path. Use when you need to compare
    /// floor effects numerically.
    #[must_use]
    pub fn without_chebyshev_sampling(mut self) -> Self {
        self.chebyshev_sampling = false;
        self
    }

    /// Opt-in to `OctonicHermite` (degree-9) spatial sampling (ADR-0117).
    ///
    /// Sets O(dx¹⁰) virtual-node floor ≈ 9.1e-16 at N=512 — required for ζ⁶/ζ⁸
    /// `TRUTHFUL_ORDER` gate at N=4096/T=10 (ADR-0119 GO). Overrides Chebyshev/Quintic
    /// flags when active; both can be set but octonic takes priority in `apply_into`.
    #[must_use]
    pub fn with_octonic_sampling(mut self) -> Self {
        self.octonic_sampling = true;
        self
    }

    /// Apply `D_ζ⁴(τ)` to `f` — generic scalar path for non-f64 types.
    ///
    /// Uses `sample_generic` (scalar interpolation) for boundary dispatch.
    /// For `F = f64`, use `ChernoffFunction::apply` which preserves the SIMD path.
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

    /// Consistency order: 2 (ζ⁴, variable `a ∈ C⁵`).
    pub fn order_val(&self) -> u32 {
        2
    }

    /// Growth bound `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    pub fn growth_val(&self) -> (f64, f64) {
        (1.0, 0.0)
    }

    /// Allocation-free variant: writes `D_ζ⁴(τ) src` directly into `dst.values`.
    ///
    /// Uses the generic scalar path (same as [`apply_f`]) but avoids the
    /// intermediate `Vec` allocation by writing results element-by-element into
    /// the pre-allocated `dst` buffer. Required by `SchrodingerChernoff<F>`
    /// to satisfy the R4 zero-alloc invariant for non-f64 types.
    ///
    /// `dst.values` is resized to `src.values.len()` if needed (uses existing
    /// capacity from `ScratchPool`-managed slice — no extra allocation after
    /// warm-up).
    ///
    /// # Errors
    ///
    /// Same as [`apply_f`].
    pub fn apply_into_f(
        &self,
        tau: F,
        src: &GridFn1D<F>,
        dst: &mut GridFn1D<F>,
    ) -> Result<(), SemiflowError> {
        validate_tau_generic(tau)?;
        let n = src.values.len();
        dst.values.resize(n, F::zero());
        for i in 0..n {
            dst.values[i] = apply_at_node_generic(self, tau, src, i)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl for Diffusion4thChernoff<f64>
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for Diffusion4thChernoff<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order **2** (τ-axis: `S(τ)f = e^{τA}f + O(τ²)` for variable
    /// `a ∈ C⁵`). Spatial accuracy O(dx⁴) is **independent** of this and is
    /// verified by gate G3⁴ (convergence slope ≥ 3.95 on the heat oracle), not by
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
    /// When `chebyshev_sampling` is set (ADR-0090/ADR-0104), `src` is re-viewed through a
    /// `ChebyshevSpectralWithBC { m, oob_policy: Inherit }` grid (effective floor ≈ 1e-10; ADR-0104 §H4).
    /// When `octonic_sampling` is set (ADR-0117), uses `OctonicHermite` O(dx¹⁰).
    /// Chebyshev takes priority over Octonic when both flags are set.
    /// The default (`CubicHermite`) is bit-identical to v0.6.
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
        if self.octonic_sampling {
            // ADR-0117: re-view src through OctonicHermite (degree-9) grid.
            let oct_grid = self.grid.with_interp(InterpKind::OctonicHermite);
            let src_o = GridFn1D {
                grid: oct_grid,
                values: src.values.clone(),
            };
            crate::parallel1d::parallel_eval_into(&mut dst.values, |i| {
                apply_at_node_f64(self, tau, &src_o, i)
            })
        } else if self.chebyshev_sampling {
            // ADR-0090 / ADR-0104: re-view src through a ChebyshevSpectralWithBC grid.
            let cheb_grid = self.grid.with_interp(InterpKind::ChebyshevSpectralWithBC {
                m: self.chebyshev_m,
                oob_policy: crate::boundary::OobPolicy::Inherit,
            });
            let src_c = GridFn1D {
                grid: cheb_grid,
                values: src.values.clone(),
            };
            crate::parallel1d::parallel_eval_into(&mut dst.values, |i| {
                apply_at_node_f64(self, tau, &src_c, i)
            })
        } else {
            crate::parallel1d::parallel_eval_into(&mut dst.values, |i| {
                apply_at_node_f64(self, tau, src, i)
            })
        }
    }
}

// Phase 5a: additive impl — delegates to generic scalar apply_f path.
impl ChernoffFunction<f32> for Diffusion4thChernoff<f32> {
    type S = GridFn1D<f32>;

    /// Consistency order 2 (mirrors f64 impl; see math.md §11.1.bis).
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    fn growth(&self) -> Growth<f32> {
        Growth::contraction()
    }

    /// Scalar apply: delegates to `apply_into_f` (allocation-free generic path).
    fn apply_into(
        &self,
        tau: f32,
        src: &GridFn1D<f32>,
        dst: &mut GridFn1D<f32>,
        _scratch: &mut ScratchPool<f32>,
    ) -> Result<(), SemiflowError> {
        self.apply_into_f(tau, src, dst)
    }
}

// ---------------------------------------------------------------------------
// Private helpers — extracted to child modules to keep file under 500 lines
// ---------------------------------------------------------------------------

#[path = "diffusion4_helpers.rs"]
mod helpers_f64;
use helpers_f64::{apply_at_node_f64, validate_tau_f64};

#[path = "diffusion4_generic.rs"]
mod helpers_generic;
use helpers_generic::{apply_at_node_generic, validate_tau_generic};
