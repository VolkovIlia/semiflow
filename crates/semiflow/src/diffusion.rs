//! [`DiffusionChernoff`] — ζ-A Chernoff for `A_self = ∂_x(a(x)·∂_x)` (v0.3.0, ADR-0008).
//!
//! ζ-A formula (math.md §9.2.3.B, NORMATIVE):
//!
//! ```text
//! D_ζ(τ) f(x) = D_γ(τ) f(x)
//!             + τ² · [ a·a'·f''' + ½·a·a''·f'' + ¼·a'·a''·f' ]
//! ```
//!
//! γ-A baseline (§9.2.3.A): `D_γ = S(τ/2) ∘ K(τ;a) ∘ S(τ/2)` where
//! `(S(s) g)(x) := g(x + s·a'(x))` and K uses weights `(7/12, 3/16, 1/48)`.
//!
//! Sympy gates (`verify_v0_3_0_zeta.py`): `Z_τ⁰` ✓  `Z_τ¹` ✓  `Z_τ²` ✓  Z_const-a ✓.
//!
//! **API break**: `new` takes 5 args `(a, a_prime, a_double_prime, a_norm_bound, grid)`.
//! Constant-`a` migration: insert `|_| 0.0` for both derivative args.
//!
//! ## v0.12.0 — `with_closure` API (ADR-0034)
//!
//! `DiffusionChernoff::with_closure` accepts owned closures. Storage lives in
//! `diffusion_storage`. Generic over `F: SemiflowFloat = f64` (ADR-0025);
//! the `f64` monomorphic path preserves bit-equality with `Diffusion4thChernoff`.

use alloc::sync::Arc;

use num_traits::Float;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion_storage::Storage,
    error::SemiflowError,
    float::{from_f64, half, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// Private helpers (f64 and generic) extracted to keep this file ≤500 lines (batch H8).
// Included directly so helpers share the same module namespace without path gymnastics.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/diffusion_helpers.rs"
));

// Fourier-symbol weights — DO NOT CHANGE (§9.2.1 derivation, unique solution).
pub(super) const W0: f64 = 7.0 / 12.0;
pub(super) const W1: f64 = 3.0 / 16.0;
pub(super) const W2: f64 = 1.0 / 48.0;

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Chernoff function for `A_self f = ∂_x(a(x)·∂_x f) = a·f'' + a'·f'`.
///
/// Implements the ζ-A formula (math.md §9.2.3.B): γ-A baseline plus τ²-correction,
/// achieving consistency order 2 for variable `a ∈ C³`.
///
/// ## Constructors
///
/// - [`DiffusionChernoff::new`] — takes three `fn(F) -> F` pointers (zero-heap,
///   backward-compatible with all v0.11.0 call-sites).
/// - [`DiffusionChernoff::new_const_a`] — constant `a(x) ≡ a_value` fast path
///   (v0.13.0, D1). Skips S-shift inner loop and ζ-A τ²-correction automatically.
/// - [`DiffusionChernoff::with_closure`] — takes owned closures `Fn(F) -> F + Send + Sync`
///   (v0.12.0, ADR-0034). Enables variable `a(x)` via FFI / `PyO3` / `WASM` callbacks.
/// - [`DiffusionChernoff::with_closure_local`] — like `with_closure` but without
///   `Send + Sync` bounds (WASM single-threaded path only).
///
/// ## `Copy` removal (v0.12.0, ADR-0034)
///
/// `DiffusionChernoff` no longer implements `Copy`. `Clone` is preserved (cheap
/// via `Arc` on the closure variant; bitwise on the fn-ptr variant). External
/// callers that relied on `Copy` (none in-tree) should replace `let dc2 = dc;`
/// with `let dc2 = dc.clone();`.
///
/// ## Generic-over-Float (ADR-0025)
///
/// `DiffusionChernoff<F: SemiflowFloat = f64>`.  `DiffusionChernoff<f64>` implements
/// [`ChernoffFunction`] for use with [`crate::ChernoffSemigroup`].
///
/// # Caller invariants
/// - `a(x) > 0` everywhere (strict ellipticity; required by `validate_a_x`).
/// - `a ∈ C³(ℝ)` with bounded derivatives through order 4.
/// - Constant-`a`: prefer [`Self::new_const_a`] (fast path); or pass `|_| F::zero()`
///   for both derivative args to [`Self::new`] (both are correct, fast path is faster).
///
/// # Example
///
/// ```rust
/// use semiflow_core::{Grid1D, GridFn1D, DiffusionChernoff};
/// let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
/// // Constant-a diffusion: a=1.0
/// let diff = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let u1 = diff.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct DiffusionChernoff<F: SemiflowFloat = f64> {
    /// Internal coefficient storage — fn-ptrs or heap-owned closures.
    storage: Storage<F>,
    /// Upper bound for `‖a‖_∞` (diagnostics only; not used in compute).
    pub a_norm_bound: f64,
    /// Reference grid geometry (node iteration and output allocation).
    pub grid: Grid1D<F>,
}

impl<F: SemiflowFloat> DiffusionChernoff<F> {
    /// Construct a `DiffusionChernoff` from fn-pointers (v0.3.0 ζ-A, 5-arg constructor).
    ///
    /// All 39 existing call-sites compile unchanged. Internally stores `Storage::FnPtr`.
    #[must_use]
    pub fn new(
        a: fn(F) -> F,
        a_prime: fn(F) -> F,
        a_double_prime: fn(F) -> F,
        a_norm_bound: f64,
        grid: Grid1D<F>,
    ) -> Self {
        Self {
            storage: Storage::FnPtr {
                a,
                a_prime,
                a_double_prime,
            },
            a_norm_bound,
            grid,
        }
    }

    /// Construct a `DiffusionChernoff` from owned closures with `Send + Sync` (ADR-0034).
    ///
    /// Enables variable `a(x)` across FFI / `PyO3` / `WASM` boundaries. Internally
    /// stores `Storage::Closure` with `Arc`-wrapped closures for cheap `Clone`.
    ///
    /// Use this for runtime-parameterised `a` (e.g. Python callbacks, CFL adaptation,
    /// WASM JS callbacks wrapped in a `Send + Sync` newtype by the binding crate).
    /// Use [`Self::new`] for compile-time-constant `a` (simpler, zero-heap).
    ///
    /// ## WASM note (ADR-0034)
    ///
    /// `js_sys::Function` is not `Send + Sync`. The `semiflow-wasm` crate wraps it
    /// in a newtype with `unsafe impl Send + Sync` before calling this constructor.
    /// That unsafe is confined to the WASM crate (which has `#![allow(unsafe_code)]`).
    #[must_use]
    pub fn with_closure<A, P, D>(
        a: A,
        a_prime: P,
        a_double_prime: D,
        a_norm_bound: f64,
        grid: Grid1D<F>,
    ) -> Self
    where
        A: Fn(F) -> F + Send + Sync + 'static,
        P: Fn(F) -> F + Send + Sync + 'static,
        D: Fn(F) -> F + Send + Sync + 'static,
    {
        Self {
            storage: Storage::Closure {
                a: Arc::new(a),
                a_prime: Arc::new(a_prime),
                a_double_prime: Arc::new(a_double_prime),
            },
            a_norm_bound,
            grid,
        }
    }

    /// Construct a `DiffusionChernoff` from owned closures WITHOUT `Send + Sync`.
    ///
    /// Convenience alias for [`Self::with_closure`] intended for WASM callers.
    /// Because `semiflow-core` enforces `#![deny(unsafe_code)]`, closures stored
    /// here MUST still be `Send + Sync`. The WASM binding crate provides the
    /// `unsafe impl Send + Sync` wrapper for `js_sys::Function` before calling
    /// this function — the wrapper is in `semiflow-wasm` which allows unsafe.
    ///
    /// On multi-threaded targets this is identical to `with_closure`.
    #[must_use]
    pub fn with_closure_local<A, P, D>(
        a: A,
        a_prime: P,
        a_double_prime: D,
        a_norm_bound: f64,
        grid: Grid1D<F>,
    ) -> Self
    where
        A: Fn(F) -> F + Send + Sync + 'static,
        P: Fn(F) -> F + Send + Sync + 'static,
        D: Fn(F) -> F + Send + Sync + 'static,
    {
        // Same storage as with_closure. The "local" distinction is in the WASM
        // binding layer (unsafe wrapper), not here in semiflow-core.
        Self::with_closure(a, a_prime, a_double_prime, a_norm_bound, grid)
    }

    /// Construct a `DiffusionChernoff` for constant diffusion `a(x) ≡ a_value`.
    ///
    /// Fast path added in v0.13.0 (Wave D, D1): because `a'(x) ≡ 0` and
    /// `a''(x) ≡ 0`, the inner Strang shift (`x_pre = x + τ/2·a'(x)`) reduces
    /// to `x_pre = x`, and the entire ζ-A τ²-correction evaluates to zero.
    /// Both reductions are applied automatically at call-time via the `ConstA`
    /// storage variant — no extra configuration needed.
    ///
    /// ## When to use
    ///
    /// Use whenever `a(x)` is a compile-time or runtime constant and does not
    /// depend on the spatial coordinate `x`. For spatially-varying `a(x)`, use
    /// [`Self::new`] (fn-ptr) or [`Self::with_closure`] (heap closure).
    ///
    /// # Example
    ///
    /// ```rust
    /// use semiflow_core::{Grid1D, GridFn1D, DiffusionChernoff};
    /// let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
    /// // Constant-a diffusion a=0.5 (heat equation).
    /// let diff = DiffusionChernoff::new_const_a(0.5, 1.0, grid);
    /// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    /// let u1 = diff.apply_chernoff(0.01, &u0).unwrap();
    /// assert_eq!(u1.values.len(), 64);
    /// ```
    #[must_use]
    pub fn new_const_a(a_value: F, a_norm_bound: f64, grid: Grid1D<F>) -> Self {
        Self {
            storage: Storage::ConstA { a_value },
            a_norm_bound,
            grid,
        }
    }

    /// Returns `true` if this instance was created via [`Self::new_const_a`].
    ///
    /// Useful for diagnostics; the fast-path selection is automatic.
    #[must_use]
    #[inline]
    pub fn is_const_a(&self) -> bool {
        matches!(self.storage, Storage::ConstA { .. })
    }

    /// Evaluate `a(x)` — dispatches on storage variant (fn-ptr, heap closure, or constant).
    #[inline]
    pub fn call_a(&self, x: F) -> F {
        match &self.storage {
            Storage::FnPtr { a, .. } => a(x),
            Storage::Closure { a, .. } => a(x),
            Storage::ConstA { a_value } => *a_value,
        }
    }

    /// Evaluate `a'(x)` — dispatches on storage variant.
    ///
    /// Returns `F::zero()` immediately for the `ConstA` variant (no function call).
    #[inline]
    pub fn call_a_prime(&self, x: F) -> F {
        match &self.storage {
            Storage::FnPtr { a_prime, .. } => a_prime(x),
            Storage::Closure { a_prime, .. } => a_prime(x),
            Storage::ConstA { .. } => F::zero(),
        }
    }

    /// Evaluate `a''(x)` — dispatches on storage variant.
    ///
    /// Returns `F::zero()` immediately for the `ConstA` variant (no function call).
    #[inline]
    pub fn call_a_double_prime(&self, x: F) -> F {
        match &self.storage {
            Storage::FnPtr { a_double_prime, .. } => a_double_prime(x),
            Storage::Closure { a_double_prime, .. } => a_double_prime(x),
            Storage::ConstA { .. } => F::zero(),
        }
    }

    /// Apply `D_ζ(τ)` to `f` — generic scalar path for non-f64 types.
    ///
    /// Uses `sample_generic` (scalar interpolation) for boundary dispatch.
    /// For `F = f64`, use `ChernoffFunction::apply` which preserves the SIMD
    /// `catmull_rom` path (bit-equal with v0.8.x).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0`, non-finite, or
    ///   `a(x_pre) ≤ 0` / non-finite at `x_pre = x + (τ/2)·a'(x)`.
    /// - [`SemiflowError::Unsupported`] propagated from `f.sample_generic()`.
    pub fn apply_f(&self, tau: F, f: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError> {
        validate_tau_generic(tau)?;
        let mut out = f.zeroed_like();
        for i in 0..f.values.len() {
            out.values[i] = apply_at_node_generic(self, tau, f, i)?;
        }
        Ok(out)
    }

    /// Consistency order: 2 (ζ-A, variable `a ∈ C³`).
    pub fn order_val(&self) -> u32 {
        2
    }

    /// Growth bound `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    pub fn growth_val(&self) -> (f64, f64) {
        (1.0, 0.0)
    }
}

// ChernoffFunction impl — v3.0 surface (ADR-0074): apply removed, growth -> Growth<f64>.
impl ChernoffFunction<f64> for DiffusionChernoff<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order 2 (global O(τ²) for variable `a ∈ C³`, sympy gate `Z_τ²`).
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }

    /// Allocation-free apply: writes directly into `dst.values` via `parallel_eval_into`.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0`, non-finite, or
    ///   `a(x_pre) ≤ 0` / non-finite at `x_pre`.
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

// Phase 5a: additive impl — delegates to generic scalar apply_f path.
impl ChernoffFunction<f32> for DiffusionChernoff<f32> {
    type S = GridFn1D<f32>;

    /// Consistency order 2 (mirrors f64 impl).
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    fn growth(&self) -> Growth<f32> {
        Growth::contraction()
    }

    /// Scalar apply: delegates to `apply_f` (no SIMD — that is Phase 5b).
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

// Inherent convenience (v3.0 apply_chernoff = v2.x apply).
impl DiffusionChernoff<f64> {
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
