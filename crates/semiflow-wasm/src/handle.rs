//! Inner state construction helpers for `semiflow-wasm`.
//!
//! Mirrors `crates/semiflow-py/src/handle.rs` and
//! `crates/semiflow-ffi/src/handle.rs`. Per ADR-0028 §5, `build_heat_unit`
//! is duplicated (not shared through a common crate) to avoid cross-binding
//! dependencies. This is the third copy; the per-binding duplication is the
//! documented price of zero shared glue.
//!
//! Unlike Wave A/B, there is no `SemiflowState` opaque handle here —
//! `wasm-bindgen` owns the memory via `#[wasm_bindgen] pub struct Heat1D`.
//!
//! ## `JsCallback` newtype (ADR-0034 — WASM design deviation)
//!
//! `js_sys::Function` is not `Send + Sync` (JS callbacks are pinned to the JS
//! thread via a thread-local registry). `DiffusionChernoff::with_closure`
//! requires `Send + Sync`. We bridge this gap with `JsCallback`, a `repr(transparent)`
//! newtype that carries `unsafe impl Send + Sync`.
//!
//! **Safety contract** (from ADR-0034 §"Per-binding plan" and `diffusion_storage.rs` module-doc):
//! - `wasm32-unknown-unknown` is single-threaded by spec. There are no OS threads
//!   sharing linear memory, so `Send` and `Sync` are vacuously safe.
//! - The `--features threads` WebAssembly target (not used here) would require
//!   `SharedArrayBuffer` and explicit opt-in; it is out of scope for this binding.
//! - This crate has `#![allow(unsafe_code)]` at crate level; the WASM crate is
//!   the designated holder of this safety contract per the `diffusion_storage.rs`
//!   module-doc note.

#![allow(unsafe_code)]

use semiflow::{ChernoffSemigroup, DiffusionChernoff, Grid1D, GridFn1D, SemiflowError};
use wasm_bindgen::JsValue;

// ---------------------------------------------------------------------------
// Inner state (Rust-private, wrapped by Heat1D in state.rs)
// ---------------------------------------------------------------------------

/// Heap-allocated semigroup state owned by `Heat1D`.
pub(crate) struct SemiflowStateInner {
    /// The Chernoff semigroup (function + iteration count).
    pub semigroup: ChernoffSemigroup<DiffusionChernoff<f64>, GridFn1D<f64>>,
    /// Current function state (updated in-place by `Heat1D::evolve`).
    pub current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Static function pointers (a = 1.0, a' = 0, a'' = 0)
// ---------------------------------------------------------------------------

/// Diffusion coefficient `a(x) = 1.0` (hardcoded in v0.10.0).
///
/// Variable-coefficient support requires a runtime closure, which
/// `DiffusionChernoff::new` does not accept in this version. Deferred v0.11.0.
extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}

/// Derivative `a'(x) = 0` (constant `a`).
extern "Rust" fn zero_deriv(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// JsCallback newtype — ADR-0034 WASM design deviation
// ---------------------------------------------------------------------------

/// Transparent wrapper around a `js_sys::Function` for use as a Chernoff coefficient.
///
/// # Safety
///
/// `js_sys::Function` is not `Send + Sync` on multi-threaded targets because JS
/// callbacks are pinned to their originating JS thread via a thread-local registry.
/// However `wasm32-unknown-unknown` (used by wasm-bindgen) is single-threaded by spec:
/// there is exactly one thread sharing the WASM linear memory. `Send` and `Sync` are
/// therefore vacuously safe. This `unsafe impl` is intentional and documents that
/// the safety argument depends on the single-threaded WASM target. Do NOT apply this
/// pattern to multi-threaded targets.
#[repr(transparent)]
pub(crate) struct JsCallback(pub(crate) js_sys::Function);

// Safety: wasm32-unknown-unknown is single-threaded by spec (no Workers can share
// the same WASM linear memory by default). Send + Sync are vacuously safe here.
// ADR-0034 §"Per-binding plan" designates the WASM binding crate as the holder of
// this safety contract; semiflow-core carries #![deny(unsafe_code)] and cannot do this.
unsafe impl Send for JsCallback {}
unsafe impl Sync for JsCallback {}

impl JsCallback {
    /// Call the wrapped JS function with one `f64` argument.
    ///
    /// Returns `f64::NAN` if the call throws a JS exception or returns a
    /// non-numeric value. Callers (the integrator's coefficient paths) treat
    /// NaN as a signal that the user-supplied function is broken; the resulting
    /// grid function will also be NaN, which is caught by downstream checks.
    pub(crate) fn call(&self, x: f64) -> f64 {
        let arg = JsValue::from_f64(x);
        match self.0.call1(&JsValue::NULL, &arg) {
            Ok(v) => v.as_f64().unwrap_or(f64::NAN),
            Err(_) => f64::NAN,
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

/// Build a `SemiflowStateInner` for the 1-D heat equation with `a = 1.0`.
///
/// `u0_slice` is validated for finiteness before construction.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Grid1D::new`, `GridFn1D::new`, or
/// `ChernoffSemigroup::new`.
pub(crate) fn build_heat_unit(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0_slice: &[f64],
) -> Result<SemiflowStateInner, SemiflowError> {
    validate_u0_finite(u0_slice)?;
    let grid = Grid1D::new(xmin, xmax, n)?;
    let chernoff = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0_slice.to_vec())?;
    Ok(SemiflowStateInner { semigroup, current })
}

/// Bundled JS callback parameters for `build_heat_with_closure`.
///
/// Groups the three coefficient callbacks and the norm bound to avoid a
/// >6-argument function (suckless / clippy::too_many_arguments constraint).
pub(crate) struct ClosureParams {
    /// JS callback for `a(x)`.
    pub a: JsCallback,
    /// JS callback for `a'(x)`.
    pub a_prime: JsCallback,
    /// JS callback for `a''(x)`.
    pub a_double_prime: JsCallback,
    /// Upper bound for `‖a‖∞` (diagnostics).
    pub a_norm_bound: f64,
}

/// Build a `SemiflowStateInner` with variable diffusion coefficient `a(x)` via JS callbacks.
///
/// The three `JsCallback` values inside `params` are consumed and wrapped in closures
/// that call through to JS. Uses `DiffusionChernoff::with_closure` which requires
/// `Send + Sync`; the `JsCallback` newtype provides the required unsafe impls.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Grid1D::new`, `GridFn1D::new`, or
/// `ChernoffSemigroup::new`.
pub(crate) fn build_heat_with_closure(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0_slice: &[f64],
    params: ClosureParams,
) -> Result<SemiflowStateInner, SemiflowError> {
    validate_u0_finite(u0_slice)?;
    let grid = Grid1D::new(xmin, xmax, n)?;
    let chernoff = DiffusionChernoff::with_closure(
        move |x| params.a.call(x),
        move |x| params.a_prime.call(x),
        move |x| params.a_double_prime.call(x),
        params.a_norm_bound,
        grid,
    );
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0_slice.to_vec())?;
    Ok(SemiflowStateInner { semigroup, current })
}

/// Return `DomainViolation` if any element of `u0` is non-finite.
fn validate_u0_finite(u0: &[f64]) -> Result<(), SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}
