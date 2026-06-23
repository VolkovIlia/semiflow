//! Opaque handle type and inner state construction helpers.
//!
//! `SemiflowState` is exposed to C as an opaque pointer; the real data lives
//! in `SemiflowStateInner` allocated on the Rust heap via `Box`.
//! `ffi.rs` performs the `Box::into_raw` / `Box::from_raw` round-trip.
//! All raw pointer operations are confined to `ffi.rs` (the sole `allow` site).

use semiflow::{ChernoffSemigroup, DiffusionChernoff, Grid1D, GridFn1D, SemiflowError};

// ---------------------------------------------------------------------------
// Opaque C handle
// ---------------------------------------------------------------------------

/// Opaque handle to a semiflow semigroup state.
///
/// C callers receive a `SemiflowState *` from `smf_state_new_*` and pass it
/// to other API functions.  The struct body is zero-sized and must not be
/// dereferenced or heap-allocated by the caller; all allocation is managed by
/// the Rust runtime.  Free with `smf_state_free`.
///
/// The actual data is a `Box<SemiflowStateInner>` whose raw pointer is cast to
/// `*mut SemiflowState` by `ffi.rs`.
#[repr(C)]
pub struct SemiflowState {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state (Rust-private)
// ---------------------------------------------------------------------------

/// Inner semigroup state allocated on the Rust heap.
pub(crate) struct SemiflowStateInner {
    /// The Chernoff semigroup (function + iteration count).
    pub semigroup: ChernoffSemigroup<DiffusionChernoff<f64>, GridFn1D<f64>>,
    /// Current function state (updated by `smf_evolve`).
    pub current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Static function pointers (a = 1.0, a' = 0, a'' = 0)
// ---------------------------------------------------------------------------

/// Diffusion coefficient `a(x) = 1.0` (hardcoded in v0.10.0).
///
/// Variable-coefficient support (`a: fn(f64) -> f64` callback) requires a
/// runtime fn-pointer which the current `DiffusionChernoff::new` API cannot
/// accept without modifying `semiflow-core`. Deferred to v0.11.0.
extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}

/// Derivative `a'(x) = 0` (constant `a`).
extern "Rust" fn zero_deriv(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Constructor helper
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

/// Return `DomainViolation` if any element of `u0` is non-finite.
pub(crate) fn validate_u0_finite(u0: &[f64]) -> Result<(), SemiflowError> {
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

/// Build a `SemiflowStateInner` with a variable diffusion coefficient via Rust closures.
// Nine arguments is unavoidable here: grid geometry (3) + Chernoff parameters (3 closures
// + 1 norm bound) + initial condition (2).  The function is `pub(crate)` and called from
// a single site in `ffi.rs` where each argument is named explicitly.
#[allow(clippy::too_many_arguments)]
///
/// The three closures (`a`, `a_prime`, `a_double_prime`) are forwarded directly
/// to `DiffusionChernoff::with_closure`.  All unsafe bridging from C fn-pointers
/// to these closures is performed in `ffi.rs` (the sole `#![allow(unsafe_code)]`
/// site), keeping this helper safe.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Grid1D::new`, `GridFn1D::new`, or
/// `ChernoffSemigroup::new`.
pub(crate) fn build_heat_with_closure(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    a: Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>,
    a_prime: Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>,
    a_double_prime: Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>,
    a_norm_bound: f64,
    u0_slice: &[f64],
) -> Result<SemiflowStateInner, SemiflowError> {
    validate_u0_finite(u0_slice)?;
    let grid = Grid1D::new(xmin, xmax, n)?;
    let chernoff = DiffusionChernoff::with_closure(a, a_prime, a_double_prime, a_norm_bound, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0_slice.to_vec())?;
    Ok(SemiflowStateInner { semigroup, current })
}
