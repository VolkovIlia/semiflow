//! Destructor, evolution, inspection, and diagnostics FFI entry points.

#![allow(unsafe_code)]

use std::ffi::c_char;

use crate::{
    handle::{SemiflowState, SemiflowStateInner},
    status::SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

/// Free a state previously allocated by `smf_state_new_*`.
///
/// Null-safe: passing `NULL` is a no-op.
///
/// ## Preconditions
/// - `state` is null, or a live pointer previously returned by
///   `smf_state_new_*` that has not yet been freed.
///
/// ## Postconditions
/// - The heap memory for the state is released.
/// - After this call `state` is dangling; do not use it again.
///
/// ## Return values
/// This function is `void`.  Errors (including internal panics) are silently
/// discarded because there is no meaningful recovery in a destructor.
///
/// ## Ownership
/// Takes ownership and destroys the handle.
///
/// # Safety
/// - `state` must be either null or a pointer obtained from
///   `smf_state_new_*` that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn smf_state_free(state: *mut SemiflowState) {
    if state.is_null() {
        return;
    }
    // SAFETY: caller guarantees `state` is a live Box<SemiflowStateInner>.
    // Wrap Drop in catch_unwind: panicking through an FFI boundary is UB,
    // even from a destructor. The unwrapped result is discarded — there's
    // no recovery semantically possible inside a destructor anyway.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<SemiflowStateInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// Evolution
// ---------------------------------------------------------------------------

/// Advance the state by time `t` using `n_steps` Chernoff iterations.
///
/// Mutates the state in place.  Call `smf_state_values` afterwards to
/// copy the updated grid values.
///
/// ## Preconditions
/// - `state` is non-null and was obtained from `smf_state_new_*`.
/// - `t >= 0.0` and finite.
/// - `n_steps >= 1`.
///
/// ## Postconditions
/// - On `Ok`: the internal current function is replaced by the Chernoff
///   approximation advanced by time `t` using `n_steps` iterations.
///   The semigroup step count is updated to `n_steps` for subsequent calls.
/// - On any error: the state is left unchanged.
///
/// ## Edge cases
/// - `t = 0.0` is **accepted** but does NOT guarantee an identity transform.
///   Applying the kernel `n_steps` times with `tau = 0` produces numerically
///   underflowed values rather than `u0` exactly.  Callers needing identity
///   should skip the call instead of passing `t = 0.0`.
/// - `t < 0.0` — heat equation is ill-posed backwards.
/// - `n_steps = 0` — meaningless; treated as a precondition violation.
/// - NaN or Inf `t` — routed via internal validation.
///
/// ## Return values
/// - `Ok` (0) — success; state updated.
/// - `NullPtr` (5) — `state` is null.
/// - `OutOfDomain` (3) — `t < 0`, `t` is NaN/Inf, or `n_steps == 0`.
/// - `CflViolated` (6) — CFL constraint violated for the chosen `t`/`n_steps`.
/// - `BoundaryFailure` (4) — grid resolution too coarse for the Chernoff shift.
/// - `ConvergenceFailed` (7) — iterative solver did not converge.
/// - `Panic` (99) — internal Rust panic caught at boundary (file a bug).
///
/// ## Ownership
/// Borrows `state` for the duration of the call; does not transfer ownership.
///
/// # Safety
/// - `state` must be a valid non-null pointer obtained from
///   `smf_state_new_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_evolve(
    state: *mut SemiflowState,
    t: f64,
    n_steps: usize,
) -> SemiflowStatus {
    if state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<SemiflowStateInner>.
        let inner = unsafe { &mut *state.cast::<SemiflowStateInner>() };
        if n_steps == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        // Rebuild semigroup with requested n_steps.
        let chernoff = inner.semigroup.func.clone();
        match semiflow_core::ChernoffSemigroup::new(chernoff, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(sg) => match sg.evolve(t, &inner.current) {
                Err(e) => SemiflowStatus::from(&e),
                Ok(next) => {
                    inner.current = next;
                    inner.semigroup = sg;
                    SemiflowStatus::Ok
                }
            },
        }
    })
}

/// Advance state in place using a caller-provided buffer (Wave 5, ADR-0045 §2).
///
/// Borrows `buf` for the duration of the call only; no ownership transfer.
/// On `Ok`, `buf[0..buf_len]` contains the evolved values.  On error, `buf`
/// may be partially written (indeterminate state).
///
/// ## Preconditions
/// - `state` is non-null and was obtained from `smf_state_new_*`.
/// - `buf` is non-null and writable for at least `buf_len` `f64` values,
///   well-aligned for `f64`.
/// - `buf_len == smf_state_size(state)`.
/// - `buf` must not alias the state's internal buffer (UB if violated).
/// - `tau > 0` and finite; `n_steps >= 1`.
///
/// ## Return values
/// - `Ok` (0)              — `buf` now holds the evolved values.
/// - `NullPtr` (5)         — `state` or `buf` is null.
/// - `OutOfDomain` (3)     — `tau <= 0`, non-finite, or `n_steps == 0`.
/// - `GridMismatch` (1)    — `buf_len != smf_state_size(state)`.
/// - `Panic` (99)          — internal Rust panic caught at boundary (file a bug).
///
/// # Safety
/// - `state` must be a valid non-null pointer obtained from `smf_state_new_*`.
/// - `buf` must be valid for `buf_len` contiguous, writable, well-aligned `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_evolve_inplace(
    state: *mut SemiflowState,
    buf: *mut f64,
    buf_len: usize,
    tau: f64,
    n_steps: usize,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<SemiflowStateInner>.
        let inner = unsafe { &mut *state.cast::<SemiflowStateInner>() };
        if n_steps == 0 || !tau.is_finite() || tau <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let expected = inner.current.values.len();
        if buf_len != expected {
            return SemiflowStatus::GridMismatch;
        }
        // SAFETY: caller-validated pointer + length; non-aliasing per docstring.
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf, buf_len) };

        // Copy caller buf → internal state, run evolve, copy back.
        inner.current.values.copy_from_slice(buf_slice);
        let chernoff = inner.semigroup.func.clone();
        match semiflow_core::ChernoffSemigroup::new(chernoff, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(sg) => match sg.evolve(tau, &inner.current) {
                Err(e) => SemiflowStatus::from(&e),
                Ok(next) => {
                    buf_slice.copy_from_slice(&next.values);
                    inner.current = next;
                    inner.semigroup = sg;
                    SemiflowStatus::Ok
                }
            },
        }
    })
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

/// Copy current grid values into `out_buf`.
///
/// Writes exactly `smf_state_size(state)` `f64` values starting at
/// `out_buf`.  The write is a flat memcopy; no partial writes occur on error.
///
/// ## Preconditions
/// - `state` is non-null and was obtained from `smf_state_new_*`.
/// - `out_buf` is non-null and writable for at least `out_buf_len` `f64`
///   values (≥ `smf_state_size(state)` elements).
/// - `out_buf_len >= smf_state_size(state)`.
///
/// ## Postconditions
/// - On `Ok`: `out_buf[0..n]` contains the current grid values.
/// - On error: `out_buf` is left unchanged.
///
/// ## Return values
/// - `Ok` (0) — values copied.
/// - `NullPtr` (5) — `state` or `out_buf` is null.
/// - `GridMismatch` (1) — `out_buf_len < smf_state_size(state)`.
/// - `Panic` (99) — internal Rust panic caught at boundary (file a bug).
///
/// ## Ownership
/// Borrows `state` and writes to `out_buf`; does not transfer ownership.
///
/// # Safety
/// - `state` must be valid and non-null.
/// - `out_buf` must be valid for `out_buf_len` `f64` writes.
#[no_mangle]
pub unsafe extern "C" fn smf_state_values(
    state: *const SemiflowState,
    out_buf: *mut f64,
    out_buf_len: usize,
) -> SemiflowStatus {
    if state.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<SemiflowStateInner>.
        let inner = unsafe { &*state.cast::<SemiflowStateInner>() };
        let vals = &inner.current.values;
        if out_buf_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return the number of grid nodes (length of the value array).
///
/// Returns `0` if `state` is null.  Always safe to call before allocating an
/// output buffer for `smf_state_values`.
///
/// ## Preconditions
/// - `state` is null or a live pointer from `smf_state_new_*`.
///
/// ## Postconditions
/// - Returns the grid size `n` passed to the constructor, or `0` on null.
///
/// ## Return values
/// This function returns `usize`, not `SemiflowStatus`; it cannot fail.
///
/// ## Ownership
/// Borrows `state` for the duration of the call; does not transfer ownership.
///
/// # Safety
/// - `state` must be null or a valid pointer obtained from `smf_state_new_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_state_size(state: *const SemiflowState) -> usize {
    if state.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live Box<SemiflowStateInner>.
    let inner = unsafe { &*state.cast::<SemiflowStateInner>() };
    inner.current.values.len()
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Return a static null-terminated C string describing `s`.
///
/// The returned pointer is valid for the lifetime of the process; do not free.
///
/// ## Preconditions
/// - `s` is any valid `SemiflowStatus` discriminant.
///
/// ## Postconditions
/// - Returns a pointer to a static ASCII string matching the variant name
///   (e.g. `"Ok"`, `"GridMismatch"`, `"Panic"`).
///
/// ## Return values
/// Always returns a non-null pointer; cannot fail.
#[no_mangle]
pub extern "C" fn smf_status_str(s: SemiflowStatus) -> *const c_char {
    let msg: &[u8] = match s {
        SemiflowStatus::Ok => b"Ok\0",
        SemiflowStatus::GridMismatch => b"GridMismatch\0",
        SemiflowStatus::NanInf => b"NanInf\0",
        SemiflowStatus::OutOfDomain => b"OutOfDomain\0",
        SemiflowStatus::BoundaryFailure => b"BoundaryFailure\0",
        SemiflowStatus::NullPtr => b"NullPtr\0",
        SemiflowStatus::CflViolated => b"CflViolated\0",
        SemiflowStatus::ConvergenceFailed => b"ConvergenceFailed\0",
        SemiflowStatus::Unsupported => b"Unsupported\0",
        SemiflowStatus::Panic => b"Panic\0",
    };
    msg.as_ptr().cast::<c_char>()
}

/// Return the crate version string as a static null-terminated C string.
///
/// Example: `"0.10.0"`.  The returned pointer is valid for the lifetime of
/// the process; do not free.
///
/// ## Return values
/// Always returns a non-null pointer; cannot fail.
#[no_mangle]
pub extern "C" fn smf_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0")
        .as_bytes()
        .as_ptr()
        .cast::<c_char>()
}
