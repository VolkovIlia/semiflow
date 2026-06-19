//! All `extern "C"` entry points for the `semiflow-ffi` cdylib.
//!
//! # Ownership model
//!
//! - `smf_state_new_heat_1d_unit` allocates a `Box<SemiflowStateInner>` and
//!   transfers ownership to the caller as a `*mut SemiflowState`.
//! - The caller owns the handle until it calls `smf_state_free`, after which
//!   the pointer is dangling and must not be used.
//! - Functions that take `*const SemiflowState` borrow the handle for the
//!   duration of the call only; they do not take ownership.
//!
//! # Thread safety
//!
//! A `SemiflowState *` handle encapsulates mutable Rust heap data and is
//! **not** thread-safe.  Concurrent calls on the same handle from multiple
//! threads produce data races and undefined behaviour.  Use one handle per
//! thread, or guard access with an external lock.  Status-code constants
//! (`SemiflowStatus`) are integer values and are safe to share across threads.
//!
//! # Reentrancy / panic safety
//!
//! Every entry point except `smf_state_free` wraps its body in
//! `catch_unwind` via the `catch_panic!` macro.  A Rust panic is caught,
//! converted to `SemiflowStatus::Panic` (99), and returned to the caller.
//! This prevents unwinding across the FFI boundary (which is undefined
//! behaviour under the C ABI).  `smf_state_free` also wraps its drop
//! in `catch_unwind`; the result is discarded.
//!
//! # Safety invariants (enforced per function)
//!
//! 1. Null-check BEFORE `catch_panic!` (fast non-panicking early return).
//! 2. Pointer validity: `*mut SemiflowState` is always a live `Box<SemiflowStateInner>`.
//! 3. No double-free: `smf_state_free` is nullable and idempotent.
//! 4. Slice validity: `(ptr, len)` pairs are caller-guaranteed valid for reads.
//! 5. Output-buffer validity: `out_buf` is caller-guaranteed valid for `out_buf_len` writes.

#![allow(unsafe_code)]

use crate::{
    handle::{build_heat_unit, build_heat_with_closure, SemiflowState},
    status::SemiflowStatus,
};

mod evolution_inspect_diag;

pub use evolution_inspect_diag::*;

// ---------------------------------------------------------------------------
// C callback type alias.
// ---------------------------------------------------------------------------

/// C function-pointer type for a diffusion coefficient callback.
///
/// `x` is the spatial coordinate; `user_data` is the opaque pointer supplied
/// by the caller.  Functions MUST be pure, panic-free, and may not throw.
///
/// cbindgen renders this as `double (*SemiflowAFn)(double, void *)` in
/// `semiflow.h` via the `after_includes` typedef (see `cbindgen.toml`).
pub(crate) type SemiflowAFn = unsafe extern "C" fn(x: f64, user_data: *mut ()) -> f64;

/// Wrap a C callback into a `Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>`.
///
/// On an unexpected Rust panic inside the callback the sentinel `0.0` is
/// returned; C exceptions crossing FFI are UB and cannot be caught here —
/// callers MUST NOT throw from these functions.
///
/// # Thread-safety / Send + Sync
///
/// `user_data` is stored as `usize` (the pointer address cast to integer).
/// `usize: Send + Sync`, so the closure satisfies `Send + Sync` without
/// requiring `unsafe impl` for a newtype.  The raw pointer is reconstructed
/// from the integer immediately before calling the C function.
///
/// **Safety invariant (caller's responsibility)**: the underlying C memory
/// must remain valid and thread-safe for the lifetime of the closure.
pub(crate) fn make_callback(
    f: SemiflowAFn,
    user_data: *mut (),
) -> Box<dyn Fn(f64) -> f64 + Send + Sync + 'static> {
    // Cast the raw pointer to an integer address.  usize is Send + Sync, so
    // closures capturing it satisfy the Send + Sync bounds required by
    // DiffusionChernoff::with_closure (ADR-0034).
    //
    // SAFETY: usize round-trip through raw pointer is well-defined on all
    // Rust-supported platforms (provenance: the closure reconstructs the
    // pointer via addr_of + pointer arithmetic, which is correct for calling
    // the C function with the original address).
    let addr = user_data as usize;
    Box::new(move |x: f64| {
        // SAFETY: `f` is a valid non-null C fn-ptr (checked by caller before
        // `make_callback` is called); `addr` round-trips to the original `void *`.
        let ptr = addr as *mut ();
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { f(x, ptr) }));
        result.unwrap_or(0.0)
    })
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a heat-equation state with diffusion coefficient `a(x) = 1.0`.
///
/// Solves `∂_t u = ∂_xx u` (unit diffusion) on `[xmin, xmax]` with `n` nodes.
/// The default internal step count is 100; pass a different `n_steps` to
/// `smf_evolve` to override it per call.
///
/// ## Preconditions
/// - `xmin < xmax`; both must be finite.
/// - `n >= 4` (minimum grid resolution for the Chernoff kernel).
/// - `u0_len == n`; the initial-condition length must match the grid.
/// - All elements of `u0[0..u0_len]` must be finite (no NaN, no Inf).
/// - `u0` is a valid pointer to at least `u0_len` contiguous `f64` values.
/// - `out_state` is a valid pointer to a `*mut SemiflowState` location.
/// - Neither `u0` nor `out_state` may be null.
///
/// ## Postconditions
/// - On `Ok`: `*out_state` points to a freshly allocated, heap-resident
///   `SemiflowState`.  Ownership transfers to the caller; free with
///   `smf_state_free`.
/// - On any error: `*out_state` is left unchanged; no allocation escapes.
///
/// ## Return values
/// - `Ok` (0) — success; `*out_state` is set.
/// - `NullPtr` (5) — `u0` or `out_state` is null.
/// - `GridMismatch` (1) — `n < 4`, `xmin >= xmax`, or `u0_len != n`.
/// - `NanInf` (2) — a `u0` element is NaN or Inf.
/// - `Panic` (99) — internal Rust panic caught at boundary (file a bug).
///
/// ## Ownership
/// Caller owns the returned handle.  Free with `smf_state_free`.
///
/// # Safety
/// - `u0` must be a valid pointer to `u0_len` contiguous `f64` values.
/// - `out_state` must be a valid pointer to a `*mut SemiflowState`.
/// - Returns `NullPtr` if any pointer argument is null.
#[no_mangle]
pub unsafe extern "C" fn smf_state_new_heat_1d_unit(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: *const f64,
    u0_len: usize,
    out_state: *mut *mut SemiflowState,
) -> SemiflowStatus {
    if u0.is_null() || out_state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_heat_unit(xmin, xmax, n, 100, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SemiflowState>();
                unsafe { *out_state = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Allocate a heat-equation state with a variable diffusion coefficient `a(x)`.
///
/// Solves `∂_t u = ∂_x(a(x)·∂_x u)` on `[xmin, xmax]` with `n` nodes using
/// the ζ-A Chernoff formula (ADR-0008, order-2 consistency, `a ∈ C³`).
///
/// ## Callback contract
///
/// - `a`, `a_prime`, `a_double_prime` are C function pointers of type
///   `double (*)(double x, void *user_data)`.  None may be null.
/// - `user_data` is threaded through to all three callbacks unchanged.
///   Pass `NULL` if no state is needed.
/// - Callbacks MUST be pure (no global state mutation) and panic-free.
///   They will be invoked many times per `smf_evolve` call (once per
///   grid node per Chernoff step — up to O(N × `n_steps`) calls total).
/// - Callbacks MUST NOT throw C++ exceptions across the boundary; that is UB.
/// - Returning NaN or Inf from a callback causes `DomainViolation` from the
///   integrator, surfaced as `OutOfDomain` or `NanInf` from `smf_evolve`.
///
/// ## Lifetime / thread-safety
///
/// `user_data` MUST remain valid until `smf_state_free(*out_state)`
/// returns.  Violation is undefined behaviour.
///
/// The Rust wrapper marks `user_data` as `Send + Sync`.  Thread-safety is the
/// **caller's** responsibility:
/// - Read-only data (e.g. `const double *params`) is always safe.
/// - Shared mutable data without external synchronisation is UB.
///
/// ## Panic safety
///
/// Each callback invocation is wrapped in `catch_unwind`.  On an unexpected
/// Rust panic inside the callback, the sentinel value `0.0` is returned and
/// the panic is not propagated.  The entire constructor body is also wrapped
/// in `catch_unwind`; a panic there returns `Panic` (99).
///
/// ## Preconditions
///
/// Same as `smf_state_new_heat_1d_unit`, plus:
/// - `a`, `a_prime`, `a_double_prime` must be non-null function pointers.
/// - `a_norm_bound > 0` (upper bound for `‖a‖∞`; used for diagnostics only).
/// - `a(x) > 0` for all `x ∈ [xmin, xmax]` (strict ellipticity).
///
/// ## Return values
/// - `Ok` (0) — success; `*out_state` is set.
/// - `NullPtr` (5) — any pointer argument is null.
/// - `GridMismatch` (1) — `n < 4`, `xmin >= xmax`, or `u0_len != n`.
/// - `NanInf` (2) — a `u0` element is NaN or Inf.
/// - `OutOfDomain` (3) — `a(x) <= 0` or non-finite detected by integrator.
/// - `Panic` (99) — internal Rust panic caught at boundary (file a bug).
///
/// ## Ownership
/// Caller owns the returned handle.  Free with `smf_state_free`.
///
/// # Safety
/// - `a`, `a_prime`, `a_double_prime` must be valid, non-null C function
///   pointers that remain callable for the lifetime of `*out_state`.
/// - `user_data` must remain valid until `smf_state_free(*out_state)`.
/// - `u0` must be a valid pointer to `u0_len` contiguous `f64` values.
/// - `out_state` must be a valid pointer to a `*mut SemiflowState`.
#[no_mangle]
pub unsafe extern "C" fn smf_state_new_with_closure(
    xmin: f64,
    xmax: f64,
    n: usize,
    a: Option<unsafe extern "C" fn(f64, *mut ()) -> f64>,
    a_prime: Option<unsafe extern "C" fn(f64, *mut ()) -> f64>,
    a_double_prime: Option<unsafe extern "C" fn(f64, *mut ()) -> f64>,
    user_data: *mut (),
    a_norm_bound: f64,
    u0: *const f64,
    u0_len: usize,
    out_state: *mut *mut SemiflowState,
) -> SemiflowStatus {
    // Null-check before catch_panic (fast, non-panicking).
    let (Some(fn_a), Some(fn_ap), Some(fn_app)) = (a, a_prime, a_double_prime) else {
        return SemiflowStatus::NullPtr;
    };
    // Cast to SemiflowAFn for use in make_callback.
    let fn_a: SemiflowAFn = fn_a;
    let fn_ap: SemiflowAFn = fn_ap;
    let fn_app: SemiflowAFn = fn_app;
    if u0.is_null() || out_state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        let box_a = make_callback(fn_a, user_data);
        let box_ap = make_callback(fn_ap, user_data);
        let box_app = make_callback(fn_app, user_data);
        match build_heat_with_closure(
            xmin,
            xmax,
            n,
            100,
            box_a,
            box_ap,
            box_app,
            a_norm_bound,
            slice,
        ) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SemiflowState>();
                unsafe { *out_state = raw };
                SemiflowStatus::Ok
            }
        }
    })
}
