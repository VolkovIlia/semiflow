//! `GraphHeatChernoff` and `MagnusGraphHeatChernoff` FFI entry points.

#![allow(unsafe_code)]

use std::sync::Arc;

use semiflow_core::scratch::ScratchPool;
use semiflow_core::{ChernoffSemigroup, GraphHeatChernoff, Laplacian, MagnusGraphHeatChernoff};

use crate::status::SemiflowStatus;

use super::make_lap_at_t;
use super::{
    GhcInner, GraphInner, GraphSigInner, MghcInner, SmfGhc, SmfGraph, SmfGraphSig, SmfLapAtTFn,
    SmfMghc,
};

// ---------------------------------------------------------------------------
// GraphHeatChernoff
// ---------------------------------------------------------------------------

/// Construct a `GraphHeatChernoff` (order-1) state from a graph + initial signal.
///
/// Solves `∂ₜu = −L_G u` (order-1 Chernoff, Wave 2.1A).
///
/// ## Preconditions
/// - `graph` is non-null; obtained from a `smf_graph_*` constructor.
/// - `init_sig` is non-null; obtained from `smf_graphsig_new` with the same graph.
/// - `n_steps >= 1`.
/// - `out` is a valid non-null pointer to `*mut SmfGhc`.
///
/// ## Postconditions
/// - On `Ok`: `*out` is a freshly allocated `SmfGhc` handle. The initial
///   condition is a copy of `init_sig`. Ownership transfers to the caller;
///   free with `smf_ghc_drop`.
/// - On error: `*out` is left unchanged.
///
/// ## Return values
/// - `Ok` (0)           — success.
/// - `NullPtr` (5)      — any pointer is null.
/// - `OutOfDomain` (3)  — `n_steps == 0`.
/// - `Panic` (99)       — internal Rust panic.
///
/// ## Ownership
/// Caller owns the returned handle. Free with `smf_ghc_drop`.
///
/// # Safety
/// - `graph` and `init_sig` must be valid non-null pointers from their constructors.
/// - `out` must be a valid pointer to `*mut SmfGhc`.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc_new(
    graph: *const SmfGraph,
    init_sig: *const SmfGraphSig,
    n_steps: u32,
    out: *mut *mut SmfGhc,
) -> SemiflowStatus {
    if graph.is_null() || init_sig.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_steps == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        // SAFETY: caller guarantees live Box<GraphInner> / Box<GraphSigInner>.
        let g_inner = unsafe { &*graph.cast::<GraphInner>() };
        let sig_inner = unsafe { &*init_sig.cast::<GraphSigInner>() };
        let lap = Laplacian::assemble_combinatorial(&g_inner.graph);
        let chernoff = GraphHeatChernoff::from_owned(lap);
        let semigroup = match ChernoffSemigroup::new(chernoff, n_steps as usize) {
            Ok(s) => s,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let current = sig_inner.signal.clone();
        let raw = Box::into_raw(Box::new(GhcInner { semigroup, current })).cast::<SmfGhc>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Evolve the graph-heat state by `tau` using `n_steps` Chernoff steps.
///
/// Mutates state in place. Call `smf_ghc_current` afterwards to read values.
///
/// ## Preconditions
/// - `state` is non-null and was obtained from `smf_ghc_new`.
/// - `tau > 0` and finite.
/// - `n_steps >= 1`.
///
/// ## Postconditions
/// - On `Ok`: internal signal updated; `n_steps` stored for next call.
/// - On error: state is left unchanged.
///
/// ## Return values
/// - `Ok` (0)           — success.
/// - `NullPtr` (5)      — `state` is null.
/// - `OutOfDomain` (3)  — `tau <= 0`, non-finite, or `n_steps == 0`.
/// - `Panic` (99)       — internal Rust panic.
///
/// ## Ownership
/// Borrows `state` for the duration; does not transfer ownership.
///
/// # Safety
/// - `state` must be a valid non-null pointer from `smf_ghc_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc_apply_into(
    state: *mut SmfGhc,
    tau: f64,
    n_steps: u32,
) -> SemiflowStatus {
    if state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_steps == 0 || !tau.is_finite() || tau <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        // SAFETY: caller guarantees live Box<GhcInner>.
        let inner = unsafe { &mut *state.cast::<GhcInner>() };
        let chernoff = inner.semigroup.func.clone();
        let sg = match ChernoffSemigroup::new(chernoff, n_steps as usize) {
            Ok(s) => s,
            Err(e) => return SemiflowStatus::from(&e),
        };
        match sg.evolve(tau, &inner.current) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(next) => {
                inner.current = next;
                inner.semigroup = sg;
                SemiflowStatus::Ok
            }
        }
    })
}

/// Copy current signal values from the graph-heat state into `buf`.
///
/// ## Preconditions
/// - `state` is non-null and was obtained from `smf_ghc_new`.
/// - `buf` is non-null and writable for at least `buf_len` `f64` values.
/// - `buf_len >= n_nodes`.
///
/// ## Return values
/// - `Ok` (0)           — values copied.
/// - `NullPtr` (5)      — `state` or `buf` is null.
/// - `GridMismatch` (1) — `buf_len` too small.
/// - `Panic` (99)       — internal Rust panic.
///
/// # Safety
/// - `state` must be a valid non-null pointer from `smf_ghc_new`.
/// - `buf` must be valid for `buf_len` `f64` writes.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc_current(
    state: *const SmfGhc,
    buf: *mut f64,
    buf_len: u32,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<GhcInner>.
        let inner = unsafe { &*state.cast::<GhcInner>() };
        let vals = inner.current.values();
        if (buf_len as usize) < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfGhc` handle. Null-safe.
///
/// # Safety
/// - `state` must be null or a pointer from `smf_ghc_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc_drop(state: *mut SmfGhc) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<GhcInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// MagnusGraphHeatChernoff
// ---------------------------------------------------------------------------

/// Construct a `MagnusGraphHeatChernoff` (K=4) state.
///
/// Solves `∂ₜu = −L_G(t) u` using fourth-order Magnus + GL₄ quadrature.
///
/// ## Callback contract (`lap_at_t_fn`)
///
/// `lap_at_t_fn(t, user_data, out_graph)` is called with sample time `t`.
/// The C callback MUST:
/// - Allocate a new graph handle (e.g. via `smf_graph_path`) and write its
///   pointer to `*out_graph`.
/// - Return 0 on success; non-zero on error.
/// - Preserve the topology (`row_ptr`/`col_idx`) of the base `graph`.
/// - Not throw C++ exceptions across the boundary (UB).
///
/// The Rust side takes ownership of `*out_graph` immediately after the
/// callback returns — the C caller MUST NOT free it separately.
///
/// `user_data` MUST remain valid until `smf_mghc_drop(*out)` returns.
/// It is stored as a `usize` cast so the closure is `Send + Sync`. Thread-
/// safety is the **caller's** responsibility.
///
/// ## Preconditions
/// - `graph` is non-null; obtained from a `smf_graph_*` constructor.
/// - `init_sig` is non-null; obtained from `smf_graphsig_new`.
/// - `lap_at_t_fn` is non-null.
/// - `rho_bar_max > 0` and finite.
/// - `out` is non-null.
///
/// ## Return values
/// - `Ok` (0)           — success.
/// - `NullPtr` (5)      — any pointer is null.
/// - `OutOfDomain` (3)  — `rho_bar_max <= 0` or non-finite.
/// - `Panic` (99)       — internal Rust panic.
///
/// ## Ownership
/// Caller owns the returned handle. Free with `smf_mghc_drop`.
///
/// # Safety
/// - `graph`, `init_sig`, and `out` must be valid non-null pointers.
/// - `lap_at_t_fn` must be a valid C function pointer for the handle's lifetime.
/// - `user_data` must remain valid until `smf_mghc_drop(*out)` returns.
#[no_mangle]
pub unsafe extern "C" fn smf_mghc_new(
    graph: *const SmfGraph,
    init_sig: *const SmfGraphSig,
    lap_at_t_fn: Option<SmfLapAtTFn>,
    user_data: *mut (),
    rho_bar_max: f64,
    convergence_radius_check: i32,
    out: *mut *mut SmfMghc,
) -> SemiflowStatus {
    let Some(cb) = lap_at_t_fn else {
        return SemiflowStatus::NullPtr;
    };
    if graph.is_null() || init_sig.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<GraphInner> / Box<GraphSigInner>.
        let g_inner = unsafe { &*graph.cast::<GraphInner>() };
        let sig_inner = unsafe { &*init_sig.cast::<GraphSigInner>() };
        let lap_at_t = make_lap_at_t(cb, user_data);
        let do_check = convergence_radius_check != 0;
        let func = match MagnusGraphHeatChernoff::new(
            Arc::clone(&g_inner.graph),
            lap_at_t,
            rho_bar_max,
            do_check,
        ) {
            Ok(f) => f,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let current = sig_inner.signal.clone();
        let inner = MghcInner {
            func,
            current,
            scratch: ScratchPool::new(),
            t_cursor: 0.0,
        };
        let raw = Box::into_raw(Box::new(inner)).cast::<SmfMghc>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Evolve the Magnus state by `tau` using `n_steps` Chernoff sub-steps.
///
/// Each sub-step calls `lap_at_t_fn` at two GL₄ quadrature points within its
/// sub-interval. The internal time cursor advances by `tau` on success.
///
/// ## Preconditions
/// - `state` is non-null and was obtained from `smf_mghc_new`.
/// - `tau > 0` and finite.
/// - `n_steps >= 1`.
///
/// ## Postconditions
/// - On `Ok`: current signal updated; time cursor advanced by `tau`.
/// - On error: state is partially updated (best-effort).
///
/// ## Return values
/// - `Ok` (0)               — success; time cursor advanced.
/// - `NullPtr` (5)          — `state` is null.
/// - `OutOfDomain` (3)      — `tau <= 0`, non-finite, or `n_steps == 0`.
/// - `ConvergenceFailed` (7)— Magnus convergence-radius check failed.
/// - `Panic` (99)           — internal Rust panic.
///
/// ## Ownership
/// Borrows `state` for the duration; does not transfer ownership.
///
/// # Safety
/// - `state` must be a valid non-null pointer from `smf_mghc_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_mghc_apply_into(
    state: *mut SmfMghc,
    tau: f64,
    n_steps: u32,
) -> SemiflowStatus {
    if state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_steps == 0 || !tau.is_finite() || tau <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        // SAFETY: caller guarantees live Box<MghcInner>.
        let inner = unsafe { &mut *state.cast::<MghcInner>() };
        let step_tau = tau / f64::from(n_steps);
        for step_idx in 0..n_steps {
            let t_start = inner.t_cursor + f64::from(step_idx) * step_tau;
            let src = inner.current.clone();
            let mut dst = src.clone();
            if let Err(e) =
                inner
                    .func
                    .apply_into_at(t_start, step_tau, &src, &mut dst, &mut inner.scratch)
            {
                return SemiflowStatus::from(&e);
            }
            inner.current = dst;
        }
        inner.t_cursor += tau;
        SemiflowStatus::Ok
    })
}

/// Copy current signal values from the Magnus state into `buf`.
///
/// ## Preconditions
/// - `state` is non-null and was obtained from `smf_mghc_new`.
/// - `buf` is non-null and writable for at least `buf_len` `f64` values.
/// - `buf_len >= n_nodes`.
///
/// ## Return values
/// - `Ok` (0)           — values copied.
/// - `NullPtr` (5)      — `state` or `buf` is null.
/// - `GridMismatch` (1) — `buf_len` too small.
/// - `Panic` (99)       — internal Rust panic.
///
/// # Safety
/// - `state` must be a valid non-null pointer from `smf_mghc_new`.
/// - `buf` must be valid for `buf_len` `f64` writes.
#[no_mangle]
pub unsafe extern "C" fn smf_mghc_current(
    state: *const SmfMghc,
    buf: *mut f64,
    buf_len: u32,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<MghcInner>.
        let inner = unsafe { &*state.cast::<MghcInner>() };
        let vals = inner.current.values();
        if (buf_len as usize) < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfMghc` handle. Null-safe.
///
/// # Safety
/// - `state` must be null or a pointer from `smf_mghc_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_mghc_drop(state: *mut SmfMghc) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<MghcInner>())) };
    }));
}
