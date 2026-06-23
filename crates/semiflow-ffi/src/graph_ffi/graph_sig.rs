//! Graph constructors and `GraphSignal` FFI entry points.

#![allow(unsafe_code)]

use std::sync::Arc;

use semiflow::{Graph, GraphSignal};

use crate::status::SemiflowStatus;

use super::{GraphInner, GraphSigInner, SmfGraph, SmfGraphSig};

// ---------------------------------------------------------------------------
// Graph constructors
// ---------------------------------------------------------------------------

/// Construct a path graph `P_n` with `n_nodes` nodes and unit edge weights.
///
/// ## Preconditions
/// - `n_nodes >= 1`.
/// - `out` is a valid non-null pointer to a `*mut SmfGraph` location.
///
/// ## Postconditions
/// - On `Ok`: `*out` points to a freshly allocated graph handle.
///   Ownership transfers to the caller; free with `smf_graph_drop`.
/// - On error: `*out` is left unchanged.
///
/// ## Return values
/// - `Ok` (0)           — success; `*out` is set.
/// - `NullPtr` (5)      — `out` is null.
/// - `OutOfDomain` (3)  — `n_nodes == 0`.
/// - `Panic` (99)       — internal Rust panic caught at boundary.
///
/// ## Ownership
/// Caller owns the returned handle. Free with `smf_graph_drop`.
///
/// # Safety
/// - `out` must be a valid pointer to a `*mut SmfGraph` location.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_path(n_nodes: u32, out: *mut *mut SmfGraph) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_nodes == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        let graph = Arc::new(Graph::<f64>::path(n_nodes as usize));
        let raw = Box::into_raw(Box::new(GraphInner { graph })).cast::<SmfGraph>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Return the number of nodes in `g`.
///
/// Returns `0` if `g` is null.
///
/// ## Preconditions
/// - `g` is null or a live pointer from a `smf_graph_*` constructor.
///
/// ## Return values
/// `u32` node count, or `0` on null. Cannot fail.
///
/// ## Ownership
/// Borrows `g`; does not transfer ownership.
///
/// # Safety
/// - `g` must be null or a valid pointer from `smf_graph_path`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_n_nodes(g: *const SmfGraph) -> u32 {
    if g.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live Box<GraphInner>.
    let inner = unsafe { &*g.cast::<GraphInner>() };
    // Graphs are bounded to u32::MAX nodes by construction; truncation is intentional.
    #[allow(clippy::cast_possible_truncation)]
    {
        inner.graph.n_nodes() as u32
    }
}

/// Free a graph handle previously allocated by `smf_graph_path`.
///
/// Null-safe: passing `NULL` is a no-op.
///
/// ## Postconditions
/// - Heap memory released. After this call `g` is dangling.
///
/// ## Ownership
/// Takes ownership and destroys the handle.
///
/// # Safety
/// - `g` must be null or a pointer from `smf_graph_path` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_drop(g: *mut SmfGraph) {
    if g.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(g.cast::<GraphInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// GraphSignal
// ---------------------------------------------------------------------------

/// Allocate a graph signal of `n_nodes` nodes initialised from `vals`.
///
/// ## Preconditions
/// - `graph` is non-null and was obtained from a `smf_graph_*` constructor.
/// - `vals` is a valid pointer to exactly `n_nodes` contiguous `f64` values.
/// - `n_nodes` equals `smf_graph_n_nodes(graph)`.
/// - `out` is a valid non-null pointer to `*mut SmfGraphSig`.
/// - All elements of `vals[0..n_nodes]` must be finite.
///
/// ## Return values
/// - `Ok` (0)           — success; `*out` is set.
/// - `NullPtr` (5)      — any pointer argument is null.
/// - `GridMismatch` (1) — `n_nodes != smf_graph_n_nodes(graph)`.
/// - `NanInf` (2)       — a `vals` element is NaN or Inf.
/// - `Panic` (99)       — internal Rust panic.
///
/// ## Ownership
/// Caller owns the returned handle. Free with `smf_graphsig_drop`.
///
/// # Safety
/// - `graph` must be a valid non-null pointer from a `smf_graph_*` constructor.
/// - `vals` must be valid for `n_nodes` contiguous `f64` reads.
/// - `out` must be a valid pointer to `*mut SmfGraphSig`.
#[no_mangle]
pub unsafe extern "C" fn smf_graphsig_new(
    graph: *const SmfGraph,
    vals: *const f64,
    n_nodes: u32,
    out: *mut *mut SmfGraphSig,
) -> SemiflowStatus {
    if graph.is_null() || vals.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<GraphInner>.
        let g_inner = unsafe { &*graph.cast::<GraphInner>() };
        // n_nodes fits u32 by the C API contract (callers pass u32).
        #[allow(clippy::cast_possible_truncation)]
        let expected = g_inner.graph.n_nodes() as u32;
        if n_nodes != expected {
            return SemiflowStatus::GridMismatch;
        }
        // SAFETY: caller-validated pointer + length.
        let slice = unsafe { std::slice::from_raw_parts(vals, n_nodes as usize) };
        for &v in slice {
            if !v.is_finite() {
                return SemiflowStatus::NanInf;
            }
        }
        let signal = GraphSignal::from_fn(Arc::clone(&g_inner.graph), |i| slice[i as usize]);
        let raw = Box::into_raw(Box::new(GraphSigInner { signal })).cast::<SmfGraphSig>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Copy signal values into a caller-owned buffer.
///
/// ## Preconditions
/// - `sig` is non-null and was obtained from `smf_graphsig_new`.
/// - `buf` is non-null and writable for at least `buf_len` `f64` values.
/// - `buf_len >= signal length`.
///
/// ## Return values
/// - `Ok` (0)           — values copied.
/// - `NullPtr` (5)      — `sig` or `buf` is null.
/// - `GridMismatch` (1) — `buf_len < signal length`.
/// - `Panic` (99)       — internal Rust panic.
///
/// ## Ownership
/// Borrows `sig`; does not transfer ownership.
///
/// # Safety
/// - `sig` must be a valid non-null pointer from `smf_graphsig_new`.
/// - `buf` must be valid for `buf_len` `f64` writes.
#[no_mangle]
pub unsafe extern "C" fn smf_graphsig_values(
    sig: *const SmfGraphSig,
    buf: *mut f64,
    buf_len: u32,
) -> SemiflowStatus {
    if sig.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<GraphSigInner>.
        let inner = unsafe { &*sig.cast::<GraphSigInner>() };
        let vals = inner.signal.values();
        if (buf_len as usize) < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return the number of nodes (signal length). Returns 0 if null.
///
/// ## Ownership
/// Borrows `sig`; does not transfer ownership.
///
/// # Safety
/// - `sig` must be null or a valid pointer from `smf_graphsig_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_graphsig_len(sig: *const SmfGraphSig) -> u32 {
    if sig.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live Box<GraphSigInner>.
    let inner = unsafe { &*sig.cast::<GraphSigInner>() };
    // Signal length is bounded to u32::MAX by construction.
    #[allow(clippy::cast_possible_truncation)]
    {
        inner.signal.values().len() as u32
    }
}

/// Free a graph signal handle. Null-safe.
///
/// # Safety
/// - `sig` must be null or a pointer from `smf_graphsig_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_graphsig_drop(sig: *mut SmfGraphSig) {
    if sig.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(sig.cast::<GraphSigInner>())) };
    }));
}
