//! `extern "C"` entry points for Graph PDE bindings (v2.2 Wave C, ADR-0059).
//!
//! Mirrors the 1D heat ABI in `ffi.rs`. Each entry point:
//! 1. Null-checks input pointers before `catch_panic!`.
//! 2. Wraps its body in `catch_panic!` to convert Rust panics to
//!    `SemiflowStatus::Panic` (UB-safe ABI boundary).
//! 3. Uses `Box::into_raw` / `Box::from_raw` for the opaque handle idiom.
//!
//! # Ownership model
//!
//! - `smf_graph_path` → caller owns `*SmfGraph`; free with `smf_graph_drop`.
//! - `smf_ghc_new`    → caller owns `*SmfGhc`; free with `smf_ghc_drop`.
//! - `smf_mghc_new`   → caller owns `*SmfMghc`; free with `smf_mghc_drop`.
//! - `smf_graphsig_new` / accessors follow the same pattern.
//!
//! # Thread safety
//!
//! Handles are **not** thread-safe. Each handle must be used from one thread
//! at a time. The `user_data` pointer passed to `smf_mghc_new` is stored as
//! a `usize` cast so the closure satisfies `Send + Sync`; thread-safety is
//! the **caller's** responsibility.
//!
//! # Panic policy
//!
//! Build the cdylib with `--profile release-ffi` (workspace `[profile.release-ffi]`
//! sets `panic = "unwind"`). Every entry point uses `catch_panic!`; a panic is
//! caught and returned as `SemiflowStatus::Panic` (value 99).
//!
//! # Safety invariants (per function)
//!
//! 1. Null-check BEFORE `catch_panic!` (fast non-panicking early return).
//! 2. `*mut Rmz*` pointers are always live `Box<Inner*>` casts.
//! 3. Destructors wrap `drop` in `catch_unwind`; result discarded.
//! 4. Slice validity: `(ptr, len)` pairs are caller-guaranteed valid.
//! 5. Output-pointer validity: `out` pointers are caller-guaranteed valid.

#![allow(unsafe_code)]

use std::sync::Arc;

use semiflow::{
    scratch::ScratchPool, ChernoffSemigroup, Graph, GraphHeatChernoff, GraphSignal, Laplacian,
    LaplacianAtTime, MagnusGraphHeatChernoff,
};

mod ghc_mghc;
mod graph_sig;

pub use ghc_mghc::*;
pub use graph_sig::*;

// ---------------------------------------------------------------------------
// Opaque handle structs exposed to C.
// The actual data lives in inner wrapper structs allocated via Box.
// ---------------------------------------------------------------------------

/// Opaque handle to a `Graph<f64>`.
///
/// C callers must not dereference this struct. Allocate with `smf_graph_path`,
/// free with `smf_graph_drop`.
#[repr(C)]
pub struct SmfGraph {
    _private: [u8; 0],
}

/// Opaque handle to a `GraphHeatChernoff<f64>` semigroup state.
///
/// Allocate with `smf_ghc_new`, free with `smf_ghc_drop`.
#[repr(C)]
pub struct SmfGhc {
    _private: [u8; 0],
}

/// Opaque handle to a `MagnusGraphHeatChernoff<f64>` state.
///
/// Allocate with `smf_mghc_new`, free with `smf_mghc_drop`.
#[repr(C)]
pub struct SmfMghc {
    _private: [u8; 0],
}

/// Opaque handle to a `GraphSignal<f64>`.
///
/// Allocate with `smf_graphsig_new`, free with `smf_graphsig_drop`.
#[repr(C)]
pub struct SmfGraphSig {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrapper structs (Rust-private, pub(crate) for submodule access)
// ---------------------------------------------------------------------------

pub(crate) struct GraphInner {
    pub(crate) graph: Arc<Graph<f64>>,
}

pub(crate) struct GhcInner {
    pub(crate) semigroup: ChernoffSemigroup<GraphHeatChernoff<f64>, GraphSignal<f64>>,
    pub(crate) current: GraphSignal<f64>,
}

pub(crate) struct MghcInner {
    pub(crate) func: MagnusGraphHeatChernoff<f64>,
    pub(crate) current: GraphSignal<f64>,
    pub(crate) scratch: ScratchPool<f64>,
    pub(crate) t_cursor: f64,
}

pub(crate) struct GraphSigInner {
    pub(crate) signal: GraphSignal<f64>,
}

// ---------------------------------------------------------------------------
// C callback type for `LaplacianAtTime`
// ---------------------------------------------------------------------------

/// C function-pointer type for a time-to-Laplacian callback.
///
/// `t` is the sample time; `user_data` is the opaque pointer supplied by
/// the caller; `*out_graph` must be set to a freshly allocated `SmfGraph`
/// (e.g. via `smf_graph_path`). The callback MUST:
/// - Be pure (same `t` → same topology + weights).
/// - Preserve the topology (`row_ptr`/`col_idx`) of the base graph.
/// - Not throw C++ exceptions across the boundary (UB).
/// - Return 0 on success, non-zero on error.
///
/// The Rust side takes ownership of `*out_graph` immediately after the
/// callback returns — the caller MUST NOT free it separately.
pub(crate) type SmfLapAtTFn =
    unsafe extern "C" fn(t: f64, user_data: *mut (), out_graph: *mut *mut SmfGraph) -> i32;

/// Wrap a C `(SmfLapAtTFn, user_data)` pair into a `LaplacianAtTime<f64>`.
///
/// Stores `user_data` as `usize` so the closure is `Send + Sync`.
pub(crate) fn make_lap_at_t(cb: SmfLapAtTFn, user_data: *mut ()) -> LaplacianAtTime<f64> {
    let addr = user_data as usize;
    Box::new(move |t: f64| {
        let ptr = addr as *mut ();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut out_graph: *mut SmfGraph = std::ptr::null_mut();
            // SAFETY: cb is a valid non-null C fn-ptr; addr round-trips.
            let rc = unsafe { cb(t, ptr, &mut out_graph) };
            assert!(
                rc == 0 && !out_graph.is_null(),
                "smf_lap_at_t_fn callback failed (rc={rc})"
            );
            // Take ownership of the returned Box<GraphInner>.
            // SAFETY: out_graph is a live Box<GraphInner> cast to *mut SmfGraph.
            let inner = unsafe { Box::from_raw(out_graph.cast::<GraphInner>()) };
            Arc::new(Laplacian::assemble_combinatorial(&inner.graph))
        }));
        result.unwrap_or_else(|_| panic!("smf_lap_at_t_fn: callback panicked"))
    })
}
