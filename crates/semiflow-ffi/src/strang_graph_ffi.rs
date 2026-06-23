//! FFI surface for `StrangSplitGraph` (Round 11, M22).
//!
//! Mirrors Python `StrangGraph` class (`structured_traj.rs`, M22).
//!
//! Palindromic Strang split `S(τ) = A(τ/2) ∘ B(τ) ∘ A(τ/2)` on
//! `GraphSignal<f64>`. Uses edge-parity 2-coloring to guarantee commutativity.
//!
//! Reuses `SmfGraph` opaque handle from `graph_ffi.rs` for the input graph.
//!
//! ## Entry points
//!
//! - `smf_strang_graph_path_new(graph, out)` — path graph bipartite split
//! - `smf_strang_graph_cycle_new(graph, out)` — cycle graph bipartite split
//! - `smf_strang_graph_evolve(ev, t_final, n_steps, f0, f0_len, dst, dst_len)`
//! - `smf_strang_graph_n_nodes(ev)` — node count (0 if null)
//! - `smf_strang_graph_order(ev)` — approximation order (2 or 1; 0 if null)
//! - `smf_strang_graph_drop(ev)` — null-safe
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
)]

use std::sync::Arc;
use std::os::raw::c_double;

use semiflow::{
    ChernoffSemigroup, ChernoffFunction, Graph, GraphHeatChernoff,
    GraphSignal, strang_graph::StrangSplitGraph,
};

use crate::graph_ffi::{GraphInner, SmfGraph};
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `StrangSplitGraph<GraphHeatChernoff, GraphHeatChernoff, f64>`.
///
/// Allocate with `smf_strang_graph_path_new` / `smf_strang_graph_cycle_new`;
/// free with `smf_strang_graph_drop`.
#[repr(C)]
pub struct SmfStrangGraph {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrapper
// ---------------------------------------------------------------------------

struct StrangGraphInner {
    strang: StrangSplitGraph<GraphHeatChernoff<f64>, GraphHeatChernoff<f64>, f64>,
    graph: Arc<Graph<f64>>,
    n_nodes: usize,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Build a path-graph palindromic Strang split from `SmfGraph`.
///
/// Requires `n_nodes >= 2` in the graph.
///
/// # Safety
/// `graph` and `out` must be non-null valid pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_strang_graph_path_new(
    graph: *const SmfGraph,
    out: *mut *mut SmfStrangGraph,
) -> SemiflowStatus {
    if graph.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let gi = unsafe { &*graph.cast::<GraphInner>() };
        let g = Arc::clone(&gi.graph);
        let n = g.n_nodes();
        match StrangSplitGraph::new_bipartite_path(&g) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(strang) => {
                let raw = Box::into_raw(Box::new(StrangGraphInner {
                    strang,
                    graph: g,
                    n_nodes: n,
                }))
                .cast::<SmfStrangGraph>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Build a cycle-graph palindromic Strang split from `SmfGraph`.
///
/// Requires `n_nodes >= 4` and `n_nodes % 2 == 0`.
///
/// # Safety
/// `graph` and `out` must be non-null valid pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_strang_graph_cycle_new(
    graph: *const SmfGraph,
    out: *mut *mut SmfStrangGraph,
) -> SemiflowStatus {
    if graph.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let gi = unsafe { &*graph.cast::<GraphInner>() };
        let g = Arc::clone(&gi.graph);
        let n = g.n_nodes();
        match StrangSplitGraph::new_bipartite_cycle(&g) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(strang) => {
                let raw = Box::into_raw(Box::new(StrangGraphInner {
                    strang,
                    graph: g,
                    n_nodes: n,
                }))
                .cast::<SmfStrangGraph>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Evolve initial signal `f0` to time `t_final` with `n_steps` Strang steps.
///
/// `f0` is a flat `f64` array of length `n_nodes`.
/// On success `dst` receives the evolved signal (length `n_nodes`).
///
/// # Safety
/// `ev` live from constructor; `f0` readable `f0_len` f64s;
/// `dst` writable `dst_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_strang_graph_evolve(
    ev: *const SmfStrangGraph,
    t_final: c_double,
    n_steps: u32,
    f0: *const c_double,
    f0_len: usize,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || f0.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_steps == 0 || !t_final.is_finite() || t_final < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let inner = unsafe { &*ev.cast::<StrangGraphInner>() };
        let n = inner.n_nodes;
        if f0_len != n || dst_len < n {
            return SemiflowStatus::GridMismatch;
        }
        let input = unsafe { std::slice::from_raw_parts(f0, n) };
        let strang = inner.strang.clone();
        let g = Arc::clone(&inner.graph);
        let sg = match ChernoffSemigroup::new(strang, n_steps as usize) {
            Ok(s) => s,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let f0_sig = GraphSignal::from_fn(Arc::clone(&g), |i| input[i as usize]);
        match sg.evolve(t_final, &f0_sig) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                let out = unsafe { std::slice::from_raw_parts_mut(dst, n) };
                out.copy_from_slice(result.values());
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

/// Return node count; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from a constructor.
#[no_mangle]
pub unsafe extern "C" fn smf_strang_graph_n_nodes(ev: *const SmfStrangGraph) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<StrangGraphInner>() };
    inner.n_nodes
}

/// Return approximation order (2 when commutativity holds, 1 otherwise); 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from a constructor.
#[no_mangle]
pub unsafe extern "C" fn smf_strang_graph_order(ev: *const SmfStrangGraph) -> u32 {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<StrangGraphInner>() };
    inner.strang.order()
}

/// Free a `SmfStrangGraph`. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from a constructor not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_strang_graph_drop(ev: *mut SmfStrangGraph) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<StrangGraphInner>())) };
    }));
}
