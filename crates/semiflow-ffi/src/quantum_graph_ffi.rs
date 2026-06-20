//! FFI surface for `QuantumGraph` + `QuantumGraphHeatChernoff` (Round 11).
//!
//! Mirrors Python `QuantumGraph` + `QuantumGraphHeat` classes (M16, ADR-0078).
//!
//! ## Signal layout
//!
//! Flat `f64` buffer; concatenation of per-edge sampled values in edge order.
//! Edge `e` occupies `values[e * n_per_edge .. (e+1) * n_per_edge]` where
//! `n_per_edge` is uniform across all edges (path / star constructors enforce
//! equal grid sizes; `from_edges` uses the same `n_grid` for all edges).
//!
//! ## Entry points
//!
//! **`QuantumGraph`** (topology data):
//! - `smf_qgraph_path(n_edges, edge_length, n_grid, out)` → `SemiflowStatus`
//! - `smf_qgraph_star(n_arms, edge_length, n_grid, out)` → `SemiflowStatus`
//! - `smf_qgraph_from_edges(edges, n_triplets, n_grid, out)` → `SemiflowStatus`
//! - `smf_qgraph_n_edges(g)`, `smf_qgraph_n_per_edge(g)`, `smf_qgraph_total_len(g)`
//! - `smf_qgraph_drop(g)` — null-safe
//!
//! **`QuantumGraphHeat`** (evolver):
//! - `smf_qgheat_new(qgraph, out)` → `SemiflowStatus`
//! - `smf_qgheat_set_state(ev, u0, len)` → `SemiflowStatus`
//! - `smf_qgheat_evolve(ev, t, n_steps)` → `SemiflowStatus`
//! - `smf_qgheat_values(ev, dst, dst_len)` → `SemiflowStatus`
//! - `smf_qgheat_size(ev)` → total number of grid nodes (0 if null)
//! - `smf_qgheat_drop(ev)` — null-safe
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
)]

use std::sync::Arc;
use std::os::raw::c_double;

use semiflow_core::{
    ChernoffSemigroup,
    quantum_graph::{QuantumGraph, QuantumGraphHeatChernoff, QuantumGraphSignal},
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque handle to a `QuantumGraph<f64>`.
///
/// Allocate with `smf_qgraph_*`; free with `smf_qgraph_drop`.
#[repr(C)]
pub struct SmfQuantumGraph {
    _private: [u8; 0],
}

/// Opaque handle to a `QuantumGraphHeatChernoff` evolver state.
///
/// Allocate with `smf_qgheat_new`; free with `smf_qgheat_drop`.
#[repr(C)]
pub struct SmfQuantumGraphHeat {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrapper structs (Rust-private)
// ---------------------------------------------------------------------------

struct QGraphInner {
    graph: Arc<QuantumGraph<f64>>,
    n_per_edge: usize,
}

struct QGHeatInner {
    graph: Arc<QuantumGraph<f64>>,
    kernel: QuantumGraphHeatChernoff<f64>,
    current: QuantumGraphSignal<f64>,
    n_per_edge: usize,
}

// ---------------------------------------------------------------------------
// QuantumGraph constructors
// ---------------------------------------------------------------------------

/// Build a path graph `P_{n_edges+1}` and write the handle to `*out`.
///
/// # Safety
/// `out` must be a valid writable `*mut *mut SmfQuantumGraph`.
#[no_mangle]
pub unsafe extern "C" fn smf_qgraph_path(
    n_edges: usize,
    edge_length: c_double,
    n_grid: usize,
    out: *mut *mut SmfQuantumGraph,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match QuantumGraph::<f64>::path(n_edges, edge_length, n_grid) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(g) => {
                let n_per_edge = g.edge_grids[0].n;
                let raw = Box::into_raw(Box::new(QGraphInner {
                    graph: Arc::new(g),
                    n_per_edge,
                }))
                .cast::<SmfQuantumGraph>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Build a star graph (1 hub + `n_arms` leaves) and write handle to `*out`.
///
/// # Safety
/// `out` must be a valid writable `*mut *mut SmfQuantumGraph`.
#[no_mangle]
pub unsafe extern "C" fn smf_qgraph_star(
    n_arms: usize,
    edge_length: c_double,
    n_grid: usize,
    out: *mut *mut SmfQuantumGraph,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match QuantumGraph::<f64>::star(n_arms, edge_length, n_grid) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(g) => {
                let n_per_edge = g.edge_grids[0].n;
                let raw = Box::into_raw(Box::new(QGraphInner {
                    graph: Arc::new(g),
                    n_per_edge,
                }))
                .cast::<SmfQuantumGraph>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Build from explicit edge triplets `[vertex_a, vertex_b, edge_length, ...]`.
///
/// `edges` must be a pointer to `n_triplets * 3` `f64` values.
///
/// # Safety
/// `edges` must point to `n_triplets * 3` readable `f64`s.
/// `out` must be a valid writable `*mut *mut SmfQuantumGraph`.
#[no_mangle]
pub unsafe extern "C" fn smf_qgraph_from_edges(
    edges: *const c_double,
    n_triplets: usize,
    n_grid: usize,
    out: *mut *mut SmfQuantumGraph,
) -> SemiflowStatus {
    if edges.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let raw = unsafe { std::slice::from_raw_parts(edges, n_triplets * 3) };
        let ep: Vec<(usize, usize)> = raw
            .chunks_exact(3)
            .map(|c| (c[0] as usize, c[1] as usize))
            .collect();
        let lengths: Vec<f64> = raw.chunks_exact(3).map(|c| c[2]).collect();
        match QuantumGraph::<f64>::new(ep, lengths, n_grid) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(g) => {
                let n_per_edge = g.edge_grids[0].n;
                let raw_ptr = Box::into_raw(Box::new(QGraphInner {
                    graph: Arc::new(g),
                    n_per_edge,
                }))
                .cast::<SmfQuantumGraph>();
                unsafe { *out = raw_ptr };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return number of edges; 0 if null.
///
/// # Safety
/// `g` must be null or a live pointer from `smf_qgraph_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_qgraph_n_edges(g: *const SmfQuantumGraph) -> usize {
    if g.is_null() {
        return 0;
    }
    let inner = unsafe { &*g.cast::<QGraphInner>() };
    inner.graph.n_edges
}

/// Return grid points per edge; 0 if null.
///
/// # Safety
/// `g` must be null or a live pointer from `smf_qgraph_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_qgraph_n_per_edge(g: *const SmfQuantumGraph) -> usize {
    if g.is_null() {
        return 0;
    }
    let inner = unsafe { &*g.cast::<QGraphInner>() };
    inner.n_per_edge
}

/// Return sum of all edge lengths; 0.0 if null.
///
/// # Safety
/// `g` must be null or a live pointer from `smf_qgraph_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_qgraph_total_len(g: *const SmfQuantumGraph) -> c_double {
    if g.is_null() {
        return 0.0;
    }
    let inner = unsafe { &*g.cast::<QGraphInner>() };
    inner.graph.edge_lengths.iter().sum()
}

/// Free a `SmfQuantumGraph`. Null-safe.
///
/// # Safety
/// `g` must be null or a live pointer from `smf_qgraph_*` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_qgraph_drop(g: *mut SmfQuantumGraph) {
    if g.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(g.cast::<QGraphInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// QuantumGraphHeat evolver
// ---------------------------------------------------------------------------

/// Construct a `QuantumGraphHeat` evolver from a `SmfQuantumGraph`.
///
/// # Safety
/// `qgraph` and `out` must be non-null valid pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_qgheat_new(
    qgraph: *const SmfQuantumGraph,
    out: *mut *mut SmfQuantumGraphHeat,
) -> SemiflowStatus {
    if qgraph.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let qi = unsafe { &*qgraph.cast::<QGraphInner>() };
        let g = Arc::clone(&qi.graph);
        match QuantumGraphHeatChernoff::new((*g).clone()) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(kernel) => {
                let n_per_edge = qi.n_per_edge;
                let current = QuantumGraphSignal::zeroed_for_graph(&g);
                let raw = Box::into_raw(Box::new(QGHeatInner {
                    graph: g,
                    kernel,
                    current,
                    n_per_edge,
                }))
                .cast::<SmfQuantumGraphHeat>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Set initial state from a flat `f64` buffer.
///
/// `len` must equal `n_edges * n_per_edge`.
///
/// # Safety
/// `ev` live from `smf_qgheat_new`; `u0` points to `len` readable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_qgheat_set_state(
    ev: *mut SmfQuantumGraphHeat,
    u0: *const c_double,
    len: usize,
) -> SemiflowStatus {
    if ev.is_null() || u0.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<QGHeatInner>() };
        let total = inner.graph.n_edges * inner.n_per_edge;
        if len != total {
            return SemiflowStatus::GridMismatch;
        }
        let slice = unsafe { std::slice::from_raw_parts(u0, len) };
        for &v in slice {
            if !v.is_finite() {
                return SemiflowStatus::NanInf;
            }
        }
        scatter_flat(&mut inner.current, slice, inner.n_per_edge);
        SemiflowStatus::Ok
    })
}

/// Advance state by time `t` using `n_steps` Chernoff iterations.
///
/// `t` must be finite and >= 0; `n_steps` >= 1.
///
/// # Safety
/// `ev` must be a live pointer from `smf_qgheat_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_qgheat_evolve(
    ev: *mut SmfQuantumGraphHeat,
    t: c_double,
    n_steps: usize,
) -> SemiflowStatus {
    if ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_steps == 0 || !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let inner = unsafe { &mut *ev.cast::<QGHeatInner>() };
        let kernel = inner.kernel.clone();
        let sg = match ChernoffSemigroup::new(kernel, n_steps) {
            Ok(s) => s,
            Err(e) => return SemiflowStatus::from(&e),
        };
        match sg.evolve(t, &inner.current) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(next) => {
                inner.current = next;
                SemiflowStatus::Ok
            }
        }
    })
}

/// Copy current signal to `dst` (flat `f64`, length `n_edges * n_per_edge`).
///
/// # Safety
/// `ev` live from `smf_qgheat_new`; `dst` valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_qgheat_values(
    ev: *const SmfQuantumGraphHeat,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<QGHeatInner>() };
        let total = inner.graph.n_edges * inner.n_per_edge;
        if dst_len < total {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, total) };
        gather_flat(&inner.current, out, inner.n_per_edge);
        SemiflowStatus::Ok
    })
}

/// Return total grid nodes (`n_edges * n_per_edge`); 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_qgheat_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_qgheat_size(ev: *const SmfQuantumGraphHeat) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<QGHeatInner>() };
    inner.graph.n_edges * inner.n_per_edge
}

/// Free a `SmfQuantumGraphHeat`. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_qgheat_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_qgheat_drop(ev: *mut SmfQuantumGraphHeat) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<QGHeatInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// Signal helpers (private)
// ---------------------------------------------------------------------------

fn gather_flat(sig: &QuantumGraphSignal<f64>, dst: &mut [f64], n_per_edge: usize) {
    for (e_idx, edge) in sig.per_edge.iter().enumerate() {
        let base = e_idx * n_per_edge;
        dst[base..base + n_per_edge].copy_from_slice(&edge.values);
    }
}

fn scatter_flat(sig: &mut QuantumGraphSignal<f64>, flat: &[f64], n_per_edge: usize) {
    for (e_idx, edge) in sig.per_edge.iter_mut().enumerate() {
        let base = e_idx * n_per_edge;
        edge.values.copy_from_slice(&flat[base..base + n_per_edge]);
    }
}
