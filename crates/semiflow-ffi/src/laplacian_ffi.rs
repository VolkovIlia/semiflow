//! FFI surface for `Laplacian<f64>` introspection + `GraphTraj` (C-parity, ADR-0028/0171).
//!
//! **`SmfLaplacian`**: `combinatorial`/`normalized` ctors, `free`, `n_nodes`,
//! `is_combinatorial`, `is_normalized`, `spectral_bound`, `row_ptr`, `col_idx`,
//! `vals`, `to_dense`; buffers freed via `smf_free_buf_usize`/`smf_free_buf_f64`.
//!
//! **`SmfGraphTraj`**: degenerate (fixed-topology, 1-segment) `new`/`free`/
//! `n_nodes`/`n_segments`/`t_horizon`. Build with `--profile release-ffi`.

#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use std::sync::Arc;

use semiflow::{Graph, Laplacian, LaplacianKind};

use crate::{
    graph_ffi::{GraphInner, SmfGraph},
    status::SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Opaque handle structs
// ---------------------------------------------------------------------------

/// Opaque handle to a `Laplacian<f64>`.
///
/// Allocate with `smf_graph_laplacian_combinatorial` or
/// `smf_graph_laplacian_normalized`; free with `smf_laplacian_free`.
#[repr(C)]
pub struct SmfLaplacian {
    _private: [u8; 0],
}

/// Opaque handle to a degenerate `GraphTraj<f64>` (fixed topology,
/// single segment, constant Laplacian).
///
/// Allocate with `smf_graph_traj_new`; free with `smf_graph_traj_free`.
#[repr(C)]
pub struct SmfGraphTraj {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrappers (Rust-private)
// ---------------------------------------------------------------------------

struct LaplacianInner {
    lap: Arc<Laplacian<f64>>,
}

struct GraphTrajInner {
    n_nodes: usize,
    n_segments: usize,
    t_horizon: f64,
    // Hold the graph alive; not read but keeps refcount up for correctness.
    #[allow(dead_code)]
    graph: Arc<Graph<f64>>,
}

// ---------------------------------------------------------------------------
// SmfLaplacian — constructors
// ---------------------------------------------------------------------------

/// Assemble the combinatorial Laplacian `L = D − W` from `graph`.
///
/// ## Preconditions
/// - `graph` is non-null; obtained from `smf_graph_path` or similar.
/// - `out` is a valid non-null pointer to `*mut SmfLaplacian`.
///
/// ## Return values
/// - `Ok` (0)      — success; `*out` is set.
/// - `NullPtr` (5) — `graph` or `out` is null.
/// - `Panic` (99)  — internal Rust panic.
///
/// ## Ownership
/// Caller owns the returned handle. Free with `smf_laplacian_free`.
///
/// # Safety
/// - `graph` must be a valid pointer from a `smf_graph_*` constructor.
/// - `out` must be a valid pointer to `*mut SmfLaplacian`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_laplacian_combinatorial(
    graph: *const SmfGraph,
    out: *mut *mut SmfLaplacian,
) -> SemiflowStatus {
    if graph.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<GraphInner>.
        let g_inner = unsafe { &*graph.cast::<GraphInner>() };
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g_inner.graph));
        let raw = Box::into_raw(Box::new(LaplacianInner { lap })).cast::<SmfLaplacian>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Assemble the symmetric normalized Laplacian `L_sym = I − D^{−½} W D^{−½}`
/// from `graph`.
///
/// ## Preconditions / Return values / Ownership — same as
/// `smf_graph_laplacian_combinatorial`.
///
/// # Safety
/// - `graph` must be a valid pointer from a `smf_graph_*` constructor.
/// - `out` must be a valid pointer to `*mut SmfLaplacian`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_laplacian_normalized(
    graph: *const SmfGraph,
    out: *mut *mut SmfLaplacian,
) -> SemiflowStatus {
    if graph.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<GraphInner>.
        let g_inner = unsafe { &*graph.cast::<GraphInner>() };
        let lap = Arc::new(Laplacian::assemble_normalized(&g_inner.graph));
        let raw = Box::into_raw(Box::new(LaplacianInner { lap })).cast::<SmfLaplacian>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Free a `SmfLaplacian` handle. Null-safe.
///
/// # Safety
/// - `lap` must be null or a live pointer from `smf_graph_laplacian_*` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_free(lap: *mut SmfLaplacian) {
    if lap.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(lap.cast::<LaplacianInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// SmfLaplacian — scalar introspection
// ---------------------------------------------------------------------------

/// Number of nodes. Returns `0` if `lap` is null.
///
/// # Safety
/// - `lap` must be null or a valid pointer from `smf_graph_laplacian_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_n_nodes(lap: *const SmfLaplacian) -> usize {
    if lap.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live Box<LaplacianInner>.
    let inner = unsafe { &*lap.cast::<LaplacianInner>() };
    inner.lap.n_nodes()
}

/// Returns `true` iff the Laplacian is combinatorial (`L = D − W`).
/// Returns `false` if `lap` is null.
///
/// # Safety
/// - `lap` must be null or a valid pointer from `smf_graph_laplacian_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_is_combinatorial(lap: *const SmfLaplacian) -> bool {
    if lap.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live Box<LaplacianInner>.
    let inner = unsafe { &*lap.cast::<LaplacianInner>() };
    inner.lap.kind() == LaplacianKind::Combinatorial
}

/// Returns `true` iff the Laplacian is symmetric-normalized.
/// Returns `false` if `lap` is null.
///
/// # Safety
/// - `lap` must be null or a valid pointer from `smf_graph_laplacian_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_is_normalized(lap: *const SmfLaplacian) -> bool {
    if lap.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live Box<LaplacianInner>.
    let inner = unsafe { &*lap.cast::<LaplacianInner>() };
    inner.lap.kind() == LaplacianKind::SymNormalized
}

/// Gershgorin spectral-radius upper bound `ρ̄ ≥ ρ(L_G)` (cached in core).
/// Returns `0.0` if `lap` is null.
///
/// # Safety
/// - `lap` must be null or a valid pointer from `smf_graph_laplacian_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_spectral_bound(lap: *const SmfLaplacian) -> f64 {
    if lap.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live Box<LaplacianInner>.
    let inner = unsafe { &*lap.cast::<LaplacianInner>() };
    inner.lap.spectral_radius_bound()
}

// ---------------------------------------------------------------------------
// SmfLaplacian — CSR read-back (heap-allocated copies)
// ---------------------------------------------------------------------------
//
// Caller frees each buffer with `smf_free_buf_usize` / `smf_free_buf_f64`.
// Do NOT use C `free()` — the allocator must match Rust's global allocator.

/// Copy CSR row-pointer array (`len = n_nodes + 1`) into a heap `*usize` buffer.
/// Caller frees with `smf_free_buf_usize(*out, *len)`.
///
/// # Safety
/// `lap` is a valid `smf_graph_laplacian_*` pointer; `out`/`len` are non-null.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_row_ptr(
    lap: *const SmfLaplacian,
    out: *mut *mut usize,
    len: *mut usize,
) -> SemiflowStatus {
    if lap.is_null() || out.is_null() || len.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<LaplacianInner>.
        let inner = unsafe { &*lap.cast::<LaplacianInner>() };
        let src = inner.lap.row_ptr();
        let boxed: Box<[usize]> = src.to_vec().into_boxed_slice();
        let n = boxed.len();
        let ptr = Box::into_raw(boxed).cast::<usize>();
        unsafe {
            *out = ptr;
            *len = n;
        }
        SemiflowStatus::Ok
    })
}

/// Copy CSR column-index array (`len = n_directed_edges`, u32 widened to usize).
/// Caller frees with `smf_free_buf_usize(*out, *len)`.
///
/// # Safety
/// Same as `smf_laplacian_row_ptr`.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_col_idx(
    lap: *const SmfLaplacian,
    out: *mut *mut usize,
    len: *mut usize,
) -> SemiflowStatus {
    if lap.is_null() || out.is_null() || len.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<LaplacianInner>.
        let inner = unsafe { &*lap.cast::<LaplacianInner>() };
        let src = inner.lap.col_idx();
        // Widen u32 → usize (mirrors i64 widening in PyO3 layer).
        let v: Vec<usize> = src.iter().map(|&x| x as usize).collect();
        let boxed: Box<[usize]> = v.into_boxed_slice();
        let n = boxed.len();
        let ptr = Box::into_raw(boxed).cast::<usize>();
        unsafe {
            *out = ptr;
            *len = n;
        }
        SemiflowStatus::Ok
    })
}

/// Copy CSR value array (`len = n_directed_edges`) into a heap `*f64` buffer.
/// Caller frees with `smf_free_buf_f64(*out, *len)`.
///
/// # Safety
/// Same as `smf_laplacian_row_ptr`.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_vals(
    lap: *const SmfLaplacian,
    out: *mut *mut f64,
    len: *mut usize,
) -> SemiflowStatus {
    if lap.is_null() || out.is_null() || len.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<LaplacianInner>.
        let inner = unsafe { &*lap.cast::<LaplacianInner>() };
        let src = inner.lap.vals();
        let boxed: Box<[f64]> = src.to_vec().into_boxed_slice();
        let n = boxed.len();
        let ptr = Box::into_raw(boxed).cast::<f64>();
        unsafe {
            *out = ptr;
            *len = n;
        }
        SemiflowStatus::Ok
    })
}

/// Reconstruct the dense `n × n` row-major matrix from the Laplacian CSR.
///
/// Allocates `n * n` `f64` values. `*out` is set to the buffer start;
/// `*n` is set to the edge length of the square matrix.
///
/// ## Return values
/// - `Ok` (0)           — success.
/// - `NullPtr` (5)      — any argument is null.
/// - `OutOfDomain` (3)  — `n * n` would overflow `usize`.
/// - `Panic` (99)       — internal Rust panic.
///
/// ## Ownership
/// Caller frees with `smf_free_buf_f64(*out, (*n) * (*n))`.
///
/// # Safety
/// - `lap` must be a valid pointer from `smf_graph_laplacian_*`.
/// - `out` and `n` must be valid non-null write pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_laplacian_to_dense(
    lap: *const SmfLaplacian,
    out: *mut *mut f64,
    n: *mut usize,
) -> SemiflowStatus {
    if lap.is_null() || out.is_null() || n.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<LaplacianInner>.
        let inner = unsafe { &*lap.cast::<LaplacianInner>() };
        let nn = inner.lap.n_nodes();
        let Some(size) = nn.checked_mul(nn) else {
            return SemiflowStatus::OutOfDomain;
        };
        let mut buf = vec![0.0_f64; size];
        let row_ptr = inner.lap.row_ptr();
        let col_idx = inner.lap.col_idx();
        let vals = inner.lap.vals();
        for row in 0..nn {
            for k in row_ptr[row]..row_ptr[row + 1] {
                let col = col_idx[k] as usize;
                buf[row * nn + col] = vals[k];
            }
        }
        let boxed: Box<[f64]> = buf.into_boxed_slice();
        let ptr = Box::into_raw(boxed).cast::<f64>();
        unsafe {
            *out = ptr;
            *n = nn;
        }
        SemiflowStatus::Ok
    })
}

/// Free a `usize` buffer from `smf_laplacian_row_ptr` or `smf_laplacian_col_idx`. Null-safe.
///
/// # Safety
/// `buf` is null or a matching `(ptr, len)` from the read-back function.
#[no_mangle]
pub unsafe extern "C" fn smf_free_buf_usize(buf: *mut usize, len: usize) {
    if buf.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let slice = unsafe { std::slice::from_raw_parts_mut(buf, len) };
        unsafe { drop(Box::from_raw(slice)) };
    }));
}

/// Free a `f64` buffer from `smf_laplacian_vals`/`smf_laplacian_to_dense`. Null-safe.
///
/// # Safety
/// `buf` is null or a matching `(ptr, len)` from the read-back function.
#[no_mangle]
pub unsafe extern "C" fn smf_free_buf_f64(buf: *mut f64, len: usize) {
    if buf.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let slice = unsafe { std::slice::from_raw_parts_mut(buf, len) };
        unsafe { drop(Box::from_raw(slice)) };
    }));
}

// ---------------------------------------------------------------------------
// SmfGraphTraj — degenerate fixed-topology constructor
// ---------------------------------------------------------------------------
/// Build a degenerate fixed-topology `GraphTraj` (1 segment, constant
/// combinatorial Laplacian, horizon `t_horizon`).
///
/// Mirrors Python `GraphTraj(graph, t_horizon)` degenerate constructor.
///
/// ## Preconditions
/// - `graph` is non-null; obtained from a `smf_graph_*` constructor.
/// - `t_horizon > 0` and finite.
/// - `out` is a valid non-null pointer to `*mut SmfGraphTraj`.
///
/// ## Return values
/// - `Ok` (0)           — success; `*out` is set.
/// - `NullPtr` (5)      — `graph` or `out` is null.
/// - `NanInf` (2)       — `t_horizon` is NaN or Inf.
/// - `OutOfDomain` (3)  — `t_horizon <= 0`.
/// - `Panic` (99)       — internal Rust panic.
///
/// ## Ownership
/// Caller owns the returned handle. Free with `smf_graph_traj_free`.
///
/// # Safety
/// - `graph` must be a valid pointer from a `smf_graph_*` constructor.
/// - `out` must be a valid pointer to `*mut SmfGraphTraj`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_traj_new(
    graph: *const SmfGraph,
    t_horizon: f64,
    out: *mut *mut SmfGraphTraj,
) -> SemiflowStatus {
    if graph.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if !t_horizon.is_finite() {
        return SemiflowStatus::NanInf;
    }
    if t_horizon <= 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<GraphInner>.
        let g_inner = unsafe { &*graph.cast::<GraphInner>() };
        let n = g_inner.graph.n_nodes();
        let g = Arc::clone(&g_inner.graph);
        let inner = GraphTrajInner {
            n_nodes: n,
            n_segments: 1,
            t_horizon,
            graph: g,
        };
        let raw = Box::into_raw(Box::new(inner)).cast::<SmfGraphTraj>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Free a `SmfGraphTraj` handle. Null-safe.
///
/// # Safety
/// - `traj` must be null or a pointer from `smf_graph_traj_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_traj_free(traj: *mut SmfGraphTraj) {
    if traj.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(traj.cast::<GraphTrajInner>())) };
    }));
}

/// Number of nodes in the trajectory's graph. Returns `0` if null.
///
/// # Safety
/// - `traj` must be null or a valid pointer from `smf_graph_traj_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_traj_n_nodes(traj: *const SmfGraphTraj) -> usize {
    if traj.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live Box<GraphTrajInner>.
    let inner = unsafe { &*traj.cast::<GraphTrajInner>() };
    inner.n_nodes
}

/// Number of segments (always 1 for the degenerate constructor).
/// Returns `0` if null.
///
/// # Safety
/// - `traj` must be null or a valid pointer from `smf_graph_traj_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_traj_n_segments(traj: *const SmfGraphTraj) -> usize {
    if traj.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live Box<GraphTrajInner>.
    let inner = unsafe { &*traj.cast::<GraphTrajInner>() };
    inner.n_segments
}

/// Total time horizon of the trajectory. Returns `0.0` if null.
///
/// # Safety
/// - `traj` must be null or a valid pointer from `smf_graph_traj_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_traj_t_horizon(traj: *const SmfGraphTraj) -> f64 {
    if traj.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live Box<GraphTrajInner>.
    let inner = unsafe { &*traj.cast::<GraphTrajInner>() };
    inner.t_horizon
}
