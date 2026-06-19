//! `extern "C"` entry points for v2.4 graph kernels (ADR-0062, ADR-0063,
//! ADR-0064). Hosts the FFI bindings for the **two new** kernels introduced
//! in v2.4:
//!
//! - [`smf_ghc6_*`] — `GraphHeat6thChernoff` (order-6 spatial; ADR-0062).
//! - [`smf_vc_mghc_*`] — `VarCoefMagnusGraphHeatChernoff` (order-4 variable-
//!   coefficient × time-dependent Magnus; ADR-0063).
//!
//! The pre-existing K=4 / Magnus K=4 / Magnus K=6 / VarCoef-constant FFI
//! coverage (and the v2.4 expansion thereof) lives in `graph_ffi.rs`.
//!
//! See `graph_ffi.rs` for the shared opaque-handle conventions, ownership
//! model, thread-safety policy, and `catch_panic!` discipline — this file
//! follows them identically.

#![allow(unsafe_code)]

use std::sync::Arc;

use semiflow_core::scratch::ScratchPool;
use semiflow_core::{
    varcoef_magnus_graph::{VarCoefMagnusGraphHeatChernoff, WeightAtTime},
    ChernoffSemigroup, Graph, GraphHeat6thChernoff, GraphSignal, Laplacian, LaplacianAtTime,
};

use crate::graph_ffi::{SmfGraph, SmfGraphSig};
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque handle to a `GraphHeat6thChernoff<f64>` semigroup state.
#[repr(C)]
pub struct SmfGhc6 {
    _private: [u8; 0],
}

/// Opaque handle to a `VarCoefMagnusGraphHeatChernoff<f64>` state.
#[repr(C)]
pub struct SmfVcMghc {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrapper structs (Rust-private). Mirror the layout used by
// `graph_ffi.rs::GhcInner` / `MghcInner`.
// ---------------------------------------------------------------------------

struct Ghc6Inner {
    semigroup: ChernoffSemigroup<GraphHeat6thChernoff<f64>, GraphSignal<f64>>,
    current: GraphSignal<f64>,
}

struct VcMghcInner {
    func: VarCoefMagnusGraphHeatChernoff<f64>,
    current: GraphSignal<f64>,
    scratch: ScratchPool<f64>,
    t_cursor: f64,
}

// Hidden duplicate of `graph_ffi.rs::GraphInner` for `Box::from_raw` in callbacks.
// Kept private; same memory layout as `graph_ffi::GraphInner`.
#[repr(C)]
struct GraphInnerView {
    graph: Arc<Graph<f64>>,
}

#[repr(C)]
struct GraphSigInnerView {
    signal: GraphSignal<f64>,
}

// ---------------------------------------------------------------------------
// Callback types
// ---------------------------------------------------------------------------

/// C function-pointer type for a time-to-Laplacian callback (mirrors
/// `graph_ffi.rs::SmfLapAtTFn`). The callback returns a freshly allocated
/// `SmfGraph` whose ownership transfers to Rust.
type SmfLapAtTFn =
    unsafe extern "C" fn(t: f64, user_data: *mut (), out_graph: *mut *mut SmfGraph) -> i32;

/// C function-pointer type for a time-to-weight-vector callback. The
/// callback MUST fill `out_vals[0..n_nodes]` with strictly-positive finite
/// `f64` weights. Returns 0 on success, non-zero on error.
type SmfWeightAtTFn =
    unsafe extern "C" fn(t: f64, user_data: *mut (), out_vals: *mut f64, n_nodes: u32) -> i32;

fn make_lap_at_t(cb: SmfLapAtTFn, user_data: *mut ()) -> LaplacianAtTime<f64> {
    let addr = user_data as usize;
    Box::new(move |t: f64| {
        let ptr = addr as *mut ();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut out_graph: *mut SmfGraph = std::ptr::null_mut();
            // SAFETY: cb is a valid C fn-ptr; addr round-trips.
            let rc = unsafe { cb(t, ptr, &mut out_graph) };
            assert!(
                rc == 0 && !out_graph.is_null(),
                "smf_lap_at_t_fn failed (rc={rc})"
            );
            // SAFETY: out_graph is a live Box<GraphInnerView> by ADR-0059 contract.
            let inner = unsafe { Box::from_raw(out_graph.cast::<GraphInnerView>()) };
            Arc::new(Laplacian::assemble_combinatorial(&inner.graph))
        }));
        result.unwrap_or_else(|_| panic!("smf_lap_at_t_fn callback panicked"))
    })
}

fn make_a_at_t(cb: SmfWeightAtTFn, user_data: *mut (), n_nodes: usize) -> WeightAtTime<f64> {
    let addr = user_data as usize;
    Box::new(move |t: f64| {
        let ptr = addr as *mut ();
        let mut buf = vec![0.0_f64; n_nodes];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // u32 truncation is enforced by Graph::n_nodes() <= u32::MAX (CSR invariant).
            #[allow(clippy::cast_possible_truncation)]
            let n_u32 = n_nodes as u32;
            // SAFETY: cb is a valid C fn-ptr; buf is owned + sized to n_nodes.
            let rc = unsafe { cb(t, ptr, buf.as_mut_ptr(), n_u32) };
            assert!(rc == 0, "smf_a_at_t_fn failed (rc={rc})");
            buf
        }));
        result.unwrap_or_else(|_| panic!("smf_a_at_t_fn callback panicked"))
    })
}

// ---------------------------------------------------------------------------
// smf_ghc6_*: GraphHeat6thChernoff bindings (ADR-0062)
// ---------------------------------------------------------------------------

/// Construct a `GraphHeat6thChernoff<f64>` state from a graph and initial signal.
///
/// ## Preconditions
/// - `graph`, `init_sig`, `out` non-null.
/// - `n_steps >= 1`.
///
/// ## Return values
/// - `Ok` (0) — `*out` is set.
/// - `NullPtr` (5) — any pointer is null.
/// - `OutOfDomain` (3) — `n_steps == 0`.
/// - `Panic` (99) — internal panic.
///
/// ## Ownership
/// Caller owns `*out`; free with `smf_ghc6_drop`.
///
/// # Safety
/// All pointers must come from this crate's constructors and be valid.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc6_new(
    graph: *const SmfGraph,
    init_sig: *const SmfGraphSig,
    n_steps: u32,
    out: *mut *mut SmfGhc6,
) -> SemiflowStatus {
    if graph.is_null() || init_sig.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_steps == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        // SAFETY: caller guarantees live handles. Memory layout of
        // GraphInnerView/GraphSigInnerView matches graph_ffi.rs.
        let g_inner = unsafe { &*graph.cast::<GraphInnerView>() };
        let sig_inner = unsafe { &*init_sig.cast::<GraphSigInnerView>() };
        let lap = Laplacian::assemble_combinatorial(&g_inner.graph);
        let chernoff = GraphHeat6thChernoff::from_owned(lap);
        let semigroup = match ChernoffSemigroup::new(chernoff, n_steps as usize) {
            Ok(s) => s,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let current = sig_inner.signal.clone();
        let raw = Box::into_raw(Box::new(Ghc6Inner { semigroup, current })).cast::<SmfGhc6>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Advance the `GraphHeat6` semigroup by `tau` using `n_steps` Chernoff steps.
///
/// ## Return values
/// `Ok` (0), `NullPtr` (5), `OutOfDomain` (3), `Panic` (99).
///
/// # Safety
/// `state` must be a valid non-null pointer from `smf_ghc6_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc6_apply_into(
    state: *mut SmfGhc6,
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
        // SAFETY: caller guarantees live Box<Ghc6Inner>.
        let inner = unsafe { &mut *state.cast::<Ghc6Inner>() };
        // Clone the kernel and rebuild the semigroup with the new n_steps.
        // GraphHeat6thChernoff is Clone (Arc<Laplacian> bump only — cheap).
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

/// Copy current `GraphHeat6` signal values into `buf`.
///
/// # Safety
/// `state` must be a valid non-null pointer from `smf_ghc6_new`. `buf` must
/// be valid for `buf_len` `f64` writes.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc6_current(
    state: *const SmfGhc6,
    buf: *mut f64,
    buf_len: u32,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<Ghc6Inner>.
        let inner = unsafe { &*state.cast::<Ghc6Inner>() };
        let vals = inner.current.values();
        if (buf_len as usize) < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfGhc6` handle. Null-safe.
///
/// # Safety
/// `state` must be null or a pointer from `smf_ghc6_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc6_drop(state: *mut SmfGhc6) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<Ghc6Inner>())) };
    }));
}

// ---------------------------------------------------------------------------
// smf_vc_mghc_*: VarCoefMagnusGraphHeatChernoff bindings (ADR-0063)
// ---------------------------------------------------------------------------

/// Construct a `VarCoefMagnusGraphHeatChernoff<f64>` state.
///
/// Wraps two C callbacks: `lap_at_t_fn` (time → Laplacian) and `a_at_t_fn`
/// (time → length-N node-weight vector). Both are called repeatedly during
/// every `apply_into` step at the GL₂ quadrature points.
///
/// ## Preconditions
/// - `init_sig`, `out` non-null.
/// - `lap_at_t_fn`, `a_at_t_fn` non-null.
/// - `rho_bar_max > 0`, `a_sup_max > 0`, both finite.
/// - `n_nodes >= 1`.
///
/// ## Return values
/// - `Ok` (0), `NullPtr` (5), `OutOfDomain` (3), `Panic` (99).
///
/// # Safety
/// `init_sig` must be from `smf_graphsig_new`. `user_data_lap` and
/// `user_data_a` must remain valid until `smf_vc_mghc_drop(*out)` returns.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn smf_vc_mghc_new(
    n_nodes: u32,
    init_sig: *const SmfGraphSig,
    lap_at_t_fn: Option<SmfLapAtTFn>,
    user_data_lap: *mut (),
    a_at_t_fn: Option<SmfWeightAtTFn>,
    user_data_a: *mut (),
    rho_bar_max: f64,
    a_sup_max: f64,
    convergence_radius_check: i32,
    out: *mut *mut SmfVcMghc,
) -> SemiflowStatus {
    let (Some(lap_cb), Some(a_cb)) = (lap_at_t_fn, a_at_t_fn) else {
        return SemiflowStatus::NullPtr;
    };
    if init_sig.is_null() || out.is_null() || n_nodes == 0 {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<GraphSigInner>.
        let sig_inner = unsafe { &*init_sig.cast::<GraphSigInnerView>() };
        let lap_at_t = make_lap_at_t(lap_cb, user_data_lap);
        let a_at_t = make_a_at_t(a_cb, user_data_a, n_nodes as usize);
        let func = match VarCoefMagnusGraphHeatChernoff::new(
            n_nodes as usize,
            lap_at_t,
            a_at_t,
            rho_bar_max,
            a_sup_max,
        ) {
            Ok(f) => f.with_radius_check(convergence_radius_check != 0),
            Err(e) => return SemiflowStatus::from(&e),
        };
        let current = sig_inner.signal.clone();
        let inner = VcMghcInner {
            func,
            current,
            scratch: ScratchPool::new(),
            t_cursor: 0.0,
        };
        let raw = Box::into_raw(Box::new(inner)).cast::<SmfVcMghc>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Advance the `VarCoef` Magnus state by `tau` using `n_steps` Magnus steps.
///
/// Each sub-step samples `lap_at_t_fn` and `a_at_t_fn` at two GL₂ points
/// within its sub-interval. Time cursor advances by `tau` on success.
///
/// # Safety
/// `state` must be from `smf_vc_mghc_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_vc_mghc_apply_into(
    state: *mut SmfVcMghc,
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
        // SAFETY: caller guarantees live Box<VcMghcInner>.
        let inner = unsafe { &mut *state.cast::<VcMghcInner>() };
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

/// Copy current signal values from the `VarCoef` Magnus state into `buf`.
///
/// # Safety
/// `state` must be from `smf_vc_mghc_new`; `buf` must be valid for
/// `buf_len` `f64` writes.
#[no_mangle]
pub unsafe extern "C" fn smf_vc_mghc_current(
    state: *const SmfVcMghc,
    buf: *mut f64,
    buf_len: u32,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<VcMghcInner>.
        let inner = unsafe { &*state.cast::<VcMghcInner>() };
        let vals = inner.current.values();
        if (buf_len as usize) < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfVcMghc` handle. Null-safe.
///
/// # Safety
/// `state` must be null or a pointer from `smf_vc_mghc_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_vc_mghc_drop(state: *mut SmfVcMghc) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<VcMghcInner>())) };
    }));
}
