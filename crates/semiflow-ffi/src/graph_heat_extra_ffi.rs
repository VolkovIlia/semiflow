//! Graph-heat FFI parity (Round 10): `GraphHeat4thChernoff` and
//! `MagnusGraphHeat6thChernoff`.
//!
//! - `smf_ghc4_*`  — `GraphHeat4thChernoff` (order-4, static Laplacian).
//! - `smf_mghc6_*` — `MagnusGraphHeat6thChernoff` (K=6 time-varying Magnus).
//!
//! See `graph_vc_ghc_ffi.rs` for `smf_vc_ghc_*` (variable-coefficient engine).
//! All functions reuse the `SmfGraph` / `SmfGraphSig` opaque handles from
//! `graph_ffi.rs` and follow its ABI conventions identically.
//!
//! # Safety invariants (per function)
//! 1. Null-check BEFORE `catch_panic!`.
//! 2. `*mut Smf*` pointers are always live `Box<Inner*>` casts.
//! 3. Destructors wrap `drop` in `catch_unwind`; result discarded.
//! 4. Slice validity: `(ptr, len)` pairs are caller-guaranteed valid.
//! 5. Output-pointer validity: `out` pointers are caller-guaranteed valid.

#![allow(unsafe_code)]

use std::sync::Arc;

use semiflow::{
    scratch::ScratchPool, ChernoffSemigroup, Graph, GraphHeat4thChernoff, GraphSignal, Laplacian,
    LaplacianAtTime, MagnusGraphHeat6thChernoff,
};

use crate::{
    graph_ffi::{SmfGraph, SmfGraphSig},
    status::SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque handle to a `GraphHeat4thChernoff<f64>` state.
///
/// Allocate with `smf_ghc4_new`, free with `smf_ghc4_drop`.
#[repr(C)]
pub struct SmfGhc4 {
    _private: [u8; 0],
}

/// Opaque handle to a `MagnusGraphHeat6thChernoff<f64>` state.
///
/// Allocate with `smf_mghc6_new`, free with `smf_mghc6_drop`.
#[repr(C)]
pub struct SmfMghc6 {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrapper structs (Rust-private)
// ---------------------------------------------------------------------------

/// Stores `Arc<Laplacian>` to reconstruct the kernel on each `apply_into`
/// call without requiring `Clone` on `GraphHeat4thChernoff`.
struct Ghc4Inner {
    laplacian: Arc<Laplacian<f64>>,
    current: GraphSignal<f64>,
}

struct Mghc6Inner {
    func: MagnusGraphHeat6thChernoff<f64>,
    current: GraphSignal<f64>,
    scratch: ScratchPool<f64>,
    t_cursor: f64,
}

// Private view structs — same layout as graph_ffi::{GraphInner,GraphSigInner}.
// Used only to borrow handles; never freed via these types.
#[repr(C)]
struct GraphInnerView {
    graph: Arc<Graph<f64>>,
}

#[repr(C)]
struct GraphSigInnerView {
    signal: GraphSignal<f64>,
}

// ---------------------------------------------------------------------------
// C callback type for time-varying Laplacian (mirrors graph_ffi_v2_4.rs)
// ---------------------------------------------------------------------------

type SmfLapAtTFn =
    unsafe extern "C" fn(t: f64, user_data: *mut (), out_graph: *mut *mut SmfGraph) -> i32;

fn make_lap_at_t(cb: SmfLapAtTFn, user_data: *mut ()) -> LaplacianAtTime<f64> {
    let addr = user_data as usize;
    Box::new(move |t: f64| {
        let ptr = addr as *mut ();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut out: *mut SmfGraph = std::ptr::null_mut();
            // SAFETY: cb is a valid C fn-ptr; addr round-trips through usize.
            let rc = unsafe { cb(t, ptr, &mut out) };
            assert!(
                rc == 0 && !out.is_null(),
                "smf_lap_at_t_fn failed (rc={rc})"
            );
            // SAFETY: out is a live Box<GraphInnerView> by ABI contract.
            let inner = unsafe { Box::from_raw(out.cast::<GraphInnerView>()) };
            Arc::new(Laplacian::assemble_combinatorial(&inner.graph))
        }));
        result.unwrap_or_else(|_| panic!("smf_lap_at_t_fn callback panicked"))
    })
}

// ---------------------------------------------------------------------------
// smf_ghc4_*: GraphHeat4thChernoff (order-4 static Laplacian)
// ---------------------------------------------------------------------------

/// Construct a `GraphHeat4thChernoff` (order-4) state.
///
/// Solves `∂ₜu = −L_G u` via the order-4 Taylor–Padé kernel.
///
/// ## Preconditions
/// - `graph`, `init_sig`, `out` non-null.
///
/// ## Return values
/// - `Ok` (0), `NullPtr` (5), `Panic` (99).
///
/// ## Ownership
/// Caller owns `*out`; free with `smf_ghc4_drop`.
///
/// # Safety
/// All pointers must come from this crate's constructors and be valid.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc4_new(
    graph: *const SmfGraph,
    init_sig: *const SmfGraphSig,
    out: *mut *mut SmfGhc4,
) -> SemiflowStatus {
    if graph.is_null() || init_sig.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live handles with matching layout.
        let g = unsafe { &*graph.cast::<GraphInnerView>() };
        let s = unsafe { &*init_sig.cast::<GraphSigInnerView>() };
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g.graph));
        let raw = Box::into_raw(Box::new(Ghc4Inner {
            laplacian: lap,
            current: s.signal.clone(),
        }))
        .cast::<SmfGhc4>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Advance the `GraphHeat4` state by `tau` using `n_steps` Chernoff steps.
///
/// Reconstructs the kernel from the stored `Arc<Laplacian>` (Arc bump only).
///
/// ## Return values
/// `Ok` (0), `NullPtr` (5), `OutOfDomain` (3), `Panic` (99).
///
/// # Safety
/// `state` must be a valid non-null pointer from `smf_ghc4_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc4_apply_into(
    state: *mut SmfGhc4,
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
        // SAFETY: caller guarantees live Box<Ghc4Inner>.
        let inner = unsafe { &mut *state.cast::<Ghc4Inner>() };
        let chernoff = GraphHeat4thChernoff::new(Arc::clone(&inner.laplacian));
        let sg = match ChernoffSemigroup::new(chernoff, n_steps as usize) {
            Ok(s) => s,
            Err(e) => return SemiflowStatus::from(&e),
        };
        match sg.evolve(tau, &inner.current) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(next) => {
                inner.current = next;
                SemiflowStatus::Ok
            }
        }
    })
}

/// Copy current `GraphHeat4` signal values into `buf`.
///
/// ## Return values
/// `Ok` (0), `NullPtr` (5), `GridMismatch` (1), `Panic` (99).
///
/// # Safety
/// `state` from `smf_ghc4_new`; `buf` valid for `buf_len` f64 writes.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc4_current(
    state: *const SmfGhc4,
    buf: *mut f64,
    buf_len: u32,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<Ghc4Inner>.
        let inner = unsafe { &*state.cast::<Ghc4Inner>() };
        let vals = inner.current.values();
        if (buf_len as usize) < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfGhc4` handle. Null-safe.
///
/// # Safety
/// `state` must be null or a pointer from `smf_ghc4_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_ghc4_drop(state: *mut SmfGhc4) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<Ghc4Inner>())) };
    }));
}

// ---------------------------------------------------------------------------
// smf_mghc6_*: MagnusGraphHeat6thChernoff (K=6 time-varying Magnus)
// ---------------------------------------------------------------------------

/// Construct a `MagnusGraphHeat6thChernoff` (K=6) state.
///
/// Solves `∂ₜu = −L_G(t) u` using sixth-order GL₆ three-point Magnus.
///
/// ## Callback contract (`lap_at_t_fn`)
/// Identical to `smf_mghc_new`: the C callback writes a freshly allocated
/// `SmfGraph` to `*out_graph`; Rust takes ownership immediately.
///
/// ## Preconditions
/// - `graph`, `init_sig`, `out` non-null; `lap_at_t_fn` non-null.
/// - `rho_bar_max > 0` and finite.
///
/// ## Return values
/// - `Ok` (0), `NullPtr` (5), `OutOfDomain` (3), `Panic` (99).
///
/// ## Ownership
/// Caller owns `*out`; free with `smf_mghc6_drop`.
///
/// # Safety
/// - `graph`, `init_sig`, `out` must be valid non-null pointers.
/// - `lap_at_t_fn` and `user_data` must remain valid until `smf_mghc6_drop`.
#[no_mangle]
pub unsafe extern "C" fn smf_mghc6_new(
    graph: *const SmfGraph,
    init_sig: *const SmfGraphSig,
    lap_at_t_fn: Option<SmfLapAtTFn>,
    user_data: *mut (),
    rho_bar_max: f64,
    convergence_radius_check: i32,
    out: *mut *mut SmfMghc6,
) -> SemiflowStatus {
    let Some(cb) = lap_at_t_fn else {
        return SemiflowStatus::NullPtr;
    };
    if graph.is_null() || init_sig.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if !rho_bar_max.is_finite() || rho_bar_max <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        // SAFETY: caller guarantees live handles with matching layout.
        let g = unsafe { &*graph.cast::<GraphInnerView>() };
        let s = unsafe { &*init_sig.cast::<GraphSigInnerView>() };
        let lap_at_t = make_lap_at_t(cb, user_data);
        let func = match MagnusGraphHeat6thChernoff::new(
            Arc::clone(&g.graph),
            lap_at_t,
            rho_bar_max,
            convergence_radius_check != 0,
        ) {
            Ok(f) => f,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let inner = Mghc6Inner {
            func,
            current: s.signal.clone(),
            scratch: ScratchPool::new(),
            t_cursor: 0.0,
        };
        let raw = Box::into_raw(Box::new(inner)).cast::<SmfMghc6>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Advance the Magnus K=6 state by `tau` using `n_steps` sub-steps.
///
/// Each sub-step calls `lap_at_t_fn` three times (GL₆ abscissas).
/// The internal time cursor advances by `tau` on success.
///
/// ## Return values
/// `Ok`(0), `NullPtr`(5), `OutOfDomain`(3), `ConvergenceFailed`(7), `Panic`(99).
///
/// # Safety
/// `state` must be a valid non-null pointer from `smf_mghc6_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_mghc6_apply_into(
    state: *mut SmfMghc6,
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
        // SAFETY: caller guarantees live Box<Mghc6Inner>.
        let inner = unsafe { &mut *state.cast::<Mghc6Inner>() };
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

/// Copy current Magnus K=6 signal values into `buf`.
///
/// ## Return values
/// `Ok` (0), `NullPtr` (5), `GridMismatch` (1), `Panic` (99).
///
/// # Safety
/// `state` from `smf_mghc6_new`; `buf` valid for `buf_len` f64 writes.
#[no_mangle]
pub unsafe extern "C" fn smf_mghc6_current(
    state: *const SmfMghc6,
    buf: *mut f64,
    buf_len: u32,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<Mghc6Inner>.
        let inner = unsafe { &*state.cast::<Mghc6Inner>() };
        let vals = inner.current.values();
        if (buf_len as usize) < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfMghc6` handle. Null-safe.
///
/// # Safety
/// `state` must be null or a pointer from `smf_mghc6_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_mghc6_drop(state: *mut SmfMghc6) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<Mghc6Inner>())) };
    }));
}
