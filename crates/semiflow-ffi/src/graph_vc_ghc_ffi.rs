//! Graph-heat FFI (Round 10): `VarCoefGraphHeatChernoff` — `smf_vc_ghc_*`.
//!
//! Solves `∂ₜu = −L_a u`, `L_a = A^{1/2} L_G A^{1/2}`, `A = diag(a)`.
//! Order-2 Chernoff approximation with static (time-independent) conductivity.
//!
//! Reuses `SmfGraph` / `SmfGraphSig` handles from `graph_ffi.rs`.
//!
//! # Safety invariants
//! 1. Null-check BEFORE `catch_panic!`.
//! 2. `*mut SmfVcGhc` is always a live `Box<VcGhcInner>` cast.
//! 3. Destructor wraps `drop` in `catch_unwind`; result discarded.

#![allow(unsafe_code)]

use std::sync::Arc;

use semiflow::{
    ChernoffSemigroup, Graph, GraphSignal, VarCoefGraphHeatChernoff,
};

use crate::graph_ffi::{SmfGraph, SmfGraphSig};
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `VarCoefGraphHeatChernoff<f64>` state.
///
/// Allocate with `smf_vc_ghc_new`, free with `smf_vc_ghc_drop`.
#[repr(C)]
pub struct SmfVcGhc {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrapper struct (Rust-private)
// ---------------------------------------------------------------------------

/// Stores graph + coefficient vector + `rho_bar` so the kernel can be
/// reconstructed per `apply_into` call (no `Clone` on the Chernoff type).
struct VcGhcInner {
    graph: Arc<Graph<f64>>,
    a: Vec<f64>,
    rho_bar: f64,
    current: GraphSignal<f64>,
}

// Private view structs — same layout as graph_ffi::{GraphInner,GraphSigInner}.
#[repr(C)]
struct GraphInnerView {
    graph: Arc<Graph<f64>>,
}

#[repr(C)]
struct GraphSigInnerView {
    signal: GraphSignal<f64>,
}

// ---------------------------------------------------------------------------
// smf_vc_ghc_*
// ---------------------------------------------------------------------------

/// Construct a `VarCoefGraphHeatChernoff` (order-2, variable-coefficient) state.
///
/// Solves `∂ₜu = −L_a u`, `L_a = A^{1/2} L_G A^{1/2}`, `A = diag(a)`.
///
/// ## Preconditions
/// - `graph`, `init_sig`, `a_vals`, `out` non-null.
/// - `a_len == n_nodes`.
/// - All `a_vals[i] > 0` and finite; `rho_bar > 0` and finite.
///
/// ## Return values
/// - `Ok` (0), `NullPtr` (5), `OutOfDomain` (3), `GridMismatch` (1), `Panic` (99).
///
/// ## Ownership
/// Caller owns `*out`; free with `smf_vc_ghc_drop`.
///
/// # Safety
/// - `graph`, `init_sig`, `out` must be valid non-null pointers.
/// - `a_vals` must be valid for `a_len` f64 reads for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn smf_vc_ghc_new(
    graph: *const SmfGraph,
    init_sig: *const SmfGraphSig,
    a_vals: *const f64,
    a_len: u32,
    rho_bar: f64,
    out: *mut *mut SmfVcGhc,
) -> SemiflowStatus {
    if graph.is_null() || init_sig.is_null() || a_vals.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if !rho_bar.is_finite() || rho_bar <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        // SAFETY: caller guarantees live handles and valid slice.
        let g = unsafe { &*graph.cast::<GraphInnerView>() };
        let s = unsafe { &*init_sig.cast::<GraphSigInnerView>() };
        let a_slice = unsafe { std::slice::from_raw_parts(a_vals, a_len as usize) };
        if a_len as usize != g.graph.n_nodes() {
            return SemiflowStatus::GridMismatch;
        }
        let a_vec: Vec<f64> = a_slice.to_vec();
        // Validate eagerly; kernel is reconstructed cheaply in apply_into.
        if let Err(e) =
            VarCoefGraphHeatChernoff::new(Arc::clone(&g.graph), a_vec.clone(), rho_bar)
        {
            return SemiflowStatus::from(&e);
        }
        let raw = Box::into_raw(Box::new(VcGhcInner {
            graph: Arc::clone(&g.graph),
            a: a_vec,
            rho_bar,
            current: s.signal.clone(),
        }))
        .cast::<SmfVcGhc>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

/// Advance the var-coef graph heat state by `tau` using `n_steps` Chernoff steps.
///
/// Reconstructs the kernel from stored graph + `a` + `rho_bar` each call
/// (Laplacian assembly is `O(E)` — proportional to edge count, fast in practice).
///
/// ## Return values
/// `Ok` (0), `NullPtr` (5), `OutOfDomain` (3), `Panic` (99).
///
/// # Safety
/// `state` must be a valid non-null pointer from `smf_vc_ghc_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_vc_ghc_apply_into(
    state: *mut SmfVcGhc,
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
        // SAFETY: caller guarantees live Box<VcGhcInner>.
        let inner = unsafe { &mut *state.cast::<VcGhcInner>() };
        let chernoff = match VarCoefGraphHeatChernoff::new(
            Arc::clone(&inner.graph),
            inner.a.clone(),
            inner.rho_bar,
        ) {
            Ok(c) => c,
            Err(e) => return SemiflowStatus::from(&e),
        };
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

/// Copy current var-coef graph heat signal values into `buf`.
///
/// ## Return values
/// `Ok` (0), `NullPtr` (5), `GridMismatch` (1), `Panic` (99).
///
/// # Safety
/// `state` from `smf_vc_ghc_new`; `buf` valid for `buf_len` f64 writes.
#[no_mangle]
pub unsafe extern "C" fn smf_vc_ghc_current(
    state: *const SmfVcGhc,
    buf: *mut f64,
    buf_len: u32,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<VcGhcInner>.
        let inner = unsafe { &*state.cast::<VcGhcInner>() };
        let vals = inner.current.values();
        if (buf_len as usize) < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfVcGhc` handle. Null-safe.
///
/// # Safety
/// `state` must be null or a pointer from `smf_vc_ghc_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_vc_ghc_drop(state: *mut SmfVcGhc) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<VcGhcInner>())) };
    }));
}
