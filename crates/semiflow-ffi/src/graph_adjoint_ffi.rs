//! C ABI: `SmfGraphAdjoint` — pre-sampled time-varying graph state-adjoint (ADR-0180).
//!
//! No live callbacks during evolve. The host pre-samples the Laplacian weight
//! sequence at the `2·n_steps` GL₄ abscissa times (obtainable via
//! `smf_graph_adjoint_abscissa_times`) and passes the flat array at
//! construction. Evolve is a pure Rust replay.
//!
//! ## ABI conventions (ADR-0028)
//!
//! - Null-check BEFORE `catch_panic!`.
//! - `catch_panic!` wraps all fallible logic; panics → `Panic` (99).
//! - `vals_seq` is **copied** at construction; caller may free immediately.
//! - No new `SemiflowStatus` variants (ADR-0171); mismatches → `OutOfDomain`.
//!
//! ## evolve n_steps / tau discipline
//!
//! `t_horizon` is captured at construction (`tau = t_horizon / n_steps`).
//! `evolve_state_adjoint` MUST pass the same `n_steps` used at construction;
//! passing a different value returns `OutOfDomain` (3).

#![allow(unsafe_code)]

use std::sync::Arc;

use semiflow_core::{
    graph_adjoint_presampled::{
        fill_abscissa_times, PreSampledLaplacianSeq, PreSampledMagnusAdj, PreSampledVarCoefAdj,
    },
    Graph, GraphSignal, LaplacianKind, MagnusGraphHeatChernoff, VarCoefMagnusGraphHeatChernoff,
};
use semiflow_core::scratch::ScratchPool;

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a pre-sampled graph state-adjoint (ADR-0180).
///
/// Allocate with `smf_graph_adjoint_new_presampled[_varcoef]`.
/// Free with `smf_graph_adjoint_free`.
#[repr(C)]
pub struct SmfGraphAdjoint {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner (Rust-private)
// ---------------------------------------------------------------------------

enum AdjVariant {
    Magnus(PreSampledMagnusAdj<f64>),
    VarCoef(PreSampledVarCoefAdj<f64>),
}

struct GraphAdjointInner {
    /// Dummy graph used only to allocate `GraphSignal` buffers.
    graph: Arc<Graph<f64>>,
    variant: AdjVariant,
    scratch: ScratchPool<f64>,
    /// Captured at construction: `tau = t_horizon / n_steps`.
    tau: f64,
    /// n_steps captured at construction; checked in evolve.
    n_steps: usize,
}

impl GraphAdjointInner {
    fn n_nodes(&self) -> usize {
        match &self.variant {
            AdjVariant::Magnus(p) => p.n_nodes(),
            AdjVariant::VarCoef(p) => p.n_nodes(),
        }
    }
}

// ---------------------------------------------------------------------------
// smf_graph_adjoint_abscissa_times — helper for host pre-sampling
// ---------------------------------------------------------------------------

/// Fill `out` (`2 * n_steps` doubles) with the GL₄ abscissa sample times in
/// adjoint-schedule order `[(0,c₁),(0,c₂),(1,c₁),(1,c₂),…]`.
///
/// The host must pre-sample the Laplacian at exactly these times before
/// calling `smf_graph_adjoint_new_presampled`.
///
/// # Safety
/// `out` must be a valid, writable buffer of at least `2 * n_steps` doubles.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_adjoint_abscissa_times(
    t_horizon: f64,
    n_steps: usize,
    out: *mut f64,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if n_steps == 0 || !t_horizon.is_finite() || t_horizon <= 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    let slice = unsafe { core::slice::from_raw_parts_mut(out, 2 * n_steps) };
    fill_abscissa_times(t_horizon, n_steps, slice);
    SemiflowStatus::Ok
}

// ---------------------------------------------------------------------------
// smf_graph_adjoint_new_presampled — Magnus K=4 variant
// ---------------------------------------------------------------------------

/// Construct a pre-sampled Magnus K=4 graph state-adjoint (ADR-0180).
///
/// ## Parameters
/// - `n_nodes`: graph size (also `row_ptr_len = n_nodes + 1`).
/// - `row_ptr` (`len row_ptr_len`), `col_idx` (`len nnz`), `vals_seq`
///   (`len 2*n_steps*nnz`): CSR pattern + pre-sampled Laplacian weights.
/// - `n_steps`, `t_horizon`: must match the abscissa grid used to sample.
/// - `rho_bar_max > 0`: Gershgorin upper bound.
/// - `convergence_check != 0`: enable Magnus radius guard.
/// - `kind`: 0 = combinatorial, 1 = sym-normalized.
/// - `out`: receives the opaque handle on success.
///
/// ## Return values
/// `Ok`(0), `NullPtr`(5), `OutOfDomain`(3), `GridMismatch`(1), `Panic`(99).
///
/// ## Ownership
/// `vals_seq` is copied; caller may free it immediately.
/// Free the handle with `smf_graph_adjoint_free`.
///
/// # Safety
/// All pointer arguments must be valid for their stated lengths.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn smf_graph_adjoint_new_presampled(
    n_nodes: usize,
    row_ptr: *const usize,
    row_ptr_len: usize,
    col_idx: *const u32,
    nnz: usize,
    vals_seq: *const f64,
    n_steps: usize,
    t_horizon: f64,
    rho_bar_max: f64,
    convergence_check: i32,
    kind: i32,
    out: *mut *mut SmfGraphAdjoint,
) -> SemiflowStatus {
    if row_ptr.is_null() || col_idx.is_null() || vals_seq.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if n_steps == 0 || !t_horizon.is_finite() || t_horizon <= 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    catch_panic!({
        let rp = unsafe { core::slice::from_raw_parts(row_ptr, row_ptr_len) }.to_vec();
        let ci = unsafe { core::slice::from_raw_parts(col_idx, nnz) }.to_vec();
        let vs_len = 2 * n_steps * nnz;
        let vs = unsafe { core::slice::from_raw_parts(vals_seq, vs_len) }.to_vec();
        let lk = kind_from_i32(kind);
        let seq = match PreSampledLaplacianSeq::new(rp, ci, vs, n_steps, lk) {
            Ok(s) => s,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let do_check = convergence_check != 0;
        let ps = match MagnusGraphHeatChernoff::<f64>::from_presampled(seq, rho_bar_max, do_check) {
            Ok(p) => p,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let graph = Arc::new(Graph::<f64>::path(n_nodes.max(1)));
        let tau = t_horizon / n_steps as f64;
        let inner = GraphAdjointInner {
            graph, variant: AdjVariant::Magnus(ps), scratch: ScratchPool::new(), tau, n_steps,
        };
        let raw = Box::into_raw(Box::new(inner)).cast::<SmfGraphAdjoint>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_graph_adjoint_new_presampled_varcoef — VarCoef variant
// ---------------------------------------------------------------------------

/// Construct a pre-sampled VarCoef graph state-adjoint (ADR-0180).
///
/// Adds `a_seq` (len `2*n_steps*n_nodes`) and `a_sup_max` over the base
/// `smf_graph_adjoint_new_presampled` parameters.
///
/// # Safety
/// All pointer arguments must be valid for their stated lengths.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn smf_graph_adjoint_new_presampled_varcoef(
    n_nodes: usize,
    row_ptr: *const usize,
    row_ptr_len: usize,
    col_idx: *const u32,
    nnz: usize,
    vals_seq: *const f64,
    n_steps: usize,
    a_seq: *const f64,
    a_sup_max: f64,
    t_horizon: f64,
    rho_bar_max: f64,
    kind: i32,
    out: *mut *mut SmfGraphAdjoint,
) -> SemiflowStatus {
    if row_ptr.is_null() || col_idx.is_null() || vals_seq.is_null()
        || a_seq.is_null() || out.is_null()
    {
        return SemiflowStatus::NullPtr;
    }
    if n_steps == 0 || !t_horizon.is_finite() || t_horizon <= 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    catch_panic!({
        let rp = unsafe { core::slice::from_raw_parts(row_ptr, row_ptr_len) }.to_vec();
        let ci = unsafe { core::slice::from_raw_parts(col_idx, nnz) }.to_vec();
        let vs_len = 2 * n_steps * nnz;
        let vs = unsafe { core::slice::from_raw_parts(vals_seq, vs_len) }.to_vec();
        let a_v = unsafe {
            core::slice::from_raw_parts(a_seq, 2 * n_steps * n_nodes)
        }.to_vec();
        let lk = kind_from_i32(kind);
        let seq = match PreSampledLaplacianSeq::new(rp, ci, vs, n_steps, lk) {
            Ok(s) => s,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let ps = match VarCoefMagnusGraphHeatChernoff::<f64>::from_presampled(
            seq, a_v, rho_bar_max, a_sup_max,
        ) {
            Ok(p) => p,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let graph = Arc::new(Graph::<f64>::path(n_nodes.max(1)));
        let tau = t_horizon / n_steps as f64;
        let inner = GraphAdjointInner {
            graph, variant: AdjVariant::VarCoef(ps), scratch: ScratchPool::new(), tau, n_steps,
        };
        let raw = Box::into_raw(Box::new(inner)).cast::<SmfGraphAdjoint>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_graph_adjoint_evolve_state_adjoint
// ---------------------------------------------------------------------------

/// Backward costate sweep `λ_n → λ_0`.
///
/// `n_steps` MUST equal the value supplied at construction.
///
/// ## Return values
/// `Ok`(0), `NullPtr`(5), `OutOfDomain`(3), `GridMismatch`(1), `Panic`(99).
///
/// # Safety
/// `lambda_n` / `out` must be valid slices of length `>= n_nodes`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_adjoint_evolve_state_adjoint(
    h: *const SmfGraphAdjoint,
    lambda_n: *const f64,
    lambda_len: usize,
    n_steps: usize,
    out: *mut f64,
    out_len: usize,
) -> SemiflowStatus {
    if h.is_null() || lambda_n.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *h.cast::<GraphAdjointInner>().cast_mut() };
        let n = inner.n_nodes();
        if lambda_len < n || out_len < n {
            return SemiflowStatus::GridMismatch;
        }
        if n_steps != inner.n_steps {
            return SemiflowStatus::OutOfDomain;
        }
        let lam_slice = unsafe { core::slice::from_raw_parts(lambda_n, n) };
        let out_slice = unsafe { core::slice::from_raw_parts_mut(out, n) };
        // Build GraphSignal buffers using the dummy path-graph allocator.
        let src = GraphSignal::from_fn(Arc::clone(&inner.graph), |i| lam_slice[i as usize]);
        let mut dst = GraphSignal::zeros(Arc::clone(&inner.graph));
        let tau = inner.tau;
        let result = match &inner.variant {
            AdjVariant::Magnus(ps) => {
                ps.evolve_state_adjoint_into(
                    tau, n_steps, &src, &mut dst, &mut inner.scratch,
                )
            }
            AdjVariant::VarCoef(ps) => {
                ps.evolve_state_adjoint_into(
                    tau, n_steps, &src, &mut dst, &mut inner.scratch,
                )
            }
        };
        match result {
            Ok(()) => {
                out_slice.copy_from_slice(dst.values());
                SemiflowStatus::Ok
            }
            Err(e) => SemiflowStatus::from(&e),
        }
    })
}

// ---------------------------------------------------------------------------
// smf_graph_adjoint_n_nodes / smf_graph_adjoint_free
// ---------------------------------------------------------------------------

/// Return the number of graph nodes. Returns 0 if `h` is null.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_adjoint_n_nodes(h: *const SmfGraphAdjoint) -> usize {
    if h.is_null() {
        return 0;
    }
    let inner = unsafe { &*h.cast::<GraphAdjointInner>() };
    inner.n_nodes()
}

/// Free a `SmfGraphAdjoint` handle (null-safe).
///
/// # Safety
/// `h` must be null or a live pointer from `smf_graph_adjoint_new_presampled[_varcoef]`.
#[no_mangle]
pub unsafe extern "C" fn smf_graph_adjoint_free(h: *mut SmfGraphAdjoint) {
    if h.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(h.cast::<GraphAdjointInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn kind_from_i32(kind: i32) -> LaplacianKind {
    if kind == 1 { LaplacianKind::SymNormalized } else { LaplacianKind::Combinatorial }
}
