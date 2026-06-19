//! v8.1.0 FFI surface for `AdjointFokkerPlanckChernoff` (C2, ADR-0138, ADR-0107 Amdt 1).
//!
//! Exposes the adjoint (weak-*) Fokker-Planck Chernoff step on M(ℝ) via three
//! `extern "C"` entry points. The state is marshalled as two flat `f64` buffers
//! (positions, weights) of equal length — `MeasureState` never crosses the boundary.
//!
//! ## NARROW scope (§38.3, ADR-0107 AMENDMENT 1 NORMATIVE)
//!
//! Adjoint (weak-*) Fokker-Planck on M(ℝ). D=1 constant-coefficient 4-Dirac
//! pushforward (Lemma A.1, §38.3). Dirac count grows ×4 per step — bound
//! `max_steps`. Forward kernel = `DiffusionChernoff` (Brownian benchmark).
//!
//! ## ABI-safety invariant (ADR-0138 hard constraint)
//!
//! `MeasureState`<f64,1> never crosses the boundary. The input and output are
//! two parallel flat `f64` arrays (positions, weights). `max_steps` is set at
//! construction so the caller can pre-allocate output buffers of size
//! `4^max_steps * n_in`. The actual written count is returned via `out_n`.
//!
//! ## Entry points
//!
//! - `smf_adjoint_fp_new_brownian_1d_v3(a,b,c,max_steps,out_ev)` — construct
//! - `smf_adjoint_fp_step_v3(ev,tau,n_steps,pos_in,w_in,n_in,pos_out,w_out,out_cap,out_n)`
//! - `smf_adjoint_fp_free_v3(ev)` — null-safe destructor
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util between semiflow-{ffi,py,wasm}.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_truncation)]

use std::os::raw::c_double;

use semiflow_core::{
    AdjointFokkerPlanckChernoff, ChernoffFunction, DiffusionChernoff, Grid1D, MeasureState,
    ScratchPool, State,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque FFI handle to `AdjointFokkerPlanckChernoff<DiffusionChernoff<f64>, f64, 1>`.
///
/// Obtained from `smf_adjoint_fp_new_brownian_1d_v3`; passed to `_step_v3` /
/// `_free_v3`. Do not dereference or allocate this struct from C.
#[repr(C)]
pub struct SmfAdjointFpV3 {
    _private: [u8; 0],
}

/// Inner Rust state (heap-allocated, cast to/from `*mut SmfAdjointFpV3`).
struct AdjointFpInnerV3 {
    kernel: AdjointFokkerPlanckChernoff<DiffusionChernoff<f64>, f64, 1>,
    /// Maximum allowed step count (caps `4^max_steps` output growth).
    max_steps: usize,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Create an adjoint Fokker-Planck handle for the Brownian 1D benchmark.
///
/// ## Parameters
/// - `a`          — diffusion coefficient (`h = 2√(aτ)`). Must be ≥ 0 and finite.
/// - `b`          — drift coefficient (`k = 2bτ`). Must be finite.
/// - `c`          — reaction coefficient (mass factor `1 + τc`). Must be finite.
/// - `max_steps`  — `DoS` cap: output buffer must hold `4^max_steps * n_in` Diracs.
/// - `out_ev`     — receives the opaque pointer on success.
///
/// ## Return values
/// `Ok` | `NullPtr` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `out_ev` must be a valid writable `*mut *mut SmfAdjointFpV3`.
#[no_mangle]
pub unsafe extern "C" fn smf_adjoint_fp_new_brownian_1d_v3(
    a: c_double,
    b: c_double,
    c: c_double,
    max_steps: usize,
    out_ev: *mut *mut SmfAdjointFpV3,
) -> SemiflowStatus {
    if out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_adjoint_fp_inner(a, b, c, max_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfAdjointFpV3>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

/// Free an adjoint FP handle. Null-safe; do not use after this call.
///
/// # Safety
/// - `ev` must be null or a live pointer from `smf_adjoint_fp_new_brownian_1d_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_adjoint_fp_free_v3(ev: *mut SmfAdjointFpV3) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<AdjointFpInnerV3>())) };
    }));
}

// ---------------------------------------------------------------------------
// Step evaluation
// ---------------------------------------------------------------------------

/// Apply `n_steps` adjoint Fokker-Planck steps; write output Diracs into caller buffers.
///
/// Input measure: `n_in` Diracs at `pos_in[i], w_in[i]` (both length `n_in`).
/// Output measure written into `pos_out[0..out_n_written]` and `w_out[0..out_n_written]`
/// where `out_n_written` is stored in `*out_n`.
///
/// Output buffer capacity must be `out_cap >= 4^n_steps * n_in` Diracs.
/// If `out_cap < 4^n_steps * n_in`, returns `GridMismatch`.
///
/// Dirac count grows ×4 per step — bound `n_steps <= max_steps` (checked at call).
///
/// ## Preconditions
/// - `ev` non-null, from `smf_adjoint_fp_new_brownian_1d_v3`.
/// - `tau > 0`, finite.
/// - `n_steps >= 1` and `n_steps <= max_steps`.
/// - `pos_in`, `w_in` non-null, length `n_in`.
/// - `pos_out`, `w_out` non-null, length `out_cap >= 4^n_steps * n_in`.
/// - `out_n` non-null.
///
/// # Safety
/// Raw pointer dereferences guarded by null checks and length validation.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn smf_adjoint_fp_step_v3(
    ev: *mut SmfAdjointFpV3,
    tau: c_double,
    n_steps: usize,
    pos_in: *const c_double,
    w_in: *const c_double,
    n_in: usize,
    pos_out: *mut c_double,
    w_out: *mut c_double,
    out_cap: usize,
    out_n: *mut usize,
) -> SemiflowStatus {
    if ev.is_null()
        || pos_in.is_null()
        || w_in.is_null()
        || pos_out.is_null()
        || w_out.is_null()
        || out_n.is_null()
    {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<AdjointFpInnerV3>() };
        if n_steps == 0 || n_steps > inner.max_steps {
            return SemiflowStatus::OutOfDomain;
        }
        let required_cap = 4_usize.saturating_pow(n_steps as u32).saturating_mul(n_in);
        if out_cap < required_cap {
            return SemiflowStatus::GridMismatch;
        }
        let pos_slice = unsafe { std::slice::from_raw_parts(pos_in, n_in) };
        let w_slice = unsafe { std::slice::from_raw_parts(w_in, n_in) };
        let rho = match apply_adjoint_fp_steps(inner, tau, n_steps, pos_slice, w_slice) {
            Ok(r) => r,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let (out_pos_vec, out_w_vec) = rho.to_flat_buffers_d1();
        let n_out = out_pos_vec.len();
        if n_out > out_cap {
            return SemiflowStatus::GridMismatch;
        }
        unsafe { write_fp_output(pos_out, w_out, out_cap, out_n, &out_pos_vec, &out_w_vec) };
        SemiflowStatus::Ok
    })
}

/// Apply `n_steps` of the adjoint FP kernel, returning the evolved measure.
fn apply_adjoint_fp_steps(
    inner: &AdjointFpInnerV3,
    tau: f64,
    n_steps: usize,
    positions: &[f64],
    weights: &[f64],
) -> Result<MeasureState<f64, 1>, semiflow_core::SemiflowError> {
    let mut rho = build_measure_from_buffers(positions, weights);
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        let mut rho_next = MeasureState::<f64, 1>::dirac([0.0], 0.0);
        inner
            .kernel
            .apply_into(tau, &rho, &mut rho_next, &mut pool)?;
        rho = rho_next;
    }
    Ok(rho)
}

/// Write evolved measure into the caller's flat output buffers.
///
/// # Safety
/// Caller guarantees `pos_out`/`w_out` are valid for `out_cap` writes.
unsafe fn write_fp_output(
    pos_out: *mut f64,
    w_out: *mut f64,
    out_cap: usize,
    out_n: *mut usize,
    out_pos_vec: &[f64],
    out_w_vec: &[f64],
) {
    let out_pos_slice = std::slice::from_raw_parts_mut(pos_out, out_cap);
    let out_w_slice = std::slice::from_raw_parts_mut(w_out, out_cap);
    for (i, &v) in out_pos_vec.iter().enumerate() {
        out_pos_slice[i] = v;
    }
    for (i, &v) in out_w_vec.iter().enumerate() {
        out_w_slice[i] = v;
    }
    *out_n = out_pos_vec.len();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn build_adjoint_fp_inner(
    a: f64,
    b: f64,
    c: f64,
    max_steps: usize,
) -> Result<AdjointFpInnerV3, semiflow_core::SemiflowError> {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "adjoint_fp: a, b, c must be finite",
            value: a,
        });
    }
    if max_steps == 0 {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "adjoint_fp: max_steps must be >= 1",
            value: 0.0,
        });
    }
    // Unit-diffusion inner kernel (Brownian benchmark, NARROW scope §38.3).
    // DiffusionChernoff uses fn-pointers; for constant coefficients we use
    // new_const_a which takes a scalar. b and c are not part of DiffusionChernoff
    // (it models ∂_x(a∂_x)); the adjoint coefficients b,c are stored in
    // AdjointFokkerPlanckChernoff directly.
    let grid = Grid1D::new(-4.0_f64, 4.0, 32)?;
    let fwd = DiffusionChernoff::new_const_a(a, a, grid);
    let kernel = AdjointFokkerPlanckChernoff::new(fwd, a, b, c);
    Ok(AdjointFpInnerV3 { kernel, max_steps })
}

fn build_measure_from_buffers(positions: &[f64], weights: &[f64]) -> MeasureState<f64, 1> {
    let mut m = MeasureState::<f64, 1>::dirac([0.0_f64], 0.0);
    // Reset via zero then manually populate (axpy_into from scratch).
    m.zero_into();
    for (&p, &w) in positions.iter().zip(weights.iter()) {
        let atom = MeasureState::<f64, 1>::dirac([p], w);
        m.axpy_into(1.0, &atom);
    }
    m
}
