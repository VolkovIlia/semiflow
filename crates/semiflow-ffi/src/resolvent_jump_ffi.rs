//! v8.1.0 FFI surface for `ResolventJumpChernoff` (F2, ADR-0138, ADR-0134).
//!
//! Exposes the large-step resolvent time-jump approximation for the 1D
//! unit-diffusion heat kernel via three `extern "C"` entry points.
//!
//! ## NARROW scope (§47.4, ADR-0134 NORMATIVE)
//!
//! Self-adjoint / sectorial generators only (diffusion family).
//! Non-self-adjoint / advection-dominated generators are OUT of scope
//! (math.md §47.4). `m_nodes >= 6` is enforced at construction.
//!
//! ## ABI-safety invariant (ADR-0138 hard constraint)
//!
//! `Complex<f64>` / TWS contour arithmetic stays sealed inside core.
//! This surface marshals flat `f64` buffers only: input `g` as `(ptr,len)`,
//! output as a caller-owned `(out,out_len)`, both length `n_grid`.
//!
//! ## Entry points
//!
//! - `smf_resolvent_jump_new_heat_1d_unit_v3(xmin,xmax,n_grid,m_nodes,out_ev)`
//! - `smf_resolvent_jump_apply_v3(ev,t,g,g_len,out,out_len)` → jump result
//! - `smf_resolvent_jump_free_v3(ev)` — null-safe destructor
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util between
//! semiflow-{ffi,py,wasm}.  This file owns its own builder + validators.

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow_core::{DiffusionChernoff, Grid1D, GridFn1D, ResolventJumpChernoff};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque FFI handle to a `ResolventJumpChernoff<DiffusionChernoff<f64>, f64>`.
///
/// C callers receive this from `smf_resolvent_jump_new_heat_1d_unit_v3` and
/// pass it to `smf_resolvent_jump_apply_v3` / `smf_resolvent_jump_free_v3`.
/// Do not dereference or allocate this struct from C.
#[repr(C)]
pub struct SmfResolventJumpV3 {
    _private: [u8; 0],
}

/// Inner Rust state (heap-allocated, cast to/from `*mut SmfResolventJumpV3`).
struct ResolventJumpInnerV3 {
    kernel: ResolventJumpChernoff<DiffusionChernoff<f64>, f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Create a resolvent-jump handle for the unit-diffusion heat kernel.
///
/// Constructs `ResolventJumpChernoff<DiffusionChernoff<f64>>` with unit `a=1`,
/// `m_nodes` TWS contour nodes, on `[xmin, xmax]` with `n_grid` grid points.
///
/// On success `*out_ev` is set to a non-null opaque pointer.
///
/// ## Preconditions
/// - `xmin < xmax`, both finite.
/// - `n_grid >= 4`.
/// - `m_nodes >= 6` (geometric-regime floor, §47.4 NORMATIVE).
/// - `out_ev` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `out_ev` must be a valid writable `*mut *mut SmfResolventJumpV3`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_new_heat_1d_unit_v3(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    m_nodes: usize,
    out_ev: *mut *mut SmfResolventJumpV3,
) -> SemiflowStatus {
    if out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_resolvent_jump_inner(xmin, xmax, n_grid, m_nodes) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfResolventJumpV3>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

/// Free a resolvent-jump handle.  Null-safe; do not use after this call.
///
/// # Safety
/// - `ev` must be null or a live pointer from `smf_resolvent_jump_new_*_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_free_v3(ev: *mut SmfResolventJumpV3) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<ResolventJumpInnerV3>())) };
    }));
}

// ---------------------------------------------------------------------------
// Jump evaluation
// ---------------------------------------------------------------------------

/// Evaluate `e^{tA}g` via the TWS contour quadrature; write into `out`.
///
/// `g` and `out` are both length `n_grid` (the value passed to the constructor).
/// The contour arithmetic (`Complex<f64>`) stays sealed inside core — only the
/// real part is returned (ABI-safety invariant, ADR-0138).
///
/// ## Preconditions
/// - `ev` non-null, from `smf_resolvent_jump_new_*_v3`.
/// - `t > 0`, finite.
/// - `g` non-null, length `g_len == n_grid`.
/// - `out` non-null, length `out_len == n_grid`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `g` must point to `g_len` readable contiguous `f64` values.
/// - `out` must point to `out_len` writable contiguous `f64` values.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_apply_v3(
    ev: *mut SmfResolventJumpV3,
    t: c_double,
    g: *const c_double,
    g_len: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || g.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<ResolventJumpInnerV3>() };
        let n = inner.kernel.grid.n;
        if g_len != n || out_len != n {
            return SemiflowStatus::GridMismatch;
        }
        let g_slice = unsafe { std::slice::from_raw_parts(g, g_len) };
        let g_fn = match GridFn1D::new(inner.kernel.grid, g_slice.to_vec()) {
            Err(e) => return SemiflowStatus::from(&e),
            Ok(gf) => gf,
        };
        match inner.kernel.jump(t, &g_fn) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => unsafe {
                let out_slice = std::slice::from_raw_parts_mut(out, out_len);
                for (i, &v) in result.values.iter().enumerate() {
                    out_slice[i] = v;
                }
                SemiflowStatus::Ok
            },
        }
    })
}

// ---------------------------------------------------------------------------
// Private builder
// ---------------------------------------------------------------------------

fn build_resolvent_jump_inner(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    m_nodes: usize,
) -> Result<ResolventJumpInnerV3, semiflow_core::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n_grid)?;
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    let kernel = ResolventJumpChernoff::new(chernoff, m_nodes, grid)?;
    Ok(ResolventJumpInnerV3 { kernel })
}
