//! v8.3.0 FFI surface for `ResolventJumpChernoff2D`/`3D` (F2 ND, ADR-0153, ADR-0148).
//!
//! Exposes the 2D and 3D parabolic resolvent time-jump approximation via six
//! `extern "C"` entry points (three per dimension).
//!
//! ## NARROW scope (§47.8, ADR-0148 NORMATIVE)
//!
//! Self-adjoint / sectorial generators only (diffusion family, 2D/3D).
//! `m_nodes >= 6` is enforced at construction. Non-sectorial generators are
//! OUT of scope (ADR-0148 DEFER). Complex contour arithmetic stays sealed;
//! only the real-valued result crosses the ABI (ADR-0138 hard constraint).
//!
//! ## ND layout contract (§3.1, `V8_3_TIER3_BINDING_DESIGN.md` — NORMATIVE)
//!
//! Rust `GridFn2D`/`GridFn3D` use **axis-0-fastest** storage:
//!   `idx(i,j) = j·nx + i`  (2D),
//!   `idx(i,j,k) = k·nx·ny + j·nx + i`  (3D).
//! C callers MUST lay out their flat `f64` buffers in the same axis-0-fastest
//! (Fortran column-major) order.  Length check: `g_len == nx·ny[·nz]`.
//!
//! ## Entry points (2D)
//!
//! - `smf_resolvent_jump_2d_new_heat_unit_v3(xmin,xmax,nx,ymin,ymax,ny,m,out)`
//! - `smf_resolvent_jump_2d_apply_v3(ev,t,g,g_len,out,out_len)` → jump
//! - `smf_resolvent_jump_2d_free_v3(ev)` — null-safe destructor
//!
//! ## Entry points (3D)
//!
//! - `smf_resolvent_jump_3d_new_heat_unit_v3(xmin,xmax,nx,ymin,ymax,ny,zmin,zmax,nz,m,out)`
//! - `smf_resolvent_jump_3d_apply_v3(ev,t,g,g_len,out,out_len)` → jump
//! - `smf_resolvent_jump_3d_free_v3(ev)` — null-safe destructor
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
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::too_many_arguments)]

use std::os::raw::c_double;

use semiflow_core::{
    Grid1D, Grid2D, Grid3D, GridFn2D, GridFn3D, ResolventJumpChernoff2D, ResolventJumpChernoff3D,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque FFI handle to a `ResolventJumpChernoff2D<f64>`.
///
/// C callers receive this from `smf_resolvent_jump_2d_new_heat_unit_v3` and
/// pass it to `smf_resolvent_jump_2d_apply_v3` / `_free_v3`.
/// Do not dereference or allocate this struct from C.
#[repr(C)]
pub struct SmfResolventJump2DV3 {
    _private: [u8; 0],
}

/// Opaque FFI handle to a `ResolventJumpChernoff3D<f64>`.
///
/// C callers receive this from `smf_resolvent_jump_3d_new_heat_unit_v3` and
/// pass it to `smf_resolvent_jump_3d_apply_v3` / `_free_v3`.
/// Do not dereference or allocate this struct from C.
#[repr(C)]
pub struct SmfResolventJump3DV3 {
    _private: [u8; 0],
}

// Inner storage types (heap-allocated, cast to/from opaque handles).
struct Inner2D {
    kernel: ResolventJumpChernoff2D<f64>,
}

struct Inner3D {
    kernel: ResolventJumpChernoff3D<f64>,
}

// ---------------------------------------------------------------------------
// 2D constructor
// ---------------------------------------------------------------------------

/// Construct a 2D resolvent-jump handle over `[xmin,xmax]×[ymin,ymax]`.
///
/// Grid is `nx × ny` nodes (axis-0-fastest: `idx(i,j) = j·nx + i`).
/// `m_nodes >= 6` (TWS contour floor, §47.8 NORMATIVE).
///
/// ## Preconditions
/// - `xmin < xmax`, `ymin < ymax`, both finite.
/// - `nx >= 4`, `ny >= 4`.
/// - `m_nodes >= 6`.
/// - `out_ev` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `out_ev` must be a valid writable `*mut *mut SmfResolventJump2DV3`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_2d_new_heat_unit_v3(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    m_nodes: usize,
    out_ev: *mut *mut SmfResolventJump2DV3,
) -> SemiflowStatus {
    if out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_inner_2d(xmin, xmax, nx, ymin, ymax, ny, m_nodes) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfResolventJump2DV3>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// 2D destructor
// ---------------------------------------------------------------------------

/// Free a 2D resolvent-jump handle.  Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_resolvent_jump_2d_new_*_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_2d_free_v3(ev: *mut SmfResolventJump2DV3) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<Inner2D>())) };
    }));
}

// ---------------------------------------------------------------------------
// 2D jump evaluation
// ---------------------------------------------------------------------------

/// Evaluate `e^{tA}g` for the 2D unit-diffusion heat kernel; write into `out`.
///
/// `g` and `out` are flat **axis-0-fastest** `f64` buffers of length `nx·ny`.
/// The complex contour arithmetic stays sealed inside core (ADR-0138).
///
/// ## Preconditions
/// - `ev` non-null, from `smf_resolvent_jump_2d_new_*_v3`.
/// - `t > 0`, finite.
/// - `g` non-null, `g_len == nx·ny`.
/// - `out` non-null, `out_len == nx·ny`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `g` must point to `g_len` readable contiguous `f64` values.
/// - `out` must point to `out_len` writable contiguous `f64` values.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_2d_apply_v3(
    ev: *mut SmfResolventJump2DV3,
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
        let inner = unsafe { &*ev.cast::<Inner2D>() };
        let n = inner.kernel.grid.len();
        if g_len != n || out_len != n {
            return SemiflowStatus::GridMismatch;
        }
        let g_slice = unsafe { std::slice::from_raw_parts(g, g_len) };
        let g_fn = GridFn2D {
            values: g_slice.to_vec(),
            grid: inner.kernel.grid,
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
// 3D constructor
// ---------------------------------------------------------------------------

/// Construct a 3D resolvent-jump handle over `[xmin,xmax]×[ymin,ymax]×[zmin,zmax]`.
///
/// Grid is `nx × ny × nz` nodes (axis-0-fastest: `idx(i,j,k) = k·nx·ny + j·nx + i`).
/// `m_nodes >= 6` (TWS contour floor, §47.8 NORMATIVE).
///
/// ## Preconditions
/// - All bounds finite; `xmin < xmax`, `ymin < ymax`, `zmin < zmax`.
/// - `nx,ny,nz >= 4`; `m_nodes >= 6`; `out_ev` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `out_ev` must be a valid writable `*mut *mut SmfResolventJump3DV3`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_3d_new_heat_unit_v3(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    zmin: c_double,
    zmax: c_double,
    nz: usize,
    m_nodes: usize,
    out_ev: *mut *mut SmfResolventJump3DV3,
) -> SemiflowStatus {
    if out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_inner_3d(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, m_nodes) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfResolventJump3DV3>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// 3D destructor
// ---------------------------------------------------------------------------

/// Free a 3D resolvent-jump handle.  Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_resolvent_jump_3d_new_*_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_3d_free_v3(ev: *mut SmfResolventJump3DV3) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<Inner3D>())) };
    }));
}

// ---------------------------------------------------------------------------
// 3D jump evaluation
// ---------------------------------------------------------------------------

/// Evaluate `e^{tA}g` for the 3D unit-diffusion heat kernel; write into `out`.
///
/// `g` and `out` are flat **axis-0-fastest** `f64` buffers of length `nx·ny·nz`.
/// Layout: `idx(i,j,k) = k·nx·ny + j·nx + i`.  The complex contour arithmetic
/// stays sealed inside core (ADR-0138).
///
/// ## Preconditions
/// - `ev` non-null, from `smf_resolvent_jump_3d_new_*_v3`.
/// - `t > 0`, finite.
/// - `g` non-null, `g_len == nx·ny·nz`.
/// - `out` non-null, `out_len == nx·ny·nz`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `g` must point to `g_len` readable contiguous `f64` values.
/// - `out` must point to `out_len` writable contiguous `f64` values.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent_jump_3d_apply_v3(
    ev: *mut SmfResolventJump3DV3,
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
        let inner = unsafe { &*ev.cast::<Inner3D>() };
        let n = inner.kernel.grid.len();
        if g_len != n || out_len != n {
            return SemiflowStatus::GridMismatch;
        }
        let g_slice = unsafe { std::slice::from_raw_parts(g, g_len) };
        let g_fn = GridFn3D {
            values: g_slice.to_vec(),
            grid: inner.kernel.grid,
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
// Private builders
// ---------------------------------------------------------------------------

fn build_inner_2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    m_nodes: usize,
) -> Result<Inner2D, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let grid = Grid2D::new(gx, gy);
    let kernel = ResolventJumpChernoff2D::new(grid, m_nodes)?;
    Ok(Inner2D { kernel })
}

fn build_inner_3d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
    m_nodes: usize,
) -> Result<Inner3D, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let gz = Grid1D::new(zmin, zmax, nz)?;
    let grid = Grid3D::new(gx, gy, gz)?;
    let kernel = ResolventJumpChernoff3D::new(grid, m_nodes)?;
    Ok(Inner3D { kernel })
}
