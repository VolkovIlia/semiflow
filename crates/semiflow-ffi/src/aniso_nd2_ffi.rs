//! FFI surface for 2D anisotropic-shift Chernoff engine (M19, ADR-0081).
//!
//! ## Engine
//!
//! `AnisoND2` — `AnisotropicShiftChernoffND<f64, 2>`.
//! Solves `∂_t u = A(x)·∇²u + b(x)·∇u + c(x)·u` with a 2×2 SPD tensor `A`.
//!
//! ## Buffer layout (x-fastest)
//!
//! State: flat `f64[nx*ny]`, `idx(i,j) = i + j*nx`.
//! `a_values`: `4 * nx * ny` entries (2×2 matrix per point, row-major).
//! `b_values`: `2 * nx * ny` entries (drift), or null pointer → zero.
//! `c_values`: `nx * ny` entries (reaction), or null pointer → zero.
//!
//! ## Entry points
//!
//! - `smf_aniso_nd2_new(nx,ny,xmin,xmax,ymin,ymax,a,a_len,b or null,b_len,c or null,c_len,u0,u0_len,out)`
//! - `smf_aniso_nd2_evolve(ev,tau,n_steps,out,out_len)`
//! - `smf_aniso_nd2_size(ev)` → `nx*ny`
//! - `smf_aniso_nd2_values(ev,out,out_len)`
//! - `smf_aniso_nd2_free(ev)` — null-safe
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.

#![allow(unsafe_code)]
#![allow(
    clippy::assigning_clones,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
)]

use std::os::raw::c_double;
use std::sync::Arc;

use semiflow_core::{
    grid_nd::{GridFnND, GridND},
    shift_nd::AnisotropicShiftChernoffND,
    ChernoffFunction, Grid1D, ScratchPool,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle for a 2D anisotropic-shift evolver.
///
/// Obtain from `smf_aniso_nd2_new`; free with `smf_aniso_nd2_free`.
#[repr(C)]
pub struct SmfAnisoND2 {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct InnerND2 {
    kernel: Arc<AnisotropicShiftChernoffND<f64, 2>>,
    grid: GridND<f64, 2>,
    current: Vec<f64>,
    size: usize,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Construct a 2D anisotropic-shift evolver from pre-sampled coefficient arrays.
///
/// `a_values`: flat `2×2` SPD matrices, length `4 * nx * ny` (row-major).
/// `b_values`: drift vectors, length `2 * nx * ny`, or **null** → zero.
/// `c_values`: reaction scalars, length `nx * ny`, or **null** → zero.
/// `u0`: initial state, length `nx * ny`, all values must be finite.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// Pointers must be valid for their stated lengths or null.
/// `out_ev` must be a writable `*mut *mut SmfAnisoND2`.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd2_new(
    nx: usize,
    ny: usize,
    xmin: c_double,
    xmax: c_double,
    ymin: c_double,
    ymax: c_double,
    a_values: *const c_double,
    a_len: usize,
    b_values: *const c_double,
    b_len: usize,
    c_values: *const c_double,
    c_len: usize,
    u0: *const c_double,
    u0_len: usize,
    out_ev: *mut *mut SmfAnisoND2,
) -> SemiflowStatus {
    if a_values.is_null() || u0.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let n_pts = nx * ny;
        if a_len != 4 * n_pts || u0_len != n_pts {
            return SemiflowStatus::GridMismatch;
        }
        let a_sl = unsafe { std::slice::from_raw_parts(a_values, a_len) };
        let u0_sl = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        let b_vec = match parse_opt_slice(b_values, b_len, 2 * n_pts) {
            Err(st) => return st, Ok(v) => v,
        };
        let c_vec = match parse_opt_slice(c_values, c_len, n_pts) {
            Err(st) => return st, Ok(v) => v,
        };
        if let Err(st) = check_finite(a_sl) { return st; }
        if let Err(st) = check_finite(u0_sl) { return st; }
        match build_nd2(nx, ny, xmin, xmax, ymin, ymax, a_sl, b_vec, c_vec, u0_sl) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfAnisoND2>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve the 2D anisotropic state by `n_steps` of size `tau`.
///
/// Writes `nx*ny` values into `out` (x-fastest). Advances internal state.
///
/// # Safety
/// `ev` non-null from `smf_aniso_nd2_new`; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd2_evolve(
    ev: *mut SmfAnisoND2,
    tau: c_double,
    n_steps: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerND2>() };
        if out_len != inner.size { return SemiflowStatus::GridMismatch; }
        if let Err(st) = check_tau_steps(tau, n_steps) { return st; }
        match run_nd2(&inner.kernel, inner.grid.clone(), inner.current.clone(), tau, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                inner.current = result.clone();
                let sl = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                sl.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return `nx * ny`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or live from `smf_aniso_nd2_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd2_size(ev: *const SmfAnisoND2) -> usize {
    if ev.is_null() { return 0; }
    unsafe { &*ev.cast::<InnerND2>() }.size
}

/// Copy current state into `out` (x-fastest, length `out_len`).
///
/// # Safety
/// `ev` non-null; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd2_values(
    ev: *const SmfAnisoND2,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerND2>() };
        if out_len != inner.size { return SemiflowStatus::GridMismatch; }
        let sl = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        sl.copy_from_slice(&inner.current);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfAnisoND2` handle. Null-safe.
///
/// # Safety
/// `ev` must be null or live from `smf_aniso_nd2_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd2_free(ev: *mut SmfAnisoND2) {
    if ev.is_null() { return; }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerND2>())) };
    }));
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_nd2(
    nx: usize,
    ny: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    a_raw: &[f64],
    b_raw: Vec<f64>,
    c_raw: Vec<f64>,
    u0: &[f64],
) -> Result<InnerND2, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let grid_nd = GridND::<f64, 2>::new([gx, gy])?;
    let a_arc = Arc::new(a_raw.to_vec());
    let b_arc = Arc::new(b_raw);
    let c_arc = Arc::new(c_raw);
    let ns = [nx, ny];
    let axes = [(xmin, xmax, nx), (ymin, ymax, ny)];
    let a2 = Arc::clone(&a_arc);
    let b2 = Arc::clone(&b_arc);
    let c2 = Arc::clone(&c_arc);
    let ax2 = axes;
    let ax3 = axes;
    let kernel = AnisotropicShiftChernoffND::<f64, 2>::new(
        move |x, mat| {
            let flat = flat2(x, &ns, &axes);
            let base = flat * 4;
            mat.set(0, 0, a2[base]);
            mat.set(0, 1, a2[base + 1]);
            mat.set(1, 0, a2[base + 2]);
            mat.set(1, 1, a2[base + 3]);
        },
        move |x, bv| {
            let flat = flat2(x, &ns, &ax2);
            bv[0] = b2[flat * 2];
            bv[1] = b2[flat * 2 + 1];
        },
        move |x| c2[flat2(x, &ns, &ax3)],
        grid_nd.clone(),
    )?;
    Ok(InnerND2 {
        kernel: Arc::new(kernel),
        grid: grid_nd,
        current: u0.to_vec(),
        size: nx * ny,
    })
}

// ---------------------------------------------------------------------------
// Compute helper
// ---------------------------------------------------------------------------

fn run_nd2(
    kernel: &Arc<AnisotropicShiftChernoffND<f64, 2>>,
    grid: GridND<f64, 2>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let mut src = GridFnND::<f64, 2>::new(grid.clone(), input)?;
    let mut dst = GridFnND::<f64, 2>::new(grid, vec![0.0; src.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Index + validation helpers (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

#[inline]
fn phys_idx(x: f64, xmin: f64, xmax: f64, n: usize) -> usize {
    if n == 1 { return 0; }
    let fi = (x - xmin) / (xmax - xmin) * (n as f64 - 1.0);
    (fi.round() as isize).clamp(0, n as isize - 1) as usize
}

#[inline]
fn flat2(x: &[f64; 2], ns: &[usize; 2], axes: &[(f64, f64, usize)]) -> usize {
    phys_idx(x[0], axes[0].0, axes[0].1, ns[0])
        + phys_idx(x[1], axes[1].0, axes[1].1, ns[1]) * ns[0]
}

fn check_tau_steps(tau: f64, n_steps: usize) -> Result<(), SemiflowStatus> {
    if n_steps == 0 || !tau.is_finite() || tau <= 0.0 {
        return Err(SemiflowStatus::OutOfDomain);
    }
    Ok(())
}

fn check_finite(vals: &[f64]) -> Result<(), SemiflowStatus> {
    for &v in vals {
        if !v.is_finite() { return Err(SemiflowStatus::NanInf); }
    }
    Ok(())
}

/// Parse an optional pointer+length into a `Vec<f64>`.
///
/// Null pointer → `vec![0.0; expected]`. Non-null with wrong length → `GridMismatch`.
///
/// # Safety
/// If non-null, `ptr` must be readable for `len` f64 values.
unsafe fn parse_opt_slice(
    ptr: *const f64,
    len: usize,
    expected: usize,
) -> Result<Vec<f64>, SemiflowStatus> {
    if ptr.is_null() {
        return Ok(vec![0.0_f64; expected]);
    }
    if len != expected {
        return Err(SemiflowStatus::GridMismatch);
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
}
