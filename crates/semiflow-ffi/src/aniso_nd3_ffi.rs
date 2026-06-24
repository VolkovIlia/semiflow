//! FFI surface for 3D anisotropic-shift Chernoff engine (M19, ADR-0081).
//!
//! ## Engine
//!
//! `AnisoND3` — `AnisotropicShiftChernoffND<f64, 3>`.
//! Identical contract to `aniso_nd2_ffi` but for three spatial dimensions.
//!
//! ## Buffer layout (x-fastest / axis-0-fastest)
//!
//! State: flat `f64[nx*ny*nz]`, `idx(i,j,k) = i + j*nx + k*nx*ny`.
//! `a_values`: `9 * nx * ny * nz` entries (3×3 per point, row-major).
//! `b_values`: `3 * nx * ny * nz` entries, or null → zero.
//! `c_values`: `nx * ny * nz` entries, or null → zero.
//!
//! ## Entry points
//!
//! - `smf_aniso_nd3_new(nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,a,a_len,b or null,b_len,c or null,c_len,u0,u0_len,out)`
//! - `smf_aniso_nd3_evolve(ev,tau,n_steps,out,out_len)`
//! - `smf_aniso_nd3_size(ev)` → `nx*ny*nz`
//! - `smf_aniso_nd3_values(ev,out,out_len)`
//! - `smf_aniso_nd3_free(ev)` — null-safe
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
    clippy::too_many_arguments
)]

use std::{os::raw::c_double, sync::Arc};

use semiflow::{
    grid_nd::{GridFnND, GridND},
    shift_nd::AnisotropicShiftChernoffND,
    ChernoffFunction, Grid1D, ScratchPool,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle for a 3D anisotropic-shift evolver.
///
/// Obtain from `smf_aniso_nd3_new`; free with `smf_aniso_nd3_free`.
#[repr(C)]
pub struct SmfAnisoND3 {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct InnerND3 {
    kernel: Arc<AnisotropicShiftChernoffND<f64, 3>>,
    grid: GridND<f64, 3>,
    current: Vec<f64>,
    size: usize,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Construct a 3D anisotropic-shift evolver.
///
/// `a_values`: flat 3×3 SPD matrices, length `9 * nx * ny * nz` (row-major).
/// `b_values`: drift, length `3 * nx * ny * nz`, or **null** → zero.
/// `c_values`: reaction, length `nx * ny * nz`, or **null** → zero.
/// `u0`: initial state, length `nx * ny * nz`, all finite.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// Pointers must be valid for their lengths or null.
/// `out_ev` must be a writable `*mut *mut SmfAnisoND3`.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd3_new(
    nx: usize,
    ny: usize,
    nz: usize,
    xmin: c_double,
    xmax: c_double,
    ymin: c_double,
    ymax: c_double,
    zmin: c_double,
    zmax: c_double,
    a_values: *const c_double,
    a_len: usize,
    b_values: *const c_double,
    b_len: usize,
    c_values: *const c_double,
    c_len: usize,
    u0: *const c_double,
    u0_len: usize,
    out_ev: *mut *mut SmfAnisoND3,
) -> SemiflowStatus {
    if a_values.is_null() || u0.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let n_pts = nx * ny * nz;
        if a_len != 9 * n_pts || u0_len != n_pts {
            return SemiflowStatus::GridMismatch;
        }
        let a_sl = unsafe { std::slice::from_raw_parts(a_values, a_len) };
        let u0_sl = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        let (b_vec, c_vec) = match parse_bc_slices_nd3(b_values, b_len, c_values, c_len, n_pts) {
            Err(st) => return st,
            Ok(pair) => pair,
        };
        if let Err(st) = check_finite(a_sl).or_else(|_| check_finite(u0_sl)) {
            return st;
        }
        match build_nd3(
            nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, a_sl, b_vec, c_vec, u0_sl,
        ) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfAnisoND3>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve the 3D anisotropic state by `n_steps` of size `tau`.
///
/// Writes `nx*ny*nz` values into `out`. Advances internal state.
///
/// # Safety
/// `ev` non-null from `smf_aniso_nd3_new`; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd3_evolve(
    ev: *mut SmfAnisoND3,
    tau: c_double,
    n_steps: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerND3>() };
        if out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        if let Err(st) = check_tau_steps(tau, n_steps) {
            return st;
        }
        match run_nd3(
            &inner.kernel,
            inner.grid.clone(),
            inner.current.clone(),
            tau,
            n_steps,
        ) {
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

/// Return `nx * ny * nz`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or live from `smf_aniso_nd3_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd3_size(ev: *const SmfAnisoND3) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<InnerND3>() }.size
}

/// Copy current state into `out` (axis-0-fastest, length `out_len`).
///
/// # Safety
/// `ev` non-null; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd3_values(
    ev: *const SmfAnisoND3,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerND3>() };
        if out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        let sl = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        sl.copy_from_slice(&inner.current);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfAnisoND3` handle. Null-safe.
///
/// # Safety
/// `ev` must be null or live from `smf_aniso_nd3_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_aniso_nd3_free(ev: *mut SmfAnisoND3) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerND3>())) };
    }));
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

type Axes3 = [(f64, f64, usize); 3];

fn make_kernel_nd3(
    a_arc: Arc<Vec<f64>>,
    b_arc: Arc<Vec<f64>>,
    c_arc: Arc<Vec<f64>>,
    ns: [usize; 3],
    axes: Axes3,
    grid_nd: GridND<f64, 3>,
) -> Result<AnisotropicShiftChernoffND<f64, 3>, semiflow::SemiflowError> {
    let a2 = Arc::clone(&a_arc);
    let b2 = Arc::clone(&b_arc);
    let c2 = Arc::clone(&c_arc);
    let ax2 = axes;
    let ax3 = axes;
    AnisotropicShiftChernoffND::<f64, 3>::new(
        move |x, mat| {
            let base = flat3(x, &ns, &axes) * 9;
            for r in 0..3 {
                for ci in 0..3 {
                    mat.set(r, ci, a2[base + r * 3 + ci]);
                }
            }
        },
        move |x, bv| {
            let base = flat3(x, &ns, &ax2) * 3;
            bv[0] = b2[base];
            bv[1] = b2[base + 1];
            bv[2] = b2[base + 2];
        },
        move |x| c2[flat3(x, &ns, &ax3)],
        grid_nd,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_nd3(
    nx: usize,
    ny: usize,
    nz: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
    a_raw: &[f64],
    b_raw: Vec<f64>,
    c_raw: Vec<f64>,
    u0: &[f64],
) -> Result<InnerND3, semiflow::SemiflowError> {
    let grid_nd = GridND::<f64, 3>::new([
        Grid1D::new(xmin, xmax, nx)?,
        Grid1D::new(ymin, ymax, ny)?,
        Grid1D::new(zmin, zmax, nz)?,
    ])?;
    let axes = [(xmin, xmax, nx), (ymin, ymax, ny), (zmin, zmax, nz)];
    let kernel = make_kernel_nd3(
        Arc::new(a_raw.to_vec()),
        Arc::new(b_raw),
        Arc::new(c_raw),
        [nx, ny, nz],
        axes,
        grid_nd.clone(),
    )?;
    Ok(InnerND3 {
        kernel: Arc::new(kernel),
        grid: grid_nd,
        current: u0.to_vec(),
        size: nx * ny * nz,
    })
}

// ---------------------------------------------------------------------------
// Compute helper
// ---------------------------------------------------------------------------

fn run_nd3(
    kernel: &Arc<AnisotropicShiftChernoffND<f64, 3>>,
    grid: GridND<f64, 3>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let mut src = GridFnND::<f64, 3>::new(grid.clone(), input)?;
    let mut dst = GridFnND::<f64, 3>::new(grid, vec![0.0; src.values.len()])?;
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
    if n == 1 {
        return 0;
    }
    let fi = (x - xmin) / (xmax - xmin) * (n as f64 - 1.0);
    (fi.round() as isize).clamp(0, n as isize - 1) as usize
}

#[inline]
fn flat3(x: &[f64; 3], ns: &[usize; 3], axes: &[(f64, f64, usize)]) -> usize {
    let i = phys_idx(x[0], axes[0].0, axes[0].1, ns[0]);
    let j = phys_idx(x[1], axes[1].0, axes[1].1, ns[1]);
    let k = phys_idx(x[2], axes[2].0, axes[2].1, ns[2]);
    i + j * ns[0] + k * ns[0] * ns[1]
}

fn check_tau_steps(tau: f64, n_steps: usize) -> Result<(), SemiflowStatus> {
    if n_steps == 0 || !tau.is_finite() || tau <= 0.0 {
        return Err(SemiflowStatus::OutOfDomain);
    }
    Ok(())
}

fn check_finite(vals: &[f64]) -> Result<(), SemiflowStatus> {
    for &v in vals {
        if !v.is_finite() {
            return Err(SemiflowStatus::NanInf);
        }
    }
    Ok(())
}

/// Parse an optional pointer+length into a `Vec<f64>`.
///
/// Null → `vec![0.0; expected]`. Non-null + wrong length → `GridMismatch`.
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

/// Parse ND3 b and c optional slices, returning `(b_vec, c_vec)` or error.
///
/// # Safety
/// Non-null pointers must be readable for their stated lengths.
unsafe fn parse_bc_slices_nd3(
    b_ptr: *const f64,
    b_len: usize,
    c_ptr: *const f64,
    c_len: usize,
    n_pts: usize,
) -> Result<(Vec<f64>, Vec<f64>), SemiflowStatus> {
    let b = unsafe { parse_opt_slice(b_ptr, b_len, 3 * n_pts) }?;
    let c = unsafe { parse_opt_slice(c_ptr, c_len, n_pts) }?;
    Ok((b, c))
}
