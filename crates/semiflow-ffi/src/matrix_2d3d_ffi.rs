//! FFI surface for `MatrixDiffusionChernoff2D` and `MatrixDiffusionChernoff3D`
//! (ADR-0124, math §33.2/33.3). Coupled 2-component diffusion via palindromic Strang.
//!
//! ## Symbol names
//!
//! 2D: `smf_matrix2d_new` / `smf_matrix2d_evolve` / `smf_matrix2d_values`
//!     / `smf_matrix2d_size` / `smf_matrix2d_free`
//!
//! 3D: `smf_matrix3d_new` / `smf_matrix3d_evolve` / `smf_matrix3d_values`
//!     / `smf_matrix3d_size` / `smf_matrix3d_free`
//!
//! ## Buffer layout
//!
//! 2D: flat `2*nx*ny` doubles, index `(j*nx+i)*2+c`.
//! 3D: flat `2*nx*ny*nz` doubles, index `(k*nx*ny+j*nx+i)*2+c`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use std::os::raw::c_double;

use semiflow::{
    matrix_2d3d::{MatrixDiffusionChernoff2D, MatrixDiffusionChernoff3D, MatrixGridFn2D, MatrixGridFn3D},
    matrix_system::MatrixDiffusionChernoff,
    BoundaryPolicy, ChernoffSemigroup, Grid1D, Grid2D, Grid3D,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Shared kernel builder (same as structured_matrix pattern)
// ---------------------------------------------------------------------------

fn build_matrix_kernel(
    a_diag: f64,
    c_coupling: f64,
    axis: Grid1D<f64>,
) -> Result<MatrixDiffusionChernoff<f64, 2>, semiflow::SemiflowError> {
    let a = a_diag;
    let c = c_coupling;
    MatrixDiffusionChernoff::<f64, 2>::new(
        move |_x, mat| {
            mat[0][0] = a;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = a;
        },
        |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = 0.0;
        },
        move |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = c;
            mat[1][0] = c;
            mat[1][1] = 0.0;
        },
        axis,
    )
}

// ===========================================================================
// 2D — smf_matrix2d_*
// ===========================================================================

/// Opaque handle to a `MatrixDiffusionChernoff2D` evolver.
#[repr(C)]
pub struct SmfMatrix2D {
    _private: [u8; 0],
}

struct InnerMatrix2D {
    a_diag: f64,
    c_coupling: f64,
    grid2d: Grid2D<f64>,
    n_steps: usize,
    current: Vec<f64>,
}

/// Allocate a `MatrixDiffusionChernoff2D` evolver (M=2, constant scalar coefficients).
///
/// Solves coupled 2-component 2D diffusion via palindromic Strang (order 2, ADR-0124).
///
/// # Safety
/// `u0` must point to `2*nx*ny` readable `f64`s.
/// `out` must be a valid `*mut *mut SmfMatrix2D`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix2d_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    a_diag: c_double,
    c_coupling: c_double,
    u0: *const c_double,
    u0_len: usize,
    n_steps: usize,
    out: *mut *mut SmfMatrix2D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if !a_diag.is_finite() || a_diag <= 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    if u0_len != 2 * nx * ny {
        return SemiflowStatus::GridMismatch;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_matrix2d(xmin, xmax, nx, ymin, ymax, ny, a_diag, c_coupling, n_steps, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfMatrix2D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `SmfMatrix2D` state by time `t`, writing `2*nx*ny` values into `dst_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_matrix2d_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix2d_evolve(
    ev: *mut SmfMatrix2D,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerMatrix2D>() };
        if dst_len != inner.current.len() {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_matrix2d(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, inner.current.len()) };
        out.copy_from_slice(&inner.current);
        SemiflowStatus::Ok
    })
}

/// Copy current `SmfMatrix2D` values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_matrix2d_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix2d_values(
    ev: *const SmfMatrix2D,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerMatrix2D>() };
        if out_len < inner.current.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, inner.current.len()) };
        out.copy_from_slice(&inner.current);
        SemiflowStatus::Ok
    })
}

/// Return `SmfMatrix2D` buffer size (`2*nx*ny`); 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_matrix2d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix2d_size(ev: *const SmfMatrix2D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerMatrix2D>() };
    inner.current.len()
}

/// Free a `SmfMatrix2D` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_matrix2d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix2d_free(ev: *mut SmfMatrix2D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerMatrix2D>())) };
    }));
}

// ---------------------------------------------------------------------------
// 2D helpers
// ---------------------------------------------------------------------------

fn build_matrix2d(
    xmin: f64, xmax: f64, nx: usize,
    ymin: f64, ymax: f64, ny: usize,
    a_diag: f64, c_coupling: f64, n_steps: usize,
    u0: &[f64],
) -> Result<InnerMatrix2D, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(BoundaryPolicy::Reflect);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(BoundaryPolicy::Reflect);
    let grid2d = Grid2D::new(gx, gy);
    // Validate kernel construction.
    build_matrix_kernel(a_diag, c_coupling, gx)?;
    Ok(InnerMatrix2D { a_diag, c_coupling, grid2d, n_steps, current: u0.to_vec() })
}

fn evolve_matrix2d(inner: &mut InnerMatrix2D, t: f64) -> Result<(), semiflow::SemiflowError> {
    let g = inner.grid2d;
    let kx = build_matrix_kernel(inner.a_diag, inner.c_coupling, g.x)?;
    let ky = build_matrix_kernel(inner.a_diag, inner.c_coupling, g.y)?;
    let kernel = MatrixDiffusionChernoff2D::new(kx, ky);
    let sg = ChernoffSemigroup::new(kernel, inner.n_steps)?;
    let mut src = MatrixGridFn2D::<f64, 2>::new(g);
    src.values.copy_from_slice(&inner.current);
    let out = sg.evolve(t, &src)?;
    inner.current.copy_from_slice(&out.values);
    Ok(())
}

// ===========================================================================
// 3D — smf_matrix3d_*
// ===========================================================================

/// Opaque handle to a `MatrixDiffusionChernoff3D` evolver.
#[repr(C)]
pub struct SmfMatrix3D {
    _private: [u8; 0],
}

struct InnerMatrix3D {
    a_diag: f64,
    c_coupling: f64,
    grid3d: Grid3D<f64>,
    n_steps: usize,
    current: Vec<f64>,
}

/// Allocate a `MatrixDiffusionChernoff3D` evolver (M=2, constant scalar coefficients).
///
/// Solves coupled 2-component 3D diffusion via palindromic Strang (order 2, ADR-0124).
///
/// # Safety
/// `u0` must point to `2*nx*ny*nz` readable `f64`s.
/// `out` must be a valid `*mut *mut SmfMatrix3D`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix3d_new(
    xmin: c_double, xmax: c_double, nx: usize,
    ymin: c_double, ymax: c_double, ny: usize,
    zmin: c_double, zmax: c_double, nz: usize,
    a_diag: c_double,
    c_coupling: c_double,
    u0: *const c_double,
    u0_len: usize,
    n_steps: usize,
    out: *mut *mut SmfMatrix3D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if !a_diag.is_finite() || a_diag <= 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    if u0_len != 2 * nx * ny * nz {
        return SemiflowStatus::GridMismatch;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_matrix3d(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, a_diag, c_coupling, n_steps, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfMatrix3D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `SmfMatrix3D` state by time `t`, writing `2*nx*ny*nz` values into `dst_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_matrix3d_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix3d_evolve(
    ev: *mut SmfMatrix3D,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerMatrix3D>() };
        if dst_len != inner.current.len() {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_matrix3d(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, inner.current.len()) };
        out.copy_from_slice(&inner.current);
        SemiflowStatus::Ok
    })
}

/// Copy current `SmfMatrix3D` values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_matrix3d_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix3d_values(
    ev: *const SmfMatrix3D,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerMatrix3D>() };
        if out_len < inner.current.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, inner.current.len()) };
        out.copy_from_slice(&inner.current);
        SemiflowStatus::Ok
    })
}

/// Return `SmfMatrix3D` buffer size (`2*nx*ny*nz`); 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_matrix3d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix3d_size(ev: *const SmfMatrix3D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerMatrix3D>() };
    inner.current.len()
}

/// Free a `SmfMatrix3D` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_matrix3d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix3d_free(ev: *mut SmfMatrix3D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerMatrix3D>())) };
    }));
}

// ---------------------------------------------------------------------------
// 3D helpers
// ---------------------------------------------------------------------------

fn build_matrix3d(
    xmin: f64, xmax: f64, nx: usize,
    ymin: f64, ymax: f64, ny: usize,
    zmin: f64, zmax: f64, nz: usize,
    a_diag: f64, c_coupling: f64, n_steps: usize,
    u0: &[f64],
) -> Result<InnerMatrix3D, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(BoundaryPolicy::Reflect);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(BoundaryPolicy::Reflect);
    let gz = Grid1D::new(zmin, zmax, nz)?.with_boundary(BoundaryPolicy::Reflect);
    let grid3d = Grid3D::new(gx, gy, gz)?;
    build_matrix_kernel(a_diag, c_coupling, gx)?;
    Ok(InnerMatrix3D { a_diag, c_coupling, grid3d, n_steps, current: u0.to_vec() })
}

fn evolve_matrix3d(inner: &mut InnerMatrix3D, t: f64) -> Result<(), semiflow::SemiflowError> {
    let g = inner.grid3d;
    let kx = build_matrix_kernel(inner.a_diag, inner.c_coupling, g.x)?;
    let ky = build_matrix_kernel(inner.a_diag, inner.c_coupling, g.y)?;
    let kz = build_matrix_kernel(inner.a_diag, inner.c_coupling, g.z)?;
    let kernel = MatrixDiffusionChernoff3D::new(kx, ky, kz);
    let sg = ChernoffSemigroup::new(kernel, inner.n_steps)?;
    let mut src = MatrixGridFn3D::<f64, 2>::new(g);
    src.values.copy_from_slice(&inner.current);
    let out = sg.evolve(t, &src)?;
    inner.current.copy_from_slice(&out.values);
    Ok(())
}
