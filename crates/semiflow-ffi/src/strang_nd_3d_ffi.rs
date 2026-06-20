//! FFI surface for 3D tensor-product diffusion: `Heat3D` and `Heat3DVarA`.
//!
//! ## Engines
//!
//! - `Heat3D`     — unit-coefficient 3D heat via `Strang3D<DiffusionChernoff>`.
//! - `Heat3DVarA` — per-axis variable-coefficient 3D heat (same splitting).
//!
//! ## Buffer layout (NORMATIVE — x-fastest / axis-0-fastest)
//!
//! Flat `f64` buffers of length `nx * ny * nz` use **x-fastest** storage:
//!   `idx(i, j, k) = k * nx * ny + j * nx + i`
//!   (`i` = x-index, `j` = y-index, `k` = z-index).
//! This matches `GridFn3D` internal layout (I-T1-3D, `grid3d.rs` §7–8).
//! C callers MUST lay out their arrays in the same order.
//!
//! ## Variable-a layout
//!
//! Constructor `smf_heat3d_vara_new` receives three per-axis coefficient arrays:
//!   `a_x[0..nx]` — `a(x_i)` on the x-axis grid,
//!   `a_y[0..ny]` — `a(y_j)` on the y-axis grid,
//!   `a_z[0..nz]` — `a(z_k)` on the z-axis grid.
//! All arrays MUST contain finite, strictly-positive values.
//!
//! ## Entry points — `Heat3D`
//!
//! - `smf_heat3d_new(xmin,xmax,nx,ymin,ymax,ny,zmin,zmax,nz,out)`
//! - `smf_heat3d_evolve(ev,u0,u0_len,tau,n_steps,out,out_len)`
//! - `smf_heat3d_size(ev)` → `nx * ny * nz`
//! - `smf_heat3d_free(ev)` — null-safe
//!
//! ## Entry points — `Heat3DVarA`
//!
//! - `smf_heat3d_vara_new(xmin,xmax,nx,ymin,ymax,ny,zmin,zmax,nz,a_x,a_y,a_z,out)`
//! - `smf_heat3d_vara_evolve(ev,u0,u0_len,tau,n_steps,out,out_len)`
//! - `smf_heat3d_vara_size(ev)` → `nx * ny * nz`
//! - `smf_heat3d_vara_free(ev)` — null-safe
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
)]

use std::os::raw::c_double;
use std::sync::Arc;

use semiflow_core::{
    BoundaryPolicy, ChernoffFunction, DiffusionChernoff, Grid1D, Grid3D, GridFn3D, ScratchPool,
    Strang3D,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque handle to a unit-coefficient `Strang3D` 3D heat evolver.
///
/// Obtain from `smf_heat3d_new`; pass to `_evolve`/`_size`/`_free`.
/// Do not dereference or allocate from C.
#[repr(C)]
pub struct SmfHeat3D {
    _private: [u8; 0],
}

/// Opaque handle to a variable-coefficient `Strang3D` 3D heat evolver.
///
/// Obtain from `smf_heat3d_vara_new`; pass to `_evolve`/`_size`/`_free`.
/// Do not dereference or allocate from C.
#[repr(C)]
pub struct SmfHeat3DVarA {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state types
// ---------------------------------------------------------------------------

type Strang3DUnit =
    Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>;

struct Inner3D {
    strang: Strang3DUnit,
    grid: Grid3D<f64>,
    size: usize, // nx * ny * nz
}

struct Inner3DVarA {
    strang: Strang3DUnit,
    grid: Grid3D<f64>,
    size: usize, // nx * ny * nz
}

// ---------------------------------------------------------------------------
// Heat3D — unit diffusion
// ---------------------------------------------------------------------------

/// Construct a unit-coefficient 3D heat evolver.
///
/// Solves `∂_t u = ∂_xx u + ∂_yy u + ∂_zz u` via palindromic `Strang3D`.
/// Buffer layout: **x-fastest**, `idx(i,j,k) = k*nx*ny + j*nx + i`.
///
/// ## Preconditions
/// All bounds finite; `xmin < xmax`, `ymin < ymax`, `zmin < zmax`.
/// `nx,ny,nz >= 4`. `out_ev` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `out_ev` must be a valid writable `*mut *mut SmfHeat3D`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat3d_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    zmin: c_double,
    zmax: c_double,
    nz: usize,
    out_ev: *mut *mut SmfHeat3D,
) -> SemiflowStatus {
    if out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_inner_3d_unit(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat3D>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `u0` (x-fastest, length `nx*ny*nz`) by `n_steps` of size `tau`.
///
/// Writes `nx*ny*nz` values into `out`. Both buffers use x-fastest layout.
///
/// ## Preconditions
/// `ev` non-null from `smf_heat3d_new`; `tau > 0` finite; `n_steps >= 1`;
/// `u0` and `out` non-null; `u0_len == out_len == nx*ny*nz`.
///
/// # Safety
/// `u0` readable for `u0_len` f64s; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat3d_evolve(
    ev: *const SmfHeat3D,
    u0: *const c_double,
    u0_len: usize,
    tau: c_double,
    n_steps: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<Inner3D>() };
        if u0_len != inner.size || out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        if validate_tau_nsteps(tau, n_steps).is_err() {
            return SemiflowStatus::OutOfDomain;
        }
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match evolve_3d(&inner.strang, inner.grid, u0_slice, tau, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                out_slice.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return `nx * ny * nz`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat3d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat3d_size(ev: *const SmfHeat3D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<Inner3D>() };
    inner.size
}

/// Free a `SmfHeat3D` handle. Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat3d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat3d_free(ev: *mut SmfHeat3D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<Inner3D>())) };
    }));
}

// ---------------------------------------------------------------------------
// Heat3DVarA — per-axis variable diffusion
// ---------------------------------------------------------------------------

/// Construct a variable-coefficient 3D heat evolver.
///
/// Solves `∂_t u = a_x(x)·∂_xx u + a_y(y)·∂_yy u + a_z(z)·∂_zz u`.
/// Buffer layout: **x-fastest**, `idx(i,j,k) = k*nx*ny + j*nx + i`.
///
/// ## Preconditions
/// All bounds finite and ordered. `nx,ny,nz >= 4`.
/// `a_x[0..nx]`, `a_y[0..ny]`, `a_z[0..nz]`: all > 0, finite.
/// `out_ev` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `a_x` readable for `nx` f64s; `a_y` for `ny`; `a_z` for `nz`.
/// `out_ev` writable as `*mut *mut SmfHeat3DVarA`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat3d_vara_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    zmin: c_double,
    zmax: c_double,
    nz: usize,
    a_x: *const c_double,
    a_y: *const c_double,
    a_z: *const c_double,
    out_ev: *mut *mut SmfHeat3DVarA,
) -> SemiflowStatus {
    if a_x.is_null() || a_y.is_null() || a_z.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let ax_slice = unsafe { std::slice::from_raw_parts(a_x, nx) };
        let ay_slice = unsafe { std::slice::from_raw_parts(a_y, ny) };
        let az_slice = unsafe { std::slice::from_raw_parts(a_z, nz) };
        if let Err(st) = validate_coeff_slice(ax_slice) {
            return st;
        }
        if let Err(st) = validate_coeff_slice(ay_slice) {
            return st;
        }
        if let Err(st) = validate_coeff_slice(az_slice) {
            return st;
        }
        match build_inner_3d_vara(
            xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, ax_slice, ay_slice, az_slice,
        ) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat3DVarA>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `u0` (x-fastest, length `nx*ny*nz`) by `n_steps` of size `tau`.
///
/// # Safety
/// `ev` non-null from `smf_heat3d_vara_new`.
/// `u0` readable for `u0_len` f64s; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat3d_vara_evolve(
    ev: *const SmfHeat3DVarA,
    u0: *const c_double,
    u0_len: usize,
    tau: c_double,
    n_steps: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<Inner3DVarA>() };
        if u0_len != inner.size || out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        if validate_tau_nsteps(tau, n_steps).is_err() {
            return SemiflowStatus::OutOfDomain;
        }
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match evolve_3d(&inner.strang, inner.grid, u0_slice, tau, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                out_slice.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return `nx * ny * nz`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat3d_vara_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat3d_vara_size(ev: *const SmfHeat3DVarA) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<Inner3DVarA>() };
    inner.size
}

/// Free a `SmfHeat3DVarA` handle. Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat3d_vara_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat3d_vara_free(ev: *mut SmfHeat3DVarA) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<Inner3DVarA>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builders
// ---------------------------------------------------------------------------

/// Unit fn-ptr for constant diffusion coefficient a ≡ 1.
extern "Rust" fn unit_a_3d(_: f64) -> f64 {
    1.0
}

/// Zero fn-ptr for derivatives of constant a.
extern "Rust" fn zero_3d(_: f64) -> f64 {
    0.0
}

fn build_inner_3d_unit(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
) -> Result<Inner3D, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(BoundaryPolicy::Reflect);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(BoundaryPolicy::Reflect);
    let gz = Grid1D::new(zmin, zmax, nz)?.with_boundary(BoundaryPolicy::Reflect);
    let grid = Grid3D::new(gx, gy, gz)?;
    let dx = DiffusionChernoff::new(unit_a_3d, zero_3d, zero_3d, 1.0, gx);
    let dy = DiffusionChernoff::new(unit_a_3d, zero_3d, zero_3d, 1.0, gy);
    let dz = DiffusionChernoff::new(unit_a_3d, zero_3d, zero_3d, 1.0, gz);
    let strang = Strang3D::new(dx, dy, dz);
    Ok(Inner3D { strang, grid, size: nx * ny * nz })
}

fn build_inner_3d_vara(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
    ax_vals: &[f64],
    ay_vals: &[f64],
    az_vals: &[f64],
) -> Result<Inner3DVarA, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(BoundaryPolicy::Reflect);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(BoundaryPolicy::Reflect);
    let gz = Grid1D::new(zmin, zmax, nz)?.with_boundary(BoundaryPolicy::Reflect);
    let grid = Grid3D::new(gx, gy, gz)?;
    let dx = build_axis_diffusion(ax_vals, xmin, xmax, nx, gx);
    let dy = build_axis_diffusion(ay_vals, ymin, ymax, ny, gy);
    let dz = build_axis_diffusion(az_vals, zmin, zmax, nz, gz);
    let strang = Strang3D::new(dx, dy, dz);
    Ok(Inner3DVarA { strang, grid, size: nx * ny * nz })
}

/// Build a `DiffusionChernoff` from a tabulated 1D coefficient array.
///
/// Linearly interpolates `a_vals[0..n]` at runtime using `Arc<Vec<f64>>`.
fn build_axis_diffusion(
    a_vals: &[f64],
    amin: f64,
    amax: f64,
    n: usize,
    grid: Grid1D<f64>,
) -> DiffusionChernoff<f64> {
    let norm = a_vals.iter().copied().fold(0.0_f64, f64::max);
    let arc = Arc::new(a_vals.to_vec());
    DiffusionChernoff::with_closure(
        move |t: f64| interp_1d_ffi(&arc, amin, amax, n, t),
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        norm,
        grid,
    )
}

// ---------------------------------------------------------------------------
// Shared compute helper
// ---------------------------------------------------------------------------

fn evolve_3d(
    strang: &Strang3DUnit,
    grid: Grid3D<f64>,
    u0: &[f64],
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let mut src = GridFn3D::new(grid, u0.to_vec())?;
    let mut dst = GridFn3D::new(grid, vec![0.0; u0.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        strang.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Inline helpers (replicate py-crate logic; no cross-crate dep per ADR-0028)
// ---------------------------------------------------------------------------

/// Linear interpolation of a pre-sampled 1D coefficient array at position `x`.
///
/// Clamps `x` to `[amin, amax]`; handles edge nodes safely.
fn interp_1d_ffi(vals: &[f64], amin: f64, amax: f64, n: usize, x: f64) -> f64 {
    if n == 1 {
        return vals[0];
    }
    let fi = ((x - amin) / (amax - amin)) * (n as f64 - 1.0);
    let fi = fi.clamp(0.0, (n - 1) as f64);
    let i0 = (fi as usize).min(n - 2);
    let t = fi - i0 as f64;
    vals[i0] * (1.0 - t) + vals[i0 + 1] * t
}

/// Validate `tau > 0` and `n_steps >= 1`.
fn validate_tau_nsteps(tau: f64, n_steps: usize) -> Result<(), ()> {
    if n_steps == 0 || !tau.is_finite() || tau <= 0.0 {
        return Err(());
    }
    Ok(())
}

/// Validate that a coefficient slice is finite and strictly positive.
fn validate_coeff_slice(vals: &[f64]) -> Result<(), SemiflowStatus> {
    for &v in vals {
        if !v.is_finite() {
            return Err(SemiflowStatus::NanInf);
        }
        if v <= 0.0 {
            return Err(SemiflowStatus::OutOfDomain);
        }
    }
    Ok(())
}
