//! FFI surface for 2D tensor-product diffusion: `Heat2D` and `Heat2DVarA`.
//!
//! ## Engines
//!
//! - `Heat2D`     — unit-coefficient 2D heat via `Strang2D<DiffusionChernoff>`.
//! - `Heat2DVarA` — per-axis variable-coefficient 2D heat (same splitting).
//!
//! ## Buffer layout (NORMATIVE — x-fastest / axis-0-fastest)
//!
//! Flat `f64` buffers of length `nx * ny` use **x-fastest** storage:
//!   `idx(i, j) = j * nx + i`   (`i` = x-index, `j` = y-index).
//! This matches `GridFn2D` internal layout (I-T1, `grid2d.rs` §7).
//! C callers MUST lay out their arrays in the same order.
//!
//! ## Variable-a layout
//!
//! Constructor `smf_heat2d_vara_new` receives two per-axis coefficient arrays:
//!   `a_x[0..nx]` — `a(x_i)` values on the x-axis grid,
//!   `a_y[0..ny]` — `a(y_j)` values on the y-axis grid.
//! Both arrays MUST contain finite, strictly-positive values.
//!
//! ## Entry points — `Heat2D`
//!
//! - `smf_heat2d_new(xmin,xmax,nx,ymin,ymax,ny,out)`
//! - `smf_heat2d_evolve(ev,u0,u0_len,tau,n_steps,out,out_len)`
//! - `smf_heat2d_size(ev)` → `nx * ny`
//! - `smf_heat2d_free(ev)` — null-safe
//!
//! ## Entry points — `Heat2DVarA`
//!
//! - `smf_heat2d_vara_new(xmin,xmax,nx,ymin,ymax,ny,a_x,nx,a_y,ny,out)`
//! - `smf_heat2d_vara_evolve(ev,u0,u0_len,tau,n_steps,out,out_len)`
//! - `smf_heat2d_vara_size(ev)` → `nx * ny`
//! - `smf_heat2d_vara_free(ev)` — null-safe
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
    clippy::too_many_arguments
)]

use std::{os::raw::c_double, sync::Arc};

use semiflow::{
    BoundaryPolicy, ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, GridFn2D, ScratchPool,
    Strang2D,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque handle to a unit-coefficient `Strang2D` 2D heat evolver.
///
/// Obtain from `smf_heat2d_new`; pass to `_evolve`/`_size`/`_free`.
/// Do not dereference or allocate from C.
#[repr(C)]
pub struct SmfHeat2D {
    _private: [u8; 0],
}

/// Opaque handle to a variable-coefficient `Strang2D` 2D heat evolver.
///
/// Obtain from `smf_heat2d_vara_new`; pass to `_evolve`/`_size`/`_free`.
/// Do not dereference or allocate from C.
#[repr(C)]
pub struct SmfHeat2DVarA {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state types
// ---------------------------------------------------------------------------

type Strang2DUnit = Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>;

struct Inner2D {
    strang: Strang2DUnit,
    grid: Grid2D<f64>,
    size: usize, // nx * ny
}

struct Inner2DVarA {
    strang: Strang2DUnit,
    grid: Grid2D<f64>,
    size: usize, // nx * ny
}

// ---------------------------------------------------------------------------
// Heat2D — unit diffusion
// ---------------------------------------------------------------------------

/// Construct a unit-coefficient 2D heat evolver on `[xmin,xmax]×[ymin,ymax]`.
///
/// Solves `∂_t u = ∂_xx u + ∂_yy u` via palindromic `Strang2D` (order 2).
/// Buffer layout: **x-fastest**, `idx(i,j) = j*nx + i`.
///
/// ## Preconditions
/// `xmin < xmax`, `ymin < ymax` (both finite); `nx >= 4`, `ny >= 4`.
/// `out_ev` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `out_ev` must be a valid writable `*mut *mut SmfHeat2D`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat2d_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    out_ev: *mut *mut SmfHeat2D,
) -> SemiflowStatus {
    if out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_inner_2d_unit(xmin, xmax, nx, ymin, ymax, ny) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat2D>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `u0` (x-fastest, length `nx*ny`) by `n_steps` of size `tau`.
///
/// Writes `nx*ny` values into `out`. Both buffers use x-fastest layout.
///
/// ## Preconditions
/// `ev` non-null from `smf_heat2d_new`; `tau > 0` finite; `n_steps >= 1`;
/// `u0` and `out` non-null; `u0_len == out_len == nx*ny`.
///
/// # Safety
/// `u0` readable for `u0_len` f64s; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat2d_evolve(
    ev: *const SmfHeat2D,
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
        let inner = unsafe { &*ev.cast::<Inner2D>() };
        if u0_len != inner.size || out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        if validate_tau_nsteps(tau, n_steps).is_err() {
            return SemiflowStatus::OutOfDomain;
        }
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match evolve_2d(&inner.strang, inner.grid, u0_slice, tau, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                out_slice.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return `nx * ny`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat2d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat2d_size(ev: *const SmfHeat2D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<Inner2D>() };
    inner.size
}

/// Free a `SmfHeat2D` handle. Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat2d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat2d_free(ev: *mut SmfHeat2D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<Inner2D>())) };
    }));
}

// ---------------------------------------------------------------------------
// Heat2DVarA — per-axis variable diffusion
// ---------------------------------------------------------------------------

/// Construct a variable-coefficient 2D heat evolver.
///
/// Solves `∂_t u = a_x(x)·∂_xx u + a_y(y)·∂_yy u` via palindromic `Strang2D`.
/// Buffer layout: **x-fastest**, `idx(i,j) = j*nx + i`.
///
/// ## Preconditions
/// `xmin < xmax`, `ymin < ymax` (finite); `nx,ny >= 4`.
/// `a_x` has length `nx`, all values > 0 and finite.
/// `a_y` has length `ny`, all values > 0 and finite.
/// `out_ev` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `a_x` readable for `nx` f64s; `a_y` readable for `ny` f64s.
/// `out_ev` writable as `*mut *mut SmfHeat2DVarA`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat2d_vara_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    a_x: *const c_double,
    a_y: *const c_double,
    out_ev: *mut *mut SmfHeat2DVarA,
) -> SemiflowStatus {
    if a_x.is_null() || a_y.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let ax_slice = unsafe { std::slice::from_raw_parts(a_x, nx) };
        let ay_slice = unsafe { std::slice::from_raw_parts(a_y, ny) };
        if let Err(st) = validate_coeff_slice(ax_slice) {
            return st;
        }
        if let Err(st) = validate_coeff_slice(ay_slice) {
            return st;
        }
        match build_inner_2d_vara(xmin, xmax, nx, ymin, ymax, ny, ax_slice, ay_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat2DVarA>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `u0` (x-fastest, length `nx*ny`) by `n_steps` of size `tau`.
///
/// Writes `nx*ny` values into `out`. Both buffers use x-fastest layout.
///
/// # Safety
/// `ev` non-null from `smf_heat2d_vara_new`.
/// `u0` readable for `u0_len` f64s; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat2d_vara_evolve(
    ev: *const SmfHeat2DVarA,
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
        let inner = unsafe { &*ev.cast::<Inner2DVarA>() };
        if u0_len != inner.size || out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        if validate_tau_nsteps(tau, n_steps).is_err() {
            return SemiflowStatus::OutOfDomain;
        }
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match evolve_2d(&inner.strang, inner.grid, u0_slice, tau, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                out_slice.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return `nx * ny`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat2d_vara_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat2d_vara_size(ev: *const SmfHeat2DVarA) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<Inner2DVarA>() };
    inner.size
}

/// Free a `SmfHeat2DVarA` handle. Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat2d_vara_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat2d_vara_free(ev: *mut SmfHeat2DVarA) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<Inner2DVarA>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builders
// ---------------------------------------------------------------------------

/// Unit fn-ptr for constant diffusion coefficient a ≡ 1.
extern "Rust" fn unit_a_2d(_: f64) -> f64 {
    1.0
}

/// Zero fn-ptr for derivatives of constant a.
extern "Rust" fn zero_2d(_: f64) -> f64 {
    0.0
}

fn build_inner_2d_unit(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
) -> Result<Inner2D, semiflow::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(BoundaryPolicy::Reflect);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(BoundaryPolicy::Reflect);
    let grid = Grid2D::new(gx, gy);
    let dx = DiffusionChernoff::new(unit_a_2d, zero_2d, zero_2d, 1.0, gx);
    let dy = DiffusionChernoff::new(unit_a_2d, zero_2d, zero_2d, 1.0, gy);
    let strang = Strang2D::new(dx, dy);
    Ok(Inner2D {
        strang,
        grid,
        size: nx * ny,
    })
}

fn build_inner_2d_vara(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    ax_vals: &[f64],
    ay_vals: &[f64],
) -> Result<Inner2DVarA, semiflow::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(BoundaryPolicy::Reflect);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(BoundaryPolicy::Reflect);
    let grid = Grid2D::new(gx, gy);
    let dx = build_axis_diffusion(ax_vals, xmin, xmax, nx, gx);
    let dy = build_axis_diffusion(ay_vals, ymin, ymax, ny, gy);
    let strang = Strang2D::new(dx, dy);
    Ok(Inner2DVarA {
        strang,
        grid,
        size: nx * ny,
    })
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

fn evolve_2d(
    strang: &Strang2DUnit,
    grid: Grid2D<f64>,
    u0: &[f64],
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let mut src = GridFn2D::new(grid, u0.to_vec())?;
    let mut dst = GridFn2D::new(grid, vec![0.0; u0.len()])?;
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

/// Validate `tau > 0` and `n_steps >= 1` for Strang-based kernels.
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
