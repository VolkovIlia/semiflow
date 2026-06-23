//! FFI surface for `ObstacleNDV8` — D=2 forward obstacle evolver (C-parity pass,
//! ADR-0028/0171, ADR-0153 TIER-2).
//!
//! Mirrors `semiflow-py` `PyObstacleNDV8` (`obstacle_gamma_py.rs`).
//!
//! ## Engine
//!
//! `ObstacleChernoffND<AnisotropicShiftChernoffND<f64,2>, ConstantObstacle<f64>, f64, 2>`
//! — unit-isotropic diffusion inner (a=I, b=0, c=0), constant obstacle floor.
//!
//! ## Buffer layout (axis-0-fastest / Fortran order)
//!
//! State: flat `f64[nx*ny]`, `idx(i,j) = i + j*nx` (axis-0 = x fastest).
//! This matches the §3.1 Fortran-order contract used by `PyObstacleNDV8`.
//!
//! ## Entry points
//!
//! - `smf_obstacle_nd2_new(xmin,xmax,nx,ymin,ymax,ny,level,out)` → `SemiflowStatus`
//! - `smf_obstacle_nd2_free(ptr)` — null-safe
//! - `smf_obstacle_nd2_shape(ptr,nx_out,ny_out)` → `SemiflowStatus`
//! - `smf_obstacle_nd2_apply(ptr,tau,v,v_len,out,out_len)` → `SemiflowStatus`
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::cast_precision_loss)]

use std::os::raw::c_double;

use semiflow_core::{
    grid_nd::{GridFnND, GridND},
    shift_nd::AnisotropicShiftChernoffND,
    ChernoffFunction, ConstantObstacle, Grid1D, ObstacleChernoffND, ScratchPool, SquareMatrix,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Type alias for the D=2 kernel
// ---------------------------------------------------------------------------

type Nd2Kernel =
    ObstacleChernoffND<AnisotropicShiftChernoffND<f64, 2>, ConstantObstacle<f64>, f64, 2>;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle for a D=2 obstacle ND evolver.
///
/// Obtain from `smf_obstacle_nd2_new`; free with `smf_obstacle_nd2_free`.
#[repr(C)]
pub struct SmfObstacleND2 {
    _private: [u8; 0],
}

struct Nd2Inner {
    level: f64,
    grid_nd: GridND<f64, 2>,
    nx: usize,
    ny: usize,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a D=2 obstacle evolver.
///
/// `level` must be finite. Both axes must have `n >= 5` recommended
/// (`nx * ny >= 25` required by `AnisotropicShiftChernoffND`).
///
/// # Safety
/// `out` must be a valid non-null `*mut *mut SmfObstacleND2`.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle_nd2_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    level: c_double,
    out: *mut *mut SmfObstacleND2,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_nd2_inner(xmin, xmax, nx, ymin, ymax, ny, level) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfObstacleND2>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Free / shape
// ---------------------------------------------------------------------------

/// Free a `SmfObstacleND2` handle. Null-safe.
///
/// # Safety
/// `ptr` must be null or a live pointer from `smf_obstacle_nd2_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle_nd2_free(ptr: *mut SmfObstacleND2) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ptr.cast::<Nd2Inner>())) };
    }));
}

/// Return the grid shape `(nx, ny)` via out-params.
///
/// ## Return values
/// - `Ok` (0)      — success; `*nx_out` and `*ny_out` are set.
/// - `NullPtr` (5) — any argument is null.
///
/// # Safety
/// `ptr` live from `smf_obstacle_nd2_new`; `nx_out`, `ny_out` writable.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle_nd2_shape(
    ptr: *const SmfObstacleND2,
    nx_out: *mut usize,
    ny_out: *mut usize,
) -> SemiflowStatus {
    if ptr.is_null() || nx_out.is_null() || ny_out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ptr.cast::<Nd2Inner>() };
        unsafe {
            *nx_out = inner.nx;
            *ny_out = inner.ny;
        }
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

/// Apply one Chernoff step `Π_g ∘ S(Δτ)` to a flat axis-0-fastest buffer.
///
/// `v` and `out` have length `nx*ny` (axis-0-fastest: `idx(i,j) = i + j*nx`).
/// `tau > 0` and finite.
///
/// ## Return values
/// - `Ok` (0)           — success.
/// - `NullPtr` (5)      — `ptr`, `v`, or `out` is null.
/// - `GridMismatch` (1) — `v_len != nx*ny` or `out_len != nx*ny`.
/// - `OutOfDomain` (3)  — `tau <= 0` or non-finite.
/// - `Panic` (99)       — internal Rust panic.
///
/// # Safety
/// - `ptr` live from `smf_obstacle_nd2_new`.
/// - `v` readable for `v_len` f64s; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle_nd2_apply(
    ptr: *const SmfObstacleND2,
    tau: c_double,
    v: *const c_double,
    v_len: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ptr.is_null() || v.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ptr.cast::<Nd2Inner>() };
        let expected = inner.nx * inner.ny;
        if v_len != expected || out_len != expected {
            return SemiflowStatus::GridMismatch;
        }
        if !tau.is_finite() || tau <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let v_slice = unsafe { std::slice::from_raw_parts(v, v_len) };
        match run_nd2_step(inner.grid_nd.clone(), v_slice, inner.level, tau) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                out_slice.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Pure-Rust step helper (mirrors run_nd_step_2d in obstacle_gamma_py.rs)
// ---------------------------------------------------------------------------

fn run_nd2_step(
    grid_nd: GridND<f64, 2>,
    v_vals: &[f64],
    level: f64,
    tau: f64,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let src = GridFnND::new(grid_nd.clone(), v_vals.to_vec())?;
    let mut dst = GridFnND::new(grid_nd.clone(), vec![0.0_f64; v_vals.len()])?;
    // Unit-isotropic diffusion inner: a = I, b = 0, c = 0.
    let inner = AnisotropicShiftChernoffND::<f64, 2>::new(
        |_x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            a.set(0, 1, 0.0);
            a.set(1, 0, 0.0);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x: &[f64; 2]| 0.0_f64,
        grid_nd.clone(),
    )?;
    let obs = ConstantObstacle::new(level)?;
    let kernel: Nd2Kernel = ObstacleChernoffND::new(inner, obs)?;
    let mut scratch = ScratchPool::new();
    kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
    Ok(dst.values)
}

// ---------------------------------------------------------------------------
// Private builder
// ---------------------------------------------------------------------------

fn build_nd2_inner(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    level: f64,
) -> Result<Nd2Inner, semiflow_core::SemiflowError> {
    validate_nd2_domain(xmin, xmax, ymin, ymax, level)?;
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let grid_nd = GridND::<f64, 2>::new([gx, gy])?;
    Ok(Nd2Inner { level, grid_nd, nx, ny })
}

fn validate_nd2_domain(
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    level: f64,
) -> Result<(), semiflow_core::SemiflowError> {
    if !xmin.is_finite() || !xmax.is_finite() || !ymin.is_finite() || !ymax.is_finite() {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "obstacle_nd2: domain bounds must be finite",
            value: f64::NAN,
        });
    }
    if xmin >= xmax {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "xmin must be < xmax",
            value: xmin,
        });
    }
    if ymin >= ymax {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "ymin must be < ymax",
            value: ymin,
        });
    }
    if !level.is_finite() {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "obstacle_nd2: level must be finite",
            value: level,
        });
    }
    Ok(())
}
