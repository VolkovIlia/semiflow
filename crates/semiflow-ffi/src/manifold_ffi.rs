//! FFI surface for the 2D Riemannian manifold Chernoff engine (Round 7).
//!
//! | C handle         | Core type                          | Python class |
//! |------------------|------------------------------------|--------------|
//! | `SmfManifold2D`  | `ManifoldChernoff<M, f64>` (enum)  | `Manifold2D` |
//!
//! ## Backend selector (`manifold_tag` — `u32`)
//!
//! Mirrors Python `Manifold2D(manifold=...)` string kwarg.
//!
//! | tag | Python string    | Core backend                | Key param       |
//! |-----|------------------|-----------------------------|-----------------|
//! | 0   | `"torus"`        | `Torus<f64, 2>::unit()`     | (none)          |
//! | 1   | `"sphere2"`      | `Sphere2::with_radius(r)`   | `radius`        |
//! | 2   | `"hyperbolic2"`  | `Hyperbolic2::with_scale(r)`| `radius`        |
//!
//! ## Constructor
//!
//! `smf_manifold2d_new(x0min, x0max, nx, x1min, x1max, ny,
//!                     manifold_tag, radius, curvature_correction,
//!                     u0, u0_len, out)`
//!
//! ## Buffer layout
//!
//! Flat `f64` buffer of length `nx * ny` in row-major order
//! (x0 fastest, matching `GridFn2D` internal layout).
//!
//! ## Evolve API
//!
//! `smf_manifold2d_evolve(ev, t, n_steps, dst, dst_len)`
//! Advances state by time `t` using `n_steps` iterations (`tau = t / n_steps`).
//! Mirrors `Manifold2D.evolve(t, n_steps)`.
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(
    clippy::assigning_clones,
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
)]

use std::os::raw::{c_double, c_uint};

use semiflow_core::{
    manifold::{Hyperbolic2, Sphere2, Torus},
    manifold_chernoff::ManifoldChernoff,
    ChernoffFunction, Grid1D, Grid2D, GridFn2D, ScratchPool,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ---------------------------------------------------------------------------
// Binding-side ManifoldEnum (mirrors PyO3 ManifoldEnum)
// ---------------------------------------------------------------------------

/// Binding-side enum wrapping the three `ManifoldChernoff` concrete types.
///
/// `BoundedGeometryManifold` is not object-safe (const-generic `D` on `Torus`).
/// This enum implements `ChernoffFunction<f64>` by matching on the variant.
#[derive(Clone)]
enum ManifoldEnum {
    Torus(ManifoldChernoff<Torus<f64, 2>, f64>),
    Sphere(ManifoldChernoff<Sphere2<f64>, f64>),
    Hyperbolic(ManifoldChernoff<Hyperbolic2<f64>, f64>),
}

// Safety: all inner types are Send+Sync (verified in semiflow-py send_assertions.rs).
unsafe impl Send for ManifoldEnum {}
unsafe impl Sync for ManifoldEnum {}

impl ChernoffFunction<f64> for ManifoldEnum {
    type S = GridFn2D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn2D<f64>,
        dst: &mut GridFn2D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow_core::SemiflowError> {
        match self {
            ManifoldEnum::Torus(k) => k.apply_into(tau, src, dst, scratch),
            ManifoldEnum::Sphere(k) => k.apply_into(tau, src, dst, scratch),
            ManifoldEnum::Hyperbolic(k) => k.apply_into(tau, src, dst, scratch),
        }
    }

    fn order(&self) -> u32 {
        match self {
            ManifoldEnum::Torus(k) => k.order(),
            ManifoldEnum::Sphere(k) => k.order(),
            ManifoldEnum::Hyperbolic(k) => k.order(),
        }
    }

    fn growth(&self) -> semiflow_core::chernoff::Growth<f64> {
        match self {
            ManifoldEnum::Torus(k) => k.growth(),
            ManifoldEnum::Sphere(k) => k.growth(),
            ManifoldEnum::Hyperbolic(k) => k.growth(),
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `ManifoldChernoff` 2D evolver.
///
/// Obtain from `smf_manifold2d_new`; free with `smf_manifold2d_free`.
#[repr(C)]
pub struct SmfManifold2D {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct ManifoldState2D {
    kernel: ManifoldEnum,
    grid: Grid2D<f64>,
    current: Vec<f64>,
    size: usize, // nx * ny
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Allocate a 2D Riemannian manifold Chernoff evolver.
///
/// `manifold_tag`: 0 = torus, 1 = sphere2, 2 = hyperbolic2.
/// `radius`      — sphere radius or hyperbolic scale; must be > 0.
///                 Ignored for torus.
/// `curvature_correction` — apply R/12 correction (MMRS 2023 Thm 1).
///
/// ## Preconditions
/// - `x0min < x0max`, `x1min < x1max`, both finite; `nx >= 4`, `ny >= 4`.
/// - `u0` non-null, `u0_len == nx*ny`, all finite.
/// - `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Unsupported` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfManifold2D`.
#[no_mangle]
pub unsafe extern "C" fn smf_manifold2d_new(
    x0min: c_double,
    x0max: c_double,
    nx: usize,
    x1min: c_double,
    x1max: c_double,
    ny: usize,
    manifold_tag: c_uint,
    radius: c_double,
    curvature_correction: bool,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfManifold2D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        let kernel = match parse_manifold(manifold_tag, radius, curvature_correction) {
            Ok(k) => k,
            Err(st) => return st,
        };
        match build_manifold2d(x0min, x0max, nx, x1min, x1max, ny, kernel, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfManifold2D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance manifold evolver by time `t` using `n_steps` iterations.
///
/// Writes `nx*ny` values into `dst` (row-major, x0-fastest).
/// `tau = t / n_steps`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_manifold2d_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_manifold2d_evolve(
    ev: *mut SmfManifold2D,
    t: c_double,
    n_steps: usize,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<ManifoldState2D>() };
        if dst_len != s.size {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if n_steps == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        let tau = t / n_steps as f64;
        match evolve_manifold(&s.kernel, s.grid, s.current.clone(), tau, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                s.current = result.clone();
                let buf = unsafe { std::slice::from_raw_parts_mut(dst, dst_len) };
                buf.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Copy current manifold values into `out` (row-major, x0-fastest).
///
/// # Safety
/// `ev` must be a live pointer from `smf_manifold2d_new`.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_manifold2d_values(
    ev: *const SmfManifold2D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<ManifoldState2D>() };
        if out_len != s.size {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        buf.copy_from_slice(&s.current);
        SemiflowStatus::Ok
    })
}

/// Return `nx * ny`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_manifold2d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_manifold2d_size(ev: *const SmfManifold2D) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<ManifoldState2D>() }.size
}

/// Free a manifold handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_manifold2d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_manifold2d_free(ev: *mut SmfManifold2D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<ManifoldState2D>())) };
    }));
}

// ---------------------------------------------------------------------------
// Parser helpers
// ---------------------------------------------------------------------------

fn parse_manifold(
    tag: u32,
    radius: f64,
    curvature_correction: bool,
) -> Result<ManifoldEnum, SemiflowStatus> {
    match tag {
        0 => Ok(ManifoldEnum::Torus(ManifoldChernoff::new(
            Torus::<f64, 2>::unit(),
            curvature_correction,
        ))),
        1 => Sphere2::with_radius(radius)
            .map(|m| ManifoldEnum::Sphere(ManifoldChernoff::new(m, curvature_correction)))
            .map_err(|e| SemiflowStatus::from(&e)),
        2 => Hyperbolic2::with_scale(radius)
            .map(|m| ManifoldEnum::Hyperbolic(ManifoldChernoff::new(m, curvature_correction)))
            .map_err(|e| SemiflowStatus::from(&e)),
        _ => Err(SemiflowStatus::Unsupported),
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_manifold2d(
    x0min: f64,
    x0max: f64,
    nx: usize,
    x1min: f64,
    x1max: f64,
    ny: usize,
    kernel: ManifoldEnum,
    u0: &[f64],
) -> Result<ManifoldState2D, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    let gx = Grid1D::new(x0min, x0max, nx)?;
    let gy = Grid1D::new(x1min, x1max, ny)?;
    let grid = Grid2D::new(gx, gy);
    let size = nx * ny;
    if u0.len() != size {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "u0 length must equal nx * ny",
            value: u0.len() as f64,
        });
    }
    Ok(ManifoldState2D {
        kernel,
        grid,
        current: u0.to_vec(),
        size,
    })
}

// ---------------------------------------------------------------------------
// Compute helper
// ---------------------------------------------------------------------------

fn evolve_manifold(
    kernel: &ManifoldEnum,
    grid: Grid2D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let mut src = GridFn2D::new_generic(grid, input)?;
    let zero = vec![0.0f64; src.values.len()];
    let mut dst = GridFn2D::new_generic(grid, zero)?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}
