//! FFI surface for BC kernels — Part 3: `DirichletHeat2nd1D` (odd-image Dirichlet, §21.9).
//!
//! | C handle                    | Core type                                                      | Python class          |
//! |-----------------------------|----------------------------------------------------------------|-----------------------|
//! | `SmfDirichletHeat2nd1D`     | `DirichletHeat2ndChernoff<DiffUnit, HalfSpaceRegion>`          | `DirichletHeat2nd1D`  |
//!
//! Part 1 (`bc_ffi.rs`): `Killing1D`, `Reflected1D`.
//! Part 2 (`bc_ffi2.rs`): `Robin1D`, `Resolvent1D`, `KilledDir1D`.
//!
//! ## Safety contract (all entry points)
//!
//! - Null-check BEFORE `catch_panic!`.
//! - `(ptr, len)` pairs are caller-guaranteed valid for that length.
//! - `_free` is always null-safe.
//! - Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use std::os::raw::c_double;

use semiflow_core::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    killing_order2::DirichletHeat2ndChernoff,
    reflection::HalfSpaceRegion,
    ChernoffSemigroup,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ---------------------------------------------------------------------------
// Unit-diffusion fn-pointers (local copies — each ffi module is standalone)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_bc3(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_bc3(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type DirichletHeat2ndKernel = DirichletHeat2ndChernoff<DiffUnit, HalfSpaceRegion<f64, 1>, f64>;

// ===========================================================================
// DirichletHeat2ndChernoff — smf_dirichlet_heat2nd1d_*
// ===========================================================================

/// Opaque handle to `DirichletHeat2ndChernoff<DiffusionChernoff, HalfSpaceRegion>`.
#[repr(C)]
pub struct SmfDirichletHeat2nd1D {
    _private: [u8; 0],
}

struct DirichletHeat2ndState {
    semigroup: ChernoffSemigroup<DirichletHeat2ndKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate an order-2 Dirichlet-BC 1D heat evolver (odd-image method, §21.9).
///
/// Absorbing boundary `u = 0` at `origin`. Unit diffusion `a = 1`.
///
/// ## Preconditions
/// - `xmin < xmax`, both finite; `n_grid >= 4`.
/// - `n_chernoff >= 1`; `u0` non-null, `u0_len == n_grid`, all finite.
/// - `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfDirichletHeat2nd1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_dirichlet_heat2nd1d_new(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    n_chernoff: usize,
    origin: c_double,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfDirichletHeat2nd1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_dirichlet_heat2nd(xmin, xmax, n_grid, n_chernoff, origin, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfDirichletHeat2nd1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance DirichletHeat2nd evolver by `t`; write values into `dst`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_dirichlet_heat2nd1d_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_dirichlet_heat2nd1d_evolve(
    ev: *mut SmfDirichletHeat2nd1D,
    t: c_double,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<DirichletHeat2ndState>() };
        let n = s.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        match s.semigroup.evolve(t, &s.current) {
            Err(e) => return SemiflowStatus::from(&e),
            Ok(next) => s.current = next,
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, n) };
        out.copy_from_slice(&s.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current values into `out`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_dirichlet_heat2nd1d_new`.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_dirichlet_heat2nd1d_values(
    ev: *const SmfDirichletHeat2nd1D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<DirichletHeat2ndState>() };
        let vals = &s.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, vals.len()) };
        buf.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_dirichlet_heat2nd1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_dirichlet_heat2nd1d_size(
    ev: *const SmfDirichletHeat2nd1D,
) -> usize {
    if ev.is_null() {
        return 0;
    }
    let s = unsafe { &*ev.cast::<DirichletHeat2ndState>() };
    s.current.values.len()
}

/// Free a DirichletHeat2nd1D handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_dirichlet_heat2nd1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_dirichlet_heat2nd1d_free(ev: *mut SmfDirichletHeat2nd1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<DirichletHeat2ndState>())) };
    }));
}

fn build_dirichlet_heat2nd(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_chernoff: usize,
    origin: f64,
    u0: &[f64],
) -> Result<DirichletHeat2ndState, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n_grid)?;
    let diff = DiffusionChernoff::new(unit_a_bc3, zero_bc3, zero_bc3, 1.0, grid);
    let region = HalfSpaceRegion::<f64, 1>::new([origin], [1.0])?;
    let kernel = DirichletHeat2ndChernoff::new(diff, region)?;
    let semigroup = ChernoffSemigroup::new(kernel, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(DirichletHeat2ndState { semigroup, current })
}
