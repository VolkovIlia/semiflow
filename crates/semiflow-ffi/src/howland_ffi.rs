//! FFI surface for the nonautonomous Howland-lift engine (Round 7).
//!
//! | C handle        | Core type                               | Python class |
//! |-----------------|-----------------------------------------|--------------|
//! | `SmfHowland1D`  | `HowlandLift<DiffusionChernoff<f64>>`   | `Howland1D`  |
//!
//! ## Constructor parameters (mirror Python `Howland1D`)
//!
//! `smf_howland1d_new(xmin, xmax, n, n_t, t_horizon, u0, u0_len, out)`
//!
//! - `n_t >= 2` temporal grid points in `[0, t_horizon]`.
//! - `t_horizon > 0` and finite.
//! - `u0` replicated across all `n_t` time slices (Howland initial state).
//!
//! ## Matched-step constraint
//!
//! `tau = delta_s = t_horizon / (n_t − 1)` is enforced internally.
//! `smf_howland1d_evolve` takes no `t` argument — it always advances by
//! one full `t_horizon` (mirrors `PyHowland1D::evolve()` being parameter-free).
//!
//! ## Read-out
//!
//! `smf_howland1d_values` copies the **last** time slice `u(t_horizon, ·)`.
//! `smf_howland1d_size` returns the spatial grid size `n`.
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments
)]

use std::os::raw::c_double;

use semiflow::{
    diffusion::DiffusionChernoff,
    howland::{HowlandLift, HowlandState},
    ChernoffSemigroup, Grid1D, GridFn1D,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ---------------------------------------------------------------------------
// Fn-pointers for unit-diffusion (a = 1, a' = 0, a'' = 0)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_hw(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_hw(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type HowlandKernel = HowlandLift<DiffUnit, f64>;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `HowlandLift<DiffusionChernoff<f64>>` evolver.
///
/// Obtain from `smf_howland1d_new`; free with `smf_howland1d_free`.
#[repr(C)]
pub struct SmfHowland1D {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct HowlandState1D {
    lift: HowlandKernel,
    state: HowlandState<GridFn1D<f64>, f64>,
    n: usize,
    n_t: usize,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Allocate a Howland-lift 1D nonautonomous heat evolver.
///
/// Unit diffusion `a = 1`. The lifted state holds `n_t` time slices of
/// `GridFn1D`, all initialised from `u0`.
///
/// ## Preconditions
/// - `xmin < xmax`, both finite; `n >= 4`.
/// - `n_t >= 2`; `t_horizon > 0` and finite.
/// - `u0` non-null, `u0_len == n`, all finite.
/// - `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfHowland1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_howland1d_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_t: usize,
    t_horizon: c_double,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfHowland1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_howland(xmin, xmax, n, n_t, t_horizon, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfHowland1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance the Howland state by one full `t_horizon`.
///
/// Uses `n_steps = n_t − 1` Chernoff iterations with `tau = delta_s`
/// (matched-step constraint, math §23.4).
///
/// ## Return values
/// `Ok` | `NullPtr` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_howland1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_howland1d_evolve(ev: *mut SmfHowland1D) -> SemiflowStatus {
    if ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<HowlandState1D>() };
        match evolve_howland(&s.lift, s.state.clone()) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(next) => {
                s.state = next;
                SemiflowStatus::Ok
            }
        }
    })
}

/// Copy the last time slice `u(t_horizon, ·)` into `out`.
///
/// `out_len` must equal the spatial grid size `n`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `Panic`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_howland1d_new`.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_howland1d_values(
    ev: *const SmfHowland1D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<HowlandState1D>() };
        let last = &s.state.samples[s.n_t - 1];
        let vals = &last.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, vals.len()) };
        buf.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return spatial grid size `n`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_howland1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_howland1d_size(ev: *const SmfHowland1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let s = unsafe { &*ev.cast::<HowlandState1D>() };
    s.n
}

/// Free a Howland handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_howland1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_howland1d_free(ev: *mut SmfHowland1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<HowlandState1D>())) };
    }));
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_howland(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_t: usize,
    t_horizon: f64,
    u0: &[f64],
) -> Result<HowlandState1D, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?;
    let diff = DiffusionChernoff::new(unit_a_hw, zero_hw, zero_hw, 1.0, grid);
    let lift = HowlandLift::new(diff, t_horizon, n_t)?;
    let slice = GridFn1D::new(grid, u0.to_vec())?;
    let samples: Vec<GridFn1D<f64>> = (0..n_t).map(|_| slice.clone()).collect();
    let state = HowlandState::new(samples)?;
    Ok(HowlandState1D {
        lift,
        state,
        n,
        n_t,
    })
}

// ---------------------------------------------------------------------------
// Compute helper
// ---------------------------------------------------------------------------

fn evolve_howland(
    lift: &HowlandKernel,
    state: HowlandState<GridFn1D<f64>, f64>,
) -> Result<HowlandState<GridFn1D<f64>, f64>, semiflow::SemiflowError> {
    let n_steps = lift.n_t() - 1;
    let sg = ChernoffSemigroup::new(lift.clone(), n_steps)?;
    sg.evolve(lift.delta_s() * n_steps as f64, &state)
}
