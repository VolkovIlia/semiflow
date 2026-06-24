//! FFI surface for `AdaptivePI` — PI-controller adaptive integrator (Round 9).
//!
//! Mirrors `semiflow-py`'s `AdaptivePI` class (`adaptive.rs`).
//!
//! ## Python constructor (authoritative spec)
//!
//! ```python
//! AdaptivePI(xmin, xmax, n, u0, *, kernel="heat2",
//!            tol_abs=1e-6, tol_rel=1e-4, boundary="reflect")
//! ```
//!
//! `kernel` ∈ `{"heat2", "heat4", "heat6", "drift", "shift"}`.
//!
//! ## FFI constructor
//!
//! ```c
//! SemiflowStatus smf_adaptive_pi_new(
//!     double xmin, double xmax, size_t n,
//!     const char *kernel,   // NUL-terminated
//!     double tol_abs,       // absolute tolerance (default 1e-6)
//!     double tol_rel,       // relative tolerance (default 1e-4)
//!     const double *u0, size_t u0_len,
//!     SmfAdaptivePI **out
//! );
//! ```
//!
//! `evolve` integrates by time `t` using adaptive substeps.  The number of
//! accepted/rejected substeps is written to the optional output pointers
//! `steps_accepted` and `steps_rejected` (may be NULL).
//!
//! ## Symbol names
//!
//! - `smf_adaptive_pi_new` / `smf_adaptive_pi_evolve` / `smf_adaptive_pi_values`
//!   / `smf_adaptive_pi_size` / `smf_adaptive_pi_free`
//!
//! ## Design notes
//!
//! `AdaptivePI::evolve_adaptive` is `&mut self` (PI controller is stateful).
//! The inner state (integrator + current grid function) is heap-allocated and
//! reached via the opaque handle, matching all other Round-N FFI modules.
//!
//! ## Panic safety
//!
//! Every `extern "C"` entry is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::cast_precision_loss)]

use std::{ffi::CStr, os::raw::c_double};

use semiflow::{
    AdaptivePI as CorePI, BoundaryPolicy, Diffusion4thChernoff, Diffusion6thChernoff,
    DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, ShiftChernoff1D,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ---------------------------------------------------------------------------
// Constant fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_adp(_: f64) -> f64 {
    1.0
}
extern "Rust" fn half_a_adp(_: f64) -> f64 {
    0.5
}
extern "Rust" fn zero_adp(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// 5-kernel enum (avoids Box<dyn ChernoffFunction>)
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
enum AdaptiveVariant {
    Diff2(CorePI<DiffusionChernoff<f64>>),
    Diff4(CorePI<Diffusion4thChernoff<f64>>),
    Diff6(CorePI<Diffusion6thChernoff<f64>>),
    DriftReaction(CorePI<DriftReactionChernoff<f64>>),
    Shift(CorePI<ShiftChernoff1D<f64>>),
}

impl AdaptiveVariant {
    /// Run adaptive integration from `u0`; return evolved `GridFn1D`.
    fn evolve(
        &mut self,
        t: f64,
        u0: &GridFn1D<f64>,
    ) -> Result<(GridFn1D<f64>, usize, usize), semiflow::SemiflowError> {
        let outcome = match self {
            Self::Diff2(k) => k.evolve_adaptive(t, u0)?,
            Self::Diff4(k) => k.evolve_adaptive(t, u0)?,
            Self::Diff6(k) => k.evolve_adaptive(t, u0)?,
            Self::DriftReaction(k) => k.evolve_adaptive(t, u0)?,
            Self::Shift(k) => k.evolve_adaptive(t, u0)?,
        };
        Ok((
            outcome.final_state,
            outcome.steps_accepted,
            outcome.steps_rejected,
        ))
    }

    fn set_tolerance(&mut self, abs: f64, rel: f64) {
        match self {
            Self::Diff2(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
            Self::Diff4(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
            Self::Diff6(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
            Self::DriftReaction(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
            Self::Shift(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque handle + inner state
// ---------------------------------------------------------------------------

/// Opaque handle to a 1-D adaptive PI integrator.
///
/// Obtained from `smf_adaptive_pi_new`; free with `smf_adaptive_pi_free`.
#[repr(C)]
pub struct SmfAdaptivePI {
    _private: [u8; 0],
}

struct AdaptivePIInner {
    integrator: AdaptiveVariant,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a 1-D adaptive PI integrator.
///
/// `kernel` must be NUL-terminated: `"heat2"`, `"heat4"`, `"heat6"`,
/// `"drift"`, or `"shift"`.
///
/// # Safety
/// `kernel` must be a valid NUL-terminated C string.
/// `u0` readable for `u0_len` f64s; `out` writable `*mut *mut SmfAdaptivePI`.
#[no_mangle]
pub unsafe extern "C" fn smf_adaptive_pi_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    kernel: *const std::os::raw::c_char,
    tol_abs: c_double,
    tol_rel: c_double,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfAdaptivePI,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() || kernel.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let kernel_str = unsafe { CStr::from_ptr(kernel) }
            .to_str()
            .unwrap_or("heat2");
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_adaptive_inner(xmin, xmax, n, kernel_str, tol_abs, tol_rel, u0_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfAdaptivePI>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Evolve by time `t` using adaptive PI substeps; write result to `dst_buf`.
///
/// `steps_accepted` and `steps_rejected` are optional — pass NULL to ignore.
///
/// # Safety
/// `ev` live from `smf_adaptive_pi_new`; `dst_buf` writable for `dst_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_adaptive_pi_evolve(
    ev: *mut SmfAdaptivePI,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
    steps_accepted: *mut usize,
    steps_rejected: *mut usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<AdaptivePIInner>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let u0_snapshot = inner.current.clone();
        let (result, acc, rej) = match inner.integrator.evolve(t, &u0_snapshot) {
            Ok(r) => r,
            Err(e) => return SemiflowStatus::from(&e),
        };
        inner.current = result;
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        if !steps_accepted.is_null() {
            unsafe { *steps_accepted = acc };
        }
        if !steps_rejected.is_null() {
            unsafe { *steps_rejected = rej };
        }
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Values / Size / Free
// ---------------------------------------------------------------------------

/// Copy current grid values into `out_buf`.
///
/// # Safety
/// `ev` live; `out_buf` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_adaptive_pi_values(
    ev: *const SmfAdaptivePI,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<AdaptivePIInner>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or live from `smf_adaptive_pi_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_adaptive_pi_size(ev: *const SmfAdaptivePI) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<AdaptivePIInner>() };
    inner.current.values.len()
}

/// Free a `SmfAdaptivePI` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or live from `smf_adaptive_pi_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_adaptive_pi_free(ev: *mut SmfAdaptivePI) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<AdaptivePIInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builder
// ---------------------------------------------------------------------------

fn build_adaptive_inner(
    xmin: f64,
    xmax: f64,
    n: usize,
    kernel: &str,
    tol_abs: f64,
    tol_rel: f64,
    u0: &[f64],
) -> Result<AdaptivePIInner, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let current = GridFn1D::new(grid, u0.to_vec())?;
    let mut iv = build_variant(grid, kernel)?;
    iv.set_tolerance(tol_abs, tol_rel);
    Ok(AdaptivePIInner {
        integrator: iv,
        current,
    })
}

fn build_variant(
    grid: Grid1D<f64>,
    kernel: &str,
) -> Result<AdaptiveVariant, semiflow::SemiflowError> {
    match kernel {
        "heat2" => {
            let inner = DiffusionChernoff::new(unit_a_adp, zero_adp, zero_adp, 1.0, grid);
            Ok(AdaptiveVariant::Diff2(CorePI::new(inner)))
        }
        "heat4" => {
            let inner = Diffusion4thChernoff::new(unit_a_adp, zero_adp, zero_adp, 1.0, grid);
            Ok(AdaptiveVariant::Diff4(CorePI::new(inner)))
        }
        "heat6" => {
            let inner = Diffusion6thChernoff::new(unit_a_adp, zero_adp, zero_adp, 1.0, grid);
            Ok(AdaptiveVariant::Diff6(CorePI::new(inner)))
        }
        "drift" => {
            let inner = DriftReactionChernoff::new(half_a_adp, zero_adp, 0.5, grid);
            Ok(AdaptiveVariant::DriftReaction(CorePI::new(inner)))
        }
        "shift" => {
            let inner = ShiftChernoff1D::new(half_a_adp, zero_adp, zero_adp, 0.5, grid);
            Ok(AdaptiveVariant::Shift(CorePI::new(inner)))
        }
        _ => Err(semiflow::SemiflowError::DomainViolation {
            what: "adaptive: unknown kernel; expected heat2|heat4|heat6|drift|shift",
            value: 0.0,
        }),
    }
}
