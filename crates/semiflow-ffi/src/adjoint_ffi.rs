//! FFI surface for `AdjointChernoff` — adjoint semigroup wrapper (Round 9).
//!
//! Mirrors `semiflow-py`'s `Adjoint` class (`adjoint.rs`).
//!
//! ## Python constructor (authoritative spec)
//!
//! ```python
//! Adjoint(xmin, xmax, n, u0, *, kernel="heat2",
//!         self_adjoint=False, boundary="reflect")
//! ```
//!
//! `kernel` ∈ `{"heat2", "heat4", "heat6", "drift", "shift"}`.
//!
//! ## FFI constructor
//!
//! ```c
//! SemiflowStatus smf_adjoint1d_new(
//!     double xmin, double xmax, size_t n,
//!     size_t n_steps,
//!     const char *kernel,      // NUL-terminated: "heat2"|"heat4"|"heat6"|"drift"|"shift"
//!     int self_adjoint,        // 0 = false, non-zero = true
//!     const double *u0, size_t u0_len,
//!     SmfAdjoint1D **out
//! );
//! ```
//!
//! ## Symbol names
//!
//! - `smf_adjoint1d_new` / `smf_adjoint1d_evolve` / `smf_adjoint1d_values`
//!   / `smf_adjoint1d_size` / `smf_adjoint1d_order` / `smf_adjoint1d_free`
//!
//! ## Scope
//!
//! 1-D adjoint only; boundary policy hardcoded to `Reflect` (matches Python
//! default `"reflect"`).  `heat4` / `heat6` use unit diffusion `a = 1`.
//!
//! ## Panic safety
//!
//! Every `extern "C"` entry is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::cast_precision_loss)]

use std::{
    ffi::CStr,
    os::raw::{c_double, c_int},
};

use semiflow::{
    AdjointChernoff, BoundaryPolicy, ChernoffFunction, Diffusion4thChernoff, Diffusion6thChernoff,
    DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, ScratchPool, ShiftChernoff1D,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ---------------------------------------------------------------------------
// Constant fn-pointers (unit a = 1.0 / half a = 0.5 / zero)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_adj(_: f64) -> f64 {
    1.0
}
extern "Rust" fn half_a_adj(_: f64) -> f64 {
    0.5
}
extern "Rust" fn zero_adj(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// 5-kernel enum (avoids Box<dyn ChernoffFunction>)
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
enum AdjKernelVariant {
    Diff2(AdjointChernoff<DiffusionChernoff<f64>>),
    Diff4(AdjointChernoff<Diffusion4thChernoff<f64>>),
    Diff6(AdjointChernoff<Diffusion6thChernoff<f64>>),
    DriftReaction(AdjointChernoff<DriftReactionChernoff<f64>>),
    Shift(AdjointChernoff<ShiftChernoff1D<f64>>),
}

impl AdjKernelVariant {
    fn apply_step(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow::SemiflowError> {
        match self {
            Self::Diff2(k) => k.apply_into(tau, src, dst, scratch),
            Self::Diff4(k) => k.apply_into(tau, src, dst, scratch),
            Self::Diff6(k) => k.apply_into(tau, src, dst, scratch),
            Self::DriftReaction(k) => k.apply_into(tau, src, dst, scratch),
            Self::Shift(k) => k.apply_into(tau, src, dst, scratch),
        }
    }

    fn order(&self) -> u32 {
        match self {
            Self::Diff2(k) => k.order(),
            Self::Diff4(k) => k.order(),
            Self::Diff6(k) => k.order(),
            Self::DriftReaction(k) => k.order(),
            Self::Shift(k) => k.order(),
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque handle + inner state
// ---------------------------------------------------------------------------

/// Opaque handle to a 1-D adjoint Chernoff evolver.
///
/// Obtained from `smf_adjoint1d_new`; free with `smf_adjoint1d_free`.
#[repr(C)]
pub struct SmfAdjoint1D {
    _private: [u8; 0],
}

struct AdjointInner {
    kernel: AdjKernelVariant,
    n_steps: usize,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a 1-D adjoint Chernoff evolver.
///
/// `kernel` must be a NUL-terminated C string: `"heat2"`, `"heat4"`,
/// `"heat6"`, `"drift"`, or `"shift"`.
/// `self_adjoint` is treated as `bool` (0 = false).
///
/// # Safety
/// `kernel` must be a valid NUL-terminated C string for its lifetime.
/// `u0` readable for `u0_len` f64s; `out` writable `*mut *mut SmfAdjoint1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_adjoint1d_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_steps: usize,
    kernel: *const std::os::raw::c_char,
    self_adjoint: c_int,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfAdjoint1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() || kernel.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let kernel_str = unsafe { CStr::from_ptr(kernel) }
            .to_str()
            .unwrap_or("heat2");
        let sa = self_adjoint != 0;
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_adjoint_inner(xmin, xmax, n, n_steps, kernel_str, sa, u0_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfAdjoint1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Evolve the adjoint evolver by time `t`; write values to `dst_buf`.
///
/// # Safety
/// `ev` live from `smf_adjoint1d_new`; `dst_buf` writable for `dst_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_adjoint1d_evolve(
    ev: *mut SmfAdjoint1D,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<AdjointInner>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = run_adjoint_evolve(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Values / Size / Order / Free
// ---------------------------------------------------------------------------

/// Copy current grid values into `out_buf`.
///
/// # Safety
/// `ev` live; `out_buf` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_adjoint1d_values(
    ev: *const SmfAdjoint1D,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<AdjointInner>() };
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
/// `ev` must be null or live from `smf_adjoint1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_adjoint1d_size(ev: *const SmfAdjoint1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<AdjointInner>() };
    inner.current.values.len()
}

/// Return approximation order of the adjoint kernel; 0 if null.
///
/// # Safety
/// `ev` must be null or live from `smf_adjoint1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_adjoint1d_order(ev: *const SmfAdjoint1D) -> u32 {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<AdjointInner>() };
    inner.kernel.order()
}

/// Free a `SmfAdjoint1D` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or live from `smf_adjoint1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_adjoint1d_free(ev: *mut SmfAdjoint1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<AdjointInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builders
// ---------------------------------------------------------------------------

fn build_adjoint_inner(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    kernel: &str,
    self_adjoint: bool,
    u0: &[f64],
) -> Result<AdjointInner, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let current = GridFn1D::new(grid, u0.to_vec())?;
    let kv = build_kernel_variant(grid, kernel, self_adjoint)?;
    Ok(AdjointInner {
        kernel: kv,
        n_steps,
        current,
    })
}

fn build_kernel_variant(
    grid: Grid1D<f64>,
    kernel: &str,
    self_adjoint: bool,
) -> Result<AdjKernelVariant, semiflow::SemiflowError> {
    match kernel {
        "heat2" => {
            let inner = DiffusionChernoff::new(unit_a_adj, zero_adj, zero_adj, 1.0, grid);
            Ok(AdjKernelVariant::Diff2(AdjointChernoff::new_self_adjoint(
                inner,
            )))
        }
        "heat4" => {
            let inner = Diffusion4thChernoff::new(unit_a_adj, zero_adj, zero_adj, 1.0, grid);
            Ok(AdjKernelVariant::Diff4(AdjointChernoff::new_self_adjoint(
                inner,
            )))
        }
        "heat6" => {
            let inner = Diffusion6thChernoff::new(unit_a_adj, zero_adj, zero_adj, 1.0, grid);
            Ok(AdjKernelVariant::Diff6(AdjointChernoff::new_self_adjoint(
                inner,
            )))
        }
        "drift" => {
            let inner = DriftReactionChernoff::new(half_a_adj, zero_adj, 0.5, grid);
            let adj = if self_adjoint {
                AdjointChernoff::new_self_adjoint(inner)
            } else {
                AdjointChernoff::new_general(inner)
            };
            Ok(AdjKernelVariant::DriftReaction(adj))
        }
        "shift" => {
            let inner = ShiftChernoff1D::new(half_a_adj, zero_adj, zero_adj, 0.5, grid);
            Ok(AdjKernelVariant::Shift(AdjointChernoff::new_self_adjoint(
                inner,
            )))
        }
        _ => Err(semiflow::SemiflowError::DomainViolation {
            what: "adjoint: unknown kernel; expected heat2|heat4|heat6|drift|shift",
            value: 0.0,
        }),
    }
}

// ---------------------------------------------------------------------------
// Evolve helper
// ---------------------------------------------------------------------------

fn run_adjoint_evolve(inner: &mut AdjointInner, t: f64) -> Result<(), semiflow::SemiflowError> {
    let tau = t / inner.n_steps as f64;
    let grid = inner.current.grid;
    let src_vals = inner.current.values.clone();
    let mut src = GridFn1D::new(grid, src_vals)?;
    let mut dst = GridFn1D::new(grid, vec![0.0; src.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..inner.n_steps {
        inner.kernel.apply_step(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    inner.current.values = src.values;
    Ok(())
}
