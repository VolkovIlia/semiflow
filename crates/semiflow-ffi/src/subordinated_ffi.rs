//! FFI surface for the subordinated heat semigroup engine (Round 7).
//!
//! | C handle            | Core type                                       | Python class    |
//! |---------------------|-------------------------------------------------|-----------------|
//! | `SmfSubordinated1D` | `SubordinatedChernoff<DiffUnit, SubEnum, f64>`  | `Subordinated1D`|
//!
//! ## Subordinator selector (`subordinator_tag` — `u32`)
//!
//! Mirrors Python `Subordinated1D(subordinator=...)` string kwarg.
//!
//! | tag | Python string       | Subordinator                | Key param |
//! |-----|---------------------|-----------------------------|-----------|
//! | 0   | `"stable"`          | `StableSubordinator<f64>`   | `alpha`   |
//! | 1   | `"gamma"`           | `GammaSubordinator<f64>`    | `c`       |
//! | 2   | `"inverse_gaussian"`| `InverseGaussianSubordinator<f64>` | `c` |
//!
//! ## Constructor
//!
//! `smf_subordinated1d_new(xmin, xmax, n, n_chernoff,
//!                         subordinator_tag, alpha, c, n_nodes,
//!                         u0, u0_len, out)`
//!
//! - `alpha` — used only for `subordinator_tag == 0`; must be in `(0, 1)`.
//! - `c`     — used only for tags 1 and 2; must be `> 0`.
//! - `n_nodes` — GL-32 quadrature nodes, `1..=32`; default via Python is 32.
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use std::os::raw::{c_double, c_uint};

use semiflow::{
    diffusion::DiffusionChernoff,
    subordinated::{
        GammaSubordinator, InverseGaussianSubordinator, LevySubordinator, StableSubordinator,
        SubordinatedChernoff,
    },
    ChernoffSemigroup, Grid1D, GridFn1D,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ---------------------------------------------------------------------------
// Fn-pointers for unit-diffusion
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_sub(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_sub(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Binding-side subordinator enum (mirrors PyO3 SubordinatorEnum)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum SubEnum {
    Stable(StableSubordinator<f64>),
    Gamma(GammaSubordinator<f64>),
    InverseGaussian(InverseGaussianSubordinator<f64>),
}

impl LevySubordinator<f64> for SubEnum {
    fn laplace_exponent(&self, lambda: f64) -> f64 {
        match self {
            SubEnum::Stable(s) => s.laplace_exponent(lambda),
            SubEnum::Gamma(s) => s.laplace_exponent(lambda),
            SubEnum::InverseGaussian(s) => s.laplace_exponent(lambda),
        }
    }

    fn quadrature(&self, tau: f64, n_nodes: usize) -> (Vec<f64>, Vec<f64>) {
        match self {
            SubEnum::Stable(s) => s.quadrature(tau, n_nodes),
            SubEnum::Gamma(s) => s.quadrature(tau, n_nodes),
            SubEnum::InverseGaussian(s) => s.quadrature(tau, n_nodes),
        }
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type SubKernel = SubordinatedChernoff<DiffUnit, SubEnum, f64>;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `SubordinatedChernoff` 1D evolver.
///
/// Obtain from `smf_subordinated1d_new`; free with `smf_subordinated1d_free`.
#[repr(C)]
pub struct SmfSubordinated1D {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct SubordinatedState1D {
    semigroup: ChernoffSemigroup<SubKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Allocate a subordinated 1D heat evolver.
///
/// `subordinator_tag`: 0 = stable, 1 = gamma, 2 = `inverse_gaussian`.
/// `alpha` used for tag 0 (must be in `(0,1)`).
/// `c`     used for tags 1, 2 (must be `> 0`).
/// `n_nodes` GL quadrature nodes; `1..=32`; typical 32.
///
/// ## Preconditions
/// - `xmin < xmax`, finite; `n >= 4`.
/// - `n_chernoff >= 1`.
/// - `u0` non-null, `u0_len == n`, all finite.
/// - `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Unsupported` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfSubordinated1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_subordinated1d_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    subordinator_tag: c_uint,
    alpha: c_double,
    c: c_double,
    n_nodes: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfSubordinated1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        let sub = match parse_subordinator(subordinator_tag, alpha, c) {
            Ok(s) => s,
            Err(st) => return st,
        };
        match build_subordinated(xmin, xmax, n, n_chernoff, sub, n_nodes, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfSubordinated1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance the subordinated evolver by `t` using `n_steps` iterations.
///
/// Writes `n` values into `dst`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_subordinated1d_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_subordinated1d_evolve(
    ev: *mut SmfSubordinated1D,
    t: c_double,
    n_steps: usize,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<SubordinatedState1D>() };
        let n = s.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if n_steps == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        match s.semigroup.evolve(t, &s.current) {
            Err(e) => return SemiflowStatus::from(&e),
            Ok(next) => s.current = next,
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(dst, n) };
        buf.copy_from_slice(&s.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current values into `out`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_subordinated1d_new`.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_subordinated1d_values(
    ev: *const SmfSubordinated1D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<SubordinatedState1D>() };
        let vals = &s.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, vals.len()) };
        buf.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return grid size `n`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_subordinated1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_subordinated1d_size(ev: *const SmfSubordinated1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let s = unsafe { &*ev.cast::<SubordinatedState1D>() };
    s.current.values.len()
}

/// Free a subordinated handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_subordinated1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_subordinated1d_free(ev: *mut SmfSubordinated1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<SubordinatedState1D>())) };
    }));
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Map `subordinator_tag` → `SubEnum`.  Returns `SemiflowStatus::Unsupported`
/// for unknown tags; `OutOfDomain` if param range is invalid.
fn parse_subordinator(tag: u32, alpha: f64, c: f64) -> Result<SubEnum, SemiflowStatus> {
    match tag {
        0 => StableSubordinator::new(alpha)
            .map(SubEnum::Stable)
            .map_err(|e| SemiflowStatus::from(&e)),
        1 => GammaSubordinator::new(c)
            .map(SubEnum::Gamma)
            .map_err(|e| SemiflowStatus::from(&e)),
        2 => InverseGaussianSubordinator::new(c)
            .map(SubEnum::InverseGaussian)
            .map_err(|e| SemiflowStatus::from(&e)),
        _ => Err(SemiflowStatus::Unsupported),
    }
}

fn build_subordinated(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    sub: SubEnum,
    n_nodes: usize,
    u0: &[f64],
) -> Result<SubordinatedState1D, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?;
    let diff = DiffusionChernoff::new(unit_a_sub, zero_sub, zero_sub, 1.0, grid);
    let kernel = SubordinatedChernoff::with_n_nodes(diff, sub, n_nodes)?;
    let semigroup = ChernoffSemigroup::new(kernel, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(SubordinatedState1D { semigroup, current })
}
