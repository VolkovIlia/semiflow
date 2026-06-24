//! FFI surface for the native-complex Schrödinger engine (`SchrödingerChernoffComplex`).
//!
//! Mirrors `semiflow-py`'s `SchrodingerComplex1D` class.
//!
//! ## Complex buffer convention
//!
//! Same as `schrodinger_ffi.rs` — **interleaved** `f64` buffer of length `2*n`:
//!
//! ```text
//! buf = [re0, im0, re1, im1, …, re_{n-1}, im_{n-1}]
//! ```
//!
//! The kernel stores the wavefunction as `Vec<num_complex::Complex<f64>>` internally
//! and converts at the FFI boundary.
//!
//! ## Entry points
//!
//! - `smf_schrodinger_cx_new(xmin, xmax, n, v, v_len, psi0, psi0_len, n_steps, out)`
//! - `smf_schrodinger_cx_evolve(ev, t, dst, dst_len)`
//! - `smf_schrodinger_cx_values(ev, dst, dst_len)`
//! - `smf_schrodinger_cx_size(ev)` → `n`
//! - `smf_schrodinger_cx_free(ev)` — null-safe
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]

use std::{os::raw::c_double, sync::Arc};

use num_complex::Complex;
use semiflow::{
    BoundaryPolicy, ChernoffSemigroup, Grid1D, GridFnComplex1D, SchrödingerChernoffComplex,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Type alias
// ---------------------------------------------------------------------------

type C64 = Complex<f64>;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a native-complex Schrödinger evolver.
///
/// Obtain from `smf_schrodinger_cx_new`; pass to `_evolve`/`_values`/`_size`/`_free`.
/// Do not dereference or allocate from C.
#[repr(C)]
pub struct SmfSchrodingerComplex1D {
    _private: [u8; 0],
}

/// Inner state (heap-allocated behind opaque pointer).
struct InnerSchrodingerCx {
    /// Pre-sampled potential values `V(x_0), …, V(x_{N-1})`.
    v_at_node: Vec<f64>,
    /// Number of Chernoff steps per `evolve` call.
    n_steps: usize,
    /// Grid geometry.
    grid: Grid1D<f64>,
    /// Left boundary stored for kernel reconstruction.
    xmin: f64,
    /// Current wavefunction `ψ ∈ ℂⁿ`.
    state: GridFnComplex1D<C64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a native-complex Schrödinger evolver.
///
/// # Parameters
///
/// - `xmin`, `xmax`: grid boundaries (must be finite; `xmin < xmax`).
/// - `n`: number of grid nodes (must be >= 4).
/// - `v`: pre-sampled `V(x_i)` values; length `n`; all finite. Pass null for
///   the free-particle case (V = 0).
/// - `v_len`: length of `v`; must equal `n` when `v` is non-null.
/// - `psi0`: interleaved complex initial state `[re0,im0,re1,im1,…]`; length `2*n`.
/// - `psi0_len`: must equal `2*n`.
/// - `n_steps`: Chernoff steps per `evolve` call (must be >= 1).
/// - `out`: non-null pointer to receive the handle.
///
/// # Safety
///
/// `v` (if non-null) must point to `v_len` readable `f64`s.
/// `psi0` must point to `psi0_len` readable `f64`s.
/// `out` must be a valid writable `*mut *mut SmfSchrodingerComplex1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_cx_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    v: *const c_double,
    v_len: usize,
    psi0: *const c_double,
    psi0_len: usize,
    n_steps: usize,
    out: *mut *mut SmfSchrodingerComplex1D,
) -> SemiflowStatus {
    if psi0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let psi_slice = unsafe { std::slice::from_raw_parts(psi0, psi0_len) };
        let v_slice: Option<&[f64]> = if v.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(v, v_len) })
        };
        match build_schrodinger_cx(xmin, xmax, n, v_slice, psi_slice, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfSchrodingerComplex1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Advance by time `t` and copy result into `dst` (interleaved re/im, length `2*n`).
///
/// `t` must be finite and >= 0.
///
/// # Safety
///
/// `ev` must be a live pointer from `smf_schrodinger_cx_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_cx_evolve(
    ev: *mut SmfSchrodingerComplex1D,
    t: c_double,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerSchrodingerCx>() };
        let n = inner.state.values.len();
        if dst_len != 2 * n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_schrodinger_cx(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, 2 * n) };
        complex_to_interleaved(out, &inner.state.values);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Read-out
// ---------------------------------------------------------------------------

/// Copy current wavefunction into `dst` (interleaved re/im, length `2*n`).
///
/// # Safety
///
/// `ev` must be a live pointer from `smf_schrodinger_cx_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_cx_values(
    ev: *const SmfSchrodingerComplex1D,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerSchrodingerCx>() };
        let n = inner.state.values.len();
        if dst_len < 2 * n {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, 2 * n) };
        complex_to_interleaved(out, &inner.state.values);
        SemiflowStatus::Ok
    })
}

/// Return the number of grid nodes; 0 if `ev` is null.
///
/// Note: the interleaved buffer length is `2 * smf_schrodinger_cx_size(ev)`.
///
/// # Safety
///
/// `ev` must be null or a live pointer from `smf_schrodinger_cx_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_cx_size(ev: *const SmfSchrodingerComplex1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerSchrodingerCx>() };
    inner.state.values.len()
}

/// Free a handle from `smf_schrodinger_cx_new`. Null-safe.
///
/// # Safety
///
/// `ev` must be null or a live pointer from `smf_schrodinger_cx_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_cx_free(ev: *mut SmfSchrodingerComplex1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerSchrodingerCx>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn build_schrodinger_cx(
    xmin: f64,
    xmax: f64,
    n: usize,
    v_slice: Option<&[f64]>,
    psi0: &[f64],
    n_steps: usize,
) -> Result<InnerSchrodingerCx, semiflow::SemiflowError> {
    validate_psi_interleaved(psi0, n)?;
    let psi_vec = deinterleave_to_complex(psi0);
    let v_at_node = build_v_at_node(v_slice, n)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    // Validate kernel construction eagerly.
    build_cx_kernel(&v_at_node, grid, xmin)?;
    let state = GridFnComplex1D::<C64>::new(grid, psi_vec)?;
    Ok(InnerSchrodingerCx {
        v_at_node,
        n_steps,
        grid,
        xmin,
        state,
    })
}

fn validate_psi_interleaved(psi0: &[f64], n: usize) -> Result<(), semiflow::SemiflowError> {
    if psi0.len() != 2 * n {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "psi0_len must equal 2*n (interleaved re/im)",
            value: psi0.len() as f64,
        });
    }
    for &v in psi0 {
        if !v.is_finite() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "psi0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

fn build_v_at_node(v_slice: Option<&[f64]>, n: usize) -> Result<Vec<f64>, semiflow::SemiflowError> {
    match v_slice {
        None => Ok(vec![0.0_f64; n]),
        Some(arr) => {
            if arr.len() != n {
                return Err(semiflow::SemiflowError::DomainViolation {
                    what: "v_len must equal n",
                    value: arr.len() as f64,
                });
            }
            for &vi in arr {
                if !vi.is_finite() {
                    return Err(semiflow::SemiflowError::DomainViolation {
                        what: "v contains NaN or Inf",
                        value: vi,
                    });
                }
            }
            Ok(arr.to_vec())
        }
    }
}

fn build_cx_kernel(
    v_at_node: &[f64],
    grid: Grid1D<f64>,
    xmin: f64,
) -> Result<SchrödingerChernoffComplex<C64>, semiflow::SemiflowError> {
    let v = Arc::new(v_at_node.to_vec());
    let v2 = v.clone();
    let dx = grid.dx();
    SchrödingerChernoffComplex::<C64>::new(grid, move |x: f64| {
        let idx = ((x - xmin) / dx).round() as usize;
        v2[idx.min(v2.len().saturating_sub(1))]
    })
}

fn deinterleave_to_complex(buf: &[f64]) -> Vec<C64> {
    buf.chunks_exact(2).map(|c| C64::new(c[0], c[1])).collect()
}

fn complex_to_interleaved(dst: &mut [f64], vals: &[C64]) {
    for (i, z) in vals.iter().enumerate() {
        dst[2 * i] = z.re;
        dst[2 * i + 1] = z.im;
    }
}

fn evolve_schrodinger_cx(
    inner: &mut InnerSchrodingerCx,
    t: f64,
) -> Result<(), semiflow::SemiflowError> {
    let kernel = build_cx_kernel(&inner.v_at_node, inner.grid, inner.xmin)?;
    let sg = ChernoffSemigroup::new(kernel, inner.n_steps)?;
    let src = GridFnComplex1D::<C64>::new(inner.grid, inner.state.values.clone())?;
    let out = sg.evolve(t, &src)?;
    inner.state.values = out.values;
    Ok(())
}
