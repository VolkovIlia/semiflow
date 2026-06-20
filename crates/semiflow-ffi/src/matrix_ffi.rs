//! FFI surface for the coupled 2-component matrix diffusion engine
//! (`MatrixDiffusionChernoff<f64, 2>`).
//!
//! Mirrors `semiflow-py`'s `MatrixDiffusion1D` class (M17, ADR-0082, math §33).
//!
//! ## State buffer layout (NORMATIVE)
//!
//! Flat `f64` buffer of length `2*n` (component-inner / row-major):
//!
//! ```text
//! buf[k*2 + i] = u_i(x_k)   (k = grid point, i = component ∈ {0,1})
//! ```
//!
//! This matches `MatrixGridFn1D<f64, 2>::values` and the Python binding's
//! documented layout.
//!
//! ## Entry points
//!
//! - `smf_matrix_diffusion_new(xmin, xmax, n, a_diag, c_coupling, u0, u0_len, n_steps, out)`
//! - `smf_matrix_diffusion_evolve(ev, t, dst, dst_len)`
//! - `smf_matrix_diffusion_values(ev, dst, dst_len)`
//! - `smf_matrix_diffusion_size(ev)` → `n` (grid nodes; buffer length = `2*n`)
//! - `smf_matrix_diffusion_free(ev)` — null-safe
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
)]

use std::os::raw::c_double;

use semiflow_core::{
    matrix_system::{MatrixDiffusionChernoff, MatrixGridFn1D},
    ChernoffSemigroup, Grid1D,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a 2-component matrix-diffusion evolver.
///
/// Obtain from `smf_matrix_diffusion_new`; pass to `_evolve`/`_values`/`_size`/`_free`.
/// Do not dereference or allocate from C.
#[repr(C)]
pub struct SmfMatrixDiffusion1D {
    _private: [u8; 0],
}

/// Inner state (heap-allocated behind opaque pointer).
struct InnerMatrix {
    /// Diagonal diffusion coefficient `a_00 = a_11`.
    a_diag: f64,
    /// Off-diagonal reaction `c_01 = c_10`.
    c_coupling: f64,
    /// Number of Chernoff steps per `evolve` call.
    n_steps: usize,
    /// Grid geometry.
    grid: Grid1D<f64>,
    /// Current state flat buffer `u[k*2+i]`.
    current: MatrixGridFn1D<f64, 2>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a 2-component matrix-diffusion evolver.
///
/// # Parameters
///
/// - `xmin`, `xmax`: grid boundaries (must be finite; `xmin < xmax`).
/// - `n`: number of grid nodes (must be >= 5 per ADR-0082).
/// - `a_diag`: diagonal diffusion coefficient (must be finite and > 0).
/// - `c_coupling`: off-diagonal reaction coefficient (any finite value).
/// - `u0`: initial condition; flat `f64` array of length `2*n`
///   (`u0[k*2+i] = u_i(x_k)`); all values finite.
/// - `u0_len`: must equal `2*n`.
/// - `n_steps`: Chernoff steps per `evolve` call (must be >= 1).
/// - `out`: non-null pointer to receive the handle.
///
/// # Safety
///
/// `u0` must point to `u0_len` readable `f64`s.
/// `out` must be a valid writable `*mut *mut SmfMatrixDiffusion1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix_diffusion_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    a_diag: c_double,
    c_coupling: c_double,
    u0: *const c_double,
    u0_len: usize,
    n_steps: usize,
    out: *mut *mut SmfMatrixDiffusion1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_matrix(xmin, xmax, n, a_diag, c_coupling, u0_slice, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfMatrixDiffusion1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Advance by time `t` and copy result into `dst` (flat buffer, length `2*n`).
///
/// `t` must be finite and >= 0.
///
/// # Safety
///
/// `ev` must be a live pointer from `smf_matrix_diffusion_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix_diffusion_evolve(
    ev: *mut SmfMatrixDiffusion1D,
    t: c_double,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerMatrix>() };
        let buf_len = inner.current.values.len();
        if dst_len != buf_len {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_matrix(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, buf_len) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Read-out
// ---------------------------------------------------------------------------

/// Copy current state into `dst` (flat buffer, length `2*n`).
///
/// # Safety
///
/// `ev` must be a live pointer from `smf_matrix_diffusion_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix_diffusion_values(
    ev: *const SmfMatrixDiffusion1D,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerMatrix>() };
        let buf_len = inner.current.values.len();
        if dst_len < buf_len {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, buf_len) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

/// Return the number of grid nodes; 0 if `ev` is null.
///
/// Note: the state buffer length is `2 * smf_matrix_diffusion_size(ev)`.
///
/// # Safety
///
/// `ev` must be null or a live pointer from `smf_matrix_diffusion_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix_diffusion_size(ev: *const SmfMatrixDiffusion1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerMatrix>() };
    // buf_len = 2 * n_grid; divide back.
    inner.current.values.len() / 2
}

/// Free a handle from `smf_matrix_diffusion_new`. Null-safe.
///
/// # Safety
///
/// `ev` must be null or a live pointer from `smf_matrix_diffusion_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_matrix_diffusion_free(ev: *mut SmfMatrixDiffusion1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerMatrix>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn build_matrix(
    xmin: f64,
    xmax: f64,
    n: usize,
    a_diag: f64,
    c_coupling: f64,
    u0: &[f64],
    n_steps: usize,
) -> Result<InnerMatrix, semiflow_core::SemiflowError> {
    if n < 5 {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "n must be >= 5 (block-CN stencil, ADR-0082)",
            value: n as f64,
        });
    }
    if !a_diag.is_finite() || a_diag <= 0.0 {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "a_diag must be finite and > 0",
            value: a_diag,
        });
    }
    if u0.len() != 2 * n {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "u0_len must equal 2*n",
            value: u0.len() as f64,
        });
    }
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?;
    // Validate kernel construction eagerly.
    build_matrix_kernel(a_diag, c_coupling, grid)?;
    let mut current = MatrixGridFn1D::<f64, 2>::new(grid);
    current.values.copy_from_slice(u0);
    Ok(InnerMatrix { a_diag, c_coupling, n_steps, grid, current })
}

/// Build `MatrixDiffusionChernoff<f64, 2>` with diagonal-a and symmetric coupling.
fn build_matrix_kernel(
    a_diag: f64,
    c_coupling: f64,
    grid: Grid1D<f64>,
) -> Result<MatrixDiffusionChernoff<f64, 2>, semiflow_core::SemiflowError> {
    let a_d = a_diag;
    let c_c = c_coupling;
    MatrixDiffusionChernoff::<f64, 2>::new(
        move |_x, mat| {
            mat[0][0] = a_d;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = a_d;
        },
        |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = 0.0;
        },
        move |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = c_c;
            mat[1][0] = c_c;
            mat[1][1] = 0.0;
        },
        grid,
    )
}

fn evolve_matrix(
    inner: &mut InnerMatrix,
    t: f64,
) -> Result<(), semiflow_core::SemiflowError> {
    let kernel = build_matrix_kernel(inner.a_diag, inner.c_coupling, inner.grid)?;
    let sg = ChernoffSemigroup::new(kernel, inner.n_steps)?;
    let mut src = MatrixGridFn1D::<f64, 2>::new(inner.grid);
    src.values.copy_from_slice(&inner.current.values);
    let out = sg.evolve(t, &src)?;
    inner.current.values.copy_from_slice(&out.values);
    Ok(())
}
