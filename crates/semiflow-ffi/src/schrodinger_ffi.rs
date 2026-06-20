//! FFI surface for the real-pair Schrödinger engine (`SchrodingerChernoff`).
//!
//! Mirrors `semiflow-py`'s `Schrodinger1D` class.
//!
//! ## Complex buffer convention (NORMATIVE)
//!
//! The wavefunction ψ ∈ ℂⁿ is represented as an **interleaved** `f64` buffer
//! of length `2*n`:
//!
//! ```text
//! buf = [re0, im0, re1, im1, …, re_{n-1}, im_{n-1}]
//! ```
//!
//! This convention is used consistently by all Schrödinger FFI functions in
//! this file and in `schrodinger_complex_ffi.rs`.
//!
//! ## Entry points
//!
//! - `smf_schrodinger_new(xmin, xmax, n, v, v_len, psi0, psi0_len, n_steps, out)`
//! - `smf_schrodinger_evolve(ev, t, dst, dst_len)`
//! - `smf_schrodinger_values(ev, dst, dst_len)` — copy current state
//! - `smf_schrodinger_size(ev)` → `n` (grid nodes; buffer length = `2*n`)
//! - `smf_schrodinger_free(ev)` — null-safe
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
    clippy::too_many_arguments,
)]

use std::os::raw::c_double;

use semiflow_core::{
    BoundaryPolicy, ChernoffFunction, Diffusion4thChernoff, Grid1D, GridFn1D, SchrodingerChernoff,
    SchrodingerState, ScratchPool,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a real-pair Schrödinger evolver.
///
/// Obtain from `smf_schrodinger_new`; pass to `_evolve`/`_values`/`_size`/`_free`.
/// Do not dereference or allocate from C.
#[repr(C)]
pub struct SmfSchrodinger1D {
    _private: [u8; 0],
}

/// Inner state (heap-allocated behind opaque pointer).
struct InnerSchrodinger {
    /// Pre-sampled potential values `V(x_0), …, V(x_{N-1})`.
    v_at_node: Vec<f64>,
    /// Number of Chernoff steps per `evolve` call.
    n_steps: usize,
    /// Grid geometry.
    grid: Grid1D<f64>,
    /// Current wavefunction real part `ψ_re`.
    psi_re: Vec<f64>,
    /// Current wavefunction imaginary part `ψ_im`.
    psi_im: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a real-pair Schrödinger evolver.
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
/// `out` must be a valid writable `*mut *mut SmfSchrodinger1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    v: *const c_double,
    v_len: usize,
    psi0: *const c_double,
    psi0_len: usize,
    n_steps: usize,
    out: *mut *mut SmfSchrodinger1D,
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
        match build_schrodinger(xmin, xmax, n, v_slice, psi_slice, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfSchrodinger1D>();
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
/// # Safety
///
/// `ev` must be a live pointer from `smf_schrodinger_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_evolve(
    ev: *mut SmfSchrodinger1D,
    t: c_double,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerSchrodinger>() };
        let n = inner.psi_re.len();
        if dst_len != 2 * n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_schrodinger(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, 2 * n) };
        interleave_into(out, &inner.psi_re, &inner.psi_im);
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
/// `ev` must be a live pointer from `smf_schrodinger_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_values(
    ev: *const SmfSchrodinger1D,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerSchrodinger>() };
        let n = inner.psi_re.len();
        if dst_len < 2 * n {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, 2 * n) };
        interleave_into(out, &inner.psi_re, &inner.psi_im);
        SemiflowStatus::Ok
    })
}

/// Return the number of grid nodes; 0 if `ev` is null.
///
/// Note: the interleaved buffer length is `2 * smf_schrodinger_size(ev)`.
///
/// # Safety
///
/// `ev` must be null or a live pointer from `smf_schrodinger_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_size(ev: *const SmfSchrodinger1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerSchrodinger>() };
    inner.psi_re.len()
}

/// Free a handle from `smf_schrodinger_new`. Null-safe.
///
/// # Safety
///
/// `ev` must be null or a live pointer from `smf_schrodinger_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_schrodinger_free(ev: *mut SmfSchrodinger1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerSchrodinger>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn build_schrodinger(
    xmin: f64,
    xmax: f64,
    n: usize,
    v_slice: Option<&[f64]>,
    psi0: &[f64],
    n_steps: usize,
) -> Result<InnerSchrodinger, semiflow_core::SemiflowError> {
    validate_psi_interleaved(psi0, n)?;
    let (psi_re, psi_im) = deinterleave(psi0);
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let v_at_node = build_v_at_node(v_slice, n)?;
    // Validate kernel construction eagerly.
    build_kernel(&v_at_node, grid)?;
    Ok(InnerSchrodinger { v_at_node, n_steps, grid, psi_re, psi_im })
}

fn validate_psi_interleaved(
    psi0: &[f64],
    n: usize,
) -> Result<(), semiflow_core::SemiflowError> {
    if psi0.len() != 2 * n {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "psi0_len must equal 2*n (interleaved re/im)",
            value: psi0.len() as f64,
        });
    }
    for &v in psi0 {
        if !v.is_finite() {
            return Err(semiflow_core::SemiflowError::DomainViolation {
                what: "psi0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

/// Split interleaved `[re0,im0,re1,im1,...]` into `(re_vec, im_vec)`.
fn deinterleave(buf: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = buf.len() / 2;
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for chunk in buf.chunks_exact(2) {
        re.push(chunk[0]);
        im.push(chunk[1]);
    }
    (re, im)
}

/// Write `(re, im)` into interleaved buffer `dst`.
fn interleave_into(dst: &mut [f64], re: &[f64], im: &[f64]) {
    for (i, (r, c)) in re.iter().zip(im.iter()).enumerate() {
        dst[2 * i] = *r;
        dst[2 * i + 1] = *c;
    }
}

/// Sample V at each grid node (zero potential when `v_slice` is `None`).
fn build_v_at_node(
    v_slice: Option<&[f64]>,
    n: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    match v_slice {
        None => Ok(vec![0.0_f64; n]),
        Some(arr) => {
            if arr.len() != n {
                return Err(semiflow_core::SemiflowError::DomainViolation {
                    what: "v_len must equal n",
                    value: arr.len() as f64,
                });
            }
            for &vi in arr {
                if !vi.is_finite() {
                    return Err(semiflow_core::SemiflowError::DomainViolation {
                        what: "v contains NaN or Inf",
                        value: vi,
                    });
                }
            }
            Ok(arr.to_vec())
        }
    }
}

/// Construct `SchrodingerChernoff` from a stored potential vector.
fn build_kernel(
    v_at_node: &[f64],
    grid: Grid1D<f64>,
) -> Result<SchrodingerChernoff<f64>, semiflow_core::SemiflowError> {
    let kinetic = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let v = v_at_node.to_vec();
    let dx = grid.dx();
    let xmin = grid.xmin;
    SchrodingerChernoff::new(kinetic, move |x: f64| {
        let idx = ((x - xmin) / dx).round() as usize;
        v[idx.min(v.len().saturating_sub(1))]
    })
}

/// One full evolution by time `t`: `n_steps` Chernoff steps of size `t/n_steps`.
fn evolve_schrodinger(
    inner: &mut InnerSchrodinger,
    t: f64,
) -> Result<(), semiflow_core::SemiflowError> {
    let kernel = build_kernel(&inner.v_at_node, inner.grid)?;
    let tau = t / inner.n_steps as f64;
    let psi_re = GridFn1D::new(inner.grid, inner.psi_re.clone())?;
    let psi_im = GridFn1D::new(inner.grid, inner.psi_im.clone())?;
    let mut state = SchrodingerState::new(psi_re, psi_im)?;
    let mut scratch = ScratchPool::new();
    for _ in 0..inner.n_steps {
        let mut next = state.clone();
        kernel.apply_into(tau, &state, &mut next, &mut scratch)?;
        state = next;
    }
    inner.psi_re = state.psi_re.values;
    inner.psi_im = state.psi_im.values;
    Ok(())
}

extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_d(_: f64) -> f64 {
    0.0
}
