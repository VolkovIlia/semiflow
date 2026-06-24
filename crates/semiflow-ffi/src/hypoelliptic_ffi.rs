//! FFI surface for hypoelliptic / sub-Riemannian Chernoff engines (Round 8).
//!
//! Mirrors the Python classes in `semiflow-py/src/hormander_py.rs`.
//!
//! | C handle                    | Core type                          | Python class                      |
//! |-----------------------------|------------------------------------|-----------------------------------|
//! | `SmfHypoHeisenberg`         | `HypoellipticChernoff<f64, 3, 2>`  | `HypoellipticChernoffHeisenberg`  |
//! | `SmfHypoKolmogorov`         | `KolmogorovHypoelliptic<f64>`      | `HypoellipticChernoffKolmogorov`  |
//! | `SmfHypoEngel`              | `HypoellipticChernoff<f64, 4, 2>`  | `HypoellipticChernoffEngel`       |
//!
//! ## Heisenberg surface (mirrors Python `HypoellipticChernoffHeisenberg`)
//!
//! Python exposes only `new()`, `order()`, and `kernel(h, x, y, tc)`.
//! There is no grid `evolve` for Heisenberg — the WASM binding also confirms this.
//!
//! - `smf_hypo_heisenberg_new(out_handle)` — verifies bracket; no grid params.
//! - `smf_hypo_heisenberg_order(handle)` → `u32` (always 2).
//! - `smf_hypo_heisenberg_kernel(handle, h, x, y, tc, out)` → `SemiflowStatus`.
//! - `smf_hypo_heisenberg_free(handle)` — null-safe.
//!
//! ## Kolmogorov surface (unchanged — validated grid evolve exists in Python+WASM)
//!
//! - `smf_hypo_kolmogorov_new(xmin,xmax,nx,vmin,vmax,nv,u0,u0_len,out)`
//! - `smf_hypo_kolmogorov_evolve(ev,t,n_steps,dst,dst_len)`
//! - `smf_hypo_kolmogorov_values(ev,out,out_len)`
//! - `smf_hypo_kolmogorov_size(ev)` → `nx*nv`
//! - `smf_hypo_kolmogorov_free(ev)` — null-safe
//!
//! Engel lives in `hypoelliptic_engel_ffi.rs` (split for ≤500 line limit).
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
    clippy::similar_names,
    clippy::too_many_arguments
)]

extern crate alloc;

use std::os::raw::c_double;

use semiflow::{
    heisenberg_heat_kernel,
    hormander::{HypoellipticChernoff, KolmogorovHypoelliptic, KolmogorovPhaseSpace},
    ChernoffFunction, ChernoffSemigroup, Grid1D, Grid2D, GridFn2D,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ─── Opaque handles ───────────────────────────────────────────────────────────

/// Opaque handle for the Heisenberg-group Chernoff kernel (no grid state).
///
/// Obtain from `smf_hypo_heisenberg_new`; free with `smf_hypo_heisenberg_free`.
/// Provides `order` and `kernel` — mirrors Python `HypoellipticChernoffHeisenberg`.
#[repr(C)]
pub struct SmfHypoHeisenberg {
    _private: [u8; 0],
}

/// Opaque handle for Kolmogorov-equation hypoelliptic Chernoff evolver.
///
/// Obtain from `smf_hypo_kolmogorov_new`; free with `smf_hypo_kolmogorov_free`.
#[repr(C)]
pub struct SmfHypoKolmogorov {
    _private: [u8; 0],
}

// ─── Inner states ─────────────────────────────────────────────────────────────

/// Heisenberg kernel wrapper — no grid, only `order` + `kernel` oracle.
struct HeisenbergState {
    inner: HypoellipticChernoff<f64, 3, 2>,
}

struct KolmogorovState {
    grid: Grid2D<f64>,
    current: Vec<f64>,
    size: usize, // nx * nv
}

// ─── Heisenberg: smf_hypo_heisenberg_* ───────────────────────────────────────

/// Construct the Heisenberg-group Chernoff kernel.
///
/// Verifies the step-2 Carnot bracket `[X₁, X₂] = ∂_t` at the origin
/// (mirrors Python `HypoellipticChernoffHeisenberg.__new__()`).
///
/// No grid parameters — the Heisenberg binding provides only `order` and `kernel`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `OutOfDomain` (bracket failure) | `Panic`.
///
/// # Safety
/// `out` must be a valid writable `*mut *mut SmfHypoHeisenberg`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_heisenberg_new(
    out: *mut *mut SmfHypoHeisenberg,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match HypoellipticChernoff::<f64, 3, 2>::new_heisenberg() {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let state = Box::new(HeisenbergState { inner });
                let raw = Box::into_raw(state).cast::<SmfHypoHeisenberg>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return the approximation order (always 2 for palindromic Strang-Hörmander).
///
/// Returns 0 if `handle` is null.
///
/// # Safety
/// `handle` must be null or a live pointer from `smf_hypo_heisenberg_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_heisenberg_order(handle: *const SmfHypoHeisenberg) -> u32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { &*handle.cast::<HeisenbergState>() }.inner.order()
}

/// Evaluate the Heisenberg heat kernel oracle `p_h(x, y, tc)`.
///
/// Delegates to `heisenberg_heat_kernel(h, x, y, tc)` (Gaveau-Hulanicki,
/// 32-pt Gauss-Legendre quadrature, math.md §28 AMENDMENT 2).
///
/// ## Parameters
/// - `h` — step parameter h > 0; writes 0.0 for h ≤ 0 (matches Python).
/// - `x`, `y` — horizontal coordinates.
/// - `tc` — vertical (centre) coordinate.
/// - `out` — writable pointer for the kernel value.
///
/// ## Return values
/// `Ok` | `NullPtr` | `Panic`.
///
/// # Safety
/// `handle` must be a live pointer from `smf_hypo_heisenberg_new`.
/// `out` must point to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_heisenberg_kernel(
    handle: *const SmfHypoHeisenberg,
    h: c_double,
    x: c_double,
    y: c_double,
    tc: c_double,
    out: *mut c_double,
) -> SemiflowStatus {
    if handle.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let val = heisenberg_heat_kernel(h, x, y, tc);
        unsafe { *out = val };
        SemiflowStatus::Ok
    })
}

/// Free a Heisenberg handle. Null-safe; do not use after this call.
///
/// # Safety
/// `handle` must be null or a live pointer from `smf_hypo_heisenberg_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_heisenberg_free(handle: *mut SmfHypoHeisenberg) {
    if handle.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(handle.cast::<HeisenbergState>())) };
    }));
}

// ─── Kolmogorov: smf_hypo_kolmogorov_* ───────────────────────────────────────

/// Allocate a Kolmogorov-equation hypoelliptic Chernoff evolver.
///
/// Solves `∂_t p = v·∂_x p + ½·∂²_v p` on ℝ² via the palindromic
/// Strang-Hörmander decomposition (Kolmogorov 1934, math.md §28.4.A).
///
/// ## Buffer layout (x-fastest)
/// Flat f64 array of length `nx*nv`:
///   `idx(i,j) = j*nx + i`   (`i` = x-index, `j` = v-index).
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable f64 values.
/// `out` must be a valid writable `*mut *mut SmfHypoKolmogorov`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_kolmogorov_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    vmin: c_double,
    vmax: c_double,
    nv: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfHypoKolmogorov,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_kolmogorov(xmin, xmax, nx, vmin, vmax, nv, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfHypoKolmogorov>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance Kolmogorov state by time `t` using `n_steps` iterations (τ = `t/n_steps`).
///
/// Writes `nx*nv` values into `dst` (x-fastest). Updates internal state.
///
/// # Safety
/// `ev` must be a live pointer from `smf_hypo_kolmogorov_new`.
/// `dst` must be valid for `dst_len` writable f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_kolmogorov_evolve(
    ev: *mut SmfHypoKolmogorov,
    t: c_double,
    n_steps: usize,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<KolmogorovState>() };
        if dst_len != s.size {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 || n_steps == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        let tau = t / n_steps as f64;
        match evolve_kolmogorov(s.grid, s.current.clone(), tau, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                s.current.clone_from(&result);
                let buf = unsafe { std::slice::from_raw_parts_mut(dst, dst_len) };
                buf.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Copy current Kolmogorov state into `out` (x-fastest, length `nx*nv`).
///
/// # Safety
/// `ev` must be a live pointer from `smf_hypo_kolmogorov_new`.
/// `out` must be valid for `out_len` writable f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_kolmogorov_values(
    ev: *const SmfHypoKolmogorov,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<KolmogorovState>() };
        if out_len != s.size {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        buf.copy_from_slice(&s.current);
        SemiflowStatus::Ok
    })
}

/// Return `nx * nv`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_hypo_kolmogorov_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_kolmogorov_size(ev: *const SmfHypoKolmogorov) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<KolmogorovState>() }.size
}

/// Free a Kolmogorov handle. Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_hypo_kolmogorov_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_kolmogorov_free(ev: *mut SmfHypoKolmogorov) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<KolmogorovState>())) };
    }));
}

// ─── Builder ──────────────────────────────────────────────────────────────────

fn build_kolmogorov(
    xmin: f64,
    xmax: f64,
    nx: usize,
    vmin: f64,
    vmax: f64,
    nv: usize,
    u0: &[f64],
) -> Result<KolmogorovState, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gv = Grid1D::new(vmin, vmax, nv)?;
    let grid = Grid2D::new(gx, gv);
    let size = nx * nv;
    check_len("u0 length must equal nx*nv", u0.len(), size)?;
    // Verify Hörmander bracket condition at construction.
    let x0 = alloc::boxed::Box::new(KolmogorovPhaseSpace::x0_drift());
    let x1 = alloc::boxed::Box::new(KolmogorovPhaseSpace::x1_diffusion());
    let _ = KolmogorovHypoelliptic::<f64>::new(x0, [x1])?;
    Ok(KolmogorovState {
        grid,
        current: u0.to_vec(),
        size,
    })
}

// ─── Compute helper ───────────────────────────────────────────────────────────

fn evolve_kolmogorov(
    grid: Grid2D<f64>,
    values: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let x0 = alloc::boxed::Box::new(KolmogorovPhaseSpace::x0_drift());
    let x1 = alloc::boxed::Box::new(KolmogorovPhaseSpace::x1_diffusion());
    let kernel = KolmogorovHypoelliptic::<f64>::new(x0, [x1])?;
    let f = GridFn2D::new_generic(grid, values)?;
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    Ok(sg.evolve(tau * n_steps as f64, &f)?.values)
}

// ─── Utility (pub for hypoelliptic_engel_ffi.rs) ─────────────────────────────

pub(crate) fn check_len(
    what: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), semiflow::SemiflowError> {
    if got != expected {
        return Err(semiflow::SemiflowError::DomainViolation {
            what,
            value: got as f64,
        });
    }
    Ok(())
}
