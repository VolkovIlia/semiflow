//! FFI surface for `ObstacleGammaV8` — inactive-set Γ = V″ (C-parity pass,
//! ADR-0028/0171, ADR-0153 TIER-2).
//!
//! Mirrors `semiflow-py` `PyObstacleGammaV8` (`obstacle_gamma_py.rs`).
//!
//! ## Entry points
//!
//! - `smf_obstacle_gamma_new_const(xmin,xmax,n,level,out)` → `SemiflowStatus`
//! - `smf_obstacle_gamma_new_array(xmin,xmax,n,obs,obs_len,out)` → `SemiflowStatus`
//! - `smf_obstacle_gamma_free(ptr)` — null-safe
//! - `smf_obstacle_gamma_size(ptr)` → `usize` (0 if null)
//! - `smf_obstacle_gamma_inactive_gamma(ptr,v,v_len,gamma_out,defined_out,count_out)` → `SemiflowStatus`
//!
//! ## Memory model for gamma / defined read-back
//!
//! `gamma_out` is set to a freshly Box-allocated `*mut f64` of length `v_len`.
//! `defined_out` is set to a freshly Box-allocated `*mut u8` of length `v_len`
//! (0 = false, 1 = true). The caller frees each buffer with
//! `smf_free_buf_f64` (for gamma) and `smf_free_buf_u8` (for defined).
//!
//! ## Honesty (NORMATIVE, math §44.5.bis)
//!
//! `defined[i] == 0` means Γ is REFUSED at node `i` — NOT "Γ = 0".
//! Callers MUST check `defined[i]` before reading `gamma[i]`.
//! D = 1 only; multi-asset Γ deferred (§44.5.ter, ADR-0153).
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::cast_precision_loss)]

use std::os::raw::c_double;

use semiflow_core::{
    ConstantObstacle, DiffusionChernoff, Grid1D, GridFn1D, ObstacleChernoff,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// FfiArrayObstacle (mirrors obstacle_ffi.rs — per ADR-0028 Amdt 2, no sharing)
// ---------------------------------------------------------------------------

struct GammaArrayObstacle {
    values: Vec<f64>,
}

impl GammaArrayObstacle {
    fn new(values: Vec<f64>) -> Result<Self, semiflow_core::SemiflowError> {
        validate_u0_finite(&values)?;
        Ok(Self { values })
    }
}

impl semiflow_core::Obstacle<f64> for GammaArrayObstacle {
    fn value_at(&self, _point: &[f64]) -> f64 {
        0.0
    }

    fn project_in_place(
        &self,
        dst: &mut GridFn1D<f64>,
    ) -> Result<(), semiflow_core::SemiflowError> {
        if dst.values.len() != self.values.len() {
            return Err(semiflow_core::SemiflowError::DomainViolation {
                what: "GammaArrayObstacle: length mismatch",
                value: self.values.len() as f64,
            });
        }
        for (v, &g) in dst.values.iter_mut().zip(self.values.iter()) {
            if *v < g {
                *v = g;
            }
        }
        Ok(())
    }

    fn active_set_into(
        &self,
        w: &GridFn1D<f64>,
        active: &mut [bool],
    ) -> Result<(), semiflow_core::SemiflowError> {
        if active.len() != w.grid.n || active.len() != self.values.len() {
            return Err(semiflow_core::SemiflowError::DomainViolation {
                what: "GammaArrayObstacle::active_set_into: length mismatch",
                value: active.len() as f64,
            });
        }
        for (flag, (wv, gv)) in active
            .iter_mut()
            .zip(w.values.iter().zip(self.values.iter()))
        {
            *flag = *wv > *gv;
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type GammaConst = ObstacleChernoff<DiffusionChernoff<f64>, ConstantObstacle<f64>, f64>;
type GammaArray = ObstacleChernoff<DiffusionChernoff<f64>, GammaArrayObstacle, f64>;

// ---------------------------------------------------------------------------
// GammaVariant — avoids Box<dyn …>
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
enum GammaVariant {
    Const(GammaConst),
    Array(GammaArray),
}

impl GammaVariant {
    fn apply_gamma(
        &self,
        v: &GridFn1D<f64>,
        gamma: &mut GridFn1D<f64>,
        defined: &mut [bool],
    ) -> Result<usize, semiflow_core::SemiflowError> {
        match self {
            Self::Const(k) => k.apply_inactive_gamma_into(v, gamma, defined),
            Self::Array(k) => k.apply_inactive_gamma_into(v, gamma, defined),
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to an `ObstacleGammaV8` (inactive-set Γ = V″).
///
/// Obtain from `smf_obstacle_gamma_new_const` or `smf_obstacle_gamma_new_array`.
/// Free with `smf_obstacle_gamma_free`.
#[repr(C)]
pub struct SmfObstacleGamma {
    _private: [u8; 0],
}

struct GammaInner {
    kernel: GammaVariant,
    grid: Grid1D<f64>,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Allocate an `ObstacleGamma` with a constant obstacle floor.
///
/// `level` must be finite.  `n >= 4` (Grid1D requirement).
///
/// # Safety
/// `out` must be a valid non-null `*mut *mut SmfObstacleGamma`.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle_gamma_new_const(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    level: c_double,
    out: *mut *mut SmfObstacleGamma,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_gamma_const(xmin, xmax, n, level) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfObstacleGamma>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Allocate an `ObstacleGamma` with a per-node array obstacle floor.
///
/// `obstacle` must be non-null, length `obstacle_len == n`, all finite.
/// `n >= 4`.
///
/// # Safety
/// `obstacle` readable for `obstacle_len` f64s; `out` writable `*mut *mut SmfObstacleGamma`.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle_gamma_new_array(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    obstacle: *const c_double,
    obstacle_len: usize,
    out: *mut *mut SmfObstacleGamma,
) -> SemiflowStatus {
    if obstacle.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let obs_slice = unsafe { std::slice::from_raw_parts(obstacle, obstacle_len) };
        match build_gamma_array(xmin, xmax, n, obs_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfObstacleGamma>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Free / size
// ---------------------------------------------------------------------------

/// Free a `SmfObstacleGamma` handle. Null-safe.
///
/// # Safety
/// `ptr` must be null or a live pointer from `smf_obstacle_gamma_new_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle_gamma_free(ptr: *mut SmfObstacleGamma) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ptr.cast::<GammaInner>())) };
    }));
}

/// Return grid size (n). Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be null or a live pointer from `smf_obstacle_gamma_new_*`.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle_gamma_size(ptr: *const SmfObstacleGamma) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let inner = unsafe { &*ptr.cast::<GammaInner>() };
    inner.grid.n
}

// ---------------------------------------------------------------------------
// inactive_gamma
// ---------------------------------------------------------------------------

/// Compute inactive-set Γ = V″ on the OPEN continuation set.
///
/// On success:
/// - `*gamma_out` is set to a Box-allocated `f64[v_len]` (central-diff V″;
///   valid where `defined[i] == 1`).
/// - `*defined_out` is set to a Box-allocated `uint8_t[v_len]` (1 = defined,
///   0 = REFUSED).
/// - `*count_out` is set to the number of nodes where Γ is defined.
///
/// Caller frees `*gamma_out` with `smf_free_buf_f64(*gamma_out, v_len)` and
/// `*defined_out` with `smf_free_buf_u8(*defined_out, v_len)`.
///
/// ## Return values
/// - `Ok` (0)           — success.
/// - `NullPtr` (5)      — any pointer argument is null.
/// - `GridMismatch` (1) — `v_len != n`.
/// - `Panic` (99)       — internal Rust panic.
///
/// # Safety
/// - `ptr` live pointer from `smf_obstacle_gamma_new_*`.
/// - `v` readable for `v_len` f64s.
/// - `gamma_out`, `defined_out`, `count_out` writable non-null pointers.
#[no_mangle]
#[allow(clippy::cast_precision_loss)]
pub unsafe extern "C" fn smf_obstacle_gamma_inactive_gamma(
    ptr: *const SmfObstacleGamma,
    v: *const c_double,
    v_len: usize,
    gamma_out: *mut *mut c_double,
    defined_out: *mut *mut u8,
    count_out: *mut usize,
) -> SemiflowStatus {
    if ptr.is_null() || v.is_null() || gamma_out.is_null()
        || defined_out.is_null() || count_out.is_null()
    {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ptr.cast::<GammaInner>() };
        let n = inner.grid.n;
        if v_len != n {
            return SemiflowStatus::GridMismatch;
        }
        let v_slice = unsafe { std::slice::from_raw_parts(v, v_len) };
        let v_fn = match GridFn1D::new(inner.grid, v_slice.to_vec()) {
            Ok(f) => f,
            Err(e) => return SemiflowStatus::from(&e),
        };
        let mut gamma_fn = v_fn.zeroed_like();
        let mut defined_bool = vec![false; n];
        let count = match inner.kernel.apply_gamma(&v_fn, &mut gamma_fn, &mut defined_bool) {
            Ok(c) => c,
            Err(e) => return SemiflowStatus::from(&e),
        };
        // Allocate gamma buffer (Box<[f64]>).
        let gamma_boxed: Box<[f64]> = gamma_fn.values.into_boxed_slice();
        let gamma_ptr = Box::into_raw(gamma_boxed) as *mut f64;
        // Allocate defined buffer (Box<[u8]>): 1 = true, 0 = false.
        let defined_u8: Vec<u8> = defined_bool.iter().map(|&b| b as u8).collect();
        let defined_boxed: Box<[u8]> = defined_u8.into_boxed_slice();
        let defined_ptr = Box::into_raw(defined_boxed) as *mut u8;
        unsafe {
            *gamma_out = gamma_ptr;
            *defined_out = defined_ptr;
            *count_out = count;
        }
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Buffer free helper for u8 (defined mask)
// ---------------------------------------------------------------------------

/// Free a `uint8_t` buffer previously returned by `smf_obstacle_gamma_inactive_gamma`.
/// Null-safe.
///
/// `len` must exactly match `v_len` passed to `smf_obstacle_gamma_inactive_gamma`.
///
/// # Safety
/// `buf` must be null or a pointer from `smf_obstacle_gamma_inactive_gamma`
/// with the matching `len`.
#[no_mangle]
pub unsafe extern "C" fn smf_free_buf_u8(buf: *mut u8, len: usize) {
    if buf.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let slice = unsafe { std::slice::from_raw_parts_mut(buf, len) };
        unsafe { drop(Box::from_raw(slice)) };
    }));
}

// ---------------------------------------------------------------------------
// Private builders
// ---------------------------------------------------------------------------

fn build_gamma_const(
    xmin: f64,
    xmax: f64,
    n: usize,
    level: f64,
) -> Result<GammaInner, semiflow_core::SemiflowError> {
    validate_gamma_domain(xmin, xmax, n)?;
    if !level.is_finite() {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "obstacle_gamma: level must be finite",
            value: level,
        });
    }
    let grid = Grid1D::new(xmin, xmax, n)?;
    let diff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0_f64, grid);
    let obs = ConstantObstacle::new(level)?;
    let kernel: GammaConst = ObstacleChernoff::new(diff, obs)?;
    Ok(GammaInner { kernel: GammaVariant::Const(kernel), grid })
}

fn build_gamma_array(
    xmin: f64,
    xmax: f64,
    n: usize,
    obstacle: &[f64],
) -> Result<GammaInner, semiflow_core::SemiflowError> {
    validate_gamma_domain(xmin, xmax, n)?;
    if obstacle.len() != n {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "obstacle_gamma: obstacle_array length must equal n",
            value: obstacle.len() as f64,
        });
    }
    let grid = Grid1D::new(xmin, xmax, n)?;
    let diff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0_f64, grid);
    let obs = GammaArrayObstacle::new(obstacle.to_vec())?;
    let kernel: GammaArray = ObstacleChernoff::new(diff, obs)?;
    Ok(GammaInner { kernel: GammaVariant::Array(kernel), grid })
}

#[allow(clippy::cast_precision_loss)]
fn validate_gamma_domain(
    xmin: f64,
    xmax: f64,
    n: usize,
) -> Result<(), semiflow_core::SemiflowError> {
    if !xmin.is_finite() || !xmax.is_finite() {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "obstacle_gamma: domain bounds must be finite",
            value: f64::NAN,
        });
    }
    if xmin >= xmax {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "obstacle_gamma: xmin must be < xmax",
            value: xmin,
        });
    }
    if n < 4 {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "obstacle_gamma: n must be >= 4 (Grid1D requirement)",
            value: n as f64,
        });
    }
    Ok(())
}
