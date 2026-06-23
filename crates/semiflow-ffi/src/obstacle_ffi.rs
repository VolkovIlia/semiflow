//! FFI surface for `ObstacleChernoff` — projective-splitting Chernoff (Round 9).
//!
//! Mirrors `semiflow-py`'s `PyObstacleChernoff` (`obstacle_py.rs`).
//!
//! ## Python constructor (authoritative spec)
//!
//! ```python
//! ObstacleChernoff(xmin, xmax, n, u0, *, a=1.0, b=0.0, c=0.0,
//!                  level=NaN, obstacle_array=None)
//! ```
//!
//! ## FFI constructor
//!
//! ```c
//! SemiflowStatus smf_obstacle1d_new(
//!     double xmin, double xmax, size_t n,
//!     size_t n_steps,
//!     double a, double b, double c,
//!     double level,          // pass NaN to use obstacle_array
//!     const double *obs_buf, // NULL → constant-level obstacle
//!     size_t obs_len,        // 0 when obs_buf is NULL
//!     const double *u0, size_t u0_len,
//!     SmfObstacle1D **out
//! );
//! ```
//!
//! Level-based: `obs_buf == NULL` → `ConstantObstacle(level)`.
//! Array-based: `obs_buf != NULL` → `ArrayObstacle(obs_buf[0..obs_len])`.
//!
//! `b`, `c` default to 0 in caller.  When both are zero the fast-path pure
//! diffusion kernel is used; otherwise a Strang-split inner is used.
//!
//! ## Symbol names
//!
//! - `smf_obstacle1d_new` / `smf_obstacle1d_evolve` / `smf_obstacle1d_values`
//!   / `smf_obstacle1d_size` / `smf_obstacle1d_free`
//!
//! ## Gamma / `ObstacleND` deferral (ADR-0153)
//!
//! `ObstacleGamma` (inactive-set Γ) and `ObstacleNDV8` (D=2) are TIER-2
//! opportunistic and remain PyO3-only per ADR-0153 §4.  The FFI exposes the
//! base `ObstacleChernoff` (1-D, constant or array obstacle) only.
//!
//! ## Panic safety
//!
//! Every `extern "C"` entry is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::cast_precision_loss)]

use std::os::raw::c_double;

use semiflow::{
    BoundaryPolicy, ConstantObstacle, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D,
    ObstacleChernoff, ScratchPool, StrangSplit,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Local obstacle types (mirrors obstacle_py.rs, per ADR-0028 Amendment 2)
// ---------------------------------------------------------------------------

/// Per-node array obstacle — mirrors `obstacle_py::ArrayObstacle`.
struct FfiArrayObstacle {
    values: Vec<f64>,
}

impl FfiArrayObstacle {
    fn new(values: Vec<f64>) -> Result<Self, semiflow::SemiflowError> {
        validate_u0_finite(&values)?;
        Ok(Self { values })
    }
}

impl semiflow::Obstacle<f64> for FfiArrayObstacle {
    fn value_at(&self, _point: &[f64]) -> f64 {
        0.0
    }

    fn project_in_place(
        &self,
        dst: &mut GridFn1D<f64>,
    ) -> Result<(), semiflow::SemiflowError> {
        if dst.values.len() != self.values.len() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "FfiArrayObstacle: length mismatch",
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
    ) -> Result<(), semiflow::SemiflowError> {
        if active.len() != w.grid.n || active.len() != self.values.len() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "FfiArrayObstacle::active_set_into: length mismatch",
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
// Type aliases (mirrors obstacle_py.rs type aliases)
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type DrUnit = DriftReactionChernoff<f64>;
type StrangUnit = StrangSplit<DiffUnit, DrUnit, f64>;
type ConstKernel = ObstacleChernoff<DiffUnit, ConstantObstacle<f64>, f64>;
type ArrayKernel = ObstacleChernoff<DiffUnit, FfiArrayObstacle, f64>;
type StrangConstKernel = ObstacleChernoff<StrangUnit, ConstantObstacle<f64>, f64>;
type StrangArrayKernel = ObstacleChernoff<StrangUnit, FfiArrayObstacle, f64>;

// ---------------------------------------------------------------------------
// Obstacle variant enum (avoids Box<dyn ChernoffFunction>)
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
enum ObstacleVariant {
    Const(ConstKernel),
    Array(ArrayKernel),
    Strang(StrangConstKernel),
    StrangArray(StrangArrayKernel),
}

impl ObstacleVariant {
    fn apply_step(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow::SemiflowError> {
        use semiflow::ChernoffFunction as CF;
        match self {
            Self::Const(k) => k.apply_into(tau, src, dst, scratch),
            Self::Array(k) => k.apply_into(tau, src, dst, scratch),
            Self::Strang(k) => k.apply_into(tau, src, dst, scratch),
            Self::StrangArray(k) => k.apply_into(tau, src, dst, scratch),
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque handle + inner state
// ---------------------------------------------------------------------------

/// Opaque handle to a 1-D obstacle Chernoff evolver.
///
/// Obtained from `smf_obstacle1d_new`; free with `smf_obstacle1d_free`.
#[repr(C)]
pub struct SmfObstacle1D {
    _private: [u8; 0],
}

struct ObstacleInner {
    kernel: ObstacleVariant,
    n_steps: usize,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a 1-D obstacle Chernoff evolver.
///
/// - `obs_buf == NULL` → constant obstacle at `level`.
/// - `obs_buf != NULL` → array obstacle; `obs_len` must equal `n`.
/// - `b == 0 && c == 0` → pure-diffusion fast path.
/// - Otherwise → Strang-split inner.
///
/// # Safety
/// `u0` readable for `u0_len` f64s; `out` writable `*mut *mut SmfObstacle1D`.
/// `obs_buf` readable for `obs_len` f64s when non-null.
#[no_mangle]
#[allow(clippy::many_single_char_names)]
pub unsafe extern "C" fn smf_obstacle1d_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_steps: usize,
    a: c_double,
    b: c_double,
    c: c_double,
    level: c_double,
    obs_buf: *const c_double,
    obs_len: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfObstacle1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        let obs_opt = if obs_buf.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(obs_buf, obs_len) })
        };
        match build_obstacle_inner(xmin, xmax, n, n_steps, a, b, c, level, obs_opt, u0_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfObstacle1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Evolve the obstacle evolver by time `t` and write grid values to `dst_buf`.
///
/// # Safety
/// `ev` live pointer from `smf_obstacle1d_new`; `dst_buf` writable for `dst_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle1d_evolve(
    ev: *mut SmfObstacle1D,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<ObstacleInner>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = run_obstacle_evolve(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Values / Size / Free
// ---------------------------------------------------------------------------

/// Copy current grid values into `out_buf`.
///
/// # Safety
/// `ev` live pointer; `out_buf` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle1d_values(
    ev: *const SmfObstacle1D,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<ObstacleInner>() };
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
/// `ev` must be null or live from `smf_obstacle1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle1d_size(ev: *const SmfObstacle1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<ObstacleInner>() };
    inner.current.values.len()
}

/// Free a `SmfObstacle1D` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or live from `smf_obstacle1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_obstacle1d_free(ev: *mut SmfObstacle1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<ObstacleInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builders
// ---------------------------------------------------------------------------

#[allow(clippy::many_single_char_names)]
fn build_obstacle_inner(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    a: f64,
    b: f64,
    c: f64,
    level: f64,
    obs_opt: Option<&[f64]>,
    u0: &[f64],
) -> Result<ObstacleInner, semiflow::SemiflowError> {
    validate_params(a, b, c)?;
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let current = GridFn1D::new(grid, u0.to_vec())?;
    let use_strang = b != 0.0 || c != 0.0;
    let kernel = match obs_opt {
        Some(obs) => build_array_variant(a, b, c, obs, grid, use_strang)?,
        None => build_const_variant(a, b, c, level, grid, use_strang)?,
    };
    Ok(ObstacleInner { kernel, n_steps, current })
}

fn build_const_variant(
    a: f64,
    b: f64,
    c: f64,
    level: f64,
    grid: Grid1D<f64>,
    use_strang: bool,
) -> Result<ObstacleVariant, semiflow::SemiflowError> {
    let obs = ConstantObstacle::new(level)?;
    if use_strang {
        let inner = build_strang_inner(a, b, c, grid);
        let k: StrangConstKernel = ObstacleChernoff::new(inner, obs)?;
        Ok(ObstacleVariant::Strang(k))
    } else {
        let diff = build_diff_inner(a, grid);
        let k: ConstKernel = ObstacleChernoff::new(diff, obs)?;
        Ok(ObstacleVariant::Const(k))
    }
}

fn build_array_variant(
    a: f64,
    b: f64,
    c: f64,
    obs: &[f64],
    grid: Grid1D<f64>,
    use_strang: bool,
) -> Result<ObstacleVariant, semiflow::SemiflowError> {
    if obs.len() != grid.n {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "obstacle array length must equal n",
            value: obs.len() as f64,
        });
    }
    let ao = FfiArrayObstacle::new(obs.to_vec())?;
    if use_strang {
        let inner = build_strang_inner(a, b, c, grid);
        let k: StrangArrayKernel = ObstacleChernoff::new(inner, ao)?;
        Ok(ObstacleVariant::StrangArray(k))
    } else {
        let diff = build_diff_inner(a, grid);
        let k: ArrayKernel = ObstacleChernoff::new(diff, ao)?;
        Ok(ObstacleVariant::Array(k))
    }
}

fn build_diff_inner(a: f64, grid: Grid1D<f64>) -> DiffUnit {
    DiffusionChernoff::new_const_a(a, a, grid)
}

fn build_strang_inner(a: f64, b: f64, c: f64, grid: Grid1D<f64>) -> StrangUnit {
    let diff = DiffusionChernoff::new_const_a(a, a, grid);
    let c_bound = c.abs();
    let drift = DriftReactionChernoff::with_closure(move |_| b, move |_| c, c_bound, grid);
    StrangSplit::new(diff, drift)
}

fn validate_params(a: f64, b: f64, c: f64) -> Result<(), semiflow::SemiflowError> {
    if !a.is_finite() || a <= 0.0 {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "obstacle: a must be finite and > 0",
            value: a,
        });
    }
    if !b.is_finite() || !c.is_finite() {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "obstacle: b and c must be finite",
            value: b,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Evolve helper
// ---------------------------------------------------------------------------

fn run_obstacle_evolve(
    inner: &mut ObstacleInner,
    t: f64,
) -> Result<(), semiflow::SemiflowError> {
    #[allow(clippy::cast_precision_loss)]
    let tau = t / inner.n_steps as f64;
    let grid = inner.current.grid;
    let src_vals = inner.current.values.clone();
    let mut src = GridFn1D::new(grid, src_vals)?;
    let mut dst = src.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..inner.n_steps {
        inner.kernel.apply_step(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    inner.current.values = src.values;
    Ok(())
}
