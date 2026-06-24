//! FFI surface for BC kernels — Part 2: `Robin1D`, `Resolvent1D`, `KilledDir1D` (Round 5).
//! Split from `bc_ffi.rs` (suckless ≤ 500 lines). Part 1: `bc_ffi.rs` (`Killing1D`, `Reflected1D`).
//! Safety: null-check before `catch_panic!`; `_free` null-safe; build `--profile release-ffi`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use std::os::raw::c_double;

use semiflow::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
    robin::{HalfSpaceRobin, RobinHeatChernoff},
    ChernoffSemigroup, Evolver, InterpKind, KilledDirichletChernoff,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

extern "Rust" fn unit_a_bc2(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_bc2(_: f64) -> f64 {
    0.0
}

type DiffUnit2 = DiffusionChernoff<f64>;
type RobinKernel = RobinHeatChernoff<DiffUnit2, HalfSpaceRobin<f64, 1>, f64>;
type ResolventKernel = LaplaceChernoffResolvent<DiffUnit2, f64>;

// --- RobinHeatChernoff — smf_robin1d_* ---

/// Opaque handle to `RobinHeatChernoff<DiffusionChernoff, HalfSpaceRobin>`.
#[repr(C)]
pub struct SmfRobin1D {
    _private: [u8; 0],
}

struct RobinState {
    semigroup: ChernoffSemigroup<RobinKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a Robin-BC 1D heat evolver (order 1).
///
/// Robin condition `alpha * u(origin) - beta * ∂_n u(origin) = 0`.
/// Unit diffusion `a = 1`. Uses `CubicHermite` interpolation (required by
/// `RobinHeatChernoff`; mirrors `bc_kernels2.rs`).
///
/// ## Preconditions
/// - `xmin < xmax`, both finite; `n_grid >= 4`.
/// - `alpha >= 0`, `beta > 0`, both finite.
/// - `n_chernoff >= 1`; `u0` non-null, `u0_len == n_grid`, all finite.
/// - `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfRobin1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_robin1d_new(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    n_chernoff: usize,
    alpha: c_double,
    beta: c_double,
    origin: c_double,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfRobin1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_robin(xmin, xmax, n_grid, n_chernoff, alpha, beta, origin, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfRobin1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance Robin evolver by `t`; write values into `dst`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_robin1d_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_robin1d_evolve(
    ev: *mut SmfRobin1D,
    t: c_double,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<RobinState>() };
        let n = s.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        match s.semigroup.evolve(t, &s.current) {
            Err(e) => return SemiflowStatus::from(&e),
            Ok(next) => s.current = next,
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, n) };
        out.copy_from_slice(&s.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current values into `out`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_robin1d_new`.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_robin1d_values(
    ev: *const SmfRobin1D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<RobinState>() };
        let vals = &s.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, vals.len()) };
        buf.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_robin1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_robin1d_size(ev: *const SmfRobin1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let s = unsafe { &*ev.cast::<RobinState>() };
    s.current.values.len()
}

/// Free a Robin handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_robin1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_robin1d_free(ev: *mut SmfRobin1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<RobinState>())) };
    }));
}

fn build_robin(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_chernoff: usize,
    alpha: f64,
    beta: f64,
    origin: f64,
    u0: &[f64],
) -> Result<RobinState, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    // CubicHermite required by RobinHeatChernoff (mirrors bc_kernels2.rs).
    let grid = Grid1D::new(xmin, xmax, n_grid)?.with_interp(InterpKind::CubicHermite);
    let diff = DiffusionChernoff::new(unit_a_bc2, zero_bc2, zero_bc2, 1.0, grid);
    let region = HalfSpaceRobin::<f64, 1>::new([origin], [1.0], alpha, beta)?;
    let kernel = RobinHeatChernoff::new(diff, region)?;
    let semigroup = ChernoffSemigroup::new(kernel, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(RobinState { semigroup, current })
}

// --- LaplaceChernoffResolvent — smf_resolvent1d_* ---

/// Opaque handle to `LaplaceChernoffResolvent<DiffusionChernoff, f64>`.
#[repr(C)]
pub struct SmfResolvent1D {
    _private: [u8; 0],
}

struct ResolventState {
    kernel: ResolventKernel,
    grid: Grid1D<f64>,
}

/// Allocate a Laplace-Chernoff resolvent handle.
///
/// Computes `R̃(λ) g = ∫₀^∞ exp(−λt) S(t)g dt` via Gauss-Laguerre-32.
/// Unit diffusion `a = 1`.
///
/// ## Preconditions
/// - `xmin < xmax`, both finite; `n_grid >= 4`; `n_chernoff >= 1`; `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `out` must be a valid writable `*mut *mut SmfResolvent1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent1d_new(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    n_chernoff: usize,
    out: *mut *mut SmfResolvent1D,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        match build_resolvent(xmin, xmax, n_grid, n_chernoff) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfResolvent1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evaluate `R̃(lambda) g`; write `n_grid` values into `out`.
///
/// `lambda > 0`; `g` non-null, length `g_len == n_grid`, all finite.
/// `out` non-null, length `out_len >= n_grid`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `g` must point to `g_len` readable `f64` values.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent1d_eval(
    ev: *const SmfResolvent1D,
    lambda: c_double,
    g: *const c_double,
    g_len: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || g.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<ResolventState>() };
        let n = s.grid.n;
        if g_len != n || out_len < n {
            return SemiflowStatus::GridMismatch;
        }
        if !lambda.is_finite() || lambda <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let g_slice = unsafe { std::slice::from_raw_parts(g, g_len) };
        if let Err(e) = validate_u0_finite(g_slice) {
            return SemiflowStatus::from(&e);
        }
        let g_fn = match GridFn1D::new(s.grid, g_slice.to_vec()) {
            Err(e) => return SemiflowStatus::from(&e),
            Ok(gf) => gf,
        };
        match s.kernel.eval(lambda, &g_fn) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => unsafe {
                std::slice::from_raw_parts_mut(out, n).copy_from_slice(&result.values);
                SemiflowStatus::Ok
            },
        }
    })
}

/// Return grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_resolvent1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent1d_size(ev: *const SmfResolvent1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<ResolventState>() }.grid.n
}

/// Free a resolvent handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_resolvent1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_resolvent1d_free(ev: *mut SmfResolvent1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<ResolventState>())) };
    }));
}

fn build_resolvent(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_chernoff: usize,
) -> Result<ResolventState, semiflow::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n_grid)?;
    let diff = DiffusionChernoff::new(unit_a_bc2, zero_bc2, zero_bc2, 1.0, grid);
    let kernel =
        LaplaceChernoffResolvent::new(diff, n_chernoff, LaplaceQuadrature::GaussLaguerre32)?;
    Ok(ResolventState { kernel, grid })
}

// --- KilledDirichletChernoff — smf_killed_dir1d_* ---

/// Opaque handle to `KilledDirichletChernoff` (via `Evolver`).
#[repr(C)]
pub struct SmfKilledDir1D {
    _private: [u8; 0],
}

struct KilledDirState {
    evolver: Evolver<KilledDirichletChernoff>,
    current: GridFn1D<f64>,
}

/// Allocate a killed-Dirichlet 1D heat evolver (order 2).
///
/// Absorbing endpoints `u = 0` at both domain boundaries.
/// Unit diffusion `a = 1`, zero drift `b = 0`.
///
/// ## Preconditions
/// - `domain_lo < domain_hi`, both finite; `n_grid >= 3`.
/// - `n_chernoff >= 1`; `u0` non-null, `u0_len == n_grid`, all finite.
/// - `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfKilledDir1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_killed_dir1d_new(
    domain_lo: c_double,
    domain_hi: c_double,
    n_grid: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfKilledDir1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_killed_dir(domain_lo, domain_hi, n_grid, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfKilledDir1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Apply the killed-Dirichlet semigroup for time `t`; write values into `dst`.
///
/// Updates internal state in-place (chainable).
///
/// # Safety
/// `ev` must be a live pointer from `smf_killed_dir1d_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_killed_dir1d_apply(
    ev: *mut SmfKilledDir1D,
    t: c_double,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<KilledDirState>() };
        let n = s.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        match s.evolver.evolve(t, &s.current) {
            Err(e) => return SemiflowStatus::from(&e),
            Ok(next) => s.current = next,
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, n) };
        out.copy_from_slice(&s.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current values into `out`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_killed_dir1d_new`.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_killed_dir1d_values(
    ev: *const SmfKilledDir1D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<KilledDirState>() };
        let vals = &s.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, vals.len()) };
        buf.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_killed_dir1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_killed_dir1d_size(ev: *const SmfKilledDir1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<KilledDirState>() }
        .current
        .values
        .len()
}

/// Free a killed-Dirichlet handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_killed_dir1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_killed_dir1d_free(ev: *mut SmfKilledDir1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<KilledDirState>())) };
    }));
}

fn build_killed_dir(
    domain_lo: f64,
    domain_hi: f64,
    n_grid: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<KilledDirState, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(domain_lo, domain_hi, n_grid)?;
    let kernel = KilledDirichletChernoff::new(|_| 1.0_f64, |_| 0.0_f64, grid)?;
    let evolver = Evolver::new(kernel, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(KilledDirState { evolver, current })
}
