//! Inner state construction helpers for `semiflow-py`.
//!
//! Mirrors `crates/semiflow-ffi/src/handle.rs` with one difference: there is no
//! opaque C handle type here — `PyO3` owns the memory via `#[pyclass]`.
//! `SemiflowStateInner` is wrapped directly by `state::Heat1D`.
//!
//! Per ADR-0028 §5, `build_heat_unit` is duplicated (not shared through a common
//! crate) to avoid creating a `semiflow-py → semiflow-ffi` dependency.  Both copies
//! call into `semiflow-core` independently.  The rule-of-three has not been hit yet.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::too_many_arguments)]

use semiflow::{
    chernoff::ApplyChernoffExt, BoundaryPolicy, ChernoffFunction, ChernoffSemigroup,
    DiffusionChernoff, Grid1D, Grid2D, Grid3D, GridFn1D, GridFn2D, GridFn3D, SemiflowError,
    ScratchPool, Strang2D, Strang3D,
};

// ---------------------------------------------------------------------------
// Closure type alias (ADR-0034)
// ---------------------------------------------------------------------------

/// Owned, thread-safe, heap-allocated coefficient closure (ADR-0034).
///
/// `Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>` wraps Python callables
/// in `with_a_function`. The `'static` + `Send + Sync` bounds are required by
/// `DiffusionChernoff::with_closure` and by the `py.detach` GIL-release path.
pub(crate) type CoeffClosure = Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>;

// ---------------------------------------------------------------------------
// Inner state (Rust-private, wrapped by Heat1D in state.rs)
// ---------------------------------------------------------------------------

/// Heap-allocated semigroup state owned by `Heat1D`.
pub(crate) struct SemiflowStateInner {
    /// The Chernoff semigroup (function + iteration count).
    pub semigroup: ChernoffSemigroup<DiffusionChernoff<f64>, GridFn1D<f64>>,
    /// Current function state (updated in-place by `Heat1D::evolve`).
    pub current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Static function pointers (a = 1.0, a' = 0, a'' = 0)
// ---------------------------------------------------------------------------

/// Diffusion coefficient `a(x) = 1.0` (hardcoded in v0.10.0).
///
/// Variable-coefficient support requires a runtime closure, which
/// `DiffusionChernoff::new` does not accept in this version. Deferred v0.11.0.
extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}

/// Derivative `a'(x) = 0` (constant `a`).
extern "Rust" fn zero_deriv(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Constructor helper
// ---------------------------------------------------------------------------

/// Build a `SemiflowStateInner` for the 1-D heat equation with `a = 1.0`.
///
/// Validates `u0_slice` for finiteness before construction.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Grid1D::new`, `GridFn1D::new`, or
/// `ChernoffSemigroup::new`.
pub(crate) fn build_heat_unit(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0_slice: &[f64],
    boundary: BoundaryPolicy,
) -> Result<SemiflowStateInner, SemiflowError> {
    validate_u0_finite(u0_slice)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let chernoff = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0_slice.to_vec())?;
    Ok(SemiflowStateInner { semigroup, current })
}

/// Build a `SemiflowStateInner` with Python-callable coefficients (ADR-0034).
///
/// `a`, `a_prime`, `a_double_prime` are owned closures wrapping Python callables
/// (see `state::make_coeff_closure`).  The `DiffusionChernoff::with_closure`
/// constructor stores them as `Arc<dyn Fn>` for cheap `Clone`.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Grid1D::new`, `GridFn1D::new`, or
/// `ChernoffSemigroup::new`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_heat_closure(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    a_norm_bound: f64,
    u0_slice: &[f64],
    a: CoeffClosure,
    a_prime: CoeffClosure,
    a_double_prime: CoeffClosure,
    boundary: BoundaryPolicy,
) -> Result<SemiflowStateInner, SemiflowError> {
    validate_u0_finite(u0_slice)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let chernoff = DiffusionChernoff::with_closure(a, a_prime, a_double_prime, a_norm_bound, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0_slice.to_vec())?;
    Ok(SemiflowStateInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// 2D state type (wrapped by Heat2D in state.rs)
// ---------------------------------------------------------------------------

/// Heap-allocated 2D Strang operator state owned by `Heat2D`.
pub(crate) struct Semiflow2DStateInner {
    /// The composed `Strang2D` operator (x-axis + y-axis `DiffusionChernoff`).
    pub strang: Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    /// The 2D grid geometry.
    pub grid: Grid2D<f64>,
}

/// Build a `Semiflow2DStateInner` for the 2D heat equation with `a = 1.0`.
///
/// Creates one `DiffusionChernoff` per axis with unit coefficient and
/// composes them via `Strang2D::new`. The same `boundary` policy applies
/// to both axes.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Grid1D::new`.
pub(crate) fn build_heat_2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    boundary: BoundaryPolicy,
) -> Result<Semiflow2DStateInner, SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(boundary);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(boundary);
    let grid = Grid2D::new(gx, gy);
    let dx = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, gx);
    let dy = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, gy);
    let strang = Strang2D::new(dx, dy);
    Ok(Semiflow2DStateInner { strang, grid })
}

/// Apply `n_steps` `Strang2D` steps of size `tau` to the initial flat buffer.
///
/// Called from within the `py.detach` (GIL-released) window of `Heat2D::evolve`.
/// No Python types cross this boundary.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Strang2D::apply` or `GridFn2D::new`.
pub(crate) fn compute_evolve_2d(
    strang: &Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    grid: Grid2D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    let mut state = GridFn2D::new(grid, input)?;
    for _ in 0..n_steps {
        state = strang.apply_chernoff(tau, &state)?;
    }
    Ok(state.values)
}

/// In-place zero-copy variant: advance `state` via `apply_into`, writing final
/// result into `buf`.
///
/// Called from within the `py.detach` window of `Heat2D::evolve_into` (Wave 5,
/// ADR-0045 §1). Avoids the O(N) `Vec<f64>` extraction that `compute_evolve_2d`
/// performs.  `scratch` is allocated once by the caller and reused.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `GridFn2D::new` or `apply_into`.
pub(crate) fn compute_evolve_2d_inplace(
    strang: &Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    grid: Grid2D<f64>,
    buf: &mut [f64],
    tau: f64,
    n_steps: usize,
    scratch: &mut ScratchPool<f64>,
) -> Result<(), SemiflowError> {
    let mut src = GridFn2D::new(grid, buf.to_vec())?;
    let mut dst = GridFn2D::new(grid, vec![0.0; buf.len()])?;
    for _ in 0..n_steps {
        strang.apply_into(tau, &src, &mut dst, scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    buf.copy_from_slice(&src.values);
    Ok(())
}

// ---------------------------------------------------------------------------
// 3D state type (wrapped by Heat3D in state.rs)
// ---------------------------------------------------------------------------

/// Heap-allocated 3D Strang operator state owned by `Heat3D`.
pub(crate) struct Semiflow3DStateInner {
    /// The composed `Strang3D` operator (x/y/z-axis `DiffusionChernoff`).
    pub strang: Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    /// The 3D grid geometry.
    pub grid: Grid3D<f64>,
}

/// Build a `Semiflow3DStateInner` for the 3D heat equation with `a = 1.0`.
///
/// Creates one `DiffusionChernoff` per axis with unit coefficient and
/// composes them via `Strang3D::new`. The same `boundary` policy applies
/// to all three axes.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Grid1D::new` or `Grid3D::new`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_heat_3d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
    boundary: BoundaryPolicy,
) -> Result<Semiflow3DStateInner, SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(boundary);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(boundary);
    let gz = Grid1D::new(zmin, zmax, nz)?.with_boundary(boundary);
    let grid = Grid3D::new(gx, gy, gz)?;
    let dx = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, gx);
    let dy = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, gy);
    let dz = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, gz);
    let strang = Strang3D::new(dx, dy, dz);
    Ok(Semiflow3DStateInner { strang, grid })
}

/// Apply `n_steps` `Strang3D` steps of size `tau` to the initial flat buffer.
///
/// Called from within the `py.detach` (GIL-released) window of `Heat3D::evolve`.
/// No Python types cross this boundary.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `Strang3D::apply` or `GridFn3D::new`.
pub(crate) fn compute_evolve_3d(
    strang: &Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    grid: Grid3D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    let mut state = GridFn3D::new(grid, input)?;
    for _ in 0..n_steps {
        state = strang.apply_chernoff(tau, &state)?;
    }
    Ok(state.values)
}

/// In-place zero-copy variant: advance `state` via `apply_into`, writing final
/// result into `buf`.
///
/// Called from within the `py.detach` window of `Heat3D::evolve_into` (Wave 5,
/// ADR-0045 §1). Avoids the O(N) `Vec<f64>` extraction.
///
/// # Errors
/// Propagates any [`SemiflowError`] from `GridFn3D::new` or `apply_into`.
pub(crate) fn compute_evolve_3d_inplace(
    strang: &Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    grid: Grid3D<f64>,
    buf: &mut [f64],
    tau: f64,
    n_steps: usize,
    scratch: &mut ScratchPool<f64>,
) -> Result<(), SemiflowError> {
    let mut src = GridFn3D::new(grid, buf.to_vec())?;
    let mut dst = GridFn3D::new(grid, vec![0.0; buf.len()])?;
    for _ in 0..n_steps {
        strang.apply_into(tau, &src, &mut dst, scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    buf.copy_from_slice(&src.values);
    Ok(())
}

/// Build a unit-coefficient `DiffusionChernoff` on `grid` (`a ≡ 1`, no drift).
///
/// Convenience for constructing the per-axis kernels of `NonSeparableMixedChernoff`
/// (used by `NonSeparable2D`).
pub(crate) fn unit_diffusion_1d(grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, grid)
}

/// Return `DomainViolation` if any element of `u0` is non-finite.
fn validate_u0_finite(u0: &[f64]) -> Result<(), SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}
