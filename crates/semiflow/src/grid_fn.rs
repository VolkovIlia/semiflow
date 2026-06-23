//! [`GridFn1D`] — a function sampled on a [`Grid1D`], implementing [`State`].
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D pilot)
//!
//! `GridFn1D<F: SemiflowFloat = f64>` — the `= f64` default keeps all existing
//! call-sites compiling unchanged.
//!
//! ### Backward-compatible API
//!
//! Existing code using `GridFn1D::new(grid, values)` and `GridFn1D::from_fn(grid, f)`
//! on f64 grids is NOT affected: concrete f64 implementations are provided on a
//! separate `impl GridFn1D<f64>` block with the original `f64` signatures.
//!
//! For non-f64 types, use `GridFn1D::<F>::new_generic(grid, values)` and
//! `GridFn1D::<F>::from_fn_generic(grid, f)`.

use alloc::{vec, vec::Vec};

use crate::{
    chernoff::ChernoffFunction, error::SemiflowError, float::SemiflowFloat, grid::Grid1D,
    scratch::ScratchPool, state::State,
};

/// A function sampled on a uniform 1-D grid.
///
/// Holds `values[i] ≈ f(x_i)` for `i = 0..n`. Implements [`State`] for use
/// in the Chernoff iteration.
///
/// ## Generic-over-Float (ADR-0025)
///
/// `GridFn1D<F: SemiflowFloat = f64>` — the `= f64` default keeps all existing
/// call-sites compiling unchanged via the concrete `impl GridFn1D<f64>` block.
///
/// # Example
///
/// ```rust
/// use semiflow_core::{Grid1D, GridFn1D, State};
/// let grid = Grid1D::new(-2.0, 2.0, 32).unwrap();
/// let u = GridFn1D::from_fn(grid, |x| x * x);
/// assert_eq!(u.values.len(), 32);
/// assert!((u.norm_sup() - 4.0).abs() < 1e-12); // max x² on [-2,2] = 4
/// let sample = u.sample(1.0).unwrap();
/// assert!((sample - 1.0).abs() < 1e-4);
/// ```
#[derive(Debug, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct GridFn1D<F: SemiflowFloat = f64> {
    /// Function values at grid nodes. Length equals `grid.n`.
    pub values: Vec<F>,
    /// Grid geometry (owned; cheap to clone because `Grid1D: Copy`).
    pub grid: Grid1D<F>,
}

// ---------------------------------------------------------------------------
// Generic impl — for non-f64 types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> GridFn1D<F> {
    /// Construct with validation (generic version for non-f64 types).
    ///
    /// For `F = f64`, the backward-compatible `GridFn1D::new` on
    /// `impl GridFn1D<f64>` should be preferred.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `values.len() != grid.n`.
    /// - [`SemiflowError::DomainViolation`] if any value is `NaN` or `Inf`.
    pub fn new_generic(grid: Grid1D<F>, values: Vec<F>) -> Result<Self, SemiflowError> {
        if values.len() != grid.n {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "values.len() must equal grid.n",
                value: values.len() as f64,
            });
        }
        if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
            return Err(SemiflowError::DomainViolation {
                what: "all values must be finite (no NaN/Inf)",
                value: bad.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self { values, grid })
    }

    /// Sample a closure at every grid node (generic version for non-f64 types).
    ///
    /// For `F = f64`, the backward-compatible `GridFn1D::from_fn` on
    /// `impl GridFn1D<f64>` should be preferred.
    pub fn from_fn_generic<C: FnMut(F) -> F>(grid: Grid1D<F>, mut f: C) -> Self {
        let values: Vec<F> = (0..grid.n).map(|i| f(grid.x_at(i))).collect();
        Self { values, grid }
    }

    /// Evaluate the represented function at arbitrary `x` via boundary +
    /// interpolation dispatch (generic, scalar-only path).
    ///
    /// For `F = f64`, the `sample` method on `impl GridFn1D<f64>` uses the
    /// SIMD-capable path (including `SepticHermite` and `OctonicHermite`).
    ///
    /// # Errors
    /// Propagates [`SemiflowError::Unsupported`] for unimplemented policies.
    pub fn sample_generic(&self, x: F) -> Result<F, SemiflowError> {
        self.grid.interp_generic(&self.values, x)
    }
}

// ---------------------------------------------------------------------------
// Concrete impl for GridFn1D<f64> — backward-compatible API + SIMD paths
// ---------------------------------------------------------------------------

impl GridFn1D<f64> {
    /// Construct with validation (backward-compatible f64 version).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `values.len() != grid.n`.
    /// - [`SemiflowError::DomainViolation`] if any value is `NaN` or `Inf`.
    pub fn new(grid: Grid1D<f64>, values: Vec<f64>) -> Result<Self, SemiflowError> {
        if values.len() != grid.n {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "values.len() must equal grid.n",
                value: values.len() as f64,
            });
        }
        if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
            return Err(SemiflowError::DomainViolation {
                what: "all values must be finite (no NaN/Inf)",
                value: *bad,
            });
        }
        Ok(Self { values, grid })
    }

    /// Sample a closure at every grid node and return a `GridFn1D<f64>`.
    ///
    /// Backward-compatible signature: takes `impl Fn(f64) -> f64` which
    /// preserves type inference at existing call-sites (e.g.
    /// `GridFn1D::from_fn(grid, |x| (-x * x).exp())`).
    pub fn from_fn<C: FnMut(f64) -> f64>(grid: Grid1D<f64>, mut f: C) -> Self {
        let values: Vec<f64> = (0..grid.n).map(|i| f(grid.x_at(i))).collect();
        Self { values, grid }
    }

    /// Evaluate the represented function at arbitrary `x` via boundary +
    /// interpolation dispatch (f64 path with SIMD `catmull_rom`, `SepticHermite`, and `OctonicHermite`).
    ///
    /// # Errors
    /// Propagates [`SemiflowError::Unsupported`] for unimplemented policies.
    pub fn sample(&self, x: f64) -> Result<f64, SemiflowError> {
        self.grid.interp(&self.values, x)
    }
}

// ---------------------------------------------------------------------------
// Wave 2 slice-level helper (ADR-0042)
// ---------------------------------------------------------------------------

/// Apply `inner` to a source slice `src_slot`, writing into `dst_slot`,
/// using pool buffers to avoid per-call allocation.
///
/// Wraps the `ChernoffFunction<F, S = GridFn1D<F>>::apply_into` interface
/// for callers that hold slices rather than `GridFn1D` structs (i.e.
/// `AxisLift::apply_into` and `AxisLift3D::apply_into`).
///
/// Both `src_slot` and `dst_slot` must have length `n == grid.n`.
/// The function borrows two temporary `Vec<F>` from `scratch`; in steady
/// state, both are served from the pool free-list — 0 allocations.
///
/// # Errors
/// Propagates any error returned by `inner.apply_into`.
#[allow(clippy::module_name_repetitions)]
pub(crate) fn apply_into_via_view<C, F>(
    inner: &C,
    tau: F,
    src_slot: &[F],
    dst_slot: &mut [F],
    grid: Grid1D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let n = src_slot.len();
    // Borrow two pool buffers.
    let mut src_vec = scratch.take_vec(n);
    src_vec.copy_from_slice(src_slot);
    let mut src_gf = GridFn1D {
        values: src_vec,
        grid,
    };
    let dst_vec = scratch.take_vec(n);
    let mut dst_gf = GridFn1D {
        values: dst_vec,
        grid,
    };
    // Fire the Wave 1 apply_into override on the leaf kernel.
    inner.apply_into(tau, &src_gf, &mut dst_gf, scratch)?;
    dst_slot.copy_from_slice(&dst_gf.values);
    // Reclaim pool buffers.
    scratch.return_vec(core::mem::take(&mut src_gf.values));
    scratch.return_vec(dst_gf.values);
    Ok(())
}

// ---------------------------------------------------------------------------
// State<F> + HilbertState<F> impl for GridFn1D<F> (Wave 3, ADR-0043)
// ---------------------------------------------------------------------------

crate::impl_state_for_gridfn!(GridFn1D<F>);

// ---------------------------------------------------------------------------
// v1.x source-compatibility inherent methods
// ---------------------------------------------------------------------------
//
// These are NOT trait methods. Concrete GridFn1D<F> callers can continue to
// use `u.axpy(a, &v)`, `u.scale(k)`, and `u.zeroed_like()` unchanged.
// Generic bounds `T: State<F>` must migrate to `axpy_into`/`scale_into`.

impl<F: SemiflowFloat> GridFn1D<F> {
    /// v1.x compat shim: `self ← self + a · x`. Delegates to `axpy_into`.
    #[inline]
    pub fn axpy(&mut self, a: F, x: &Self) {
        State::axpy_into(self, a, x);
    }

    /// v1.x compat shim: `self ← k · self`. Delegates to `scale_into`.
    #[inline]
    pub fn scale(&mut self, k: F) {
        State::scale_into(self, k);
    }

    /// v1.x compat shim: allocate same-shape zero state.
    ///
    /// Allocates `O(N)`. In hot paths prefer:
    /// `let mut z = scratch.take_vec(n); z.fill(F::zero());`
    #[must_use]
    #[inline]
    pub fn zeroed_like(&self) -> Self {
        Self {
            values: vec![F::zero(); self.values.len()],
            grid: self.grid,
        }
    }
}
