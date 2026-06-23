//! [`Axis`] discriminator and [`AxisLift`] per-axis adapter.
//!
//! `AxisLift<C, F>` lifts a 1D [`ChernoffFunction`]`<F, S = GridFn1D<F>>` to
//! act on [`GridFn2D<F>`] row-by-row (X) or column-by-column (Y), reusing
//! the 1D kernel verbatim. This is the core adapter enabling the tensor-product
//! 2D extension (ADR-0012).
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 4)
//!
//! `AxisLift<C, F: SemiflowFloat = f64>` — the `= f64` default keeps all
//! existing call-sites compiling unchanged.
//!
//! See `contracts/semiflow-core.tensor.yaml` §2 (`Axis`) and §3 (`AxisLift`),
//! invariant I-T4, and `docs/adr/0012-tensor-product-2d.md`.

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    pencil,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Axis
// ---------------------------------------------------------------------------

/// Discriminator for the per-axis lift.
///
/// `Axis::X` lifts a 1D Chernoff function to act row-by-row at fixed `j`
/// (X is the fast axis, stride = 1, count = nx, repetitions = ny).
///
/// `Axis::Y` lifts column-by-column at fixed `i`
/// (stride = nx, count = ny, repetitions = nx).
///
/// `Axis::Z` lifts slab-by-slab at fixed `(i, j)`
/// (stride = nx·ny, count = nz, repetitions = nx·ny).
/// Added in v0.9.0 Block C for 3D tensor-product splitting (`Strang3D`).
///
/// # Option A rationale (ADR-0024)
/// The `Axis::Z` variant extends the existing enum rather than introducing a
/// separate `Axis3` type. Only one match arm in `axis.rs` required updating
/// (≤5 threshold); the 2D `AxisLift<C>` struct reuses `Axis::X` / `Axis::Y`
/// unchanged. `Strang3D` passes `Axis::Z` to a *separate* `AxisLift3D` struct
/// (defined in `strang3d.rs`) so no 2D-path code changes were necessary.
///
/// See `contracts/semiflow-core.tensor.yaml` §2 `Axis`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Lift acts row-wise: stride = 1, count = nx, repetitions = ny.
    X,
    /// Lift acts column-wise: stride = nx, count = ny, repetitions = nx.
    Y,
    /// Lift acts slab-wise in 3D: stride = nx·ny, count = nz, repetitions = nx·ny.
    /// Used by [`crate::AxisLift3D`] and [`crate::Strang3D`] (v0.9.0).
    Z,
}

// ---------------------------------------------------------------------------
// AxisLift
// ---------------------------------------------------------------------------

/// Generic adapter that lifts a 1D Chernoff function `C` to act on
/// [`GridFn2D<F>`] along a single axis.
///
/// Implements [`ChernoffFunction<F>`]`<S = GridFn2D<F>>` so that
/// [`crate::Strang2D`] can compose two `AxisLift`s. Per-axis order and
/// growth are inherited from the inner `C`.
///
/// ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 4)
///
/// `AxisLift<C, F: SemiflowFloat = f64>` — the `= f64` default keeps all
/// existing call-sites compiling unchanged.
///
/// # Example
///
/// ```rust
/// use semiflow_core::{
///     chernoff::ApplyChernoffExt, Grid1D, Grid2D, GridFn2D, DiffusionChernoff, AxisLift, Axis,
/// };
/// let gx = Grid1D::new(-3.0, 3.0, 16).unwrap();
/// let gy = Grid1D::new(-3.0, 3.0, 16).unwrap();
/// let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx);
/// let lift = AxisLift::new(diff, Axis::X);
/// let grid2 = Grid2D::new(gx, gy);
/// let u0 = GridFn2D::from_fn(grid2, |x, y| (-(x * x + y * y)).exp());
/// let u1 = lift.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 16 * 16);
/// ```
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `AxisLift`,
/// invariant I-T4, and `math.md` §10 Lemma 10.2.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug)]
pub struct AxisLift<C, F: SemiflowFloat = f64> {
    /// The 1D Chernoff function being lifted. Must implement
    /// `ChernoffFunction<F, S = GridFn1D<F>>`.
    pub inner: C,
    /// Which axis the lift acts on (X or Y).
    pub axis: Axis,
    /// Float type marker.
    _float: core::marker::PhantomData<F>,
}

impl<C, F: SemiflowFloat> AxisLift<C, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    /// Construct an `AxisLift`.
    ///
    /// No validation — `inner` is already a valid [`ChernoffFunction<F>`].
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `AxisLift::new`.
    #[must_use]
    pub fn new(inner: C, axis: Axis) -> Self {
        Self {
            inner,
            axis,
            _float: core::marker::PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<C, F: SemiflowFloat> ChernoffFunction<F> for AxisLift<C, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    type S = GridFn2D<F>;

    /// In-place allocation-free override (Wave 2, ADR-0042).
    ///
    /// Applies the lifted kernel per-pencil (row or column), reusing pool
    /// buffers from `scratch`. Zero heap allocations per call in steady state
    /// (after the pool warms up).
    ///
    /// - X-pass: contiguous row slices via `pencil::row_2d_mut`.
    /// - Y-pass: strided gather/scatter per column; `core::mem::take` reclaim.
    ///
    /// # Errors
    /// Same conditions as [`AxisLift::apply`].
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn2D<F>,
        dst: &mut GridFn2D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        validate_axis_lift_input(tau, src)?;
        dst.grid = src.grid;
        if dst.values.len() != src.values.len() {
            dst.values.resize(src.values.len(), F::zero());
        }
        let nx = src.grid.nx();
        let ny = src.grid.ny();
        match self.axis {
            Axis::X => apply_into_x(
                &self.inner,
                tau,
                &src.values,
                &mut dst.values,
                nx,
                ny,
                src.grid.x,
                scratch,
            ),
            Axis::Y => apply_into_y(
                &self.inner,
                tau,
                &src.values,
                &mut dst.values,
                nx,
                ny,
                src.grid.y,
                scratch,
            ),
            Axis::Z => Err(SemiflowError::DomainViolation {
                what: "AxisLift (2D): Axis::Z is not valid for GridFn2D; \
                       use AxisLift3D for 3D lifts",
                value: 0.0,
            }),
        }
    }

    /// Per-axis lift preserves the lifted axis's consistency order.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `AxisLift::order`.
    fn order(&self) -> u32 {
        self.inner.order()
    }

    /// Per-axis lift preserves the 1D growth bound (orthogonal axis is identity).
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `AxisLift::growth`.
    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Validation helper (extracted batch H9b)
// ---------------------------------------------------------------------------

/// Validate `tau` and shape consistency for `AxisLift::apply_into`.
#[allow(clippy::cast_precision_loss)]
fn validate_axis_lift_input<F: SemiflowFloat>(
    tau: F,
    src: &GridFn2D<F>,
) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "AxisLift::apply_into: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    if src.values.len() != src.grid.len() {
        return Err(SemiflowError::DomainViolation {
            what: "AxisLift::apply_into: f.values.len() != f.grid.len() (I-T3)",
            value: src.values.len() as f64,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Wave 2 in-place per-pencil helpers (ADR-0042)
// ---------------------------------------------------------------------------

/// X-pass in-place: apply `inner` row-by-row using pool buffers.
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_into_x<C, F: SemiflowFloat>(
    inner: &C,
    tau: F,
    src: &[F],
    dst: &mut [F],
    nx: usize,
    ny: usize,
    gx: crate::grid::Grid1D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    for j in 0..ny {
        let src_row = pencil::row_2d(src, nx, j);
        let dst_row = pencil::row_2d_mut(dst, nx, j);
        crate::grid_fn::apply_into_via_view(inner, tau, src_row, dst_row, gx, scratch)?;
    }
    Ok(())
}

/// Y-pass in-place: apply `inner` column-by-column using pool buffers.
///
/// Strided gather/scatter with `core::mem::take` reclaim (mirrors the
/// Block A column-buf reuse in `strang2d_parallel.rs:147-155`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_into_y<C, F: SemiflowFloat>(
    inner: &C,
    tau: F,
    src: &[F],
    dst: &mut [F],
    nx: usize,
    ny: usize,
    gy: crate::grid::Grid1D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let mut src_col = scratch.take_vec(ny);
    let mut dst_col = scratch.take_vec(ny);
    for i in 0..nx {
        pencil::gather_y_2d_into(src, nx, ny, i, &mut src_col);
        let src_gf = GridFn1D {
            values: core::mem::take(&mut src_col),
            grid: gy,
        };
        let mut dst_gf = GridFn1D {
            values: core::mem::take(&mut dst_col),
            grid: gy,
        };
        inner.apply_into(tau, &src_gf, &mut dst_gf, scratch)?;
        pencil::scatter_y_2d_from(dst, nx, ny, i, &dst_gf.values);
        src_col = src_gf.values;
        dst_col = dst_gf.values;
    }
    scratch.return_vec(src_col);
    scratch.return_vec(dst_col);
    Ok(())
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{diffusion::DiffusionChernoff, grid::Grid1D, grid2d::Grid2D};

    fn make_lift() -> (AxisLift<DiffusionChernoff>, GridFn2D) {
        let gx = Grid1D::new(-3.0, 3.0, 16).unwrap();
        let gy = Grid1D::new(-3.0, 3.0, 12).unwrap();
        let g2 = Grid2D::new(gx, gy);
        let f = GridFn2D::from_fn(g2, |x, _y| (-x * x).exp());
        let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx);
        let lift = AxisLift::new(diff, Axis::X);
        (lift, f)
    }

    #[test]
    fn order_and_growth() {
        let (lift, _) = make_lift();
        assert_eq!(lift.order(), 2);
        let g = lift.growth();
        assert!(g.multiplier >= 1.0);
        assert!(g.omega.is_finite());
    }

    #[test]
    fn apply_x_preserves_shape() {
        use crate::chernoff::ApplyChernoffExt;
        let (lift, f) = make_lift();
        let out = lift.apply_chernoff(0.01, &f).unwrap();
        assert_eq!(out.values.len(), f.values.len());
        assert_eq!(out.grid, f.grid);
    }

    #[test]
    fn apply_y_preserves_shape() {
        use crate::chernoff::ApplyChernoffExt;
        let gx = Grid1D::new(-3.0, 3.0, 8).unwrap();
        let gy = Grid1D::new(-3.0, 3.0, 10).unwrap();
        let g2 = Grid2D::new(gx, gy);
        let f = GridFn2D::from_fn(g2, |_x, y| (-y * y).exp());
        let diff_y = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gy);
        let lift_y = AxisLift::new(diff_y, Axis::Y);
        let out = lift_y.apply_chernoff(0.01, &f).unwrap();
        assert_eq!(out.values.len(), f.values.len());
        assert_eq!(out.grid, f.grid);
    }

    #[test]
    fn neg_tau_rejected() {
        use crate::chernoff::ApplyChernoffExt;
        let (lift, f) = make_lift();
        assert!(lift.apply_chernoff(-1.0, &f).is_err());
    }
}
