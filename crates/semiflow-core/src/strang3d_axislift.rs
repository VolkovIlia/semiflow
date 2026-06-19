//! [`AxisLift3D`] — per-axis adapter lifting a 1D [`ChernoffFunction`] to 3D.
//!
//! Split from `strang3d.rs` in Wave 2 (ADR-0042) to keep `strang3d.rs`
//! within the 700-LoC constitution carve-out.
//!
//! Adds `AxisLift3D::apply_into` override (pool-reuse, 0 allocs per pencil
//! in steady state).  The existing `apply` body is unchanged and remains
//! the reference path for bit-equality tests.
//!
//! **Stride math** (normative, matches `pencil.rs`):
//!
//! | Axis | Stride   | Length | Pencil base                |
//! |------|----------|--------|----------------------------|
//! | X    | 1        | `nx`   | `k*nx*ny + j*nx`           |
//! | Y    | `nx`     | `ny`   | `k*nx*ny + i`              |
//! | Z    | `nx*ny`  | `nz`   | `j*nx + i`                 |

use crate::{
    axis::Axis,
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    grid_fn3d::GridFn3D,
    pencil,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Re-export for backward compat — callers import AxisLift3D from lib.rs
// ---------------------------------------------------------------------------

/// Generic adapter that lifts a 1D Chernoff function `C` to act on
/// [`GridFn3D<F>`] along a single axis (`Axis::X`, `Axis::Y`, or `Axis::Z`).
///
/// Implements [`ChernoffFunction<F>`]`<S = GridFn3D<F>>` so that [`crate::Strang3D`]
/// can compose three `AxisLift3D`s.  Per-axis order and growth are inherited
/// from the inner `C`.
///
/// **Public composition primitive**: callers may use `AxisLift3D` directly to
/// build custom 3D operators outside [`crate::Strang3D`]. This is a first-class
/// public API frozen at v1.0.0.
///
/// ## Wave 2 (ADR-0042): `apply_into` override
///
/// `apply_into` bypasses the allocating `apply` path by calling the inner
/// `C::apply_into` per pencil, with per-pencil scratch borrowed from the
/// caller's `ScratchPool`.  In steady state (pool warmed up), 0 allocations
/// per pencil.
///
/// ## Generic-over-Float (ADR-0025)
///
/// `AxisLift3D<C, F: SemiflowFloat = f64>` — the `= f64` default keeps all
/// existing call-sites compiling unchanged.
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `AxisLift3D`.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
pub struct AxisLift3D<C, F: SemiflowFloat = f64> {
    /// The 1D Chernoff function being lifted.
    pub inner: C,
    /// Which axis the lift acts on (`Axis::X`, `Axis::Y`, or `Axis::Z`).
    pub axis: Axis,
    /// Float type marker.
    _float: core::marker::PhantomData<F>,
}

impl<C, F: SemiflowFloat> AxisLift3D<C, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    /// Construct an `AxisLift3D<C, F>`.
    #[must_use]
    pub fn new(inner: C, axis: Axis) -> Self {
        Self {
            inner,
            axis,
            _float: core::marker::PhantomData,
        }
    }
}

impl<C, F: SemiflowFloat> ChernoffFunction<F> for AxisLift3D<C, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    type S = GridFn3D<F>;

    /// Wave 2 (ADR-0042): pool-reuse `apply_into` override.
    ///
    /// Per-pencil scratch is borrowed from `scratch`.  In steady state,
    /// 0 allocations per pencil.  Palindromic leg order in [`crate::Strang3D`]
    /// is preserved; only the storage strategy changes.
    ///
    /// # Errors
    /// Propagates any error from `self.inner.apply_into`.
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn3D<F>,
        dst: &mut GridFn3D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let nx = src.grid.nx();
        let ny = src.grid.ny();
        let nz = src.grid.nz();
        if dst.values.len() != src.values.len() {
            dst.values.resize(src.values.len(), F::zero());
        }
        dst.grid = src.grid;
        match self.axis {
            Axis::X => apply_into_x_3d(
                &self.inner,
                tau,
                &src.values,
                &mut dst.values,
                nx,
                ny,
                nz,
                src.grid.x,
                scratch,
            ),
            Axis::Y => apply_into_y_3d(
                &self.inner,
                tau,
                &src.values,
                &mut dst.values,
                nx,
                ny,
                nz,
                src.grid.y,
                scratch,
            ),
            Axis::Z => apply_into_z_3d(
                &self.inner,
                tau,
                &src.values,
                &mut dst.values,
                nx,
                ny,
                nz,
                src.grid.z,
                scratch,
            ),
        }
    }

    fn order(&self) -> u32 {
        self.inner.order()
    }

    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Pool-reuse apply_into helpers (Wave 2, ADR-0042)
// ---------------------------------------------------------------------------

/// X-pass: contiguous pencils — use slice directly via `apply_into_via_view`.
#[allow(clippy::too_many_arguments)]
fn apply_into_x_3d<C, F: SemiflowFloat>(
    inner: &C,
    tau: F,
    src: &[F],
    dst: &mut [F],
    nx: usize,
    ny: usize,
    nz: usize,
    gx: crate::grid::Grid1D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    for k in 0..nz {
        for j in 0..ny {
            let src_p = pencil::pencil_x_3d(src, nx, ny, j, k);
            let dst_p = pencil::pencil_x_3d_mut(dst, nx, ny, j, k);
            crate::grid_fn::apply_into_via_view(inner, tau, src_p, dst_p, gx, scratch)?;
        }
    }
    Ok(())
}

/// Y-pass: strided pencils — gather, `apply_into`, scatter with `mem::take` reuse.
#[allow(clippy::too_many_arguments)]
fn apply_into_y_3d<C, F: SemiflowFloat>(
    inner: &C,
    tau: F,
    src: &[F],
    dst: &mut [F],
    nx: usize,
    ny: usize,
    nz: usize,
    gy: crate::grid::Grid1D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let mut src_col = scratch.take_vec(ny);
    let mut dst_col = scratch.take_vec(ny);
    for k in 0..nz {
        for i in 0..nx {
            pencil::gather_y_3d_into(src, nx, ny, i, k, &mut src_col);
            let src_gf = GridFn1D {
                values: core::mem::take(&mut src_col),
                grid: gy,
            };
            let mut dst_gf = GridFn1D {
                values: core::mem::take(&mut dst_col),
                grid: gy,
            };
            inner.apply_into(tau, &src_gf, &mut dst_gf, scratch)?;
            pencil::scatter_y_3d_from(dst, nx, ny, i, k, &dst_gf.values);
            src_col = src_gf.values;
            dst_col = dst_gf.values;
        }
    }
    scratch.return_vec(src_col);
    scratch.return_vec(dst_col);
    Ok(())
}

/// Z-pass: strided pencils — gather, `apply_into`, scatter with `mem::take` reuse.
#[allow(clippy::too_many_arguments)]
fn apply_into_z_3d<C, F: SemiflowFloat>(
    inner: &C,
    tau: F,
    src: &[F],
    dst: &mut [F],
    nx: usize,
    ny: usize,
    nz: usize,
    gz: crate::grid::Grid1D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let mut src_col = scratch.take_vec(nz);
    let mut dst_col = scratch.take_vec(nz);
    for j in 0..ny {
        for i in 0..nx {
            pencil::gather_z_3d_into(src, nx, ny, nz, i, j, &mut src_col);
            let src_gf = GridFn1D {
                values: core::mem::take(&mut src_col),
                grid: gz,
            };
            let mut dst_gf = GridFn1D {
                values: core::mem::take(&mut dst_col),
                grid: gz,
            };
            inner.apply_into(tau, &src_gf, &mut dst_gf, scratch)?;
            pencil::scatter_z_3d_from(dst, nx, ny, nz, i, j, &dst_gf.values);
            src_col = src_gf.values;
            dst_col = dst_gf.values;
        }
    }
    scratch.return_vec(src_col);
    scratch.return_vec(dst_col);
    Ok(())
}
