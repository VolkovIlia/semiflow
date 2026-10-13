//! Per-species diffusion engines of the Python reaction–diffusion classes
//! (ADR-0209): `D Δ` by `DiffusionChernoff` per axis (Strang across axes), the exact
//! identity for `D = 0`. With cross-diffusion (ADR-0213) `D` is a mode eigenvalue.

use std::marker::PhantomData;

use semiflow::{
    ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, Grid3D, GridFn1D, GridFn2D, GridFn3D,
    Growth, ScratchPool, SemiflowError as CoreError, SpeciesEngine, State, Strang2D, Strang3D,
};

/// Diffusion engine along one axis (`D > 0`).
fn axis_engine(d: f64, g: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new_const_a(d, d, g)
}

/// The exact `D = 0` "diffusion": the identity (bit-exact, no sampling).
struct Frozen<S>(PhantomData<fn() -> S>);

impl<S: State<f64>> ChernoffFunction<f64> for Frozen<S> {
    type S = S;
    fn apply_into(
        &self,
        _tau: f64,
        src: &S,
        dst: &mut S,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), CoreError> {
        dst.copy_from(src);
        Ok(())
    }
    fn order(&self) -> u32 {
        u32::MAX
    }
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

/// `Frozen` for `D = 0`, else `build()`.
fn engine_or_frozen<S: State<f64> + 'static>(
    d: f64,
    build: impl FnOnce() -> SpeciesEngine<'static, f64, S>,
) -> SpeciesEngine<'static, f64, S> {
    if d == 0.0 {
        Box::new(Frozen::<S>(PhantomData))
    } else {
        build()
    }
}

/// `D_k ∂ₓₓ` engine of a 1-D species.
pub(crate) fn engine_1d(d: f64, g: Grid1D<f64>) -> SpeciesEngine<'static, f64, GridFn1D<f64>> {
    engine_or_frozen(d, || Box::new(axis_engine(d, g)))
}

/// `D_k Δ` engine of a 2-D species (Strang across the axes).
pub(crate) fn engine_2d(d: f64, g: Grid2D<f64>) -> SpeciesEngine<'static, f64, GridFn2D<f64>> {
    engine_or_frozen(d, || {
        Box::new(Strang2D::new(axis_engine(d, g.x), axis_engine(d, g.y)))
    })
}

/// `D_k Δ` engine of a 3-D species (Strang across the axes).
pub(crate) fn engine_3d(
    diff: f64,
    grid: Grid3D<f64>,
) -> SpeciesEngine<'static, f64, GridFn3D<f64>> {
    let axis = |g: Grid1D<f64>| axis_engine(diff, g);
    engine_or_frozen(diff, || {
        Box::new(Strang3D::new(axis(grid.x), axis(grid.y), axis(grid.z)))
    })
}
