//! [`NodalField`] for the library's states: grids in 1-D/2-D/3-D and graph signals.
//!
//! [`NodalField::mixes_with`] (cross-diffusion, ADR-0213): two grid fields mix when
//! every axis has the same geometry, interpolant and boundary policy, and that policy
//! is homogeneous linear — every policy except `Dirichlet` with a non-zero value,
//! whose constant ghost value a species mixing would scale. Two graph signals mix on
//! the same graph (one shared `Arc`).

use alloc::vec::Vec;

use super::NodalField;
use crate::{
    boundary::BoundaryPolicy, float::SemiflowFloat, graph_signal::GraphSignal, grid::Grid1D,
    grid_fn::GridFn1D, grid_fn2d::GridFn2D, grid_fn3d::GridFn3D,
};

/// Same axis geometry, interpolant and boundary, and that boundary is homogeneous
/// linear (a non-zero Dirichlet value is affine).
fn axis_mixes<F: SemiflowFloat>(a: &Grid1D<F>, b: &Grid1D<F>) -> bool {
    let homogeneous =
        !matches!(a.boundary, BoundaryPolicy::Dirichlet { value } if value != F::zero());
    homogeneous
        && a.xmin == b.xmin
        && a.xmax == b.xmax
        && a.n == b.n
        && a.boundary == b.boundary
        && a.interp == b.interp
}

impl<F: SemiflowFloat> NodalField<F> for GridFn1D<F> {
    fn node_values(&self) -> &[F] {
        &self.values
    }
    fn node_values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
    fn node_coordinates(&self) -> Vec<F> {
        (0..self.values.len()).map(|i| self.grid.x_at(i)).collect()
    }
    fn mixes_with(&self, other: &Self) -> bool {
        axis_mixes(&self.grid, &other.grid)
    }
}

impl<F: SemiflowFloat> NodalField<F> for GridFn2D<F> {
    fn node_values(&self) -> &[F] {
        &self.values
    }
    fn node_values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
    /// Row-major, `x` fastest: node `iy·nx + ix`.
    fn node_coordinates(&self) -> Vec<F> {
        let (gx, gy) = (self.grid.x, self.grid.y);
        let (xs, ys): (Vec<F>, Vec<F>) = (0..gy.n)
            .flat_map(|iy| (0..gx.n).map(move |ix| (gx.x_at(ix), gy.x_at(iy))))
            .unzip();
        [xs, ys].concat()
    }
    fn mixes_with(&self, other: &Self) -> bool {
        axis_mixes(&self.grid.x, &other.grid.x) && axis_mixes(&self.grid.y, &other.grid.y)
    }
}

impl<F: SemiflowFloat> NodalField<F> for GridFn3D<F> {
    fn node_values(&self) -> &[F] {
        &self.values
    }
    fn node_values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
    /// `x` fastest, then `y`, then `z`: node `(iz·ny + iy)·nx + ix`.
    fn node_coordinates(&self) -> Vec<F> {
        let (gx, gy, gz) = (self.grid.x, self.grid.y, self.grid.z);
        let n = gx.n * gy.n * gz.n;
        let mut out = Vec::with_capacity(3 * n);
        let node = |i: usize| (i % gx.n, (i / gx.n) % gy.n, i / (gx.n * gy.n));
        out.extend((0..n).map(|i| gx.x_at(node(i).0)));
        out.extend((0..n).map(|i| gy.x_at(node(i).1)));
        out.extend((0..n).map(|i| gz.x_at(node(i).2)));
        out
    }
    fn mixes_with(&self, other: &Self) -> bool {
        let (a, b) = (&self.grid, &other.grid);
        axis_mixes(&a.x, &b.x) && axis_mixes(&a.y, &b.y) && axis_mixes(&a.z, &b.z)
    }
}

impl<F: SemiflowFloat> NodalField<F> for GraphSignal<F> {
    fn node_values(&self) -> &[F] {
        &self.values
    }
    fn node_values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
    fn node_coordinates(&self) -> Vec<F> {
        Vec::new()
    }
    /// The same graph (one shared `Arc<Graph>`): graph engines are linear.
    fn mixes_with(&self, other: &Self) -> bool {
        core::ptr::eq(self.graph(), other.graph())
    }
}
