//! [`GridFn3D`] — a function sampled on a [`Grid3D`], implementing [`State`].
//!
//! Holds a single `Vec<F>` of length `nx * ny * nz`, indexed by
//! `idx(i, j, k) = k * nx * ny + j * nx + i` (I-T1-3D, x-fastest row-major).
//!
//! All [`State`] operations (`axpy`, `scale`, `norm_sup`, `zeroed_like`)
//! operate on the flat `Vec<F>` — O(nx * ny * nz) with no branching on (i, j, k).
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 3)
//!
//! `GridFn3D<F: SemiflowFloat = f64>` — the `= f64` default keeps all existing
//! call-sites compiling unchanged. Generic `*_generic` constructors mirror the
//! Wave-2 pattern for non-f64 types.
//!
//! See `contracts/semiflow-core.tensor.yaml` §3 `GridFn3D`,
//! `docs/adr/0024-tensor-3d.md`, and `contracts/semiflow-core.math.md` §10.8.

use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError, float::SemiflowFloat, grid3d::Grid3D, grid_fn::GridFn1D, state::State,
};

// ---------------------------------------------------------------------------
// GridFn3D
// ---------------------------------------------------------------------------

/// A function sampled on a 3D tensor-product grid.
///
/// Flat x-fastest row-major storage: `values[k*nx*ny + j*nx + i] ≈ f(x_i, y_j, z_k)`.
/// Single `Vec<F>` allocation (no nested `Vec` — mirrors ADR-0012 for 2D).
///
/// Implements [`State<F>`] for use in the Chernoff iteration via [`crate::Strang3D`].
///
/// ## Generic-over-Float (ADR-0025, Wave 3)
///
/// `GridFn3D<F: SemiflowFloat = f64>` — `= f64` default keeps existing call-sites
/// unchanged. For f32 grids use `GridFn3D::<f32>` explicitly.
///
/// # Example
///
/// ```rust
/// use semiflow::{Grid1D, Grid3D, GridFn3D};
/// let g = Grid1D::new(-1.0, 1.0, 8).unwrap();
/// let grid = Grid3D::new(g, g, g).unwrap();
/// let u = GridFn3D::from_fn(grid, |x, y, z| x * x + y * y + z * z);
/// assert_eq!(u.values.len(), 512);
/// ```
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn3D`.
#[derive(Debug, Clone)]
pub struct GridFn3D<F: SemiflowFloat = f64> {
    /// Flat x-fastest storage. Length equals `grid.nx() * grid.ny() * grid.nz()`.
    pub values: Vec<F>,
    /// 3D geometry. Owned by value (cheap to clone — three `Grid1D: Copy`).
    pub grid: Grid3D<F>,
}

// ---------------------------------------------------------------------------
// Generic impl — constructors + pencil access for all SemiflowFloat types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> GridFn3D<F> {
    /// Construct with shape and finiteness validation (generic version).
    ///
    /// For `F = f64`, the backward-compatible `GridFn3D::new` on the concrete
    /// `impl GridFn3D<f64>` block should be preferred at existing call-sites.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `values.len() != grid.len()`.
    /// - [`SemiflowError::DomainViolation`] if any value is `NaN` or `Inf`.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn3D::new`.
    pub fn new_generic(grid: Grid3D<F>, values: Vec<F>) -> Result<Self, SemiflowError> {
        if values.len() != grid.len() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "GridFn3D::new_generic: values.len() must equal nx*ny*nz",
                value: values.len() as f64,
            });
        }
        if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
            return Err(SemiflowError::DomainViolation {
                what: "GridFn3D::new_generic: all values must be finite (no NaN/Inf)",
                value: bad.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self { values, grid })
    }

    /// Sample a 3D closure `f(x, y, z)` at every grid node (generic version).
    ///
    /// Iteration order: k outer, j middle, i inner (x-fastest). Infallible.
    ///
    /// For `F = f64`, the backward-compatible `GridFn3D::from_fn` on the
    /// concrete `impl GridFn3D<f64>` block should be preferred.
    pub fn from_fn_generic<C: Fn(F, F, F) -> F>(grid: Grid3D<F>, f: C) -> Self {
        let nx = grid.nx();
        let ny = grid.ny();
        let nz = grid.nz();
        let mut values = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            let zk = grid.z.x_at(k);
            for j in 0..ny {
                let yj = grid.y.x_at(j);
                for i in 0..nx {
                    let xi = grid.x.x_at(i);
                    values.push(f(xi, yj, zk));
                }
            }
        }
        Self { values, grid }
    }

    /// Extract the X-pencil at `(j, k)` as a [`GridFn1D<F>`] on `self.grid.x`.
    ///
    /// The X-pencil at fixed `(j, k)` is the contiguous slice
    /// `values[k*nx*ny + j*nx .. k*nx*ny + (j+1)*nx]`.
    ///
    /// Used by [`crate::Strang3D`] for the X-leg passes.
    #[must_use]
    pub(crate) fn pencil_x_generic(&self, j: usize, k: usize) -> GridFn1D<F> {
        let nx = self.grid.nx();
        let ny = self.grid.ny();
        let start = k * nx * ny + j * nx;
        GridFn1D {
            values: self.values[start..start + nx].to_vec(),
            grid: self.grid.x,
        }
    }

    /// Scatter values from an X-pencil back at `(j, k)` (generic version).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != nx`.
    pub(crate) fn write_pencil_x_generic(
        &mut self,
        j: usize,
        k: usize,
        src: &GridFn1D<F>,
    ) -> Result<(), SemiflowError> {
        let nx = self.grid.nx();
        if src.values.len() != nx {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "GridFn3D::write_pencil_x_generic: src.len() != nx",
                value: src.values.len() as f64,
            });
        }
        let ny = self.grid.ny();
        let start = k * nx * ny + j * nx;
        self.values[start..start + nx].copy_from_slice(&src.values);
        Ok(())
    }

    /// Extract the Y-pencil at `(i, k)` as a [`GridFn1D<F>`] on `self.grid.y`.
    ///
    /// Gathered with stride `nx`.
    #[must_use]
    pub(crate) fn pencil_y_generic(&self, i: usize, k: usize) -> GridFn1D<F> {
        let ny = self.grid.ny();
        let mut vals = Vec::with_capacity(ny);
        for j in 0..ny {
            vals.push(self.values[self.grid.idx(i, j, k)]);
        }
        GridFn1D {
            values: vals,
            grid: self.grid.y,
        }
    }

    /// Scatter values from a Y-pencil back at `(i, k)` (generic version).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != ny`.
    pub(crate) fn write_pencil_y_generic(
        &mut self,
        i: usize,
        k: usize,
        src: &GridFn1D<F>,
    ) -> Result<(), SemiflowError> {
        let ny = self.grid.ny();
        if src.values.len() != ny {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "GridFn3D::write_pencil_y_generic: src.len() != ny",
                value: src.values.len() as f64,
            });
        }
        for j in 0..ny {
            let k_idx = self.grid.idx(i, j, k);
            self.values[k_idx] = src.values[j];
        }
        Ok(())
    }

    /// Extract the Z-pencil at `(i, j)` as a [`GridFn1D<F>`] on `self.grid.z`.
    ///
    /// Gathered with stride `nx*ny`.
    #[must_use]
    pub(crate) fn pencil_z_generic(&self, i: usize, j: usize) -> GridFn1D<F> {
        let nz = self.grid.nz();
        let mut vals = Vec::with_capacity(nz);
        for k in 0..nz {
            vals.push(self.values[self.grid.idx(i, j, k)]);
        }
        GridFn1D {
            values: vals,
            grid: self.grid.z,
        }
    }

    /// Scatter values from a Z-pencil back at `(i, j)` (generic version).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != nz`.
    pub(crate) fn write_pencil_z_generic(
        &mut self,
        i: usize,
        j: usize,
        src: &GridFn1D<F>,
    ) -> Result<(), SemiflowError> {
        let nz = self.grid.nz();
        if src.values.len() != nz {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "GridFn3D::write_pencil_z_generic: src.len() != nz",
                value: src.values.len() as f64,
            });
        }
        for k in 0..nz {
            let k_idx = self.grid.idx(i, j, k);
            self.values[k_idx] = src.values[k];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Concrete backward-compatible impl for GridFn3D<f64>
// ---------------------------------------------------------------------------

// f64 convenience wrappers: called from semiflow-py (handle.rs) and benches;
// rustc only sees dead_code here because it does not cross crate boundaries in this analysis.
#[allow(dead_code)]
impl GridFn3D<f64> {
    /// Construct with shape and finiteness validation (backward-compatible f64).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `values.len() != grid.len()`.
    /// - [`SemiflowError::DomainViolation`] if any value is `NaN` or `Inf`.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn3D::new`.
    pub fn new(grid: Grid3D<f64>, values: Vec<f64>) -> Result<Self, SemiflowError> {
        Self::new_generic(grid, values)
    }

    /// Sample a 3D closure `f(x, y, z)` at every grid node and return a `GridFn3D`.
    ///
    /// Iteration order: k outer, j middle, i inner (x-fastest). Infallible —
    /// the closure is assumed to return finite values; use [`GridFn3D::new`] if
    /// you need validation.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn3D::from_fn`.
    pub fn from_fn<C: Fn(f64, f64, f64) -> f64>(grid: Grid3D<f64>, f: C) -> Self {
        Self::from_fn_generic(grid, f)
    }

    /// Extract the X-pencil at `(j, k)` as a [`GridFn1D`] on `self.grid.x`.
    ///
    /// Used by [`crate::Strang3D`] for the X-leg passes.
    #[must_use]
    pub(crate) fn pencil_x(&self, j: usize, k: usize) -> GridFn1D {
        self.pencil_x_generic(j, k)
    }

    /// Scatter values from a X-pencil back at `(j, k)`.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != nx`.
    pub(crate) fn write_pencil_x(
        &mut self,
        j: usize,
        k: usize,
        src: &GridFn1D,
    ) -> Result<(), SemiflowError> {
        self.write_pencil_x_generic(j, k, src)
    }

    /// Extract the Y-pencil at `(i, k)` as a [`GridFn1D`] on `self.grid.y`.
    ///
    /// Gathered with stride `nx`. Used by [`crate::Strang3D`] for the Y-leg passes.
    #[must_use]
    pub(crate) fn pencil_y(&self, i: usize, k: usize) -> GridFn1D {
        self.pencil_y_generic(i, k)
    }

    /// Scatter values from a Y-pencil back at `(i, k)`.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != ny`.
    pub(crate) fn write_pencil_y(
        &mut self,
        i: usize,
        k: usize,
        src: &GridFn1D,
    ) -> Result<(), SemiflowError> {
        self.write_pencil_y_generic(i, k, src)
    }

    /// Extract the Z-pencil at `(i, j)` as a [`GridFn1D`] on `self.grid.z`.
    ///
    /// Gathered with stride `nx*ny`. Used by [`crate::Strang3D`] for the Z-leg.
    #[must_use]
    pub(crate) fn pencil_z(&self, i: usize, j: usize) -> GridFn1D {
        self.pencil_z_generic(i, j)
    }

    /// Scatter values from a Z-pencil back at `(i, j)`.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != nz`.
    pub(crate) fn write_pencil_z(
        &mut self,
        i: usize,
        j: usize,
        src: &GridFn1D,
    ) -> Result<(), SemiflowError> {
        self.write_pencil_z_generic(i, j, src)
    }
}

// ---------------------------------------------------------------------------
// State<F> + HilbertState<F> impl for GridFn3D<F> (Wave 3, ADR-0043)
// ---------------------------------------------------------------------------

crate::impl_state_for_gridfn!(GridFn3D<F>);

// ---------------------------------------------------------------------------
// v1.x source-compatibility inherent methods
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> GridFn3D<F> {
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
    #[must_use]
    #[inline]
    pub fn zeroed_like(&self) -> Self {
        Self {
            values: vec![F::zero(); self.values.len()],
            grid: self.grid,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid1D;

    fn make_grid() -> Grid3D {
        let x = Grid1D::new(0.0, 1.0, 4).unwrap();
        let y = Grid1D::new(0.0, 2.0, 5).unwrap();
        let z = Grid1D::new(0.0, 3.0, 4).unwrap();
        Grid3D::new(x, y, z).unwrap()
    }

    /// `from_fn_evaluates_correctly` — checks x-fastest storage order.
    #[test]
    fn from_fn_evaluates_correctly() {
        let g = make_grid();
        let nx = g.nx();
        let ny = g.ny();
        // f(x, y, z) = x + 10*y + 100*z
        let f = GridFn3D::from_fn(g, |x, y, z| x + 10.0 * y + 100.0 * z);
        // Check index 0: (i=0, j=0, k=0)
        let x0 = g.x.x_at(0);
        let y0 = g.y.x_at(0);
        let z0 = g.z.x_at(0);
        assert!((f.values[0] - (x0 + 10.0 * y0 + 100.0 * z0)).abs() < 1e-14);
        // Check idx(1, 0, 0) = 1: x-stride = 1
        let x1 = g.x.x_at(1);
        assert!((f.values[1] - (x1 + 10.0 * y0 + 100.0 * z0)).abs() < 1e-14);
        // Check idx(0, 1, 0) = nx: y-stride = nx
        let y1 = g.y.x_at(1);
        assert!((f.values[nx] - (x0 + 10.0 * y1 + 100.0 * z0)).abs() < 1e-14);
        // Check idx(0, 0, 1) = nx*ny: z-stride = nx*ny
        let z1 = g.z.x_at(1);
        assert!((f.values[nx * ny] - (x0 + 10.0 * y0 + 100.0 * z1)).abs() < 1e-14);
    }

    #[test]
    fn state_axpy_scale_norm() {
        let g = make_grid();
        let mut f = GridFn3D::from_fn(g, |_, _, _| 2.0);
        let g2 = GridFn3D::from_fn(g, |_, _, _| 1.0);
        f.axpy(3.0, &g2);
        assert!((f.norm_sup() - 5.0).abs() < 1e-14);
        f.scale(2.0);
        assert!((f.norm_sup() - 10.0).abs() < 1e-14);
        let z = f.zeroed_like();
        assert!(z.norm_sup().abs() < f64::EPSILON);
    }

    #[test]
    fn pencil_roundtrip() {
        let g = make_grid();
        let f = GridFn3D::from_fn(g, |x, y, z| x + 10.0 * y + 100.0 * z);
        let nx = g.nx();
        let ny = g.ny();
        let nz = g.nz();
        // X-pencil at (j=1, k=2)
        let px = f.pencil_x(1, 2);
        assert_eq!(px.values.len(), nx);
        for i in 0..nx {
            assert!((px.values[i] - f.values[g.idx(i, 1, 2)]).abs() < 1e-14);
        }
        // Y-pencil at (i=2, k=1)
        let py = f.pencil_y(2, 1);
        assert_eq!(py.values.len(), ny);
        for j in 0..ny {
            assert!((py.values[j] - f.values[g.idx(2, j, 1)]).abs() < 1e-14);
        }
        // Z-pencil at (i=1, j=2)
        let pz = f.pencil_z(1, 2);
        assert_eq!(pz.values.len(), nz);
        for k in 0..nz {
            assert!((pz.values[k] - f.values[g.idx(1, 2, k)]).abs() < 1e-14);
        }
    }
}
