//! [`GridFn2D`] — a function sampled on a [`Grid2D`], implementing [`State`].
//!
//! Holds a single `Vec<F>` of length `grid.nx * grid.ny`, indexed by
//! `idx(i, j) = j * nx + i` (I-T1, row-major, x is the fast axis).
//!
//! All [`State`] operations (`axpy`, `scale`, `norm_sup`, `zeroed_like`)
//! operate on the flat `Vec<F>` — O(nx * ny) with no branching on (i, j).
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 2)
//!
//! `GridFn2D<F: SemiflowFloat = f64>` — the `= f64` default keeps all existing
//! call-sites compiling unchanged. Generic `*_generic` constructors mirror the
//! Wave-1 pattern for non-f64 types.
//!
//! See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D`,
//! invariants I-T1 and I-T3, and `docs/adr/0012-tensor-product-2d.md`.

use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError, float::SemiflowFloat, grid2d::Grid2D, grid_fn::GridFn1D, state::State,
};

// ---------------------------------------------------------------------------
// GridFn2D
// ---------------------------------------------------------------------------

/// A function sampled on a 2D tensor-product grid.
///
/// Flat row-major storage: `values[j*nx + i] ≈ f(x_i, y_j)`.
/// Single `Vec<F>` allocation (no `Vec<Vec<F>>` — ADR-0012 closed decision).
///
/// Implements [`State<F>`] for use in the Chernoff iteration via [`crate::Strang2D`].
///
/// ## Generic-over-Float (ADR-0025, Wave 2)
///
/// `GridFn2D<F: SemiflowFloat = f64>` — `= f64` default keeps existing call-sites
/// unchanged. For f32 grids use `GridFn2D::<f32>` explicitly.
///
/// # Example
///
/// ```rust
/// use semiflow::{Grid1D, Grid2D, GridFn2D};
/// let gx = Grid1D::new(-1.0, 1.0, 8).unwrap();
/// let gy = Grid1D::new(-1.0, 1.0, 8).unwrap();
/// let grid = Grid2D::new(gx, gy);
/// let u = GridFn2D::from_fn(grid, |x, y| x * x + y * y);
/// assert_eq!(u.values.len(), 64);
/// ```
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D`.
#[derive(Debug, Clone)]
pub struct GridFn2D<F: SemiflowFloat = f64> {
    /// Flat row-major storage. Length equals `grid.nx() * grid.ny()` (I-T3).
    pub values: Vec<F>,
    /// 2D geometry. Owned by value (cheap to clone — two `Grid1D: Copy`).
    pub grid: Grid2D<F>,
}

// ---------------------------------------------------------------------------
// Generic impl — constructors + row/col access for all SemiflowFloat types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> GridFn2D<F> {
    /// Construct with shape and finiteness validation (generic version).
    ///
    /// For `F = f64`, the backward-compatible `GridFn2D::new` on the concrete
    /// `impl GridFn2D<f64>` block should be preferred at existing call-sites.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `values.len() != grid.len()` (I-T3).
    /// - [`SemiflowError::DomainViolation`] if any value is `NaN` or `Inf`.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D::new`.
    pub fn new_generic(grid: Grid2D<F>, values: Vec<F>) -> Result<Self, SemiflowError> {
        if values.len() != grid.len() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "values.len() must equal grid.nx * grid.ny (I-T3)",
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

    /// Sample a 2D closure `f(x, y)` at every grid node (generic version).
    ///
    /// For `F = f64`, the backward-compatible `GridFn2D::from_fn` on the
    /// concrete `impl GridFn2D<f64>` block should be preferred.
    pub fn from_fn_generic<C: Fn(F, F) -> F>(grid: Grid2D<F>, f: C) -> Self {
        let nx = grid.nx();
        let ny = grid.ny();
        let mut values = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            let yj = grid.y.x_at(j);
            for i in 0..nx {
                let xi = grid.x.x_at(i);
                values.push(f(xi, yj));
            }
        }
        Self { values, grid }
    }

    /// Extract the j-th X-row as a [`GridFn1D<F>`] on `self.grid.x`.
    ///
    /// Allocates `nx` elements and copies the contiguous slice
    /// `values[j*nx..(j+1)*nx]`. Used by [`crate::AxisLift`] for the X-pass.
    #[must_use]
    pub fn row_generic(&self, j: usize) -> GridFn1D<F> {
        let nx = self.grid.nx();
        let start = j * nx;
        let row_vals = self.values[start..start + nx].to_vec();
        GridFn1D {
            values: row_vals,
            grid: self.grid.x,
        }
    }

    /// Extract the i-th Y-column as a [`GridFn1D<F>`] on `self.grid.y`.
    ///
    /// Gathers `ny` elements with stride `nx`.
    #[must_use]
    pub fn col_generic(&self, i: usize) -> GridFn1D<F> {
        let ny = self.grid.ny();
        let mut col_vals = Vec::with_capacity(ny);
        for j in 0..ny {
            col_vals.push(self.values[self.grid.idx(i, j)]);
        }
        GridFn1D {
            values: col_vals,
            grid: self.grid.y,
        }
    }

    /// Overwrite the j-th X-row with values from `src` (generic version).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != self.grid.nx()`.
    pub fn write_row_generic(&mut self, j: usize, src: &GridFn1D<F>) -> Result<(), SemiflowError> {
        let nx = self.grid.nx();
        if src.values.len() != nx {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "write_row: src.values.len() must equal grid.nx",
                value: src.values.len() as f64,
            });
        }
        let start = j * nx;
        self.values[start..start + nx].copy_from_slice(&src.values);
        Ok(())
    }

    /// Overwrite the i-th Y-column with values from `src` (generic version).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != self.grid.ny()`.
    pub fn write_col_generic(&mut self, i: usize, src: &GridFn1D<F>) -> Result<(), SemiflowError> {
        let ny = self.grid.ny();
        if src.values.len() != ny {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "write_col: src.values.len() must equal grid.ny",
                value: src.values.len() as f64,
            });
        }
        for j in 0..ny {
            let k = self.grid.idx(i, j);
            self.values[k] = src.values[j];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Concrete backward-compatible impl for GridFn2D<f64>
// ---------------------------------------------------------------------------

impl GridFn2D<f64> {
    /// Construct with shape and finiteness validation (backward-compatible f64).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `values.len() != grid.len()` (I-T3).
    /// - [`SemiflowError::DomainViolation`] if any value is `NaN` or `Inf`.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D::new`.
    pub fn new(grid: Grid2D<f64>, values: Vec<f64>) -> Result<Self, SemiflowError> {
        if values.len() != grid.len() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "values.len() must equal grid.nx * grid.ny (I-T3)",
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

    /// Sample a 2D closure `f(x, y)` at every grid node and return a `GridFn2D`.
    ///
    /// Iteration order: j outer, i inner (row-major). Infallible — the closure
    /// is assumed to return finite values; use [`GridFn2D::new`] if you need
    /// validation.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D::from_fn`.
    pub fn from_fn<C: Fn(f64, f64) -> f64>(grid: Grid2D<f64>, f: C) -> Self {
        let nx = grid.nx();
        let ny = grid.ny();
        let mut values = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            let yj = grid.y.x_at(j);
            for i in 0..nx {
                let xi = grid.x.x_at(i);
                values.push(f(xi, yj));
            }
        }
        Self { values, grid }
    }

    /// Convenience: sample tensor-product data `f(x_i, y_j) = g(x_i) * h(y_j)`.
    ///
    /// Used by Theorem 7 oracles and the `axis_lift_1d_consistency` property.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D::from_separable`.
    pub fn from_separable<G, H>(grid: Grid2D<f64>, g: G, h: H) -> Self
    where
        G: Fn(f64) -> f64,
        H: Fn(f64) -> f64,
    {
        Self::from_fn(grid, |x, y| g(x) * h(y))
    }

    /// Extract the j-th X-row as a [`GridFn1D`] on `self.grid.x`.
    ///
    /// Allocates `nx` floats and copies the contiguous slice
    /// `values[j*nx..(j+1)*nx]`. Used by [`crate::AxisLift`] for the X-pass.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D::row`.
    #[must_use]
    pub fn row(&self, j: usize) -> GridFn1D {
        self.row_generic(j)
    }

    /// Extract the i-th Y-column as a [`GridFn1D`] on `self.grid.y`.
    ///
    /// Gathers `ny` floats with stride `nx`. Used by [`crate::AxisLift`]
    /// for the Y-pass.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D::col`.
    #[must_use]
    pub fn col(&self, i: usize) -> GridFn1D {
        self.col_generic(i)
    }

    /// Overwrite the j-th X-row with values from `src`.
    ///
    /// Single contiguous memcpy. Errors on shape mismatch.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != self.grid.nx()`.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D::write_row`.
    pub fn write_row(&mut self, j: usize, src: &GridFn1D) -> Result<(), SemiflowError> {
        self.write_row_generic(j, src)
    }

    /// Overwrite the i-th Y-column with values from `src`.
    ///
    /// Strided scatter. Errors on shape mismatch.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `src.values.len() != self.grid.ny()`.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `GridFn2D::write_col`.
    pub fn write_col(&mut self, i: usize, src: &GridFn1D) -> Result<(), SemiflowError> {
        self.write_col_generic(i, src)
    }
}

// ---------------------------------------------------------------------------
// State<F> + HilbertState<F> impl for GridFn2D<F> (Wave 3, ADR-0043)
// ---------------------------------------------------------------------------

crate::impl_state_for_gridfn!(GridFn2D<F>);

// ---------------------------------------------------------------------------
// v1.x source-compatibility inherent methods
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> GridFn2D<F> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid1D;

    fn make_grid() -> Grid2D {
        let x = Grid1D::new(0.0, 1.0, 4).unwrap();
        let y = Grid1D::new(0.0, 2.0, 5).unwrap();
        Grid2D::new(x, y)
    }

    #[test]
    fn new_wrong_len() {
        let g = make_grid();
        let err = GridFn2D::new(g, vec![0.0; 3]).unwrap_err();
        matches!(err, SemiflowError::DomainViolation { .. });
    }

    #[test]
    fn from_fn_row_major() {
        let g = make_grid();
        let f = GridFn2D::from_fn(g, |x, y| x + 10.0 * y);
        // values[0] = x_at(0) + 10*y_at(0) = 0 + 0 = 0
        let x0 = g.x.x_at(0);
        let y0 = g.y.x_at(0);
        assert!((f.values[0] - (x0 + 10.0 * y0)).abs() < 1e-14);
        // values[nx] corresponds to j=1, i=0
        let y1 = g.y.x_at(1);
        assert!((f.values[g.nx()] - (x0 + 10.0 * y1)).abs() < 1e-14);
    }

    #[test]
    fn row_col_roundtrip() {
        let g = make_grid();
        let nx = g.nx();
        let ny = g.ny();
        let f = GridFn2D::from_fn(g, |x, y| x + 10.0 * y);
        // row 2
        let r = f.row(2);
        assert_eq!(r.values.len(), nx);
        for i in 0..nx {
            assert!((r.values[i] - f.values[2 * nx + i]).abs() < 1e-14);
        }
        // col 1
        let c = f.col(1);
        assert_eq!(c.values.len(), ny);
        for j in 0..ny {
            assert!((c.values[j] - f.values[g.idx(1, j)]).abs() < 1e-14);
        }
    }

    #[test]
    fn write_row_col() {
        let g = make_grid();
        let nx = g.nx();
        let ny = g.ny();
        let mut f = GridFn2D::from_fn(g, |_, _| 0.0);
        // write_row: set row 1 to ones
        let ones_row = GridFn1D {
            values: vec![1.0; nx],
            grid: g.x,
        };
        f.write_row(1, &ones_row).unwrap();
        for i in 0..nx {
            // Values are assigned as 1.0 exactly; bit-exact equality is correct here.
            assert!((f.values[nx + i] - 1.0).abs() < f64::EPSILON);
        }
        // write_col: set col 2 to twos
        let twos_col = GridFn1D {
            values: vec![2.0; ny],
            grid: g.y,
        };
        f.write_col(2, &twos_col).unwrap();
        for j in 0..ny {
            // Values are assigned as 2.0 exactly; bit-exact equality is correct here.
            assert!((f.values[g.idx(2, j)] - 2.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn state_axpy_scale_norm() {
        let g = make_grid();
        let mut f = GridFn2D::from_fn(g, |_, _| 2.0);
        let g2 = GridFn2D::from_fn(g, |_, _| 1.0);
        f.axpy(3.0, &g2);
        // all values = 2 + 3*1 = 5
        assert!((f.norm_sup() - 5.0).abs() < 1e-14);
        f.scale(2.0);
        assert!((f.norm_sup() - 10.0).abs() < 1e-14);
        let z = f.zeroed_like();
        // zeroed_like produces exactly 0.0; compare against zero tolerance.
        assert!(z.norm_sup().abs() < f64::EPSILON);
    }

    #[test]
    fn gridfn2d_f32_generic() {
        // Verify that GridFn2D<f32> compiles and basic ops work.
        let gx = Grid1D::<f32>::new_generic(0.0_f32, 1.0_f32, 4).unwrap();
        let gy = Grid1D::<f32>::new_generic(0.0_f32, 2.0_f32, 5).unwrap();
        let g = Grid2D::<f32>::new(gx, gy);
        let f = GridFn2D::<f32>::from_fn_generic(g, |x, y| x + 10.0_f32 * y);
        assert_eq!(f.values.len(), 4 * 5);
        // Sup-norm of all non-negative values should be the maximum.
        let norm = f.norm_sup();
        assert!(norm > 0.0_f32);
        let z = f.zeroed_like();
        assert!(z.norm_sup() < f32::EPSILON);
    }
}
