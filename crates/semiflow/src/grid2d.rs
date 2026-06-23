//! [`Grid2D`] — rectangular tensor-product geometry for the 2D extension.
//!
//! Owns two [`Grid1D`] instances (one per axis), each carrying its own
//! [`crate::BoundaryPolicy`] and [`crate::InterpKind`] independently
//! (ADR-0012 closed decision: "per-axis BC").
//!
//! Storage convention (I-T1): row-major, `idx(i, j) = j * nx + i`.
//! x is the **fast** axis (X-rows are contiguous in memory).
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 2)
//!
//! `Grid2D<F: SemiflowFloat = f64>` — the `= f64` default keeps all existing
//! call-sites compiling unchanged. `Grid2D<f32>` composes two `Grid1D<f32>`
//! axes.
//!
//! See `contracts/semiflow-core.tensor.yaml` §3 (`Grid2D`),
//! invariant I-T1 and I-T2, and `docs/adr/0012-tensor-product-2d.md`.

use crate::{float::SemiflowFloat, grid::Grid1D};

// ---------------------------------------------------------------------------
// Grid2D
// ---------------------------------------------------------------------------

/// Rectangular tensor-product grid `[xmin, xmax] × [ymin, ymax]`.
///
/// Owns two [`Grid1D<F>`] instances: each axis carries its own boundary policy
/// and interpolation kind independently (ADR-0012).
///
/// `Grid2D` does NOT own values; values live in [`crate::GridFn2D`].
///
/// Storage convention (I-T1): row-major, `idx(i, j) = j * nx + i`,
/// x is the fast axis. Total cell count is `nx * ny`.
///
/// ## Generic-over-Float (ADR-0025, Wave 2)
///
/// `Grid2D<F: SemiflowFloat = f64>` — `= f64` default keeps existing call-sites
/// unchanged.
///
/// # Example
///
/// ```rust
/// use semiflow::{Grid1D, Grid2D};
/// let gx = Grid1D::new(-2.0, 2.0, 32).unwrap();
/// let gy = Grid1D::new(-2.0, 2.0, 32).unwrap();
/// let grid = Grid2D::new(gx, gy);
/// assert_eq!(grid.nx(), 32);
/// assert_eq!(grid.ny(), 32);
/// assert_eq!(grid.len(), 1024);
/// ```
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `Grid2D`.
#[derive(Debug, Clone, Copy)]
pub struct Grid2D<F: SemiflowFloat = f64> {
    /// Fast-axis grid. `nx = x.n`. Carries its own `BoundaryPolicy` / `InterpKind`.
    pub x: Grid1D<F>,
    /// Slow-axis grid. `ny = y.n`. Independent BC and interp from `x`.
    pub y: Grid1D<F>,
}

impl<F: SemiflowFloat> Grid2D<F> {
    /// Construct a `Grid2D<F>` from two validated [`Grid1D<F>`] instances.
    ///
    /// No additional validation is performed: each [`Grid1D`] has already
    /// been validated by its constructor (which enforces `xmin < xmax`,
    /// `n >= 4`, finite endpoints). The 2D invariant I-T2 (`n >= 2` per
    /// axis) is therefore implied by the `n >= 4` per-axis precondition.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Grid2D::new`.
    #[must_use]
    pub fn new(x: Grid1D<F>, y: Grid1D<F>) -> Self {
        Self { x, y }
    }

    /// Number of x-axis nodes: `self.x.n`.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Grid2D::nx`.
    #[inline]
    #[must_use]
    pub fn nx(&self) -> usize {
        self.x.n
    }

    /// Number of y-axis nodes: `self.y.n`.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Grid2D::ny`.
    #[inline]
    #[must_use]
    pub fn ny(&self) -> usize {
        self.y.n
    }

    /// Total number of grid cells: `nx * ny`.
    ///
    /// Used by [`crate::GridFn2D`] shape checks (I-T3).
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Grid2D::len`.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.n * self.y.n
    }

    /// Returns `true` if the grid contains no cells (`nx == 0` or `ny == 0`).
    ///
    /// In practice `Grid1D::new` requires `n >= 4`, so this is always `false`
    /// for any valid `Grid2D`. Provided to satisfy the `len_without_is_empty`
    /// lint.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.n == 0 || self.y.n == 0
    }

    /// Row-major linear index: `j * nx + i` (I-T1).
    ///
    /// This is `pub(crate)` — callers outside the crate use [`crate::GridFn2D`]
    /// accessors. In release builds there is no bounds check (BLAS-axpy
    /// convention, ADR-0008 carried over); debug builds assert.
    ///
    /// # Preconditions
    /// `i < self.nx()` and `j < self.ny()`. Violation is caller error.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Grid2D::idx`.
    #[inline]
    pub(crate) fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(
            i < self.x.n,
            "Grid2D::idx: i={i} out of range nx={}",
            self.x.n
        );
        debug_assert!(
            j < self.y.n,
            "Grid2D::idx: j={j} out of range ny={}",
            self.y.n
        );
        j * self.x.n + i
    }
}

// ---------------------------------------------------------------------------
// PartialEq (geometry equality, used by GridFn2D::write_row / write_col)
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> PartialEq for Grid2D<F> {
    /// Two `Grid2D` instances are equal iff all geometric fields match.
    fn eq(&self, other: &Self) -> bool {
        self.x.xmin == other.x.xmin
            && self.x.xmax == other.x.xmax
            && self.x.n == other.x.n
            && self.x.boundary == other.x.boundary
            && self.x.interp == other.x.interp
            && self.y.xmin == other.y.xmin
            && self.y.xmax == other.y.xmax
            && self.y.n == other.y.n
            && self.y.boundary == other.y.boundary
            && self.y.interp == other.y.interp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid1D;

    fn make_grid() -> Grid2D {
        let x = Grid1D::new(0.0, 1.0, 8).unwrap();
        let y = Grid1D::new(0.0, 2.0, 6).unwrap();
        Grid2D::new(x, y)
    }

    #[test]
    fn nx_ny_len() {
        let g = make_grid();
        assert_eq!(g.nx(), 8);
        assert_eq!(g.ny(), 6);
        assert_eq!(g.len(), 48);
    }

    #[test]
    fn idx_row_major() {
        let g = make_grid();
        // i=0, j=0 → 0
        assert_eq!(g.idx(0, 0), 0);
        // i=1, j=0 → 1  (x is fast axis)
        assert_eq!(g.idx(1, 0), 1);
        // i=0, j=1 → nx = 8
        assert_eq!(g.idx(0, 1), 8);
        // i=3, j=2 → 2*8 + 3 = 19
        assert_eq!(g.idx(3, 2), 19);
    }

    #[test]
    fn partial_eq() {
        let g1 = make_grid();
        let g2 = make_grid();
        assert_eq!(g1, g2);
    }

    #[test]
    fn grid2d_f32_new_generic() {
        let gx = Grid1D::<f32>::new_generic(0.0_f32, 1.0_f32, 8).unwrap();
        let gy = Grid1D::<f32>::new_generic(0.0_f32, 2.0_f32, 6).unwrap();
        let g = Grid2D::<f32>::new(gx, gy);
        assert_eq!(g.nx(), 8);
        assert_eq!(g.ny(), 6);
        assert_eq!(g.len(), 48);
        assert_eq!(g.idx(3, 2), 19);
    }
}
