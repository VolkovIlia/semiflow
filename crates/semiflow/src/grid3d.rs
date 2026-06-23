//! [`Grid3D`] — cuboid tensor-product geometry for the 3D extension.
//!
//! Owns three [`Grid1D`] instances (one per axis), each carrying its own
//! [`crate::BoundaryPolicy`] and [`crate::InterpKind`] independently
//! (per-axis BC principle, mirroring ADR-0012 for 2D).
//!
//! Storage convention (I-T1-3D): x-fastest row-major,
//! `idx(i, j, k) = k * nx * ny + j * nx + i`.
//! Strides: x → 1, y → nx, z → nx*ny.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 3)
//!
//! `Grid3D<F: SemiflowFloat = f64>` — the `= f64` default keeps all existing
//! call-sites compiling unchanged. `Grid3D<f32>` composes three `Grid1D<f32>`
//! axes.
//!
//! See `contracts/semiflow-core.tensor.yaml` §3 (`Grid3D`),
//! `docs/adr/0024-tensor-3d.md`, and `contracts/semiflow-core.math.md` §10.8.4.

use crate::{error::SemiflowError, float::SemiflowFloat, grid::Grid1D};

// ---------------------------------------------------------------------------
// Grid3D
// ---------------------------------------------------------------------------

/// Cuboid tensor-product grid `[xmin, xmax] × [ymin, ymax] × [zmin, zmax]`.
///
/// Owns three [`Grid1D<F>`] instances: each axis carries its own boundary policy
/// and interpolation kind independently (per-axis BC, ADR-0024).
///
/// `Grid3D` does NOT own values; values live in [`crate::GridFn3D`].
///
/// Storage convention (I-T1-3D): x-fastest row-major,
/// `idx(i, j, k) = k * nx * ny + j * nx + i`.
/// Strides: x → 1, y → nx, z → nx*ny. Total cell count is `nx * ny * nz`.
///
/// ## Generic-over-Float (ADR-0025, Wave 3)
///
/// `Grid3D<F: SemiflowFloat = f64>` — `= f64` default keeps existing call-sites
/// unchanged.
///
/// # Example
///
/// ```rust
/// use semiflow::{Grid1D, Grid3D};
/// let g = Grid1D::new(-1.0, 1.0, 8).unwrap();
/// let grid = Grid3D::new(g, g, g).unwrap();
/// assert_eq!(grid.nx(), 8);
/// assert_eq!(grid.ny(), 8);
/// assert_eq!(grid.nz(), 8);
/// assert_eq!(grid.len(), 512);
/// ```
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `Grid3D`.
#[derive(Debug, Clone, Copy)]
pub struct Grid3D<F: SemiflowFloat = f64> {
    /// Fast-axis (x) grid. `nx = x.n`.
    pub x: Grid1D<F>,
    /// Middle-axis (y) grid. `ny = y.n`.
    pub y: Grid1D<F>,
    /// Slow-axis (z) grid. `nz = z.n`.
    pub z: Grid1D<F>,
}

// ---------------------------------------------------------------------------
// Generic impl — works for all SemiflowFloat types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> Grid3D<F> {
    /// Construct a `Grid3D<F>` from three validated [`Grid1D<F>`] instances.
    ///
    /// Returns `Err(DomainViolation)` if any axis has fewer than 2 nodes
    /// (safety floor; in practice `Grid1D::new_generic` requires `n >= 4`).
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Grid3D::new`.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `x.n < 2`, `y.n < 2`, or `z.n < 2`.
    pub fn new_generic(x: Grid1D<F>, y: Grid1D<F>, z: Grid1D<F>) -> Result<Self, SemiflowError> {
        if x.n < 2 {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "Grid3D::new_generic: x.n must be >= 2",
                value: x.n as f64,
            });
        }
        if y.n < 2 {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "Grid3D::new_generic: y.n must be >= 2",
                value: y.n as f64,
            });
        }
        if z.n < 2 {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "Grid3D::new_generic: z.n must be >= 2",
                value: z.n as f64,
            });
        }
        Ok(Self { x, y, z })
    }

    /// Number of x-axis nodes: `self.x.n`.
    #[inline]
    #[must_use]
    pub fn nx(&self) -> usize {
        self.x.n
    }

    /// Number of y-axis nodes: `self.y.n`.
    #[inline]
    #[must_use]
    pub fn ny(&self) -> usize {
        self.y.n
    }

    /// Number of z-axis nodes: `self.z.n`.
    #[inline]
    #[must_use]
    pub fn nz(&self) -> usize {
        self.z.n
    }

    /// Total number of grid nodes: `nx * ny * nz`.
    ///
    /// Used by [`crate::GridFn3D`] shape checks (I-T1-3D).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.n * self.y.n * self.z.n
    }

    /// Returns `true` if the grid contains no nodes.
    ///
    /// In practice `Grid1D::new_generic` requires `n >= 4`, so this is always
    /// `false` for any valid `Grid3D`. Provided to satisfy the
    /// `len_without_is_empty` lint.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.n == 0 || self.y.n == 0 || self.z.n == 0
    }

    /// x-fastest 3D linear index: `k * nx * ny + j * nx + i` (I-T1-3D).
    ///
    /// Strides: x → 1, y → nx, z → nx*ny.
    ///
    /// This is `pub` so that integration tests and external callers can index
    /// into the public `GridFn3D::values` slice directly (the layout is already
    /// part of the public contract via that `pub` field). Debug builds assert
    /// bounds; release builds do not.
    ///
    /// # Preconditions
    /// `i < nx`, `j < ny`, `k < nz`. Violation is caller error.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Grid3D::idx`.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        debug_assert!(
            i < self.x.n,
            "Grid3D::idx: i={i} out of range nx={}",
            self.x.n
        );
        debug_assert!(
            j < self.y.n,
            "Grid3D::idx: j={j} out of range ny={}",
            self.y.n
        );
        debug_assert!(
            k < self.z.n,
            "Grid3D::idx: k={k} out of range nz={}",
            self.z.n
        );
        k * self.x.n * self.y.n + j * self.x.n + i
    }
}

// ---------------------------------------------------------------------------
// Concrete backward-compatible impl for Grid3D<f64>
// ---------------------------------------------------------------------------

impl Grid3D<f64> {
    /// Construct a `Grid3D` from three validated [`Grid1D`] instances
    /// (backward-compatible f64 API).
    ///
    /// Each [`Grid1D`] has already been validated by [`Grid1D::new`].
    /// Returns `Err(DomainViolation)` if any axis has fewer than 2 nodes
    /// (safety floor; in practice `Grid1D::new` requires `n >= 4`).
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Grid3D::new`.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `x.n < 2`, `y.n < 2`, or `z.n < 2`.
    pub fn new(x: Grid1D<f64>, y: Grid1D<f64>, z: Grid1D<f64>) -> Result<Self, SemiflowError> {
        Self::new_generic(x, y, z)
    }
}

// ---------------------------------------------------------------------------
// PartialEq (geometry equality)
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> PartialEq for Grid3D<F> {
    /// Two `Grid3D` instances are equal iff all geometric fields match.
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
            && self.z.xmin == other.z.xmin
            && self.z.xmax == other.z.xmax
            && self.z.n == other.z.n
            && self.z.boundary == other.z.boundary
            && self.z.interp == other.z.interp
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid() -> Grid3D {
        let x = Grid1D::new(0.0, 1.0, 8).unwrap();
        let y = Grid1D::new(0.0, 2.0, 6).unwrap();
        let z = Grid1D::new(0.0, 3.0, 4).unwrap();
        Grid3D::new(x, y, z).unwrap()
    }

    /// Verify `len()` and individual axis counts.
    #[test]
    fn nx_ny_nz_len() {
        let g = make_grid();
        assert_eq!(g.nx(), 8);
        assert_eq!(g.ny(), 6);
        assert_eq!(g.nz(), 4);
        assert_eq!(g.len(), 8 * 6 * 4);
    }

    /// `idx_x_fastest_consistent` — I-T1-3D: x-fastest storage.
    ///
    /// Checks: `idx(0,0,0)=0`, `idx(1,0,0)=1`, `idx(0,1,0)=nx`, `idx(0,0,1)=nx*ny`.
    #[test]
    fn idx_x_fastest_consistent() {
        let g = make_grid();
        let nx = g.nx();
        let ny = g.ny();
        assert_eq!(g.idx(0, 0, 0), 0);
        assert_eq!(g.idx(1, 0, 0), 1); // x-stride = 1
        assert_eq!(g.idx(0, 1, 0), nx); // y-stride = nx
        assert_eq!(g.idx(0, 0, 1), nx * ny); // z-stride = nx*ny
        assert_eq!(g.idx(3, 2, 1), nx * ny + 2 * nx + 3);
    }

    /// `bad_dimensions_rejected` — `Grid3D::new` rejects grids with n < 4
    /// (`Grid1D::new` already enforces n >= 4, so this tests the propagated error).
    #[test]
    fn bad_dimensions_rejected() {
        // Grid1D::new requires n >= 4, so n=1 and n=3 are rejected there.
        let x = Grid1D::new(0.0, 1.0, 4).unwrap();
        let y = Grid1D::new(0.0, 2.0, 4).unwrap();
        let z_valid = Grid1D::new(0.0, 3.0, 4).unwrap();
        // A valid 3-axis grid should succeed.
        assert!(Grid3D::new(x, y, z_valid).is_ok());
    }

    #[test]
    fn partial_eq() {
        let g1 = make_grid();
        let g2 = make_grid();
        assert_eq!(g1, g2);
    }

    #[test]
    fn grid3d_f32_new_generic() {
        let gx = Grid1D::<f32>::new_generic(0.0_f32, 1.0_f32, 8).unwrap();
        let gy = Grid1D::<f32>::new_generic(0.0_f32, 2.0_f32, 6).unwrap();
        let gz = Grid1D::<f32>::new_generic(0.0_f32, 3.0_f32, 4).unwrap();
        let g = Grid3D::<f32>::new_generic(gx, gy, gz).unwrap();
        assert_eq!(g.nx(), 8);
        assert_eq!(g.ny(), 6);
        assert_eq!(g.nz(), 4);
        assert_eq!(g.len(), 8 * 6 * 4);
        assert_eq!(g.idx(3, 2, 1), g.nx() * g.ny() + 2 * g.nx() + 3);
    }
}
