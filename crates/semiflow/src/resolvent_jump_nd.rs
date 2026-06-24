//! F2 — 2D/3D parabolic LHP backends (math.md §47.8, ADR-0148, v8.2.0).
//!
//! Extends [`crate::resolvent_jump`] via a **banded complex LU** solve for
//! `(λI − A)⁻¹b` over the `grid2d.rs` / `grid3d.rs` Kronecker-sum Laplacians,
//! reusing the same TWS parabolic contour (dimension-blind, Hale–Weideman 2015).
//!
//! **NARROW** (§47.8): self-adjoint / sectorial generators only. Hyperbolic
//! contour (non-sectorial) is honestly deferred (ADR-0148, Weideman–Trefethen 2007).
//! Half-bandwidth `bw = nx` (2D) or `nx·ny` (3D); `O(N·bw²)` per node; `no_std`.
//!
//! - Hale–Weideman, SIAM J. Sci. Comput. 37:6 (2015) — ND extension precedent.
//! - ADR-0134 (1D NARROW-GO); ADR-0148 (2D/3D GO + hyperbolic DEFER).
//! - Gates: `G_RESOLVENT_JUMP_2D_ORDER`, `G_RESOLVENT_JUMP_3D_ORDER` (`RELEASE_BLOCKING`).

// Grid dimensions (usize) cast to f64/isize/usize for contour/stencil/index computations.
// All values are grid sizes ≪ 2^52 (precision) and ≪ isize::MAX (wrap); sign is checked
// by pre-conditions in the banded LU solver.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use num_complex::Complex;

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid2d::Grid2D,
    grid3d::Grid3D,
    grid_fn2d::GridFn2D,
    grid_fn3d::GridFn3D,
    resolvent_jump::contour_node,
};

// Banded LU helpers (storage primitives + Gaussian elimination) are in a
// sibling module to keep this file ≤500 lines.
#[path = "resolvent_jump_nd_banded.rs"]
mod resolvent_jump_nd_banded;

// ---------------------------------------------------------------------------
// Minimum node count (must match resolvent_jump.rs M_MIN).
// ---------------------------------------------------------------------------

/// Minimum allowed `m_nodes` for 2D/3D backends.
const M_MIN_ND: usize = 6;

// ---------------------------------------------------------------------------
// ResolventJumpChernoff2D
// ---------------------------------------------------------------------------

/// F2 resolvent time-jump for 2D divergence-form Laplacians (NARROW-PARABOLIC).
///
/// Computes `e^{tA}g` via the TWS parabolic-contour inverse Laplace sum
/// (math.md §47.3 / §47.8) where each resolvent `(λI − A)⁻¹g` is evaluated
/// by a **banded complex LU** direct solve over the `Grid2D` Kronecker-sum
/// Neumann Laplacian. The outer contour quadrature is **unchanged** from the 1D
/// `ResolventJumpChernoff` — dimension-blind (Hale–Weideman 2015).
///
/// **NARROW**: self-adjoint sectorial generators only (§47.8).
pub struct ResolventJumpChernoff2D<F = f64>
where
    F: SemiflowFloat,
{
    /// 2D tensor-product grid geometry.
    pub grid: Grid2D<F>,
    /// Number of contour nodes `M`. Invariant: `m_nodes >= M_MIN_ND`.
    pub m_nodes: usize,
}

impl<F: SemiflowFloat> ResolventJumpChernoff2D<F> {
    /// Construct from a `Grid2D` and node count `M ≥ 6`.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `m_nodes < 6`.
    pub fn new(grid: Grid2D<F>, m_nodes: usize) -> Result<Self, SemiflowError> {
        if m_nodes < M_MIN_ND {
            return Err(SemiflowError::DomainViolation {
                what: "ResolventJumpChernoff2D: m_nodes must be >= 6",
                value: m_nodes as f64,
            });
        }
        Ok(Self { grid, m_nodes })
    }

    /// Approximate `e^{tA}g` via the TWS parabolic-contour quadrature (§47.8).
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `t ≤ 0`, non-finite, or `g` size mismatch.
    pub fn jump(&self, t: F, g: &GridFn2D<F>) -> Result<GridFn2D<F>, SemiflowError> {
        validate_t_nd(t)?;
        let n = self.grid.len();
        if g.values.len() != n {
            return Err(SemiflowError::DomainViolation {
                what: "ResolventJumpChernoff2D::jump: g.len() != grid.len()",
                value: g.values.len() as f64,
            });
        }
        let b: Vec<f64> = g.values.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let acc = contour_sum_nd(
            self.m_nodes,
            t.to_f64().unwrap_or(f64::NAN),
            &b,
            |lam, rhs| resolve_lhp_2d(self.grid, lam, rhs),
        )?;
        let values = acc.into_iter().map(|c| from_f64::<F>(c.re)).collect();
        Ok(GridFn2D {
            values,
            grid: self.grid,
        })
    }
}

// ---------------------------------------------------------------------------
// ResolventJumpChernoff3D
// ---------------------------------------------------------------------------

/// F2 resolvent time-jump for 3D divergence-form Laplacians (NARROW-PARABOLIC).
///
/// Identical design to [`ResolventJumpChernoff2D`] but over `Grid3D`.
/// Half-bandwidth of `(λI − A)` is `nx·ny` (Kronecker-sum, z-slowest row-major).
/// The contour quadrature is dimension-blind (Hale–Weideman 2015).
///
/// **NARROW**: self-adjoint sectorial generators only (§47.8).
pub struct ResolventJumpChernoff3D<F = f64>
where
    F: SemiflowFloat,
{
    /// 3D tensor-product grid geometry.
    pub grid: Grid3D<F>,
    /// Number of contour nodes `M`. Invariant: `m_nodes >= M_MIN_ND`.
    pub m_nodes: usize,
}

impl<F: SemiflowFloat> ResolventJumpChernoff3D<F> {
    /// Construct from a `Grid3D` and node count `M ≥ 6`.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `m_nodes < 6`.
    pub fn new(grid: Grid3D<F>, m_nodes: usize) -> Result<Self, SemiflowError> {
        if m_nodes < M_MIN_ND {
            return Err(SemiflowError::DomainViolation {
                what: "ResolventJumpChernoff3D: m_nodes must be >= 6",
                value: m_nodes as f64,
            });
        }
        Ok(Self { grid, m_nodes })
    }

    /// Approximate `e^{tA}g` via the TWS parabolic-contour quadrature (§47.8).
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `t ≤ 0`, non-finite, or `g` size mismatch.
    pub fn jump(&self, t: F, g: &GridFn3D<F>) -> Result<GridFn3D<F>, SemiflowError> {
        validate_t_nd(t)?;
        let n = self.grid.len();
        if g.values.len() != n {
            return Err(SemiflowError::DomainViolation {
                what: "ResolventJumpChernoff3D::jump: g.len() != grid.len()",
                value: g.values.len() as f64,
            });
        }
        let b: Vec<f64> = g.values.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let acc = contour_sum_nd(
            self.m_nodes,
            t.to_f64().unwrap_or(f64::NAN),
            &b,
            |lam, rhs| resolve_lhp_3d(self.grid, lam, rhs),
        )?;
        let values = acc.into_iter().map(|c| from_f64::<F>(c.re)).collect();
        Ok(GridFn3D {
            values,
            grid: self.grid,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared contour quadrature (dimension-blind, §47.3 / §47.8)
// ---------------------------------------------------------------------------

/// TWS parabolic-contour midpoint sum, generic over any LHP solve.
///
/// `resolve_fn(lam, b) -> Result<Vec<Complex<f64>>, SemiflowError>` implements
/// the per-node `(λI − A)⁻¹b`. Reuses `contour_node` from `resolvent_jump.rs`.
// m, t, b, k: standard contour-quadrature names from §47.3.
#[allow(clippy::many_single_char_names)]
fn contour_sum_nd<R>(
    m: usize,
    t: f64,
    b: &[f64],
    resolve_fn: R,
) -> Result<Vec<Complex<f64>>, SemiflowError>
where
    R: Fn(Complex<f64>, &[f64]) -> Result<Vec<Complex<f64>>, SemiflowError>,
{
    let n = b.len();
    let mut acc: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    let scale = m as f64 / t;
    let step = 2.0 * core::f64::consts::PI / m as f64;
    for k in 0..m {
        let (lam, dlam) = contour_node(scale, k, step);
        let r = resolve_fn(lam, b)?;
        let weight = (lam * t).exp() * dlam * step / Complex::new(0.0, 2.0 * core::f64::consts::PI);
        for i in 0..n {
            acc[i] += weight * r[i];
        }
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// validate_t helper
// ---------------------------------------------------------------------------

fn validate_t_nd<F: SemiflowFloat>(t: F) -> Result<(), SemiflowError> {
    if !t.is_finite() || t <= F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "ResolventJumpChernoffND::jump: t must be finite and positive",
            value: t.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 2D banded LHP solve
// ---------------------------------------------------------------------------

/// Solve `(λI − A)r = b` for the 2D Neumann Laplacian (Kronecker sum, row-major).
///
/// `(λI − A)` has half-bandwidth `bw = nx`. Gaussian elimination is `O(N·bw²)`.
///
/// # Errors
/// [`SemiflowError::DomainViolation`] if a pivot is near-zero (λ on spectrum).
fn resolve_lhp_2d<F: SemiflowFloat>(
    grid: Grid2D<F>,
    lam: Complex<f64>,
    b: &[f64],
) -> Result<Vec<Complex<f64>>, SemiflowError> {
    let nx = grid.x.n;
    let ny = grid.y.n;
    let dx = grid.x.dx().to_f64().unwrap_or(f64::NAN);
    let dy = grid.y.dx().to_f64().unwrap_or(f64::NAN);
    let mat = build_2d_system(nx, ny, dx, dy, lam);
    let rhs: Vec<Complex<f64>> = b.iter().map(|v| Complex::new(*v, 0.0)).collect();
    banded_lu_solve(&mat, nx, nx * ny, &rhs)
}

/// Assemble the banded `(λI − A)` matrix for the 2D Neumann Laplacian.
///
/// Returns banded storage `mat[row * (2*bw + 1) + (bw + col - row)]`.
fn build_2d_system(nx: usize, ny: usize, dx: f64, dy: f64, lam: Complex<f64>) -> Vec<Complex<f64>> {
    let n = nx * ny;
    let bw = nx;
    let width = 2 * bw + 1;
    let mut mat = vec![Complex::new(0.0, 0.0); n * width];
    let ix2 = 1.0 / (dx * dx);
    let iy2 = 1.0 / (dy * dy);
    // Row idx(i, j) = j*nx + i; neighbors: (i±1,j) → ±1, (i,j±1) → ±nx.
    for j in 0..ny {
        for i in 0..nx {
            let row = j * nx + i;
            // Diagonal: λ − (−2/dx² − 2/dy²) except boundary rows.
            let ax = if i == 0 || i == nx - 1 {
                ix2
            } else {
                2.0 * ix2
            };
            let ay = if j == 0 || j == ny - 1 {
                iy2
            } else {
                2.0 * iy2
            };
            set_band(&mut mat, n, bw, row, row, lam + Complex::new(ax + ay, 0.0));
            // x-axis off-diagonals (bandwidth 1).
            if i > 0 {
                set_band(&mut mat, n, bw, row, row - 1, Complex::new(-ix2, 0.0));
            }
            if i < nx - 1 {
                set_band(&mut mat, n, bw, row, row + 1, Complex::new(-ix2, 0.0));
            }
            // y-axis off-diagonals (bandwidth nx).
            if j > 0 {
                set_band(&mut mat, n, bw, row, row - nx, Complex::new(-iy2, 0.0));
            }
            if j < ny - 1 {
                set_band(&mut mat, n, bw, row, row + nx, Complex::new(-iy2, 0.0));
            }
        }
    }
    mat
}

// ---------------------------------------------------------------------------
// 3D banded LHP solve
// ---------------------------------------------------------------------------

/// Solve `(λI − A)r = b` for the 3D Neumann Laplacian (Kronecker sum, x-fastest).
///
/// Half-bandwidth `bw = nx·ny`. Valid for small grids (e.g. 8³ → bw=64).
///
/// # Errors
/// [`SemiflowError::DomainViolation`] if a pivot is near-zero (λ on spectrum).
fn resolve_lhp_3d<F: SemiflowFloat>(
    grid: Grid3D<F>,
    lam: Complex<f64>,
    b: &[f64],
) -> Result<Vec<Complex<f64>>, SemiflowError> {
    let nx = grid.x.n;
    let ny = grid.y.n;
    let nz = grid.z.n;
    let dx = grid.x.dx().to_f64().unwrap_or(f64::NAN);
    let dy = grid.y.dx().to_f64().unwrap_or(f64::NAN);
    let dz = grid.z.dx().to_f64().unwrap_or(f64::NAN);
    let mat = build_3d_system(nx, ny, nz, dx, dy, dz, lam);
    let rhs: Vec<Complex<f64>> = b.iter().map(|v| Complex::new(*v, 0.0)).collect();
    banded_lu_solve(&mat, nx * ny, nx * ny * nz, &rhs)
}

/// Assemble the banded `(λI − A)` for the 3D Neumann Laplacian.
// nx,ny,nz,dx,dy,dz,λ: all 7 are required for the 3D banded system assembly.
#[allow(clippy::too_many_arguments)]
fn build_3d_system(
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    lam: Complex<f64>,
) -> Vec<Complex<f64>> {
    let n = nx * ny * nz;
    let bw = nx * ny;
    let mut mat = vec![Complex::new(0.0, 0.0); n * (2 * bw + 1)];
    let ix2 = 1.0 / (dx * dx);
    let iy2 = 1.0 / (dy * dy);
    let iz2 = 1.0 / (dz * dz);
    // idx(i,j,k) = k*nx*ny + j*nx + i; neighbors: ±1 (x), ±nx (y), ±nx*ny (z).
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let row = k * nx * ny + j * nx + i;
                fill_3d_row(
                    &mut mat, bw, n, nx, ny, nz, row, i, j, k, ix2, iy2, iz2, lam,
                );
            }
        }
    }
    mat
}

/// Fill one row of the 3D banded system: diagonal + x/y/z off-diagonals.
// All parameters required: 3D stencil needs all axis indices and inverse spacings.
#[allow(clippy::too_many_arguments)]
fn fill_3d_row(
    mat: &mut [Complex<f64>],
    bw: usize,
    n: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    row: usize,
    i: usize,
    j: usize,
    k: usize,
    ix2: f64,
    iy2: f64,
    iz2: f64,
    lam: Complex<f64>,
) {
    // Diagonal: λ + ax + ay + az (Neumann boundary halves the coefficient).
    let ax = bnd_coeff(ix2, i, nx);
    let ay = bnd_coeff(iy2, j, ny);
    let az = bnd_coeff(iz2, k, nz);
    set_band(mat, n, bw, row, row, lam + Complex::new(ax + ay + az, 0.0));
    if i > 0 {
        set_band(mat, n, bw, row, row - 1, Complex::new(-ix2, 0.0));
    }
    if i < nx - 1 {
        set_band(mat, n, bw, row, row + 1, Complex::new(-ix2, 0.0));
    }
    if j > 0 {
        set_band(mat, n, bw, row, row - nx, Complex::new(-iy2, 0.0));
    }
    if j < ny - 1 {
        set_band(mat, n, bw, row, row + nx, Complex::new(-iy2, 0.0));
    }
    if k > 0 {
        set_band(mat, n, bw, row, row - nx * ny, Complex::new(-iz2, 0.0));
    }
    if k < nz - 1 {
        set_band(mat, n, bw, row, row + nx * ny, Complex::new(-iz2, 0.0));
    }
}

/// Neumann boundary coefficient: `inv_sq` at boundary, `2·inv_sq` interior.
#[inline]
fn bnd_coeff(inv_sq: f64, idx: usize, n: usize) -> f64 {
    if idx == 0 || idx == n - 1 {
        inv_sq
    } else {
        2.0 * inv_sq
    }
}

// Banded storage helpers and Gaussian elimination live in the sibling module.
use resolvent_jump_nd_banded::{banded_lu_solve, set_band};

// ---------------------------------------------------------------------------
// Unit tests (included from resolvent_jump_nd_tests.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/resolvent_jump_nd_tests.rs"
    ));
}
