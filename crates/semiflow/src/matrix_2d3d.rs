//! `MatrixDiffusionChernoff2D/3D` — per-axis palindromic Strang for M-component
//! coupled diffusion on 2D/3D tensor-product grids (ADR-0124).
//!
//! PRE-FLIGHT (`scripts/verify_matrix_2d3d_preflight.py`) confirmed order-2:
//! - C1a: `[Lx_diff, Ly_diff] = 0` exactly (separability holds for M-component lifts).
//! - C2:  τ¹ and τ² BCH terms vanish (palindromic Strang is order-2 despite `[Cx,Cy]≠0`).
//! - C3:  per-axis matrix-exp reuse is order-2-consistent.
//!
//! 3D is inductive (Theorem 7', math §10.8): `Strang3D(τ) = X(τ/2)Y(τ/2)Z(τ)Y(τ/2)X(τ/2)`.
//!
//! ## Layout
//!
//! `MatrixGridFn2D<F, M>`: flat `Vec<F>` length `nx*ny*M`,
//!   index `(j*nx + i)*M + c` — spatial index outer, component inner.
//! `MatrixGridFn3D<F, M>`: flat `Vec<F>` length `nx*ny*nz*M`,
//!   index `(k*nx*ny + j*nx + i)*M + c`.
//!
//! Gate: `tests/g_matrix_2d3d.rs` — slope ≤ −0.80 for both 2D and 3D (ADR-0124).

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{half, SemiflowFloat},
    grid2d::Grid2D,
    grid3d::Grid3D,
    matrix_system::{MatrixDiffusionChernoff, MatrixGridFn1D},
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// MatrixGridFn2D
// ---------------------------------------------------------------------------

/// Multi-component 2D grid state `u : Grid2D → ℝᴹ`.
///
/// Storage: `values[(j*nx + i)*M + c]`. Length = `nx * ny * M`.
#[derive(Clone, Debug)]
pub struct MatrixGridFn2D<F: SemiflowFloat = f64, const M: usize = 2> {
    /// Grid geometry.
    pub grid: Grid2D<F>,
    /// Flat values — length `grid.nx() * grid.ny() * M`.
    pub values: Vec<F>,
}

impl<F: SemiflowFloat, const M: usize> MatrixGridFn2D<F, M> {
    /// Create zero-valued state on `grid`.
    pub fn new(grid: Grid2D<F>) -> Self {
        Self {
            grid,
            values: vec![F::zero(); grid.nx() * grid.ny() * M],
        }
    }

    /// Create from pointwise closure `func(x, y) -> [F; M]`.
    #[allow(clippy::many_single_char_names)]
    pub fn from_fn(grid: Grid2D<F>, mut func: impl FnMut(F, F) -> [F; M]) -> Self {
        let nx = grid.nx();
        let ny = grid.ny();
        let mut values = vec![F::zero(); nx * ny * M];
        for j in 0..ny {
            for i in 0..nx {
                let v = func(grid.x.x_at(i), grid.y.x_at(j));
                let base = (j * nx + i) * M;
                values[base..base + M].copy_from_slice(&v);
            }
        }
        Self { grid, values }
    }

    fn gather_x(&self, j: usize, out: &mut MatrixGridFn1D<F, M>) {
        let nx = self.grid.nx();
        for i in 0..nx {
            let sb = (j * nx + i) * M;
            out.values[i * M..i * M + M].copy_from_slice(&self.values[sb..sb + M]);
        }
    }

    fn scatter_x(&mut self, j: usize, src: &MatrixGridFn1D<F, M>) {
        let nx = self.grid.nx();
        for i in 0..nx {
            let db = (j * nx + i) * M;
            self.values[db..db + M].copy_from_slice(&src.values[i * M..i * M + M]);
        }
    }

    fn gather_y(&self, i: usize, out: &mut MatrixGridFn1D<F, M>) {
        let nx = self.grid.nx();
        for j in 0..self.grid.ny() {
            let sb = (j * nx + i) * M;
            out.values[j * M..j * M + M].copy_from_slice(&self.values[sb..sb + M]);
        }
    }

    fn scatter_y(&mut self, i: usize, src: &MatrixGridFn1D<F, M>) {
        let nx = self.grid.nx();
        for j in 0..self.grid.ny() {
            let db = (j * nx + i) * M;
            self.values[db..db + M].copy_from_slice(&src.values[j * M..j * M + M]);
        }
    }
}

impl<F: SemiflowFloat, const M: usize> State<F> for MatrixGridFn2D<F, M> {
    fn len(&self) -> usize {
        self.grid.nx() * self.grid.ny()
    }
    fn axpy_into(&mut self, a: F, src: &Self) {
        for (d, &s) in self.values.iter_mut().zip(src.values.iter()) {
            *d += a * s;
        }
    }
    fn copy_from(&mut self, src: &Self) {
        self.values.clone_from(&src.values);
    }
    fn zero_into(&mut self) {
        for v in &mut self.values {
            *v = F::zero();
        }
    }
    fn norm_sup(&self) -> F {
        self.values.iter().fold(
            F::zero(),
            |acc, &v| if v.abs() > acc { v.abs() } else { acc },
        )
    }
    fn scale_into(&mut self, k: F) {
        for v in &mut self.values {
            *v *= k;
        }
    }
}

// ---------------------------------------------------------------------------
// MatrixGridFn3D
// ---------------------------------------------------------------------------

/// Multi-component 3D grid state `u : Grid3D → ℝᴹ`.
///
/// Storage: `values[(k*nx*ny + j*nx + i)*M + c]`. Length = `nx * ny * nz * M`.
#[derive(Clone, Debug)]
pub struct MatrixGridFn3D<F: SemiflowFloat = f64, const M: usize = 2> {
    /// Grid geometry.
    pub grid: Grid3D<F>,
    /// Flat values — length `grid.nx() * grid.ny() * grid.nz() * M`.
    pub values: Vec<F>,
}

impl<F: SemiflowFloat, const M: usize> MatrixGridFn3D<F, M> {
    /// Create zero-valued state on `grid`.
    pub fn new(grid: Grid3D<F>) -> Self {
        Self {
            grid,
            values: vec![F::zero(); grid.nx() * grid.ny() * grid.nz() * M],
        }
    }

    /// Create from pointwise closure `func(x, y, z) -> [F; M]`.
    #[allow(clippy::many_single_char_names)]
    pub fn from_fn(grid: Grid3D<F>, mut func: impl FnMut(F, F, F) -> [F; M]) -> Self {
        let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());
        let mut values = vec![F::zero(); nx * ny * nz * M];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let v = func(grid.x.x_at(i), grid.y.x_at(j), grid.z.x_at(k));
                    let base = (k * nx * ny + j * nx + i) * M;
                    values[base..base + M].copy_from_slice(&v);
                }
            }
        }
        Self { grid, values }
    }

    /// Gather strided pencil into `out` (stride=step between spatial indices).
    fn gather(&self, stride: usize, base: usize, len: usize, out: &mut MatrixGridFn1D<F, M>) {
        for s in 0..len {
            let sb = (base + s * stride) * M;
            out.values[s * M..s * M + M].copy_from_slice(&self.values[sb..sb + M]);
        }
    }

    /// Scatter strided pencil from `src`.
    fn scatter(&mut self, stride: usize, base: usize, len: usize, src: &MatrixGridFn1D<F, M>) {
        for s in 0..len {
            let db = (base + s * stride) * M;
            self.values[db..db + M].copy_from_slice(&src.values[s * M..s * M + M]);
        }
    }
}

impl<F: SemiflowFloat, const M: usize> State<F> for MatrixGridFn3D<F, M> {
    fn len(&self) -> usize {
        self.grid.nx() * self.grid.ny() * self.grid.nz()
    }
    fn axpy_into(&mut self, a: F, src: &Self) {
        for (d, &s) in self.values.iter_mut().zip(src.values.iter()) {
            *d += a * s;
        }
    }
    fn copy_from(&mut self, src: &Self) {
        self.values.clone_from(&src.values);
    }
    fn zero_into(&mut self) {
        for v in &mut self.values {
            *v = F::zero();
        }
    }
    fn norm_sup(&self) -> F {
        self.values.iter().fold(
            F::zero(),
            |acc, &v| if v.abs() > acc { v.abs() } else { acc },
        )
    }
    fn scale_into(&mut self, k: F) {
        for v in &mut self.values {
            *v *= k;
        }
    }
}

// ---------------------------------------------------------------------------
// MatrixDiffusionChernoff2D
// ---------------------------------------------------------------------------

/// Palindromic Strang 2D composition: `Φ(τ) = Lx(τ/2) Ly(τ) Lx(τ/2)` (ADR-0124).
///
/// Order 2 is sympy-verified (C1a+C2+C3 in PRE-FLIGHT); slope gate ≤ −0.80.
pub struct MatrixDiffusionChernoff2D<F: SemiflowFloat = f64, const M: usize = 2> {
    /// X-axis kernel (applied at τ/2 twice).
    pub kernel_x: MatrixDiffusionChernoff<F, M>,
    /// Y-axis kernel (applied at τ once).
    pub kernel_y: MatrixDiffusionChernoff<F, M>,
}

impl<F: SemiflowFloat, const M: usize> MatrixDiffusionChernoff2D<F, M> {
    /// Construct from per-axis kernels.
    #[must_use]
    pub fn new(kx: MatrixDiffusionChernoff<F, M>, ky: MatrixDiffusionChernoff<F, M>) -> Self {
        Self {
            kernel_x: kx,
            kernel_y: ky,
        }
    }
}

impl<F: SemiflowFloat, const M: usize> ChernoffFunction<F> for MatrixDiffusionChernoff2D<F, M> {
    type S = MatrixGridFn2D<F, M>;
    fn order(&self) -> u32 {
        2
    }
    fn growth(&self) -> Growth<F> {
        let gx = self.kernel_x.growth();
        let gy = self.kernel_y.growth();
        Growth {
            multiplier: gx.multiplier * gx.multiplier * gy.multiplier,
            omega: gx.omega + gy.omega,
        }
    }

    fn apply_into(
        &self,
        tau: F,
        src: &MatrixGridFn2D<F, M>,
        dst: &mut MatrixGridFn2D<F, M>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let ht = half::<F>() * tau;
        let (nx, ny) = (src.grid.nx(), src.grid.ny());
        if dst.values.len() != src.values.len() {
            dst.values.resize(src.values.len(), F::zero());
        }
        dst.grid = src.grid;
        dst.values.copy_from_slice(&src.values);
        pass_2d_x(dst, &self.kernel_x, ht, ny, scratch)?;
        pass_2d_y(dst, &self.kernel_y, tau, nx, scratch)?;
        pass_2d_x(dst, &self.kernel_x, ht, ny, scratch)
    }
}

fn pass_2d_x<F: SemiflowFloat, const M: usize>(
    buf: &mut MatrixGridFn2D<F, M>,
    kernel: &MatrixDiffusionChernoff<F, M>,
    tau: F,
    ny: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let (mut p, mut q) = (
        MatrixGridFn1D::<F, M>::new(kernel.grid),
        MatrixGridFn1D::<F, M>::new(kernel.grid),
    );
    for j in 0..ny {
        buf.gather_x(j, &mut p);
        kernel.apply_into(tau, &p, &mut q, scratch)?;
        buf.scatter_x(j, &q);
    }
    Ok(())
}

fn pass_2d_y<F: SemiflowFloat, const M: usize>(
    buf: &mut MatrixGridFn2D<F, M>,
    kernel: &MatrixDiffusionChernoff<F, M>,
    tau: F,
    nx: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let (mut p, mut q) = (
        MatrixGridFn1D::<F, M>::new(kernel.grid),
        MatrixGridFn1D::<F, M>::new(kernel.grid),
    );
    for i in 0..nx {
        buf.gather_y(i, &mut p);
        kernel.apply_into(tau, &p, &mut q, scratch)?;
        buf.scatter_y(i, &q);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// MatrixDiffusionChernoff3D
// ---------------------------------------------------------------------------

/// Palindromic 5-leg Strang 3D: `Lx(τ/2) Ly(τ/2) Lz(τ) Ly(τ/2) Lx(τ/2)` (ADR-0124).
///
/// Order 2 follows inductively from C1a separability and palindromic BCH cancellation
/// (Theorem 7', math §10.8). Slope gate: ≤ −0.80.
pub struct MatrixDiffusionChernoff3D<F: SemiflowFloat = f64, const M: usize = 2> {
    /// X-axis kernel (τ/2 twice, outermost).
    pub kernel_x: MatrixDiffusionChernoff<F, M>,
    /// Y-axis kernel (τ/2 twice, middle).
    pub kernel_y: MatrixDiffusionChernoff<F, M>,
    /// Z-axis kernel (τ once, innermost).
    pub kernel_z: MatrixDiffusionChernoff<F, M>,
}

impl<F: SemiflowFloat, const M: usize> MatrixDiffusionChernoff3D<F, M> {
    /// Construct from per-axis kernels.
    #[must_use]
    pub fn new(
        kx: MatrixDiffusionChernoff<F, M>,
        ky: MatrixDiffusionChernoff<F, M>,
        kz: MatrixDiffusionChernoff<F, M>,
    ) -> Self {
        Self {
            kernel_x: kx,
            kernel_y: ky,
            kernel_z: kz,
        }
    }
}

impl<F: SemiflowFloat, const M: usize> ChernoffFunction<F> for MatrixDiffusionChernoff3D<F, M> {
    type S = MatrixGridFn3D<F, M>;
    fn order(&self) -> u32 {
        2
    }
    fn growth(&self) -> Growth<F> {
        let gx = self.kernel_x.growth();
        let gy = self.kernel_y.growth();
        let gz = self.kernel_z.growth();
        Growth {
            multiplier: gx.multiplier
                * gx.multiplier
                * gy.multiplier
                * gy.multiplier
                * gz.multiplier,
            omega: gx.omega + gy.omega + gz.omega,
        }
    }

    fn apply_into(
        &self,
        tau: F,
        src: &MatrixGridFn3D<F, M>,
        dst: &mut MatrixGridFn3D<F, M>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let ht = half::<F>() * tau;
        let (nx, ny, nz) = (src.grid.nx(), src.grid.ny(), src.grid.nz());
        if dst.values.len() != src.values.len() {
            dst.values.resize(src.values.len(), F::zero());
        }
        dst.grid = src.grid;
        dst.values.copy_from_slice(&src.values);
        pass_3d_x(dst, &self.kernel_x, ht, nx, ny, nz, scratch)?;
        pass_3d_y(dst, &self.kernel_y, ht, nx, ny, nz, scratch)?;
        pass_3d_z(dst, &self.kernel_z, tau, nx, ny, nz, scratch)?;
        pass_3d_y(dst, &self.kernel_y, ht, nx, ny, nz, scratch)?;
        pass_3d_x(dst, &self.kernel_x, ht, nx, ny, nz, scratch)
    }
}

/// X-pencils 3D: stride=1, base = k*nx*ny + j*nx, len = nx.
#[allow(clippy::too_many_arguments)]
fn pass_3d_x<F: SemiflowFloat, const M: usize>(
    buf: &mut MatrixGridFn3D<F, M>,
    kernel: &MatrixDiffusionChernoff<F, M>,
    tau: F,
    nx: usize,
    ny: usize,
    nz: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let (mut p, mut q) = (
        MatrixGridFn1D::<F, M>::new(kernel.grid),
        MatrixGridFn1D::<F, M>::new(kernel.grid),
    );
    for k in 0..nz {
        for j in 0..ny {
            let base = k * nx * ny + j * nx;
            buf.gather(1, base, nx, &mut p);
            kernel.apply_into(tau, &p, &mut q, scratch)?;
            buf.scatter(1, base, nx, &q);
        }
    }
    Ok(())
}

/// Y-pencils 3D: stride=nx, base = k*nx*ny + i, len = ny.
#[allow(clippy::too_many_arguments)]
fn pass_3d_y<F: SemiflowFloat, const M: usize>(
    buf: &mut MatrixGridFn3D<F, M>,
    kernel: &MatrixDiffusionChernoff<F, M>,
    tau: F,
    nx: usize,
    ny: usize,
    nz: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let (mut p, mut q) = (
        MatrixGridFn1D::<F, M>::new(kernel.grid),
        MatrixGridFn1D::<F, M>::new(kernel.grid),
    );
    for k in 0..nz {
        for i in 0..nx {
            let base = k * nx * ny + i;
            buf.gather(nx, base, ny, &mut p);
            kernel.apply_into(tau, &p, &mut q, scratch)?;
            buf.scatter(nx, base, ny, &q);
        }
    }
    Ok(())
}

/// Z-pencils 3D: stride=nx*ny, base = j*nx + i, len = nz.
#[allow(clippy::too_many_arguments)]
fn pass_3d_z<F: SemiflowFloat, const M: usize>(
    buf: &mut MatrixGridFn3D<F, M>,
    kernel: &MatrixDiffusionChernoff<F, M>,
    tau: F,
    nx: usize,
    ny: usize,
    nz: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let (mut p, mut q) = (
        MatrixGridFn1D::<F, M>::new(kernel.grid),
        MatrixGridFn1D::<F, M>::new(kernel.grid),
    );
    for j in 0..ny {
        for i in 0..nx {
            let base = j * nx + i;
            buf.gather(nx * ny, base, nz, &mut p);
            kernel.apply_into(tau, &p, &mut q, scratch)?;
            buf.scatter(nx * ny, base, nz, &q);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (included from matrix_2d3d_tests.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/matrix_2d3d_tests.rs"
    ));
}
