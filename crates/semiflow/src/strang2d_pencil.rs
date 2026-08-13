//! [`Strang2DPencil`] — palindromic Strang with **per-pencil** 1-D kernels
//! (ADR-0195, Issue #21).
//!
//! [`crate::strang2d::Strang2D`] holds one `X` kernel and one `Y` kernel and
//! applies the same `X` to every row (see `AxisLift`'s pencil loops in
//! `crate::axis`). A coefficient that varies along the **transverse** axis is
//! therefore inexpressible — and that is exactly what the Heston generator needs.
//! After the standard log-price + decorrelation transform `z = x − (ρ/ξ)v` the
//! generator is cross-term-free,
//!
//! ```text
//!   ∂_τ u = ½v(1−ρ²)·u_zz + ½ξ²v·u_vv + b_z(v)·u_z + κ(θ−v)·u_v − r·u
//! ```
//!
//! but **both** diagonal coefficients depend on `v`, i.e. each varies along the
//! *other* axis. `Heat2DVarA`'s `a_x(x)` / `a_y(y)` cannot say that.
//!
//! This type carries one prebuilt 1-D kernel per pencil — `ny` of them for the
//! X-legs, `nx` for the Y-leg — dispatched by pencil index. Total coefficient
//! storage is the two `nx·ny` arrays the caller already supplies; the per-kernel
//! overhead is `(nx+ny)` structs.
//!
//! ## Order (NORMATIVE — read this before trusting `order()`)
//!
//! `Strang2D`'s module doc justifies order 2 by the commutator identity
//! `[L_x ⊗ I, I ⊗ L_y] = 0`, which makes palindromic Strang *exact* at the BCH
//! level for a separable generator. **With transverse-varying coefficients that
//! premise is false**: `L_x = a_x(x,y)∂_xx + …` and `L_y = a_y(x,y)∂_yy + …` do
//! not commute. Order 2 is retained here by the *classical symmetric-splitting*
//! argument instead — the τ² term of `e^{τA/2} e^{τB} e^{τA/2}` vanishes
//! identically for arbitrary non-commuting `A`, `B`, leaving a local error
//! `(τ³/24)([B,[B,A]] − 2[A,[A,B]]) + O(τ⁴)` and hence global τ². The *slope* is
//! unchanged; the error **constant** now carries those double commutators and
//! grows with `‖∂_y a_x‖` and `‖∂_x a_y‖`. Users with strongly transverse-varying
//! coefficients should verify the slope on their own field rather than assume it.

extern crate alloc;

use alloc::vec::Vec;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{half, SemiflowFloat},
    grid2d::Grid2D,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    pencil,
    scratch::ScratchPool,
};

/// Palindromic Strang composition with one 1-D kernel per pencil.
///
/// `x[j]` acts on row `j` (so `x.len() == ny`); `y[i]` acts on column `i`
/// (`y.len() == nx`).
#[derive(Clone)]
pub struct Strang2DPencil<X, Y, F: SemiflowFloat = f64> {
    x: Vec<X>,
    y: Vec<Y>,
    grid: Grid2D<F>,
}

impl<X, Y, F> Strang2DPencil<X, Y, F>
where
    F: SemiflowFloat,
    X: ChernoffFunction<F, S = GridFn1D<F>>,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
{
    /// Construct from per-pencil kernels.
    ///
    /// # Errors
    /// `DomainViolation` if `x.len() != grid.ny` or `y.len() != grid.nx`.
    pub fn new(x: Vec<X>, y: Vec<Y>, grid: Grid2D<F>) -> Result<Self, SemiflowError> {
        if x.len() != grid.ny() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "Strang2DPencil: x kernels must number ny (one per row)",
                value: x.len() as f64,
            });
        }
        if y.len() != grid.nx() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "Strang2DPencil: y kernels must number nx (one per column)",
                value: y.len() as f64,
            });
        }
        Ok(Self { x, y, grid })
    }

    /// Grid geometry.
    #[must_use]
    pub fn grid(&self) -> Grid2D<F> {
        self.grid
    }

    /// X-leg: row `j` is contiguous, so it is handed to `x[j]` as a view.
    fn x_pass(
        &self,
        tau: F,
        src: &[F],
        dst: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let (nx, ny) = (self.grid.nx(), self.grid.ny());
        let gx = self.grid.x;
        for j in 0..ny {
            let src_row = pencil::row_2d(src, nx, j);
            let dst_row = pencil::row_2d_mut(dst, nx, j);
            crate::grid_fn::apply_into_via_view(&self.x[j], tau, src_row, dst_row, gx, scratch)?;
        }
        Ok(())
    }

    /// Y-leg: column `i` is strided, so it is gathered into a pool buffer first.
    fn y_pass(
        &self,
        tau: F,
        src: &[F],
        dst: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let (nx, ny) = (self.grid.nx(), self.grid.ny());
        let gy = self.grid.y;
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
            let step = self.y[i].apply_into(tau, &src_gf, &mut dst_gf, scratch);
            pencil::scatter_y_2d_from(dst, nx, ny, i, &dst_gf.values);
            src_col = src_gf.values;
            dst_col = dst_gf.values;
            step?;
        }
        scratch.return_vec(src_col);
        scratch.return_vec(dst_col);
        Ok(())
    }
}

impl<X, Y, F> ChernoffFunction<F> for Strang2DPencil<X, Y, F>
where
    F: SemiflowFloat,
    X: ChernoffFunction<F, S = GridFn1D<F>>,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
{
    type S = GridFn2D<F>;

    /// `Φ(τ) = X(τ/2) ∘ Y(τ) ∘ X(τ/2)`, per-pencil.
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn2D<F>,
        dst: &mut GridFn2D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let n = src.values.len();
        if dst.values.len() != n {
            dst.values.resize(n, F::zero());
        }
        let half_tau = half::<F>() * tau;
        let mut mid_a = scratch.take_vec(n);
        let mut mid_b = scratch.take_vec(n);
        mid_a.resize(n, F::zero());
        mid_b.resize(n, F::zero());

        let outcome = (|| -> Result<(), SemiflowError> {
            self.x_pass(half_tau, &src.values, &mut mid_a, scratch)?;
            self.y_pass(tau, &mid_a, &mut mid_b, scratch)?;
            self.x_pass(half_tau, &mid_b, &mut dst.values, scratch)
        })();

        scratch.return_vec(mid_a);
        scratch.return_vec(mid_b);
        outcome
    }

    /// `min` over every pencil, capped at 2 by the splitting itself.
    ///
    /// The cap is the point: a per-pencil kernel of order 4 does not lift the
    /// composition above the τ² of symmetric splitting.
    fn order(&self) -> u32 {
        let lo = self
            .x
            .iter()
            .map(ChernoffFunction::order)
            .chain(self.y.iter().map(ChernoffFunction::order))
            .min()
            .unwrap_or(1);
        lo.min(2)
    }

    /// Sup over pencils — a bound must hold for the worst one, not the first.
    fn growth(&self) -> Growth<F> {
        let mut omega = F::zero();
        let mut m = F::one();
        for g in self
            .x
            .iter()
            .map(ChernoffFunction::growth)
            .chain(self.y.iter().map(ChernoffFunction::growth))
        {
            if g.omega > omega {
                omega = g.omega;
            }
            if g.multiplier > m {
                m = g.multiplier;
            }
        }
        Growth::new(m, omega)
    }
}
