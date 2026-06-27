//! [`GeneratorAction<F>`] — minimal adapter trait over linear PDE generators
//! (ADR-0189 §58.1).
//!
//! Two concrete adapters:
//! - [`DivFormGenerator`]: wraps `Diffusion4thChernoff<f64>`, provides the
//!   divergence-form `A = L` stencil (already negative-semidefinite).
//! - [`NegLaplacianGenerator<F,Op>`]: wraps any [`SymmetricLinearOp<F>`],
//!   provides `A = −L` (negate: graphs use contracting semigroup `e^{−τL}`).

extern crate alloc;

use alloc::vec;

use crate::{
    diffusion4::Diffusion4thChernoff,
    diffusion4_zeta4::apply_div_form,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    symmetric_operator::SymmetricLinearOp,
};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Thin generator interface consumed by [`crate::phi_action`].
///
/// `apply_generator(src, dst)` computes `dst ← A·src` where `A` is the PDE
/// linear generator.  Both slices must have length `self.dim()`.
pub trait GeneratorAction<F: SemiflowFloat>: Send + Sync {
    /// Operator dimension `n`.
    fn dim(&self) -> usize;

    /// `dst ← A · src`.
    fn apply_generator(&self, src: &[F], dst: &mut [F]);

    /// Conservative upper bound on `‖A‖` (used for Horner scaling).
    fn norm_bound(&self) -> F;

    /// `dst ← Aᵀ · src`.  Defaults to `apply_generator` (self-adjoint case).
    fn apply_generator_transpose(&self, src: &[F], dst: &mut [F]) {
        self.apply_generator(src, dst);
    }
}

// ---------------------------------------------------------------------------
// DivFormGenerator
// ---------------------------------------------------------------------------

/// Adapter for the 1-D divergence-form generator `A = L` (§58.1, heat equation).
///
/// Wraps [`Diffusion4thChernoff<f64>`]; forwards `apply_generator` to
/// `apply_div_form`.  Conservative norm bound: `4·a_norm_bound / dx²`.
pub struct DivFormGenerator {
    inner: Diffusion4thChernoff<f64>,
    norm_est: f64,
}

impl DivFormGenerator {
    /// Build from a div-form kernel.  Consumes the kernel (cheap Copy inside).
    #[must_use]
    pub fn new(inner: Diffusion4thChernoff<f64>) -> Self {
        let dx = inner.grid.dx();
        let norm_est = 4.0 * inner.a_norm_bound / (dx * dx);
        Self { inner, norm_est }
    }
}

impl GeneratorAction<f64> for DivFormGenerator {
    fn dim(&self) -> usize {
        self.inner.grid.n
    }

    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        let n = self.inner.grid.n;
        // Wrap src as a temporary GridFn1D (one allocation, O(n) copy).
        let src_gfn = GridFn1D { grid: self.inner.grid, values: src[..n].to_vec() };
        let mut dst_gfn = GridFn1D { grid: self.inner.grid, values: vec![0.0_f64; n] };
        apply_div_form(&self.inner, &src_gfn, &mut dst_gfn)
            .expect("DivFormGenerator: apply_div_form");
        dst[..n].copy_from_slice(&dst_gfn.values);
    }

    fn norm_bound(&self) -> f64 {
        self.norm_est
    }
    // apply_generator_transpose = apply_generator (self-adjoint)
}

// ---------------------------------------------------------------------------
// NegLaplacianGenerator
// ---------------------------------------------------------------------------

/// Adapter for the negated Laplacian `A = −L` (§58.1, graph heat equation).
///
/// Wraps any [`SymmetricLinearOp<F>`] and negates each `apply_into_slice` result.
/// For symmetric `L`: transpose equals forward, so `apply_generator_transpose`
/// inherits the default (which calls `apply_generator`).
pub struct NegLaplacianGenerator<F: SemiflowFloat, Op: SymmetricLinearOp<F>> {
    op: Op,
    _marker: core::marker::PhantomData<F>,
}

impl<F: SemiflowFloat, Op: SymmetricLinearOp<F>> NegLaplacianGenerator<F, Op> {
    /// Wrap an operator.
    #[must_use]
    pub fn new(op: Op) -> Self {
        Self { op, _marker: core::marker::PhantomData }
    }
}

impl<F: SemiflowFloat, Op: SymmetricLinearOp<F>> GeneratorAction<F>
    for NegLaplacianGenerator<F, Op>
{
    fn dim(&self) -> usize {
        self.op.n()
    }

    fn apply_generator(&self, src: &[F], dst: &mut [F]) {
        self.op.apply_into_slice(src, dst);
        for d in dst.iter_mut() {
            *d = -*d;
        }
    }

    fn norm_bound(&self) -> F {
        self.op.lambda_max_bound()
    }
    // apply_generator_transpose = apply_generator (symmetric)
}
