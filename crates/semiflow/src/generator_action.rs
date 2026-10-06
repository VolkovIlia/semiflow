//! [`GeneratorAction<F>`] — minimal adapter trait over linear PDE generators
//! (ADR-0189 §58.1).
//!
//! Three concrete adapters:
//! - [`DivFormGenerator`]: wraps `Diffusion4thChernoff<f64>`, provides the
//!   divergence-form `A = L` stencil (already negative-semidefinite).
//! - [`NegLaplacianGenerator<F,Op>`]: wraps any [`SymmetricLinearOp<F>`],
//!   provides `A = −L` (negate: graphs use contracting semigroup `e^{−τL}`).
//! - [`CsrGenerator<F>`]: `G = −M⁻¹A` over a CSR operator (symmetric or general)
//!   with an optional diagonal mass `M` (ADR-0202 §62.3).

extern crate alloc;

use alloc::{sync::Arc, vec, vec::Vec};

use crate::{
    diffusion4::Diffusion4thChernoff,
    diffusion4_zeta4::apply_div_form,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    general_operator::GeneralOperator,
    graph::Laplacian,
    grid_fn::GridFn1D,
    symmetric_operator::{SymmetricLinearOp, SymmetricOperator},
};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Thin generator interface consumed by [`mod@crate::phi_action`].
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
        let src_gfn = GridFn1D {
            grid: self.inner.grid,
            values: src[..n].to_vec(),
        };
        let mut dst_gfn = GridFn1D {
            grid: self.inner.grid,
            values: vec![0.0_f64; n],
        };
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
        Self {
            op,
            _marker: core::marker::PhantomData,
        }
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

// ---------------------------------------------------------------------------
// CsrGenerator
// ---------------------------------------------------------------------------

/// Which CSR operator backs a [`CsrGenerator`].
enum CsrBackend<F: SemiflowFloat> {
    /// Symmetric PSD operator (`Gᵀ` has the same matrix pattern as `G`).
    Symmetric(Arc<Laplacian<F>>),
    /// General operator with its exact stored transpose.
    General(GeneralOperator<F>),
}

/// `G = −M⁻¹A` over a CSR operator, `M = diag(mass)` (§62.3).
///
/// `apply_generator(v) = −(A v) ⊘ m`; the exact transpose is `Gᵀ v = −Aᵀ(v ⊘ m)`.
/// Norm bound: with `mass = None` the operator's own bound (bit-identical to
/// [`NegLaplacianGenerator`] for a symmetric operator); otherwise the exact
/// row-wise Gershgorin bound `maxᵢ Σⱼ|aᵢⱼ|/mᵢ ≥ ‖G‖_∞`.
pub struct CsrGenerator<F: SemiflowFloat = f64> {
    backend: CsrBackend<F>,
    mass: Option<Vec<F>>,
    norm: F,
}

/// Validate a diagonal mass (length, finite, strictly positive).
fn check_mass<F: SemiflowFloat>(n: usize, mass: &[F]) -> Result<(), SemiflowError> {
    if mass.len() != n {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "CsrGenerator: mass.len() != n",
            value: mass.len() as f64,
        });
    }
    for &m in mass {
        if !m.is_finite() || m <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "CsrGenerator: non-positive or non-finite mass",
                value: m.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

/// `maxᵢ Σⱼ |aᵢⱼ| / mᵢ` over CSR rows.
fn row_gershgorin<F: SemiflowFloat>(row_ptr: &[usize], vals: &[F], mass: &[F]) -> F {
    let mut best = F::zero();
    for (i, &m) in mass.iter().enumerate() {
        let mut sum = F::zero();
        for &v in &vals[row_ptr[i]..row_ptr[i + 1]] {
            sum += v.abs();
        }
        best = best.max(sum / m);
    }
    best
}

impl<F: SemiflowFloat> CsrGenerator<F> {
    /// Generator of a symmetric operator, `G = −M⁻¹A` (`mass = None` → `M = I`).
    ///
    /// # Errors
    /// `DomainViolation`: mass length / positivity / finiteness.
    pub fn from_symmetric(
        op: &SymmetricOperator<F>,
        mass: Option<&[F]>,
    ) -> Result<Self, SemiflowError> {
        let inner = Arc::clone(&op.inner);
        let norm = match mass {
            None => inner.spectral_radius_bound(),
            Some(m) => {
                check_mass(op.n(), m)?;
                row_gershgorin(inner.row_ptr(), inner.vals(), m)
            }
        };
        Ok(Self {
            backend: CsrBackend::Symmetric(inner),
            mass: mass.map(<[F]>::to_vec),
            norm,
        })
    }

    /// Generator of a general (non-symmetric) operator, `G = −M⁻¹A`.
    ///
    /// # Errors
    /// `DomainViolation`: mass length / positivity / finiteness.
    pub fn from_general(
        op: &GeneralOperator<F>,
        mass: Option<&[F]>,
    ) -> Result<Self, SemiflowError> {
        let norm = match mass {
            None => from_f64(op.norm_inf_bound()),
            Some(m) => {
                check_mass(op.n(), m)?;
                let (row_ptr, _, vals) = op.csr_parts();
                row_gershgorin(row_ptr, vals, m)
            }
        };
        Ok(Self {
            backend: CsrBackend::General(op.clone()),
            mass: mass.map(<[F]>::to_vec),
            norm,
        })
    }

    /// `dst ← A·src` (forward matrix).
    fn apply_a(&self, src: &[F], dst: &mut [F]) {
        match &self.backend {
            CsrBackend::Symmetric(lap) => Laplacian::apply_into_slice(lap, src, dst),
            CsrBackend::General(op) => op.apply_into_slice(src, dst),
        }
    }

    /// `dst ← Aᵀ·src`.
    fn apply_a_transpose(&self, src: &[F], dst: &mut [F]) {
        match &self.backend {
            CsrBackend::Symmetric(lap) => Laplacian::apply_into_slice(lap, src, dst),
            CsrBackend::General(op) => op.apply_transpose_into_slice(src, dst),
        }
    }
}

impl<F: SemiflowFloat> GeneratorAction<F> for CsrGenerator<F> {
    fn dim(&self) -> usize {
        match &self.backend {
            CsrBackend::Symmetric(lap) => lap.n_nodes(),
            CsrBackend::General(op) => op.n(),
        }
    }

    fn apply_generator(&self, src: &[F], dst: &mut [F]) {
        self.apply_a(src, dst);
        match &self.mass {
            None => {
                for d in dst.iter_mut() {
                    *d = -*d;
                }
            }
            Some(m) => {
                for (d, &mi) in dst.iter_mut().zip(m) {
                    *d = -*d / mi;
                }
            }
        }
    }

    fn norm_bound(&self) -> F {
        self.norm
    }

    fn apply_generator_transpose(&self, src: &[F], dst: &mut [F]) {
        match &self.mass {
            None => self.apply_a_transpose(src, dst),
            Some(m) => {
                let scaled: Vec<F> = src.iter().zip(m).map(|(&x, &mi)| x / mi).collect();
                self.apply_a_transpose(&scaled, dst);
            }
        }
        for d in dst.iter_mut() {
            *d = -*d;
        }
    }
}
