//! Native complex state type for the Schrödinger Option B kernel.
//!
//! Extracted from `schrodinger_complex.rs` to keep that file within the 500-line
//! suckless budget (same pattern as `quantum_graph_data.rs`).

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use alloc::vec::Vec;

use num_traits::{Float, Zero};

use crate::{complex::SemiflowComplex, error::SemiflowError, grid::Grid1D, state::State};

// ---------------------------------------------------------------------------
// GridFnComplex1D<C> — native complex state type
// ---------------------------------------------------------------------------

/// Native complex grid function: `ψ(x_i) ∈ C` for each node `i`.
///
/// Introduced because `GridFn1D<F>` requires `F: SemiflowFloat`, which
/// `num_complex::Complex<f64>` does NOT satisfy. `GridFnComplex1D<C>` is the
/// Option B state type for [`crate::SchrödingerChernoffComplex`].
///
/// Implements [`State<C::Real>`] so it plugs into [`crate::Evolver`].
#[derive(Clone, Debug)]
pub struct GridFnComplex1D<C: SemiflowComplex> {
    /// Complex amplitudes at grid nodes. Length = `grid.n`.
    pub values: Vec<C>,
    /// Grid geometry (owned; `Grid1D<C::Real>: Copy`).
    pub grid: Grid1D<C::Real>,
}

impl<C: SemiflowComplex> GridFnComplex1D<C> {
    /// Construct from pre-computed values. Validates `values.len() == grid.n`.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `values.len() != grid.n`.
    pub fn new(grid: Grid1D<C::Real>, values: Vec<C>) -> Result<Self, SemiflowError> {
        if values.len() != grid.n {
            return Err(SemiflowError::DomainViolation {
                what: "GridFnComplex1D: values.len() must equal grid.n",
                value: values.len() as f64,
            });
        }
        Ok(Self { values, grid })
    }

    /// Construct by sampling a closure at each grid node.
    pub fn from_fn(grid: Grid1D<C::Real>, f: impl Fn(C::Real) -> C) -> Self {
        let values: Vec<C> = (0..grid.n).map(|i| f(grid.x_at(i))).collect();
        Self { values, grid }
    }

    /// Discrete L²-norm squared: `Σ |ψ_k|² · dx`.
    pub fn norm_l2_sq(&self) -> C::Real {
        let dx = self.grid.dx();
        self.values
            .iter()
            .fold(<C::Real as Zero>::zero(), |acc, &z| {
                acc + z.abs() * z.abs() * dx
            })
    }

    /// Discrete L²-norm: `√(Σ |ψ_k|² · dx)`.
    pub fn norm_l2(&self) -> C::Real {
        self.norm_l2_sq().sqrt()
    }
}

// State<C::Real> impl — required for Evolver
impl<C: SemiflowComplex> State<C::Real> for GridFnComplex1D<C> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    /// `self[k] ← self[k] + alpha · src[k]` (real scalar · complex vector).
    #[inline]
    fn axpy_into(&mut self, alpha: C::Real, src: &Self) {
        for (s, &v) in self.values.iter_mut().zip(src.values.iter()) {
            *s += C::from_real(alpha) * v;
        }
    }

    #[inline]
    fn copy_from(&mut self, src: &Self) {
        self.values.copy_from_slice(&src.values);
    }

    #[inline]
    fn zero_into(&mut self) {
        for z in &mut self.values {
            *z = C::zero();
        }
    }

    /// Sup-norm of amplitudes: `max_k |ψ_k|`.
    fn norm_sup(&self) -> C::Real {
        self.values
            .iter()
            .fold(<C::Real as Zero>::zero(), |acc, &z| {
                let a = z.abs();
                if a > acc {
                    a
                } else {
                    acc
                }
            })
    }

    #[inline]
    fn scale_into(&mut self, k: C::Real) {
        let ck = C::from_real(k);
        for z in &mut self.values {
            *z *= ck;
        }
    }
}
