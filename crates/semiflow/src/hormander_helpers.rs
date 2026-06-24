// Heisenberg group backends — moved from hormander.rs (batch H6).
// Secondary impls to keep hormander.rs ≤ 500 lines.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use core::marker::PhantomData;

use crate::{error::SemiflowError, float::SemiflowFloat};

// ─── Heisenberg group d=3 (math.md §28.4.B) ──────────────────────────────────

/// Marker for the Heisenberg group `ℍ¹ = {(x, y, t)}` (math.md §28.4.B).
///
/// Left-invariant fields:
/// - `X₁ = ∂_x − (y/2)·∂_t`
/// - `X₂ = ∂_y + (x/2)·∂_t`
/// - Bracket: `[X₁, X₂] = ∂_t` — step-2 Carnot.
///
/// Sub-Laplacian: `L = ½(X₁² + X₂²)` (Folland 1975 *Ark. Mat.* §2).
/// Heat kernel involves complex integrals (Beals-Gaveau-Greiner 1997);
/// validation deferred to v4.0+ `SemiflowComplex`. Ships in v3.1 as a
/// constructive instance (compiles, passes step-checker, NOT gated).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeisenbergGroup<F: SemiflowFloat = f64> {
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat> HeisenbergGroup<F> {
    /// Construct the Heisenberg group marker (zero-sized, infallible).
    #[must_use]
    pub fn new() -> Self {
        Self { _f: PhantomData }
    }

    /// Return the left-invariant field `X₁ = ∂_x − (y/2)·∂_t`.
    #[must_use]
    pub fn x1() -> HeisenbergX<F> {
        HeisenbergX { _f: PhantomData }
    }

    /// Return the left-invariant field `X₂ = ∂_y + (x/2)·∂_t`.
    #[must_use]
    pub fn x2() -> HeisenbergY<F> {
        HeisenbergY { _f: PhantomData }
    }
}

impl<F: SemiflowFloat> Default for HeisenbergGroup<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Left-invariant field `X₁ = ∂_x − (y/2)·∂_t` on `ℍ¹`.
///
/// Coordinate layout: `x[0]=x, x[1]=y, x[2]=t`.
/// Returns `(1, 0, −y/2)`.
///
/// Reference: Folland 1975 *Ark. Mat.* §2.3; math.md §28.4.B.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeisenbergX<F: SemiflowFloat = f64> {
    pub(crate) _f: PhantomData<F>,
}

impl<F: SemiflowFloat> super::VectorField<F, 3> for HeisenbergX<F> {
    /// `X₁(x, y, t) = (1, 0, −y/2)`.
    fn evaluate(&self, x: &[F; 3], out: &mut [F; 3]) -> Result<(), SemiflowError> {
        let half = crate::float::from_f64::<F>(0.5_f64);
        out[0] = F::one();
        out[1] = F::zero();
        out[2] = -(half * x[1]); // −y/2
        Ok(())
    }
}

/// Left-invariant field `X₂ = ∂_y + (x/2)·∂_t` on `ℍ¹`.
///
/// Coordinate layout: `x[0]=x, x[1]=y, x[2]=t`.
/// Returns `(0, 1, x/2)`.
///
/// Reference: Folland 1975 *Ark. Mat.* §2.3; math.md §28.4.B.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeisenbergY<F: SemiflowFloat = f64> {
    pub(crate) _f: PhantomData<F>,
}

impl<F: SemiflowFloat> super::VectorField<F, 3> for HeisenbergY<F> {
    /// `X₂(x, y, t) = (0, 1, x/2)`.
    fn evaluate(&self, x: &[F; 3], out: &mut [F; 3]) -> Result<(), SemiflowError> {
        let half = crate::float::from_f64::<F>(0.5_f64);
        out[0] = F::zero();
        out[1] = F::one();
        out[2] = half * x[0]; // x/2
        Ok(())
    }
}
