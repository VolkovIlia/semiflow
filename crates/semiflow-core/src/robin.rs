//! A.3 Robin BC partial-additive port (math.md §3.5.tris, ADR-0098).
//!
//! Standalone sibling of v2.8 `ReflectedHeatChernoff` (ADR-0072); REUSES `σ_R`
//! via the `ReflectingRegion<F>` trait (no new geometric-reflection trait —
//! the geometry is shared with Neumann; only the scalar mixing coefficient
//! `r(α, β, τ)` differs from Neumann's `r ≡ 1`).
//!
//! ## Order
//!
//! `RobinHeatChernoff::order() = 1`. The skew extension `1_R + r · 1_R ∘ σ_R`
//! has a NONZERO commutator with self-adjoint L for `r ≠ 1` (math §3.5.tris.5).
//! Proposition 25.1 (Neumann order-preservation) does NOT extend to Robin.
//!
//! ## Kernel (Carslaw-Jaeger 1959 §14.2 eq 5)
//!
//! The exact Robin heat kernel on `[0, ∞)` is:
//!
//! ```text
//! K^Robin(x, y; t) = K(x,y,t) + K(x,-y,t)
//!                   - (α/β)·exp((α/β)(x+y) + (α/β)²t)·erfc((x+y)/(2√t) + (α/β)√t)
//! ```
//!
//! (Factor is `(α/β)`, NOT `2·(α/β)` — see ADR-0098 Amendment 1.)
//! Satisfies: `α·K(0,y;t) − β·∂_x K(0,y;t) = 0` (outward-normal form;
//! `∂_n = −∂_x` at x=0 for `[0, ∞)`).
//!
//! ## Citations
//! - Carslaw-Jaeger 1959 §3.4 (image-method skew-r derivation) + §14.2 eq 5
//! - Walsh 1986 §3.4 (image-method generalisation to skew reflection)
//! - Engel-Nagel 2000 Ch. VI §6 (Robin generator characterisation)
//! - research1.md Part B Vector 4 (partial-additivity verdict; Dynamic BC unsettled)

use core::marker::PhantomData;

use crate::{
    boundary::BoundaryPolicy,
    chernoff::{ChernoffFunction, Growth},
    diffusion::DiffusionChernoff,
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    reflection::{HalfSpaceRegion, ReflectingRegion},
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// RobinRegion<F> — sub-trait of ReflectingRegion<F>
// ---------------------------------------------------------------------------

/// Robin region — sub-trait of `ReflectingRegion<F>` that adds the scalar
/// Robin coefficients `(α, β)` to the geometric-reflection contract.
///
/// v4.6 stores `(alpha, beta)` constant across ∂R; per-cell varying coefficients
/// deferred to v5.x via a future `robin_coeffs_at(point: &[F]) -> (F, F)` method.
pub trait RobinRegion<F: SemiflowFloat>: ReflectingRegion<F> {
    /// Return the scalar Robin coefficients `(α, β)` for the region.
    fn robin_coeffs(&self) -> (F, F);
}

// ---------------------------------------------------------------------------
// HalfSpaceRobin<F, D> — Robin BC on a half-space
// ---------------------------------------------------------------------------

/// Half-space Robin region: wraps `HalfSpaceRegion<F, D>` with scalar `(α, β)`.
///
/// Construction validates `‖normal‖₂ = 1` (delegated to `HalfSpaceRegion::new`)
/// and `alpha ≥ 0 ∧ beta > 0` (well-posedness of static Robin BC).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfSpaceRobin<F: SemiflowFloat = f64, const D: usize = 1> {
    /// Underlying half-space geometry (origin + unit outward normal).
    pub half_space: HalfSpaceRegion<F, D>,
    /// Coefficient α on u(x) at the boundary; α ≥ 0.
    pub alpha: F,
    /// Coefficient β on ∂_n u(x) at the boundary; β > 0.
    pub beta: F,
}

impl<F: SemiflowFloat, const D: usize> HalfSpaceRobin<F, D> {
    /// Construct with validated unit normal and Robin coefficients.
    ///
    /// # Errors
    /// - `DomainViolation` if `‖normal‖₂ ≠ 1` (via `HalfSpaceRegion::new`).
    /// - `DomainViolation` if `alpha < 0` or `beta ≤ 0` or any value non-finite.
    pub fn new(origin: [F; D], normal: [F; D], alpha: F, beta: F) -> Result<Self, SemiflowError> {
        let half_space = HalfSpaceRegion::new(origin, normal)?;
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "HalfSpaceRobin: alpha and beta must be finite",
                value: f64::NAN,
            });
        }
        if alpha < F::zero() || beta <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "HalfSpaceRobin: require alpha ≥ 0 and beta > 0 (well-posed Robin BC)",
                value: alpha.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self {
            half_space,
            alpha,
            beta,
        })
    }
}

impl<F: SemiflowFloat, const D: usize> ReflectingRegion<F> for HalfSpaceRobin<F, D> {
    fn dim(&self) -> usize {
        self.half_space.dim()
    }

    fn is_inside(&self, point: &[F]) -> bool {
        self.half_space.is_inside(point)
    }

    fn reflect_in_place(
        &self,
        dst: &mut GridFn1D<F>,
        src: &GridFn1D<F>,
    ) -> Result<(), SemiflowError> {
        self.half_space.reflect_in_place(dst, src)
    }
}

impl<F: SemiflowFloat, const D: usize> RobinRegion<F> for HalfSpaceRobin<F, D> {
    fn robin_coeffs(&self) -> (F, F) {
        (self.alpha, self.beta)
    }
}

// ---------------------------------------------------------------------------
// RobinHeatChernoff<C, R, F> — wrapper
// ---------------------------------------------------------------------------

/// Chernoff wrapper for Robin (mixed) BCs via skew image method (math §3.5.tris.3).
///
/// Per-step: `F_Robin(τ)f = C(τ)f + r(α, β, τ) · C(τ)(f ∘ σ_R)`
/// where `r(α, β, τ) = (β − α·√(2τ)) / (β + α·√(2τ))`.
///
/// `order()` = 1 (skew extension breaks Proposition 25.1 order-preservation;
/// the commutator `[1_R ∘ σ_R, L]` is nonzero for `r ≠ 1`).
#[derive(Debug, Clone)]
pub struct RobinHeatChernoff<C, R, F = f64>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: RobinRegion<F>,
    F: SemiflowFloat,
{
    inner: C,
    /// The Robin region providing geometry + (α, β) coefficients.
    pub region: R,
    _f: PhantomData<F>,
}

impl<C, R, F> RobinHeatChernoff<C, R, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: RobinRegion<F>,
    F: SemiflowFloat,
{
    /// Wrap `inner` Chernoff function with Robin region `region`.
    ///
    /// # Errors
    /// Currently infallible; returns `Ok` for API consistency.
    pub fn new(inner: C, region: R) -> Result<Self, SemiflowError> {
        Ok(Self {
            inner,
            region,
            _f: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Concrete ChernoffFunction<f64> impl
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64>
    for RobinHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRobin<f64, 1>, f64>
{
    type S = GridFn1D<f64>;

    /// Option A skew image, ADR-0098 Am.2.
    ///
    /// Mirrors the proven-working `ReflectedHeatChernoff` (G27/Neumann) pattern:
    /// set `src.grid.boundary = BoundaryPolicy::Robin { alpha, beta }` so the
    /// inner `DiffusionChernoff` stencil samples ghost nodes via the exponential
    /// skew image `u_{-d} = exp(-2(α/β)·d·dx)·u_{d}`. α=0 ⟹ weight 1 = even
    /// reflection = Neumann (G27 cannot regress).
    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        let (alpha, beta) = self.region.robin_coeffs();
        let mut src_robin = src.clone();
        src_robin.grid = src_robin
            .grid
            .with_boundary(BoundaryPolicy::Robin { alpha, beta });
        self.inner.apply_into(tau, &src_robin, dst, scratch)
    }

    /// Order = 1 (skew extension breaks Proposition 25.1; math §3.5.tris.5).
    fn order(&self) -> u32 {
        1
    }

    /// Growth: same as inner (skew adds bounded scalar; `r ∈ [−1, +1]`).
    fn growth(&self) -> Growth<f64> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{HalfSpaceRobin, RobinHeatChernoff, RobinRegion};
    use crate::{
        diffusion::DiffusionChernoff, error::SemiflowError, grid::Grid1D, ChernoffFunction,
    };

    #[test]
    fn half_space_robin_construction_ok() {
        let r = HalfSpaceRobin::<f64, 1>::new([0.0], [1.0], 1.0, 1.0);
        assert!(r.is_ok());
    }

    #[test]
    fn half_space_robin_negative_alpha_err() {
        let err = HalfSpaceRobin::<f64, 1>::new([0.0], [1.0], -1.0, 1.0).unwrap_err();
        assert!(matches!(err, SemiflowError::DomainViolation { .. }));
    }

    #[test]
    fn half_space_robin_zero_beta_err() {
        let err = HalfSpaceRobin::<f64, 1>::new([0.0], [1.0], 1.0, 0.0).unwrap_err();
        assert!(matches!(err, SemiflowError::DomainViolation { .. }));
    }

    #[test]
    fn robin_heat_chernoff_order_is_1() {
        let grid = Grid1D::new(0.0_f64, 10.0, 16).unwrap();
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let region = HalfSpaceRobin::<f64, 1>::new([0.0], [1.0], 1.0, 1.0).unwrap();
        let wrapper = RobinHeatChernoff::new(inner, region).unwrap();
        assert_eq!(
            wrapper.order(),
            1,
            "Robin order is 1 (skew commutator non-vanishing)"
        );
    }

    #[test]
    fn robin_alpha_zero_skew_weight_recovers_neumann() {
        // v6.2.3: apply_into uses the skew image weight w^(d)=exp(-2(α/β)·d·dx),
        // NOT the removed scalar r-formula. At α=0 the weight is 1 = even
        // reflection = Neumann (math §3.5.tris.3, ADR-0098 Amendment 2).
        let alpha = 0.0_f64;
        let beta = 1.0_f64;
        let dx = 0.1_f64;
        let depth = 5.0_f64;
        let w = libm::exp(-2.0 * (alpha / beta) * depth * dx); // e^0 == 1
        assert!(
            (w - 1.0).abs() < 1e-14,
            "alpha=0 skew weight must be 1 (Neumann limit), got {w}"
        );
    }

    #[test]
    fn robin_region_coeffs_round_trip() {
        let region = HalfSpaceRobin::<f64, 1>::new([0.0], [1.0], 2.5, 0.7).unwrap();
        let (a, b) = region.robin_coeffs();
        assert!((a - 2.5).abs() < 1e-15);
        assert!((b - 0.7).abs() < 1e-15);
    }
}
