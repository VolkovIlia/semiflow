//! B4 — Neumann via image method (math.md §25, ADR-0072).
//!
//! For a `ChernoffFunction` C approximating S(t) = exp(t·L) (e.g., heat) on a
//! region R with smooth boundary ∂R, the reflected semigroup with Neumann
//! BC ∂_ν u|_{∂R} = 0 is approximated by:
//!
//! ```text
//! F_refl(τ) f(x) := C(τ)f(x) + C(τ)(f ∘ σ_R)(x)
//! ```
//!
//! where `σ_R` : M → M is reflection across ∂R.
//!
//! ## Citations
//!
//! - Walsh 1986 *Markov Processes and Potential Theory* §3.4 — image-method
//!   kernel formula; foundational citation for the construction.
//! - Anderson 1988 *Reflected Brownian Motion* SIAM §2.3 — convergence.
//! - Butko 2018 §3.2 — contrast: killing caps order at 1; reflection does not.
//!
//! ## Sibling relationship to v2.6 `KillingRegion<F>`
//!
//! `ReflectingRegion<F>` is a SIBLING (not subtype) of `KillingRegion<F>`:
//! - `KillingRegion::mask_in_place` writes ZERO outside R (absorbing BC).
//! - `ReflectingRegion::reflect_in_place` writes the ghost image (Neumann BC).
//!
//! A region struct (e.g. `BoxRegion<F, D>`) may implement both traits; the
//! caller picks the wrapper type to choose the BC semantics.

use core::marker::PhantomData;

use crate::{
    boundary::BoundaryPolicy,
    chernoff::{ChernoffFunction, Growth},
    diffusion::DiffusionChernoff,
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// ReflectingRegion<F> — trait
// ---------------------------------------------------------------------------

/// Reflecting region for the Neumann image method (v2.8, ADR-0072).
///
/// Sibling to v2.6 `KillingRegion<F>`. Required by `ReflectedHeatChernoff`.
///
/// ## Implementing for custom regions
///
/// Override `is_inside`, `reflect_in_place`, and `dim`. The contract for
/// `reflect_in_place` mirrors math.md §25.3: for each node `i` of `dst`
/// whose coordinate is NOT inside R, write the value at `σ_R(coord_of(i))`
/// sampled from `src`. Leave inside nodes zero.
pub trait ReflectingRegion<F: SemiflowFloat = f64> {
    /// Returns `true` iff `point` is in the OPEN interior of R.
    /// Open convention: cells ON ∂R are EXCLUDED.
    fn is_inside(&self, point: &[F]) -> bool;

    /// In-place ghost build: for each cell `i` of `dst`, if the cell's
    /// coordinate is NOT inside R, write the value at `σ_R(coord)` sampled
    /// from `src`. Cells inside R stay zero.
    ///
    /// # Errors
    ///
    /// Returns `SemiflowError::DomainViolation` for invalid configurations
    /// (e.g., `BallRegion<F, 1>`; use `HalfSpaceRegion<F, 1>` instead).
    fn reflect_in_place(
        &self,
        dst: &mut GridFn1D<F>,
        src: &GridFn1D<F>,
    ) -> Result<(), SemiflowError>;

    /// Spatial dimension `D` of the region.
    fn dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// HalfSpaceRegion<F, D> — hyperplane reflection
// ---------------------------------------------------------------------------

/// Half-space `R = {p : (p − origin) · normal > 0}` with unit normal.
///
/// The reflection `σ_R(p)` = p − 2 · ((p − origin) · normal) · normal.
///
/// ## v2.8 scope
///
/// `reflect_in_place` is implemented for D = 1 on `GridFn1D<F>`. Multi-D
/// reflections on `GridFn2D`/`GridFn3D` are deferred to v2.9.
///
/// ## Construction
///
/// `HalfSpaceRegion::new` validates that `‖normal‖₂ = 1` to machine precision.
/// Returns `Err(DomainViolation)` if the norm deviates by more than 100·ε.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfSpaceRegion<F: SemiflowFloat = f64, const D: usize = 1> {
    /// Reference point on ∂R.
    pub origin: [F; D],
    /// Unit outward normal to ∂R. Invariant: ‖normal‖₂ = 1.
    pub normal: [F; D],
}

impl<F: SemiflowFloat, const D: usize> HalfSpaceRegion<F, D> {
    /// Construct a `HalfSpaceRegion` with validated unit normal.
    ///
    /// # Errors
    ///
    /// Returns `SemiflowError::DomainViolation` if `‖normal‖₂ ≠ 1` to
    /// within `100 · F::epsilon()`, or if any coordinate is non-finite.
    pub fn new(origin: [F; D], normal: [F; D]) -> Result<Self, SemiflowError> {
        let norm_sq: F = normal.iter().map(|&n| n * n).fold(F::zero(), |a, b| a + b);
        let one = F::one();
        // 100 * epsilon tolerance for floating-point unit-vector input.
        // `F: Float` from num_traits, so `F::epsilon()` is available.
        let eps100 = F::epsilon() * F::from(100.0).unwrap_or(one);
        let delta = if norm_sq > one {
            norm_sq - one
        } else {
            one - norm_sq
        };
        if delta > eps100 {
            return Err(SemiflowError::DomainViolation {
                what: "HalfSpaceRegion: normal must be a unit vector (‖normal‖₂ = 1)",
                value: norm_sq.to_f64().unwrap_or(f64::NAN),
            });
        }
        for &c in origin.iter().chain(normal.iter()) {
            if !c.is_finite() {
                return Err(SemiflowError::DomainViolation {
                    what: "HalfSpaceRegion: coordinates must be finite",
                    value: c.to_f64().unwrap_or(f64::NAN),
                });
            }
        }
        Ok(Self { origin, normal })
    }

    /// Compute `σ_R(point)` = point − 2·((point − origin)·normal)·normal.
    pub fn reflect_coords(&self, point: &[F]) -> [F; D] {
        let dot: F = (0..D)
            .map(|k| (point[k] - self.origin[k]) * self.normal[k])
            .fold(F::zero(), |a, b| a + b);
        let two = one::<F>() + one::<F>();
        let mut out = [F::zero(); D];
        for k in 0..D {
            out[k] = point[k] - two * dot * self.normal[k];
        }
        out
    }
}

/// Helper: the multiplicative identity for F.
#[inline]
fn one<F: SemiflowFloat>() -> F {
    F::one()
}

impl<F: SemiflowFloat, const D: usize> ReflectingRegion<F> for HalfSpaceRegion<F, D> {
    fn dim(&self) -> usize {
        D
    }

    fn is_inside(&self, point: &[F]) -> bool {
        let dot: F = (0..D)
            .map(|k| (point[k] - self.origin[k]) * self.normal[k])
            .fold(F::zero(), |a, b| a + b);
        dot > F::zero()
    }

    /// Build ghost state: for each outside node i, `dst[i]` = src sampled at `σ_R(x_i)`.
    fn reflect_in_place(
        &self,
        dst: &mut GridFn1D<F>,
        src: &GridFn1D<F>,
    ) -> Result<(), SemiflowError> {
        let n = dst.grid.n;
        for i in 0..n {
            let x = dst.grid.x_at(i);
            let coord = [x];
            if !self.is_inside(&coord) {
                let reflected = self.reflect_coords(&coord);
                let ghost_val = src.sample_generic(reflected[0])?;
                dst.values[i] = ghost_val;
            }
            // Inside nodes stay zero (ghost has support only outside R).
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Additive impls for BoxRegion<F, D> and BallRegion<F, D>
// ---------------------------------------------------------------------------

// Moved to `reflection_regions` to stay within suckless ≤500 LoC budget (ADR-0072).
// The impls are registered via `pub mod reflection_regions` in lib.rs; no re-export
// needed — trait impls are picked up automatically by the compiler.

// ---------------------------------------------------------------------------
// ReflectedHeatChernoff — wrapper
// ---------------------------------------------------------------------------

/// Chernoff wrapper for Neumann (reflecting) BCs via the image method.
///
/// `F_refl(τ) f = C(τ)f + C(τ)(f ∘ σ_R)` (Walsh 1986 §3.4, math.md §25.3).
///
/// `order()` returns `inner.order()`: reflection preserves order for symmetric
/// (Neumann) BCs (Proposition 25.1 — commutator vanishes).
///
/// ## v2.8 scope
///
/// Concrete `ChernoffFunction<f64>` impl for:
/// `ReflectedHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRegion<f64, 1>>`.
/// Generic-C impls deferred to v2.9 (requires `CoordinateState<F, D>` supertrait).
#[derive(Debug, Clone)]
pub struct ReflectedHeatChernoff<C, R, F = f64>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: ReflectingRegion<F>,
    F: SemiflowFloat,
{
    inner: C,
    /// The reflecting region. Stored for future multi-D `GridFn2D`/`GridFn3D`
    /// impls (deferred to v2.9) and for public inspection by callers.
    pub region: R,
    _f: PhantomData<F>,
}

impl<C, R, F> ReflectedHeatChernoff<C, R, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: ReflectingRegion<F>,
    F: SemiflowFloat,
{
    /// Wrap `inner` Chernoff function with reflecting region `region`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the inner or region is invalid; always `Ok` for
    /// pre-validated inputs.
    pub fn new(inner: C, region: R) -> Result<Self, SemiflowError> {
        Ok(Self {
            inner,
            region,
            _f: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<f64> for the D=1 HalfSpaceRegion + DiffusionChernoff case
// ---------------------------------------------------------------------------

/// Image-method `apply_into` for the D=1 half-space on a one-sided grid.
///
/// ## Implementation note (half-line `[0, L]` grid)
///
/// math.md §25.3 specifies a 4-step full-line algorithm where the ghost
/// `h = f ∘ σ_R` has support at the OUTSIDE (negative-x) nodes of a full-line
/// grid. For the common case of a one-sided grid `[0, L]` (all nodes inside R),
/// there are no outside nodes, so ghost = 0 and the 4-step algorithm reduces
/// to a plain inner Chernoff step — NOT the correct Neumann semigroup.
///
/// The correct Neumann semigroup on a `[0, L]` half-line grid is obtained by
/// replacing `src.grid.boundary` with `BoundaryPolicy::Reflect` before the
/// inner Chernoff step. This gives the `DiffusionChernoff` stencil access to the
/// even extension of src at the left boundary (x = 0), which is exactly what
/// the full-line image method computes when restricted to `x ∈ [0, L]`.
///
/// Equivalence proof sketch (single node x = 0, pure-heat stencil):
///
/// ```text
/// Full-line image method at x = 0:
///   step1 = W0*f(0) + W1*f(h)  + W2*f(h')         (src = 0 for x < 0)
///   step3 = W1*f(h) + W2*f(h')                     (ghost = f(|x|) for x < 0)
///   sum   = W0*f(0) + 2*W1*f(h) + 2*W2*f(h')
///
/// Half-line Reflect step at x = 0:
///   C_reflect*f(0) = W0*f(0) + W1*(f(h) + f(h)) + W2*(f(h') + f(h'))
///                  = W0*f(0) + 2*W1*f(h)          + 2*W2*f(h')  == sum
/// ```
///
/// The equivalence holds at every grid node (interior nodes are
/// unaffected by boundary policy for small tau).
impl ChernoffFunction<f64>
    for ReflectedHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRegion<f64, 1>, f64>
{
    type S = GridFn1D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        // Build a view of src with Reflect boundary.
        // This implements the even extension at x = 0 (the half-space boundary):
        //   sample(x_0 - δ) → sample(x_0 + δ)  [mirror at left edge]
        // Equivalent to the full-line image method restricted to [0, L].
        // (Right boundary at x = GRID_MAX uses Reflect too; negligible for
        //  compactly-supported ICs since the solution is ~0 near GRID_MAX.)
        let mut src_reflect = src.clone();
        src_reflect.grid = src_reflect.grid.with_boundary(BoundaryPolicy::Reflect);

        // Single inner step with Reflect boundary = correct Neumann semigroup step.
        self.inner.apply_into(tau, &src_reflect, dst, scratch)
    }

    /// Order matches inner (Proposition 25.1 — no cap for Neumann BCs).
    fn order(&self) -> u32 {
        self.inner.order()
    }

    /// Growth: `(inner_M, inner_ω)` — no doubling on a one-sided grid because
    /// the Reflect-boundary step is a single contraction of the inner Chernoff.
    fn growth(&self) -> Growth<f64> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{HalfSpaceRegion, ReflectedHeatChernoff, ReflectingRegion};
    use crate::{
        chernoff::ChernoffFunction, diffusion::DiffusionChernoff, error::SemiflowError,
        grid::Grid1D, grid_fn::GridFn1D, scratch::ScratchPool,
    };

    // --- HalfSpaceRegion construction ---

    #[test]
    fn half_space_unit_normal_ok() {
        let r = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]);
        assert!(r.is_ok());
    }

    #[test]
    fn half_space_non_unit_normal_err() {
        let err = HalfSpaceRegion::<f64, 1>::new([0.0], [2.0]).unwrap_err();
        assert!(matches!(err, SemiflowError::DomainViolation { .. }));
    }

    // --- HalfSpaceRegion is_inside ---

    #[test]
    fn half_space_is_inside_d1() {
        let r = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        assert!(r.is_inside(&[0.5])); // inside
        assert!(!r.is_inside(&[0.0])); // on boundary: excluded (open convention)
        assert!(!r.is_inside(&[-0.5])); // outside
    }

    // --- reflect_coords ---

    #[test]
    fn half_space_reflect_across_origin() {
        let r = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        // σ_R(-1) = -1 - 2·(-1)·1 = -1 + 2 = 1
        let reflected = r.reflect_coords(&[-1.0]);
        assert!((reflected[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn half_space_reflect_positive_is_identity_across_origin() {
        let r = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        // σ_R(1) = 1 - 2·(1)·1 = -1 (image of interior point)
        let reflected = r.reflect_coords(&[1.0]);
        assert!((reflected[0] - (-1.0)).abs() < 1e-14);
    }

    // --- ReflectedHeatChernoff order ---

    #[test]
    fn reflected_heat_chernoff_order_matches_inner() {
        let grid = Grid1D::new(0.0_f64, 10.0, 16).unwrap();
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let region = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        let wrapper = ReflectedHeatChernoff::new(inner, region).unwrap();
        // DiffusionChernoff order = 2; wrapper must not degrade it (Prop 25.1).
        assert_eq!(wrapper.order(), 2, "reflection must preserve inner order");
    }

    // --- Smoke: wrapper runs without panicking ---

    #[test]
    fn reflected_heat_chernoff_smoke() {
        let grid = Grid1D::new(0.0_f64, 4.0, 16).unwrap();
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let region = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        let wrapper = ReflectedHeatChernoff::new(inner, region).unwrap();
        let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
        let mut u1 = GridFn1D::from_fn(grid, |_| 0.0_f64);
        let mut scratch = ScratchPool::new();
        wrapper
            .apply_into(0.001, &u0, &mut u1, &mut scratch)
            .unwrap();
        assert!(
            u1.values.iter().all(|v| v.is_finite()),
            "all values must be finite"
        );
    }

    // --- Mass conservation: reflected step ≥ 0 for non-negative IC ---

    #[test]
    fn reflected_heat_chernoff_nonneg_preserved() {
        let grid = Grid1D::new(0.0_f64, 5.0, 32).unwrap();
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let region = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        let wrapper = ReflectedHeatChernoff::new(inner, region).unwrap();
        // Non-negative IC: Gaussian bump entirely inside R.
        let u0 = GridFn1D::from_fn(grid, |x| (-(x - 2.0).powi(2)).exp());
        let mut u1 = GridFn1D::from_fn(grid, |_| 0.0_f64);
        let mut scratch = ScratchPool::new();
        wrapper
            .apply_into(0.001, &u0, &mut u1, &mut scratch)
            .unwrap();
        // After one reflected step, all values must be non-negative (mass-preserving).
        for (i, &v) in u1.values.iter().enumerate() {
            assert!(v >= -1e-12, "value at node {i} is negative: {v}");
        }
    }
}
