//! §21.9 — Order-2 hard-wall Dirichlet via the ODD (antisymmetric) image method (ADR-0176).
//!
//! For a `ChernoffFunction` C approximating S(t) = exp(t·L) (e.g., heat) on a
//! region R with smooth boundary ∂R, the Dirichlet semigroup with absorbing
//! BC u|_{∂R} = 0 is approximated at order 2 by:
//!
//! ```text
//! F_D(τ) f(x) := C(τ) f̃(x),   x ∈ R
//! ```
//!
//! where `f̃` is the **odd extension** (math §21.9 (21.9.1)):
//!
//! ```text
//! f̃(x) =  f(x)        for x ∈ R
//! f̃(x) = -f(σ_R(x))   for x ∉ R
//! ```
//!
//! Realised at the stencil level by setting the grid boundary to
//! `BoundaryPolicy::OddReflect` before running one inner Chernoff step.
//! This is the minus-sign mirror of the Neumann (even) image method in §25.
//!
//! ## Mathematical justification (Proposition 21.9.1)
//!
//! `σ_R` is a Riemannian isometry; `L` is self-adjoint. The commutator
//! `[L, 𝟙_R − 𝟙_R∘σ_R] = 0` identically on the core of L (Prop 25.1 / 21.9.1
//! — the sign of the image term does not affect the vanishing argument). Hence
//! `F_D` introduces NO O(τ) commutator term and inherits the inner order.
//! The Dirichlet BC holds because the odd kernel `K^D(x₀,y;t) = 0` for
//! `x₀ ∈ ∂R` (21.9.3).
//!
//! **Self-adjoint L only.** For non-self-adjoint operators use `KillingChernoff`.
//!
//! ## Citations
//!
//! - ADR-0176 — design decision, TRIZ resolution summary.
//! - math.md §21.9 — NORMATIVE library (odd-image kernel K^D; Prop 21.9.1).
//! - math.md §25 (ADR-0072) — even-image Neumann method; this module is its `+→−` mirror.
//! - Walsh 1986 *Markov Processes and Potential Theory* §3.4 — image-method kernel.

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
// DirichletHeat2ndChernoff — wrapper
// ---------------------------------------------------------------------------

/// Chernoff wrapper for order-2 hard-wall Dirichlet BCs via the odd image method.
///
/// `F_D(τ) f = C(τ) f̃` where `f̃` is the odd extension of `f` across ∂R
/// (math §21.9 (21.9.1), ADR-0176). At the stencil level: clone `src`,
/// set boundary to `BoundaryPolicy::OddReflect`, run one inner Chernoff step.
///
/// `order()` returns `inner.order()` (Proposition 21.9.1 — no order cap).
///
/// ## Key difference from `KillingChernoff`
///
/// `KillingChernoff` (§21.3) is order-1: the discontinuous indicator `𝟙_R` has
/// an irreducible O(τ) boundary commutator. This wrapper escapes the cap by
/// replacing the indicator with an odd ghost — the commutator vanishes by
/// self-adjointness of L and isometry of σ_R.
///
/// ## Non-negativity note
///
/// The odd ghost subtracts mass — `F_D` does NOT preserve non-negativity.
/// This is physically correct: an absorbing Dirichlet wall removes mass.
///
/// ## v0.x scope
///
/// Concrete `ChernoffFunction<f64>` impl for D=1 half-line only (like §25).
/// Multi-D and generic-C generalisation deferred to a future milestone.
#[derive(Debug, Clone)]
pub struct DirichletHeat2ndChernoff<C, R, F = f64>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: ReflectingRegion<F>,
    F: SemiflowFloat,
{
    inner: C,
    /// The reflecting (half-space) region. Stored for public inspection and
    /// future multi-D impls.
    pub region: R,
    _f: PhantomData<F>,
}

impl<C, R, F> DirichletHeat2ndChernoff<C, R, F>
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
// ChernoffFunction<f64> for D=1 HalfSpaceRegion + DiffusionChernoff case
// ---------------------------------------------------------------------------

/// Odd-image `apply_into` for the D=1 half-line.
///
/// ## Implementation note (half-line `[0, L]` grid)
///
/// Mirror of `ReflectedHeatChernoff` (§25) with `Reflect → OddReflect`:
/// setting the grid boundary to `BoundaryPolicy::OddReflect` causes
/// `DiffusionChernoff`'s stencil to see `−f(|x|)` at ghost nodes left of x=0.
/// By antisymmetry, the interpolated value AT x=0 is then 0 exactly — the
/// Dirichlet BC falls out of oddness without any extra masking step.
///
/// Equivalence proof sketch (single node x=0):
///
/// ```text
/// Odd full-line at x=0:
///   C(τ) f̃ (0) = W0*f(0) + W1*(f(h) - f(h)) + W2*(f(h') - f(h')) = W0*f(0)
///              ≈ f(0) [self-consistent: f(0)=0 by IC; BC maintained]
/// OddReflect step at x=0:
///   W0*f(0) + W1*(f(h) + (−f(h))) + W2*(f(h') + (−f(h'))) = W0*f(0)
/// ```
///
/// For interior nodes the boundary policy is not accessed (stencil stays inside
/// `[0, L]`); the odd boundary only affects the single stencil cell at x=0.
impl ChernoffFunction<f64>
    for DirichletHeat2ndChernoff<DiffusionChernoff<f64>, HalfSpaceRegion<f64, 1>, f64>
{
    type S = GridFn1D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        // Build a view of src with OddReflect boundary.
        // This realises the odd extension at x = 0 (the half-space boundary):
        //   sample(x_0 - δ) → −sample(x_0 + δ)  [mirror + negate at left edge]
        // Equivalent to the full-line odd-image method restricted to [0, L].
        let mut src_odd = src.clone();
        src_odd.grid = src_odd.grid.with_boundary(BoundaryPolicy::OddReflect);

        // Single inner step with OddReflect boundary = order-2 Dirichlet step.
        self.inner.apply_into(tau, &src_odd, dst, scratch)
    }

    /// Order matches inner (Proposition 21.9.1 — no cap for Dirichlet odd-image BCs).
    fn order(&self) -> u32 {
        self.inner.order()
    }

    /// Growth: `(inner_M, inner_ω)` — no doubling on a one-sided grid because
    /// the OddReflect-boundary step is a single contraction of the inner Chernoff.
    fn growth(&self) -> Growth<f64> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{DirichletHeat2ndChernoff, HalfSpaceRegion};
    use crate::{
        chernoff::ChernoffFunction, diffusion::DiffusionChernoff, grid::Grid1D,
        grid_fn::GridFn1D, scratch::ScratchPool,
    };

    /// `order()` must match inner DiffusionChernoff (which returns 2).
    /// Mirrors `reflected_heat_chernoff_order_matches_inner` in reflection.rs.
    #[test]
    fn dirichlet_heat2nd_order_matches_inner() {
        let grid = Grid1D::new(0.0_f64, 10.0, 16).unwrap();
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let region = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        let wrapper = DirichletHeat2ndChernoff::new(inner, region).unwrap();
        assert_eq!(wrapper.order(), 2, "odd-image must preserve inner order (Prop 21.9.1)");
    }

    /// Smoke: `apply_into` runs without panic and produces finite values.
    #[test]
    fn dirichlet_heat2nd_smoke() {
        let grid = Grid1D::new(0.0_f64, 4.0, 16).unwrap();
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let region = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        let wrapper = DirichletHeat2ndChernoff::new(inner, region).unwrap();
        let u0 = GridFn1D::from_fn(grid, |x| (-(x - 2.0).powi(2)).exp());
        let mut u1 = GridFn1D::from_fn(grid, |_| 0.0_f64);
        let mut scratch = ScratchPool::new();
        wrapper.apply_into(0.001, &u0, &mut u1, &mut scratch).unwrap();
        assert!(u1.values.iter().all(|v| v.is_finite()), "all values must be finite");
    }
    // NOTE: no non-negativity test — the odd ghost subtracts mass, so u can go
    // negative near the absorbing wall. This is CORRECT (ADR-0176 §Consequences).
}
