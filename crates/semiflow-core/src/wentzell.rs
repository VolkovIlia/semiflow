//! C-9 — Dynamic Wentzell/Robin BC via implicit Cayley boundary step (math.md §49, ADR-0151).
//!
//! The dynamic Wentzell/Robin condition `∂_t u + γ(t)·∂_ν u + c·u = 0` on `∂Ω`
//! is advanced by a **bulk–boundary Lie split**:
//!   (a) One bulk Chernoff step `C(τ)` (the inner `DiffusionChernoff`).
//!   (b) An **implicit Cayley boundary sub-step** advancing the 2×2-per-boundary-DOF
//!       coupled block `K_CN = (I − τC_∂/2)⁻¹(I + τC_∂/2)` (math §49.3).
//!
//! The Cayley/Möbius map sends the closed left half-plane to the closed unit disk,
//! so the unbounded normal-derivative coupling yields `ρ(K_CN) ≤ 1` at any stiffness,
//! including time-dependent `γ(t)` (ADR-0146 stability VERDICT: GO).
//!
//! **Stability:** A-stable (unconditional), unlike the explicit Trotter/Stephan product.
//! A weak CFL `τ ≤ c·h` is an ACCURACY requirement for the Lie split, not a stability bound.
//!
//! ## 1D collapse: boundary-trace DOF = `dst.values[0]`
//!
//! In 1D on `[0, ∞)` the product space `X ⊕ L²(∂Ω)` collapses to `X ⊕ ℝ_∂`;
//! the trace DOF is exactly `dst.values[0]` (the boundary node). Multi-D true-product
//! state is deferred to v8.x (math §49.7).
//!
//! ## Order
//!
//! `DynamicWentzellChernoff::order() = 1`. The bulk↔boundary commutator is nonzero
//! and the left-endpoint freezing of `γ(t)` is order-1 (matches static Robin ADR-0098
//! and the Altmann–Verfürth Lie splitting; math §49.8).
//!
//! ## Citations
//! - Altmann–Verfürth, *IMA J. Numer. Anal.* 2023 (arXiv:2108.08147) — implicit-Euler Lie split.
//! - Kovács–Lubich, *IMA J. Numer. Anal.* 2017 (arXiv:1501.01882) — A-stable implicit BC.
//! - Stephan 2023 (arXiv:2307.00419, ZAMM 2025) — explicit product diverges (§5).
//! - math.md §49 (NORMATIVE), ADR-0151, ADR-0146.

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion::DiffusionChernoff,
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    howland::TimedChernoffFunction,
    reflection::{HalfSpaceRegion, ReflectingRegion},
    robin::RobinRegion,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// WentzellRegion<F> — sub-trait of RobinRegion<F>
// ---------------------------------------------------------------------------

/// Wentzell region — sub-trait of [`RobinRegion<F>`] adding time-dependent
/// boundary scaling `γ(t) ≥ 0` and boundary reaction `c ≥ 0`.
///
/// The dynamic Wentzell/Robin condition is `∂_t u + γ(t)·∂_ν u + c·u = 0` on `∂Ω`.
/// `gamma_at(t)` is sampled at the left endpoint of each step (Howland §23 freeze).
///
/// The default impl for `gamma_at` returns the static `β` from `robin_coeffs()`
/// (autonomous fall-back). Time-dependent kernels override `gamma_at`.
pub trait WentzellRegion<F: SemiflowFloat>: RobinRegion<F> {
    /// Time-dependent boundary scaling `γ(t) ≥ 0` at absolute time `t`.
    ///
    /// Default: returns the static `β` from `robin_coeffs()` (autonomous use).
    fn gamma_at(&self, _t: F) -> F {
        self.robin_coeffs().1
    }

    /// Boundary reaction coefficient `c ≥ 0`.
    fn reaction(&self) -> F;
}

// ---------------------------------------------------------------------------
// HalfSpaceWentzell<F, D> — concrete Wentzell region on a half-space
// ---------------------------------------------------------------------------

/// Half-space Wentzell region: wraps `HalfSpaceRegion<F, D>` with a function-pointer
/// `γ: fn(F) -> F` (no captures, matching ADR-0134 fn-ptr discipline) and scalar `c ≥ 0`.
///
/// `robin_coeffs()` returns `(c, γ(F::zero()))` for the static-Robin fall-back.
/// `gamma_at(t)` samples `γ(t)` at run-time (O(1), no heap).
///
/// Construction validates `c ≥ 0`, `c` finite, and `‖normal‖₂ = 1` (delegated).
#[derive(Debug, Clone, Copy)]
pub struct HalfSpaceWentzell<F: SemiflowFloat = f64, const D: usize = 1> {
    /// Underlying half-space geometry (origin + unit outward normal).
    pub half_space: HalfSpaceRegion<F, D>,
    /// Time-dependent boundary scaling `γ: fn(F) -> F`. Must return ≥ 0. No captures.
    pub gamma: fn(F) -> F,
    /// Boundary reaction coefficient `c ≥ 0`.
    pub c: F,
}

impl<F: SemiflowFloat, const D: usize> HalfSpaceWentzell<F, D> {
    /// Construct with validated parameters.
    ///
    /// # Errors
    /// - `DomainViolation` if `‖normal‖₂ ≠ 1` (via `HalfSpaceRegion::new`).
    /// - `DomainViolation` if `c < 0` or `c` is not finite.
    pub fn new(
        origin: [F; D],
        normal: [F; D],
        gamma: fn(F) -> F,
        c: F,
    ) -> Result<Self, SemiflowError> {
        let half_space = HalfSpaceRegion::new(origin, normal)?;
        if !c.is_finite() || c < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "HalfSpaceWentzell: c must be finite and >= 0",
                value: c.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self {
            half_space,
            gamma,
            c,
        })
    }
}

impl<F: SemiflowFloat, const D: usize> ReflectingRegion<F> for HalfSpaceWentzell<F, D> {
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

impl<F: SemiflowFloat, const D: usize> RobinRegion<F> for HalfSpaceWentzell<F, D> {
    /// Static fall-back: `(c, γ(0))`.
    fn robin_coeffs(&self) -> (F, F) {
        (self.c, (self.gamma)(F::zero()))
    }
}

impl<F: SemiflowFloat, const D: usize> WentzellRegion<F> for HalfSpaceWentzell<F, D> {
    fn gamma_at(&self, t: F) -> F {
        (self.gamma)(t)
    }

    fn reaction(&self) -> F {
        self.c
    }
}

// ---------------------------------------------------------------------------
// DynamicWentzellChernoff<C, R, F> — wrapper
// ---------------------------------------------------------------------------

/// Chernoff wrapper for dynamic Wentzell/Robin BCs via implicit Cayley boundary step.
///
/// Per-step (via [`TimedChernoffFunction::apply_at`]):
/// 1. **Bulk step**: `self.inner.apply_into(τ, src, dst, scratch)` (`DiffusionChernoff`).
/// 2. **Cayley boundary sub-step**: advance `[dst[1], dst[0]]` by
///    `K_CN = (I − τC_∂/2)⁻¹(I + τC_∂/2)` where `C_∂` is the 2×2 coupled block
///    `[[-a/dx², 1/dx], [-γ(t)/dx, -(γ(t)/dx + c)]]` (math §49.3, closed-form 2×2 inverse).
///
/// `order() = 1`. `growth()` delegates to the inner kernel (Cayley is a contraction).
///
/// See math.md §49 (NORMATIVE) and ADR-0151.
#[derive(Debug, Clone)]
pub struct DynamicWentzellChernoff<C, R, F = f64>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: WentzellRegion<F>,
    F: SemiflowFloat,
{
    /// Inner bulk Chernoff function (e.g., `DiffusionChernoff<f64>`).
    pub inner: C,
    /// Wentzell region providing geometry + `γ(t)` + `c`.
    pub region: R,
    _f: PhantomData<F>,
}

impl<C, R, F> DynamicWentzellChernoff<C, R, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: WentzellRegion<F>,
    F: SemiflowFloat,
{
    /// Wrap `inner` Chernoff function with Wentzell region `region`.
    ///
    /// # Errors
    /// Propagates `SemiflowError` from region validation (currently infallible for
    /// well-formed regions; kept for forward compatibility with multi-D validation).
    pub fn new(inner: C, region: R) -> Result<Self, SemiflowError> {
        Ok(Self {
            inner,
            region,
            _f: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers: boundary block assembly and Cayley step
// ---------------------------------------------------------------------------

/// Assemble the 2×2 near-boundary generator block `C_∂` (math §49.3).
///
/// ```text
/// C_∂ = [[ -a/dx²       +1/dx       ]   (row 0 = interior bulk node)
///        [ -γ(t)/dx   -(γ(t)/dx + c) ]] (row 1 = boundary trace DOF)
/// ```
///
/// The `(1,0)` entry `-γ/dx` is `O(1/dx) → ∞` as `dx → 0`: this is the UNBOUNDED
/// normal-derivative coupling that the explicit product formula cannot handle (Stephan
/// 2023). The Cayley map tames it: `|z_cay| ≤ 1` unconditionally (ADR-0146 witness).
#[inline]
fn assemble_boundary_block(gamma: f64, c: f64, a0: f64, dx: f64) -> [[f64; 2]; 2] {
    let g_over_dx = gamma / dx;
    [[-a0 / (dx * dx), 1.0 / dx], [-g_over_dx, -(g_over_dx + c)]]
}

/// Apply the closed-form 2×2 Cayley map `K_CN = (I − τC_∂/2)⁻¹(I + τC_∂/2)` to
/// `[u_bulk, u_bnd]` in-place (math §49.4).
///
/// The 2×2 matrix inverse is computed in closed form via the 2×2 adjugate / determinant:
/// `(I − τC/2)⁻¹ = adj(I − τC/2) / det(I − τC/2)`. No LAPACK; no_std-safe.
///
/// `u_bulk` = `dst[1]` (first interior node), `u_bnd` = `dst[0]` (boundary trace DOF).
fn cayley_boundary_step(u_bulk: &mut f64, u_bnd: &mut f64, block: [[f64; 2]; 2], tau: f64) {
    let half_tau = 0.5 * tau;

    // Left matrix: L = I - (τ/2)·C_∂
    let l00 = 1.0 - half_tau * block[0][0];
    let l01 = -half_tau * block[0][1];
    let l10 = -half_tau * block[1][0];
    let l11 = 1.0 - half_tau * block[1][1];

    // Right matrix: R = I + (τ/2)·C_∂
    let r00 = 1.0 + half_tau * block[0][0];
    let r01 = half_tau * block[0][1];
    let r10 = half_tau * block[1][0];
    let r11 = 1.0 + half_tau * block[1][1];

    // R * [u_bulk, u_bnd]ᵀ
    let rb = r00 * (*u_bulk) + r01 * (*u_bnd);
    let rd = r10 * (*u_bulk) + r11 * (*u_bnd);

    // Solve L * [new_bulk, new_bnd]ᵀ = [rb, rd] via closed-form 2×2 inverse.
    let det = l00 * l11 - l01 * l10;
    // If det ≈ 0 (pathological), leave state unchanged to avoid NaN propagation.
    if det.abs() < f64::EPSILON {
        return;
    }
    let inv_det = det.recip();
    *u_bulk = (l11 * rb - l01 * rd) * inv_det;
    *u_bnd = (-l10 * rb + l00 * rd) * inv_det;
}

// ---------------------------------------------------------------------------
// Generic ChernoffFunction<f64> + TimedChernoffFunction<f64> impls
//
// Parameterised over any `R: WentzellRegion<f64>` so that binding crates can
// supply their own per-step schedule-backed region newtype without sharing util
// (ADR-0028 Amendment 2).  The inner kernel is fixed to `DiffusionChernoff<f64>`
// because `call_a` is needed to evaluate the diffusion coefficient at the
// boundary node, and that method is only available on `DiffusionChernoff`.
// ---------------------------------------------------------------------------

impl<R> ChernoffFunction<f64> for DynamicWentzellChernoff<DiffusionChernoff<f64>, R, f64>
where
    R: WentzellRegion<f64>,
{
    type S = GridFn1D<f64>;

    /// Autonomous fall-back: delegates to `apply_at(0.0, τ, …)` using `γ(0)`.
    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        self.apply_at(0.0, tau, src, dst, scratch)
    }

    /// Order = 1 (bulk↔boundary Lie split commutator nonzero; math §49.8).
    fn order(&self) -> u32 {
        1
    }

    /// Growth: delegates to inner (Cayley boundary block is a contraction, `ρ ≤ 1`).
    fn growth(&self) -> Growth<f64> {
        self.inner.growth()
    }
}

/// Generic `TimedChernoffFunction<f64>` impl for any `WentzellRegion<f64>`.
///
/// Parameterised over `R: WentzellRegion<f64>` so binding crates can supply
/// schedule-backed region newtypes (per ADR-0028 Amendment 2 — no shared util).
///
/// The inner kernel is fixed to `DiffusionChernoff<f64>` because evaluating
/// the diffusion coefficient `a₀ = a(x_bnd)` at the boundary node requires
/// `DiffusionChernoff::call_a`, which is not available on a generic
/// `ChernoffFunction`.
impl<R> TimedChernoffFunction<f64> for DynamicWentzellChernoff<DiffusionChernoff<f64>, R, f64>
where
    R: WentzellRegion<f64>,
{
    /// Bulk–boundary Lie split at absolute time `t` (math §49.2).
    ///
    /// Steps:
    /// 1. Sample `γ = γ(t)` at the left endpoint (Howland freeze).
    /// 2. Bulk step: `inner.apply_into(τ, src, dst, scratch)`.
    /// 3. Assemble `C_∂(γ, c, a₀, dx)` and advance `[dst[1], dst[0]]` by `K_CN`.
    ///
    /// Requires `dst.values.len() >= 2` (boundary + at least one interior node).
    fn apply_at(
        &self,
        t: f64,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        let n = src.values.len();
        if n < 2 {
            return Err(SemiflowError::DomainViolation {
                what: "DynamicWentzellChernoff: grid must have >= 2 nodes",
                value: n as f64,
            });
        }
        // Sample γ(t) at the left endpoint (Howland freeze, §23.4).
        let gamma = self.region.gamma_at(t);
        let c = self.region.reaction();
        let dx = src.grid.dx();
        // Evaluate diffusion coefficient at boundary (x=0 for half-line convention).
        let x_bnd = src.grid.xmin;
        let a0 = self.inner.call_a(x_bnd);

        // (a) Bulk Chernoff step.
        self.inner.apply_into(tau, src, dst, scratch)?;

        // (b) Cayley boundary sub-step on [dst[1], dst[0]].
        let block = assemble_boundary_block(gamma, c, a0, dx);
        let mut u_bulk = dst.values[1];
        let mut u_bnd = dst.values[0];
        cayley_boundary_step(&mut u_bulk, &mut u_bnd, block, tau);
        dst.values[1] = u_bulk;
        dst.values[0] = u_bnd;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests (fast, no feature gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        diffusion::DiffusionChernoff, error::SemiflowError, grid::Grid1D, ChernoffFunction,
    };

    fn make_inner(n: usize) -> DiffusionChernoff<f64> {
        let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
        DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid)
    }

    fn make_region(gamma_fn: fn(f64) -> f64, c: f64) -> HalfSpaceWentzell<f64, 1> {
        HalfSpaceWentzell::new([0.0], [1.0], gamma_fn, c).unwrap()
    }

    #[test]
    fn half_space_wentzell_construction_ok() {
        let r = HalfSpaceWentzell::<f64, 1>::new([0.0], [1.0], |_| 1.0, 0.5);
        assert!(r.is_ok());
    }

    #[test]
    fn half_space_wentzell_negative_c_err() {
        let err = HalfSpaceWentzell::<f64, 1>::new([0.0], [1.0], |_| 1.0, -0.1).unwrap_err();
        assert!(matches!(err, SemiflowError::DomainViolation { .. }));
    }

    #[test]
    fn half_space_wentzell_nonfinite_c_err() {
        let err = HalfSpaceWentzell::<f64, 1>::new([0.0], [1.0], |_| 1.0, f64::NAN).unwrap_err();
        assert!(matches!(err, SemiflowError::DomainViolation { .. }));
    }

    #[test]
    fn wentzell_order_is_1() {
        let inner = make_inner(16);
        let region = make_region(|_| 1.0, 0.5);
        let wrapper = DynamicWentzellChernoff::new(inner, region).unwrap();
        assert_eq!(wrapper.order(), 1);
    }

    #[test]
    fn wentzell_gamma_at_samples_fn() {
        let region = make_region(|t| 0.5 + t, 0.2);
        assert!((region.gamma_at(1.0) - 1.5).abs() < 1e-14);
        assert!((region.gamma_at(0.0) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn wentzell_default_gamma_at_returns_static_beta() {
        // HalfSpaceWentzell: robin_coeffs().1 = γ(0); gamma_at(t) samples γ(t).
        let region = make_region(|_| 2.0, 0.3);
        let (_, beta_static) = region.robin_coeffs();
        assert!((beta_static - 2.0).abs() < 1e-14);
        assert!((region.gamma_at(42.0) - 2.0).abs() < 1e-14); // constant fn
    }

    #[test]
    fn cayley_step_preserves_dissipativity_smoke() {
        // Verify |K_CN * v|_2 <= |v|_2 for a dissipative block.
        let gamma = 1.0;
        let c = 0.5;
        let dx = 0.1;
        let a0 = 1.0;
        let tau = 0.004; // 0.4 * dx^2 / a (CFL scale)
        let block = assemble_boundary_block(gamma, c, a0, dx);
        let mut ub = 1.0_f64;
        let mut ud = 0.5_f64;
        let norm_before = (ub * ub + ud * ud).sqrt();
        cayley_boundary_step(&mut ub, &mut ud, block, tau);
        let norm_after = (ub * ub + ud * ud).sqrt();
        assert!(
            norm_after <= norm_before + 1e-12,
            "Cayley must not amplify: before={norm_before:.6}, after={norm_after:.6}"
        );
    }

    #[test]
    fn wentzell_apply_into_smoke() {
        let n = 32usize;
        let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let region = make_region(|_| 1.0, 0.5);
        let wrapper = DynamicWentzellChernoff::new(inner, region).unwrap();
        let src = crate::grid_fn::GridFn1D::from_fn(grid, |x| x * (1.0 - x));
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        let result = wrapper.apply_into(1e-3, &src, &mut dst, &mut scratch);
        assert!(result.is_ok(), "apply_into must not fail: {result:?}");
        assert!(
            dst.values.iter().all(|v| v.is_finite()),
            "all output values must be finite"
        );
    }

    #[test]
    fn wentzell_apply_at_time_dependent_smoke() {
        // apply_at with time-varying γ(t) must not panic and produce finite output.
        let n = 32usize;
        let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let region = make_region(|t| 0.5 + t.sin(), 0.2);
        let wrapper = DynamicWentzellChernoff::new(inner, region).unwrap();
        let src = crate::grid_fn::GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin());
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        let result = wrapper.apply_at(0.7, 1e-3, &src, &mut dst, &mut scratch);
        assert!(result.is_ok(), "apply_at must not fail: {result:?}");
        assert!(dst.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn wentzell_reaction_round_trip() {
        // WentzellRegion::reaction() must return the c value passed to new().
        let region = HalfSpaceWentzell::<f64, 1>::new([0.0], [1.0], |_| 1.0, 0.75).unwrap();
        assert!((region.reaction() - 0.75).abs() < 1e-14);
    }
}
