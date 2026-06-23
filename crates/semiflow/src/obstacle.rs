//! Projective-splitting obstacle evolver / variational inequalities (math §44).
//!
//! Composes any linear Chernoff propagator `S(τ)` with the metric projection
//! `Π_g(W) = max(W, g)` onto the convex cone `K = {V ≥ g}` to solve obstacle
//! problems / variational inequalities (VI) / optimal stopping:
//!
//! ```text
//! V^{n+1} = Π_g( S(Δτ) Vⁿ ),     Π_g(W) = max(W, g)   (post-projection).
//! ```
//!
//! The Chernoff/Remizov tangent `S(Δτ)` is the linear step the obstacle theory
//! plugs in: every existing [`ChernoffFunction`] becomes a VI engine (math §44.1).
//!
//! ## Mathematical contract (math §44)
//!
//! - `Π_g = max(·, g)` is the metric projection / proximal-resolvent of `∂I_K`:
//!   nonexpansive (1-Lipschitz, Theorem 44.1), m-accretive.
//! - The composite `Π_g ∘ S(Δτ)` is monotone + stable + consistent ⇒ converges
//!   to the unique viscosity solution of the VI (Barles–Souganidis 1991,
//!   Theorem 44.2) **when the inner is monotone and nonexpansive (`ω ≤ 0`)**.
//! - Declared `order() == 1` (honest): `O(Δτ)` for convex obstacles, degrading to
//!   `O(√Δτ)` at the free boundary — a structural property, not a defect (§44.4).
//!
//! ## Honesty notes (NORMATIVE, math §44.6 / §44.7)
//!
//! - **`growth()`**: `Π_g` is *affine*, not homogeneous (`Π_g(0) = g⁺ ≠ 0`), so the
//!   multiplicative [`Growth`] bound does NOT hold verbatim for nonzero `g`. The
//!   operative stability certificate is `Π_g`-nonexpansiveness, not `growth()`.
//!   `growth()` returns the inner's homogeneous growth (exact for `g ≤ 0`; carries
//!   an additive `‖g⁺‖∞` offset otherwise). Bounded `g⁺` is a precondition.
//! - **Non-contractive inner (`ω > 0`, discount `c(x) > 0`)**: convergence is
//!   CONJECTURAL (Trotter projection counterexample, arXiv math/0109049); allowed,
//!   doc-warned, not gated.
//! - **`D = 1` only** (the [`GridFn1D`] coordinate-access constraint shared with
//!   the killing kernel §21).

// Grid counts/lengths (usize) cast to f64 for error-reporting values; always ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

use core::marker::PhantomData;

// D≥2 forward-evolution wrapper is in obstacle_nd.rs (coherence-safe sibling).
use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    grid_nd::{enumerate_nd, GridFnND},
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Obstacle<F> — trait
// ---------------------------------------------------------------------------

/// Obstacle (lower envelope) `g(x)` for the cone `K = {V : V ≥ g}` (math §44).
///
/// Consumed by [`ObstacleChernoff`] to post-project each Chernoff step:
/// `dst[i] := max(dst[i], g(x_i))` (the metric projection `Π_g`, Theorem 44.1).
///
/// ## Shipped impls
///
/// - [`ConstantObstacle`] — a flat floor `g(x) ≡ level`.
/// - [`ClosureObstacle`] — `g` from a user closure `x ↦ g(x)`.
///
/// Mirrors [`crate::killing::KillingRegion`]: override `value_at`; the default
/// `project_in_place` / `active_set_into` iterate it (concrete impls may override
/// with tight scalar loops). All methods operate on [`GridFn1D`] node coordinates
/// (`D = 1`; higher `D` deferred, math §44.7).
pub trait Obstacle<F: SemiflowFloat = f64> {
    /// Obstacle value `g(x)` at a single point (slice of `D` coordinates).
    fn value_at(&self, point: &[F]) -> F;

    /// In-place projection `dst[i] := max(dst[i], g(x_i))` (`Π_g`, Theorem 44.1).
    ///
    /// Default iterates `value_at` per node. Concrete impls MAY override.
    ///
    /// # Errors
    /// Returns [`SemiflowError`] on grid inconsistency (concrete overrides may error).
    fn project_in_place(&self, dst: &mut GridFn1D<F>) -> Result<(), SemiflowError> {
        for i in 0..dst.grid.n {
            let x = dst.grid.x_at(i);
            let g = self.value_at(&[x]);
            if dst.values[i] < g {
                dst.values[i] = g;
            }
        }
        Ok(())
    }

    /// Continuation (active) set `active[i] := (w[i] > g(x_i))` — STRICT (the
    /// contact node `w == g` is inactive, NORMATIVE convention, math §44.5).
    ///
    /// `active.len()` MUST equal `w.grid.n`; otherwise `DomainViolation`.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `active.len() != w.grid.n`.
    fn active_set_into(&self, w: &GridFn1D<F>, active: &mut [bool]) -> Result<(), SemiflowError> {
        if active.len() != w.grid.n {
            return Err(SemiflowError::DomainViolation {
                what: "Obstacle::active_set_into: active.len() must equal w.grid.n",
                value: active.len() as f64,
            });
        }
        for (i, (a, v)) in active.iter_mut().zip(w.values.iter()).enumerate() {
            let x = w.grid.x_at(i);
            *a = *v > self.value_at(&[x]);
        }
        Ok(())
    }

    /// Spatial dimension `D` of the obstacle (informational; `D = 1` shipped).
    fn dim(&self) -> usize;

    /// D-generic in-place projection `dst[flat] := max(dst[flat], g(x))`
    /// over flat row-major `GridFnND` storage (math §44.5.ter).
    ///
    /// The projection `Π_g(W) = max(W, g)` is **elementwise** and carries no
    /// dimension assumption; this default iterates `value_at(&[F])` per node
    /// using the same row-major decode as `grid_nd::enumerate_nd`. The D=1
    /// `project_in_place` is unaffected (back-compat).
    ///
    /// # Errors
    /// `DomainViolation` if `dst.values.len() != dst.grid.len()` (structural).
    fn project_in_place_nd<const D: usize>(
        &self,
        dst: &mut GridFnND<F, D>,
    ) -> Result<(), SemiflowError> {
        if dst.values.len() != dst.grid.len() {
            return Err(SemiflowError::DomainViolation {
                what: "Obstacle::project_in_place_nd: values len != grid.len()",
                value: dst.values.len() as f64,
            });
        }
        let obstacle = self;
        enumerate_nd(&dst.grid, |flat, x| {
            let g = obstacle.value_at(x.as_slice());
            if dst.values[flat] < g {
                dst.values[flat] = g;
            }
        });
        Ok(())
    }

    /// D-generic active-set mask `active[flat] := (w[flat] > g(x))` (math §44.5.ter).
    ///
    /// STRICT convention: contact nodes `w == g` are NOT active (same as D=1).
    ///
    /// # Errors
    /// `DomainViolation` if `active.len() != w.grid.len()`.
    fn active_set_nd_into<const D: usize>(
        &self,
        w: &GridFnND<F, D>,
        active: &mut [bool],
    ) -> Result<(), SemiflowError> {
        let total = w.grid.len();
        if active.len() != total {
            return Err(SemiflowError::DomainViolation {
                what: "Obstacle::active_set_nd_into: active.len() must equal w.grid.len()",
                value: active.len() as f64,
            });
        }
        let obstacle = self;
        enumerate_nd(&w.grid, |flat, x| {
            active[flat] = w.values[flat] > obstacle.value_at(x.as_slice());
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConstantObstacle<F> — flat floor
// ---------------------------------------------------------------------------

/// Flat obstacle `g(x) ≡ level` (a uniform lower floor).
///
/// `level ≤ 0` is the sub-Markov regime where the inherited `growth()` bound is
/// exact (math §44.6); any finite `level` is accepted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantObstacle<F: SemiflowFloat = f64> {
    /// The constant obstacle value `g(x) ≡ level`.
    pub level: F,
}

impl<F: SemiflowFloat> ConstantObstacle<F> {
    /// Construct a flat obstacle at `level`.
    ///
    /// # Errors
    ///
    /// Returns `SemiflowError::DomainViolation` if `level` is not finite.
    pub fn new(level: F) -> Result<Self, SemiflowError> {
        if !level.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "ConstantObstacle: level must be finite",
                value: level.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self { level })
    }
}

impl<F: SemiflowFloat> Obstacle<F> for ConstantObstacle<F> {
    fn value_at(&self, _point: &[F]) -> F {
        self.level
    }

    fn project_in_place(&self, dst: &mut GridFn1D<F>) -> Result<(), SemiflowError> {
        for v in &mut dst.values {
            if *v < self.level {
                *v = self.level;
            }
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// ClosureObstacle<G, F> — obstacle from a closure
// ---------------------------------------------------------------------------

/// Obstacle `g(x)` supplied by a 1D closure `x ↦ g(x)` (the workhorse form).
///
/// Mirrors the coefficient-closure convention of
/// [`crate::diffusion::DiffusionChernoff`]. The closure's correctness (and the
/// bounded-`g⁺` precondition, math §44.6) is the caller's responsibility.
#[derive(Debug, Clone, Copy)]
pub struct ClosureObstacle<G, F: SemiflowFloat = f64> {
    g: G,
    _f: PhantomData<F>,
}

impl<G, F> ClosureObstacle<G, F>
where
    F: SemiflowFloat,
    G: Fn(F) -> F,
{
    /// Wrap a closure `g: x ↦ g(x)` as an obstacle.
    pub fn new(g: G) -> Self {
        Self { g, _f: PhantomData }
    }
}

impl<G, F> Obstacle<F> for ClosureObstacle<G, F>
where
    F: SemiflowFloat,
    G: Fn(F) -> F,
{
    fn value_at(&self, point: &[F]) -> F {
        (self.g)(point[0])
    }

    fn dim(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// ObstacleChernoff<C, O, F> — wrapper
// ---------------------------------------------------------------------------

/// Projective-splitting Chernoff function `Π_g ∘ S(Δτ)` (math §44).
///
/// `F(τ)f = max(C(τ)f, g)` (post-projection). Order-1 globally (declared; §44.4).
/// Stability rests on `Π_g`-nonexpansiveness (Theorem 44.1), NOT on `growth()`
/// (see the module-level honesty note). Convergence to the VI viscosity solution
/// is proven only for monotone, nonexpansive (`ω ≤ 0`) inners (Theorem 44.2).
///
/// `C::S` must be [`GridFn1D`] (`D = 1`, the coordinate-access constraint).
#[derive(Debug, Clone)]
pub struct ObstacleChernoff<C, O, F: SemiflowFloat = f64> {
    inner: C,
    obstacle: O,
    _f: PhantomData<F>,
}

impl<C, O, F> ObstacleChernoff<C, O, F>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    O: Obstacle<F>,
{
    /// Wrap `inner` with post-projection onto `{V ≥ obstacle}`.
    ///
    /// # Errors
    ///
    /// Always `Ok` for pre-validated obstacles ([`ConstantObstacle::new`] /
    /// [`ClosureObstacle::new`]); signature mirrors
    /// [`crate::killing::KillingChernoff::new`].
    pub fn new(inner: C, obstacle: O) -> Result<Self, SemiflowError> {
        Ok(Self {
            inner,
            obstacle,
            _f: PhantomData,
        })
    }

    /// Borrow the inner (linear) Chernoff propagator.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Borrow the obstacle `g`.
    pub fn obstacle(&self) -> &O {
        &self.obstacle
    }

    /// One backward (adjoint) step of the projected scheme through the active set
    /// (math §44.5, Theorem 44.3). Separate primitive — NOT
    /// `ChernoffFunction::apply_adjoint_into`; `ObstacleChernoff` does not
    /// implement `AdjointApply` (the projection is non-differentiable).
    ///
    /// Given the **pre-projection** forward state `w_fwd = S(Δτ)Vⁿ` (which freezes
    /// the active set `{w_fwd > g}`) and incoming costate `lam` (`∂/∂V^{n+1}`),
    /// produces `lam_next` (`∂/∂Vⁿ`):
    /// 1. mask: `λ_masked[i] = lam[i] · 𝟙[w_fwd[i] > g(x_i)]` (the `Π_g` Jacobian);
    /// 2. inner adjoint: `lam_next = S*(Δτ) λ_masked`.
    ///
    /// # Errors
    ///
    /// `DomainViolation` on length mismatch; `UnsupportedOperation` if the inner
    /// has no transpose-apply primitive (its `apply_adjoint_into` default).
    pub fn apply_active_set_adjoint_into(
        &self,
        tau: F,
        w_fwd: &GridFn1D<F>,
        lam: &GridFn1D<F>,
        lam_next: &mut GridFn1D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let n = w_fwd.grid.n;
        if lam.grid.n != n || lam_next.grid.n != n {
            return Err(SemiflowError::DomainViolation {
                what: "apply_active_set_adjoint_into: w_fwd/lam/lam_next length mismatch",
                value: lam.grid.n as f64,
            });
        }
        // Stage λ_masked into lam_next (reused as scratch), then adjoint into a temp.
        let mut masked = lam.zeroed_like();
        for i in 0..n {
            let x = w_fwd.grid.x_at(i);
            let active = w_fwd.values[i] > self.obstacle.value_at(&[x]);
            masked.values[i] = if active { lam.values[i] } else { F::zero() };
        }
        self.inner
            .apply_adjoint_into(tau, &masked, lam_next, scratch)
    }
}

impl<C, O, F> ChernoffFunction<F> for ObstacleChernoff<C, O, F>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    O: Obstacle<F>,
{
    type S = GridFn1D<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.inner.apply_into(tau, src, dst, scratch)?;
        self.obstacle.project_in_place(dst)
    }

    /// Order-1 globally (declared honest; `O(Δτ)` convex, `O(√Δτ)` boundary,
    /// math §44.4). The projection cap dominates any higher inner order.
    fn order(&self) -> u32 {
        1
    }

    /// Returns the inner's *homogeneous* growth (math §44.6). NOTE: `Π_g` is
    /// affine, so this is exact only for `g ≤ 0`; for general bounded `g` the true
    /// bound carries an additive `‖g⁺‖∞` offset not expressible in [`Growth`]. The
    /// operative stability certificate is `Π_g`-nonexpansiveness (Theorem 44.1).
    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}

// ObstacleChernoffND lives in obstacle_nd.rs (sibling module, suckless split).

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
#[path = "obstacle_tests.rs"]
mod tests;
