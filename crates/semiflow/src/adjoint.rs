//! Adjoint (backward) semigroup wrapper for any `ChernoffFunction<F>` whose
//! state implements `HilbertState<F>`.
//!
//! For *self-adjoint* inner generators (symmetric graph Laplacian, isotropic
//! diffusion, etc.) the wrapper is a thin re-export — `apply_into` simply
//! delegates to `inner.apply_into`.
//!
//! For *non-self-adjoint* inner generators (drift-reaction `−Δ + b·∂_x`,
//! etc.) the wrapper computes the genuine dual semigroup `S*(τ) = exp(τAᵀ)`
//! via the `AdjointApply<F>` primitive. Inners must implement `AdjointApply<F>`
//! and override `ChernoffFunction::apply_adjoint_into` to provide the correct
//! transpose action.
//!
//! # Mathematical basis
//!
//! Pazy 1983 *Semigroups of Linear Operators* §1.10 Theorem 10.4 (dual
//! semigroup). See `contracts/semiflow-core.math.md` §15 (NORMATIVE) and
//! ADR-0055, ADR-0114.
//!
//! # Honesty note (ADR-0114, 2026-05-31)
//!
//! The former `apply_dual_evolution` used a generator-independent scalar
//! correction `dst += -τ·(‖src‖²/n)·src` that violated the defining identity
//! `⟨S·u,g⟩=⟨u,S*·g⟩` by up to ~96% on drifted inners. A correct generic
//! adjoint is structurally impossible from `apply_into`+`dot`+`axpy` alone —
//! it requires an explicit transpose-apply primitive. The scalar formula is
//! deleted. `new_general` now requires `C: AdjointApply<F>`.
//!
//! # Zero-alloc steady state
//!
//! `apply_into` acquires no scratch buffers in the self-adjoint path.
//! In the general path cost equals `inner.apply_adjoint_into`'s cost.

use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    scratch::ScratchPool,
    state::HilbertState,
};

// ---------------------------------------------------------------------------
// AdjointApply<F> supertrait
// ---------------------------------------------------------------------------

/// Opt-in supertrait marking that `ChernoffFunction::apply_adjoint_into`
/// is correctly implemented for this type (not the default error stub).
///
/// # Contract
///
/// Implementors MUST:
/// 1. Override `ChernoffFunction::apply_adjoint_into` to compute
///    `dst := exp(τ Aᵀ) src` correctly.
/// 2. Satisfy `|⟨S(τ)·u, g⟩ − ⟨u, S*(τ)·g⟩| ≤ C · τ^{p+1}` for
///    seeded-random `u`, `g` at the declared consistency order `p`.
///
/// # No default impl
///
/// Implementing `AdjointApply<F>` is an explicit commitment that the
/// override is correct. No default exists — a default would mask missing
/// support and allow silent wrong numbers.
///
/// # Compile-time safety
///
/// [`AdjointChernoff::new_general`] requires `C: AdjointApply<F>`. Types
/// that cannot expose `Aᵀ` will fail to compile at `new_general` call sites.
/// For self-adjoint inners use [`AdjointChernoff::new_self_adjoint`] (no
/// `AdjointApply` bound needed).
///
/// # Implementors
///
/// - [`crate::DriftReactionChernoff`]: adjoint of `−Δ + b·∂_x` is `−Δ − b·∂_x`
///   (same kernel with negated drift coefficient).
/// - [`crate::graph_heat4::GraphHeat4thChernoff`]: autonomous order-4 graph heat
///   Chernoff via truncated Taylor series; `L_G` is a constant symmetric
///   combinatorial Laplacian ⟹ `S₄*(τ) = S₄(τ)` exactly (no time-varying
///   commutator). Delegates to `apply_into`. See ADR-0114.
///
/// # Non-implementors (intentional)
///
/// - [`crate::magnus_graph::MagnusGraphHeatChernoff`]: time-varying `L_G(t)`.
///   The Magnus exponent `Ω₄` has an antisymmetric commutator part
///   `~τ²·[L_i,L_j]`, so `exp(Ω₄)ᵀ ≠ exp(Ω₄)` at O(τ²). A correct cheap
///   transpose is not available; this type must NOT implement `AdjointApply`.
pub trait AdjointApply<F: SemiflowFloat = f64>: ChernoffFunction<F> {}

// ---------------------------------------------------------------------------
// AdjointChernoff<C, F>
// ---------------------------------------------------------------------------

/// Adjoint (backward) semigroup wrapper. See math.md §15 and ADR-0055/0114.
///
/// # Self-adjoint path — [`new_self_adjoint`]
///
/// For self-adjoint inners: `apply_into` delegates to `inner.apply_into`.
/// No extra trait bound needed.
///
/// # Non-self-adjoint path — [`new_general`]
///
/// Requires `C: AdjointApply<F>`. `apply_into` calls
/// `inner.apply_adjoint_into(...)` which the inner MUST implement correctly
/// (see [`AdjointApply`]).
///
/// # Corrected contract (ADR-0114)
///
/// The former scalar correction `dst += -τ·(‖src‖²/n)·src` is DELETED.
/// The `new_general` path now requires a correct transpose primitive.
///
/// [`new_self_adjoint`]: Self::new_self_adjoint
/// [`new_general`]: Self::new_general
#[derive(Clone, Debug)]
pub struct AdjointChernoff<C, F: SemiflowFloat = f64>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    inner: C,
    is_self_adjoint: bool,
    _f: PhantomData<F>,
}

// ---------------------------------------------------------------------------
// Constructor — self-adjoint (no AdjointApply bound)
// ---------------------------------------------------------------------------

impl<C, F: SemiflowFloat> AdjointChernoff<C, F>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    /// Construct for a known-self-adjoint inner generator.
    ///
    /// `apply_into` delegates directly to `inner.apply_into` — same cost.
    ///
    /// **Caller assertion**: library does NOT verify self-adjointness.
    /// Misuse leads to incorrect results, not crashes.
    ///
    /// `order()` = `inner.order()`.
    pub fn new_self_adjoint(inner: C) -> Self {
        Self {
            inner,
            is_self_adjoint: true,
            _f: PhantomData,
        }
    }

    /// Borrow the wrapped inner Chernoff function.
    #[must_use]
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Return the self-adjoint flag.
    #[must_use]
    pub fn is_self_adjoint(&self) -> bool {
        self.is_self_adjoint
    }

    /// Probabilistic self-adjointness check (developer tool — NOT hot-path).
    ///
    /// Stub in v7.0: always returns `Ok(false)`.
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if `n_samples == 0` or `tol <= 0`.
    pub fn detect_self_adjointness(
        _inner: &C,
        n_samples: usize,
        tol: F,
    ) -> Result<bool, SemiflowError>
    where
        C::S: Clone,
    {
        if n_samples == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "detect_self_adjointness: n_samples must be >= 1",
                value: 0.0,
            });
        }
        let tol_f64 = tol.to_f64().unwrap_or(f64::NAN);
        if !tol_f64.is_finite() || tol_f64 <= 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "detect_self_adjointness: tol must be finite and > 0",
                value: tol_f64,
            });
        }
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// Constructor — general / non-self-adjoint (requires AdjointApply<F>)
// ---------------------------------------------------------------------------

impl<C, F: SemiflowFloat> AdjointChernoff<C, F>
where
    C: AdjointApply<F>,
    C::S: HilbertState<F>,
{
    /// Construct for a non-self-adjoint inner generator.
    ///
    /// Requires `C: AdjointApply<F>` at compile time — the inner MUST
    /// override `ChernoffFunction::apply_adjoint_into` with the correct
    /// transpose action `exp(τ Aᵀ)`.
    ///
    /// `order()` = `min(inner.order(), 2)` (ADR-0055 §15.1.bis).
    ///
    /// # Design note (ADR-0114)
    ///
    /// The scalar fudge `dst += -τ·(‖src‖²/n)·src` (former implementation)
    /// violated the dual-pairing identity by up to ~96%. It is deleted.
    pub fn new_general(inner: C) -> Self {
        Self {
            inner,
            is_self_adjoint: false,
            _f: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<C, F: SemiflowFloat> ChernoffFunction<F> for AdjointChernoff<C, F>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    type S = C::S;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if self.is_self_adjoint {
            return self.inner.apply_into(tau, src, dst, scratch);
        }
        // Non-self-adjoint path: delegates to inner.apply_adjoint_into.
        // This is correct when new_general was used (requires C: AdjointApply<F>
        // which means apply_adjoint_into is properly overridden).
        // If is_self_adjoint=false but apply_adjoint_into was not overridden,
        // ChernoffFunction::apply_adjoint_into default returns UnsupportedOperation.
        self.inner.apply_adjoint_into(tau, src, dst, scratch)
    }

    /// Returns `inner.order()` for self-adjoint, `min(inner.order(), 2)` otherwise.
    ///
    /// Rationale: the transpose-apply path introduces O(τ²) discretisation
    /// error for non-symmetric inners even if `inner.order() > 2`.
    /// See math.md §15.1.bis (NORMATIVE).
    fn order(&self) -> u32 {
        if self.is_self_adjoint {
            self.inner.order()
        } else {
            core::cmp::min(self.inner.order(), 2)
        }
    }

    /// `‖exp(τA*)‖ = ‖exp(τA)‖` on Hilbert space (Pazy §1.10 Thm 10.4).
    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::*;
    use crate::{
        drift_reaction::DriftReactionChernoff,
        graph::{Graph, Laplacian},
        graph_heat::GraphHeatChernoff,
        graph_heat4::GraphHeat4thChernoff,
        graph_signal::GraphSignal,
        ChernoffFunction, Grid1D, GridFn1D, ScratchPool, State,
    };

    fn make_path_heat(n: usize) -> GraphHeatChernoff<f64> {
        let g = Arc::new(Graph::<f64>::path(n));
        let lap = Laplacian::assemble_combinatorial(&g);
        GraphHeatChernoff::from_owned(lap)
    }

    fn make_path_graph(n: usize) -> Arc<Graph<f64>> {
        Arc::new(Graph::<f64>::path(n))
    }

    // fn-pointer helpers (can't capture variables — use named fns for constants).
    fn b_half(_: f64) -> f64 {
        0.5
    }
    fn c_zero_inner(_: f64) -> f64 {
        0.0
    }

    fn make_drift_reaction(n: usize) -> (DriftReactionChernoff<f64>, Grid1D<f64>) {
        let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
        let dr = DriftReactionChernoff::new(b_half, c_zero_inner, 0.0, grid);
        (dr, grid)
    }

    #[test]
    fn order_self_adjoint_preserves_inner_order() {
        let inner = make_path_heat(8);
        let inner_order = inner.order();
        let adj = AdjointChernoff::new_self_adjoint(inner);
        assert_eq!(adj.order(), inner_order);
    }

    #[test]
    fn order_general_capped_at_2_drift_reaction() {
        // DriftReactionChernoff has order 2; general cap is min(2, 2) = 2.
        let (inner, _) = make_drift_reaction(8);
        let adj = AdjointChernoff::new_general(inner);
        assert!(adj.order() <= 2, "general order must be <= 2");
    }

    #[test]
    fn order_general_cap_applies_to_higher_order_inner() {
        // GraphHeat4thChernoff has order 4; general cap gives min(4,2)=2.
        // Honest AdjointApply implementor: autonomous truncated-Taylor with
        // constant symmetric Laplacian ⟹ S*(τ)=S(τ) (no commutator term).
        let g = Arc::new(Graph::<f64>::path(8));
        let lap = Laplacian::assemble_combinatorial(&g);
        let inner = GraphHeat4thChernoff::from_owned(lap);
        let adj = AdjointChernoff::new_general(inner);
        assert_eq!(
            adj.order(),
            2,
            "general wrapper of order-4 GraphHeat4th inner must report order 2"
        );
    }

    #[test]
    fn is_self_adjoint_flag() {
        let adj_sa = AdjointChernoff::new_self_adjoint(make_path_heat(4));
        assert!(adj_sa.is_self_adjoint());

        let (inner, _) = make_drift_reaction(4);
        let adj_gen = AdjointChernoff::new_general(inner);
        assert!(!adj_gen.is_self_adjoint());
    }

    #[test]
    fn growth_matches_inner() {
        let inner = make_path_heat(8);
        let inner_g = inner.growth();
        let adj = AdjointChernoff::new_self_adjoint(inner);
        let g = adj.growth();
        let (m, inner_m) = (g.multiplier, inner_g.multiplier);
        let (w, inner_w) = (g.omega, inner_g.omega);
        assert!(
            m.to_bits() == inner_m.to_bits(),
            "growth m differs: {m} vs {inner_m}"
        );
        assert!(
            w.to_bits() == inner_w.to_bits(),
            "growth w differs: {w} vs {inner_w}"
        );
    }

    #[test]
    fn zero_tau_self_adjoint_preserves_src() {
        let g = make_path_graph(8);
        let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
        let mut dst = src.clone();
        let mut pool = ScratchPool::<f64>::new();
        let adj = AdjointChernoff::new_self_adjoint(make_path_heat(8));
        adj.apply_into(0.0, &src, &mut dst, &mut pool).unwrap();
        let mut diff = dst.clone();
        diff.axpy_into(-1.0, &src);
        assert!(
            diff.norm_sup() < 1e-14,
            "zero-tau diff = {}",
            diff.norm_sup()
        );
    }

    #[test]
    fn detect_self_adjointness_validates_inputs() {
        let inner = make_path_heat(4);
        assert!(AdjointChernoff::detect_self_adjointness(&inner, 0, 1e-6).is_err());
        assert!(AdjointChernoff::detect_self_adjointness(&inner, 1, 0.0).is_err());
        assert!(AdjointChernoff::detect_self_adjointness(&inner, 1, 1e-6).is_ok());
    }

    // Verify that new_general on a DriftReactionChernoff inner uses the
    // genuine transpose path (apply_adjoint_into), not the fudge.
    #[test]
    fn new_general_drift_reaction_apply_succeeds() {
        let (inner, grid) = make_drift_reaction(16);
        let adj = AdjointChernoff::new_general(inner);
        let src = GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin());
        let mut dst = src.clone();
        let mut pool = ScratchPool::<f64>::new();
        let result = adj.apply_into(0.01, &src, &mut dst, &mut pool);
        assert!(
            result.is_ok(),
            "new_general on DriftReaction should succeed"
        );
        assert!(
            dst.values.iter().all(|v| v.is_finite()),
            "output must be finite"
        );
    }
}
