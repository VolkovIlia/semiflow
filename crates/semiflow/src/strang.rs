//! [`StrangSplit`] — Strang operator-splitting composer.
//!
//! Implements the palindromic Strang sandwich (ADR-0006, §9.4):
//!
//! ```text
//! Φ(τ) = D(τ/2) ∘ R(τ) ∘ D(τ/2)
//! ```
//!
//! For `D` of local order 2 (matching `e^{τA}` to O(τ³)) and exact `R`:
//! - Per-step local error: O(τ³).
//! - Global error over `n = T/τ` steps: O(τ²).
//!
//! This is the canonical Strang result (Hairer–Lubich–Wanner,
//! *Geometric Numerical Integration*, §III.5, Thm 4.1). The v0.2.0
//! acceptance gate G3-strang verifies slope ≤ −1.95 empirically.
//!
//! The struct itself implements [`ChernoffFunction`], so it composes
//! seamlessly with [`crate::ChernoffSemigroup`]:
//! `ChernoffSemigroup<StrangSplit<D, R>>` computes `(Φ(T/n))^n f`.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 4)
//!
//! `StrangSplit<D, R, F: SemiflowFloat = f64>` — the `= f64` default keeps
//! all existing call-sites compiling unchanged.

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{half, SemiflowFloat},
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Strang operator-splitting composer: `Φ(τ) = D(τ/2) ∘ R(τ) ∘ D(τ/2)`.
///
/// `D` implements `ChernoffFunction<F>` for `A = a(x)∂²_x` (typically
/// [`crate::DiffusionChernoff`]). `R` implements `ChernoffFunction<F>` for
/// `B = b(x)∂_x + c(x)` (typically [`crate::DriftReactionChernoff`]).
///
/// ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 4)
///
/// `StrangSplit<D, R, F: SemiflowFloat = f64>` — the `= f64` default keeps
/// all existing call-sites compiling unchanged.
///
/// # Order guarantee
/// Returns `order() == 2`. This assumes `D` has local order ≥ 2 and `R` is
/// exact (or has local order ≥ 2 and is time-symmetric). The canonical Strang
/// theorem (HLW §III.5) guarantees global order 2. For v0.2.0 this is verified
/// empirically (G3-strang slope ≤ −1.95).
///
/// # Growth
/// `growth()` returns `(1, c_norm_bound)` — the Strang sandwich inherits the
/// pure-contraction bound from `D` and the growth rate from `R`.
///
/// # `Copy` note
/// `Clone` is derived. `Copy` is derivable only if `D: Copy + R: Copy`; since
/// `DiffusionChernoff` and `DriftReactionChernoff` are both `Copy`, the typical
/// concrete type `StrangSplit<DiffusionChernoff, DriftReactionChernoff>` is `Copy`.
///
/// # Example
///
/// ```rust
/// use semiflow_core::{
///     Grid1D, GridFn1D, DiffusionChernoff, DriftReactionChernoff, StrangSplit,
///     ChernoffSemigroup, ChernoffFunction,
/// };
/// let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
/// let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
/// let drift = DriftReactionChernoff::new(|_| 0.0, |_| 0.0, 0.0, grid);
/// let split = StrangSplit::new(diff, drift);
/// assert_eq!(split.order(), 2);
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy)]
pub struct StrangSplit<D, R, F: SemiflowFloat = f64> {
    /// Inner Chernoff function for `A = a(x)·∂²_x`. Typically `DiffusionChernoff`.
    pub diffusion: D,
    /// Inner Chernoff function for `B = b(x)·∂_x + c(x)`. Typically `DriftReactionChernoff`.
    pub drift_reaction: R,
    /// Float type marker.
    _float: core::marker::PhantomData<F>,
}

impl<D, R, F: SemiflowFloat> StrangSplit<D, R, F>
where
    D: ChernoffFunction<F, S = GridFn1D<F>>,
    R: ChernoffFunction<F, S = GridFn1D<F>>,
{
    /// Construct a `StrangSplit<D, R, F>`.
    ///
    /// No validation — `D` and `R` already validated their own fields.
    #[must_use]
    pub fn new(diffusion: D, drift_reaction: R) -> Self {
        Self {
            diffusion,
            drift_reaction,
            _float: core::marker::PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<D, R, F: SemiflowFloat> ChernoffFunction<F> for StrangSplit<D, R, F>
where
    D: ChernoffFunction<F, S = GridFn1D<F>>,
    R: ChernoffFunction<F, S = GridFn1D<F>>,
{
    type S = GridFn1D<F>;

    /// Apply the Strang sandwich into `dst` (allocation-free).
    ///
    /// `D(τ/2) ∘ R(τ) ∘ D(τ/2)` — three sequential `apply_into` calls.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0` or non-finite.
    /// - Any [`SemiflowError`] from inner `apply_into` calls.
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn1D<F>,
        dst: &mut GridFn1D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let half_tau = half::<F>() * tau;
        let mut tmp1 = src.zeroed_like();
        let mut tmp2 = src.zeroed_like();
        self.diffusion
            .apply_into(half_tau, src, &mut tmp1, scratch)?;
        self.drift_reaction
            .apply_into(tau, &tmp1, &mut tmp2, scratch)?;
        self.diffusion.apply_into(half_tau, &tmp2, dst, scratch)?;
        Ok(())
    }

    /// Consistency order: 2.
    ///
    /// The Strang sandwich with `D` of local order ≥ 2 and exact `R` achieves
    /// global order 2 (HLW §III.5, Thm 4.1). For [A, B] ≠ 0 (variable
    /// coefficients), the palindromic structure cancels the leading commutator
    /// term, preserving order 2. This is verified empirically by G3-strang.
    fn order(&self) -> u32 {
        2
    }

    /// Growth bound `(M, ω)` for the composed Strang operator.
    ///
    /// With `diffusion.growth() = (M_D, ω_D)` and `drift_reaction.growth() = (M_R, ω_R)`:
    ///
    /// `‖Φ(τ) f‖_∞ ≤ M_D · exp(ω_D · τ/2) · M_R · exp(ω_R · τ) · M_D · exp(ω_D · τ/2) · ‖f‖_∞`
    ///
    /// For the typical concrete pair `(DiffusionChernoff, DriftReactionChernoff)`:
    /// `(M_D, ω_D) = (1, 0)` and `(M_R, ω_R) = (1, c_norm_bound)`, giving
    /// `(M, ω) = (1, c_norm_bound)`.
    fn growth(&self) -> Growth<F> {
        let gd = self.diffusion.growth();
        let gr = self.drift_reaction.growth();
        // M = M_D * M_R * M_D (three applications: D(τ/2), R(τ), D(τ/2))
        let m = gd.multiplier * gr.multiplier * gd.multiplier;
        // ω = ω_D + ω_R + ω_D (sum of growth exponents for τ/2 + τ + τ/2 = 2τ total)
        let omega = gd.omega + gr.omega + gd.omega;
        Growth {
            multiplier: m,
            omega,
        }
    }
}
