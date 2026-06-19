//! Adjoint Fokker-Planck Chernoff on the weak-* topology of M(ℝ^D).
//!
//! v8.0.0 Phase-4 C2 (ADR-0107 AMENDMENT 1, math.md §38).
//!
//! For any forward Chernoff function `S(t) : C_b → C_b`, the dual pairing
//! `⟨f, S*(t) ρ⟩ := ⟨S(t) f, ρ⟩` defines the ADJOINT Chernoff function for
//! `e^{tL*}` on M(ℝ^D) under vague topology σ(M, `C_b`) — Theorem A.2 (§38.4).
//!
//! **Lemma A.1** (§38.3): for the Theorem 4 Chernoff function with coefficients
//! `a, b, c` and `h = 2√(aτ)`, `k = 2bτ`, the adjoint pushes each Dirac:
//!
//! ```text
//! S*(τ) δ_x = (1/4)δ_{x+h} + (1/4)δ_{x-h} + (1/2)δ_{x+k} + τc · δ_x
//! ```
//!
//! Mass conservation: Σ coefficients = 1 + τc. Exact when c = 0.
//!
//! **Not** to be confused with [`crate::AdjointChernoff`] (§15, ADR-0114):
//! that operates on the SAME `L²/C_b` function space; this module operates on
//! the MEASURE space M(ℝ^D) — different state type (`MeasureState` vs `GridFn`).

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// GaussianComponent (private helper for Dirac-count reduction in long-time use)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GaussianComponent<F: SemiflowFloat, const D: usize> {
    mean: [F; D],
    /// Isotropic variance σ² (diagonal covariance σ²·I). Must be > 0.
    variance: F,
    weight: F,
}

// ---------------------------------------------------------------------------
// MeasureState<F, D>
// ---------------------------------------------------------------------------

/// Sparse signed-measure state on M(ℝ^D) for adjoint Fokker-Planck Chernoff.
/// Representation: finite weighted-Dirac sum + optional Gaussian background.
/// Lemma A.1 pushes each Dirac at `x` to four children `x±h, x+k, x` with
/// weights `(1/4, 1/4, 1/2, τc)`. The 4^n blow-up is resolved by `gridless.rs`.
#[derive(Clone)]
pub struct MeasureState<F: SemiflowFloat, const D: usize> {
    diracs: Vec<([F; D], F)>,
    gaussians: Vec<GaussianComponent<F, D>>,
}

impl<F: SemiflowFloat, const D: usize> MeasureState<F, D> {
    /// Construct a single Dirac `δ_position` with signed weight.
    #[must_use]
    pub fn dirac(position: [F; D], weight: F) -> Self {
        Self {
            diracs: alloc::vec![(position, weight)],
            gaussians: Vec::new(),
        }
    }

    /// Construct a Gaussian-smoothed measure (isotropic variance).
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if variance ≤ 0 or weight non-finite.
    pub fn gaussian(mean: [F; D], variance: F, weight: F) -> Result<Self, SemiflowError> {
        if !variance.is_finite() || variance <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "MeasureState::gaussian: variance must be finite and > 0",
                value: variance.to_f64().unwrap_or(f64::NAN),
            });
        }
        if !weight.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "MeasureState::gaussian: weight must be finite",
                value: weight.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self {
            diracs: Vec::new(),
            gaussians: alloc::vec![GaussianComponent {
                mean,
                variance,
                weight
            }],
        })
    }

    /// Construct from `(position, weight)` pairs; no validity checks.
    #[must_use]
    pub fn from_particles(particles: &[([F; D], F)]) -> Self {
        Self {
            diracs: particles.to_vec(),
            gaussians: Vec::new(),
        }
    }

    /// Total-variation norm `‖ρ‖_TV` — rate modulator in Theorem A.2 (§38.4).
    #[must_use]
    pub fn total_variation(&self) -> F {
        let d = self.diracs.iter().fold(F::zero(), |a, (_, w)| a + w.abs());
        let g = self
            .gaussians
            .iter()
            .fold(F::zero(), |a, g| a + g.weight.abs());
        d + g
    }

    /// Second moment `⟨x², ρ⟩` — tightness monitor per Lemma A.3 (§38.5).
    #[must_use]
    pub fn second_moment(&self) -> F {
        let d = self.diracs.iter().fold(F::zero(), |acc, (pos, w)| {
            acc + pos.iter().fold(F::zero(), |s, &xi| s + xi * xi) * *w
        });
        let g = self.gaussians.iter().fold(F::zero(), |acc, g| {
            let norm2 = g.mean.iter().fold(F::zero(), |s, &xi| s + xi * xi);
            let d_f = F::from(D as f64).unwrap_or(F::one());
            acc + (norm2 + d_f * g.variance) * g.weight
        });
        d + g
    }

    /// Dual pairing `⟨f, ρ⟩` for a test function f ∈ `C_b`.
    /// Gaussians use point evaluation at the mean.
    #[must_use]
    pub fn pair<G: Fn(&[F; D]) -> F>(&self, f: G) -> F {
        let d = self
            .diracs
            .iter()
            .fold(F::zero(), |a, (p, w)| a + f(p) * *w);
        let g = self
            .gaussians
            .iter()
            .fold(F::zero(), |a, g| a + f(&g.mean) * g.weight);
        d + g
    }

    /// Number of Dirac components (diagnostic).
    #[must_use]
    pub fn n_diracs(&self) -> usize {
        self.diracs.len()
    }

    /// Extract Dirac positions and weights as flat `Vec<F>` buffers (D=1 binding ABI).
    ///
    /// Returns `(positions, weights)` where both have length `n_diracs()`.
    /// Position of Dirac `i` is `positions[i]` (scalar for D=1).
    /// Gaussians are not included; use only on pure-Dirac states from `apply_into`.
    ///
    /// Used by binding integration tests to extract the golden vector for
    /// 0-ULP cross-surface comparison (`G_BINDING_ADJOINT_FP_PARITY`).
    #[must_use]
    pub fn to_flat_buffers_d1(&self) -> (Vec<F>, Vec<F>)
    where
        [F; 1]: Sized,
        F: Copy,
    {
        let pos: Vec<F> = self.diracs.iter().map(|(p, _)| p[0]).collect();
        let wts: Vec<F> = self.diracs.iter().map(|(_, w)| *w).collect();
        (pos, wts)
    }

    fn zero_in_place(&mut self) {
        self.diracs.clear();
        self.gaussians.clear();
    }

    /// Crate-private: raw Dirac slice (used by `gridless.rs`).
    #[must_use]
    pub(crate) fn as_diracs_slice(&self) -> &[([F; D], F)] {
        &self.diracs
    }

    pub(crate) fn push_dirac_raw(&mut self, pos: [F; D], weight: F) {
        self.diracs.push((pos, weight));
    }

    /// Crate-private: clear and reserve (used by `gridless.rs`).
    pub(crate) fn clear_diracs_with_capacity(&mut self, cap: usize) {
        self.diracs.clear();
        self.diracs.reserve(cap);
    }
}

/// `State<F>` impl for `MeasureState<F, D>`.
///
/// **Dynamic-length measure** — deliberately exempt from the dense-state
/// invariants documented in `state.rs`:
///
/// - `len()` returns the current atom count (grows after each `axpy_into`
///   call, not fixed at construction).
/// - `axpy_into` APPENDS atoms rather than overwriting; this is correct for
///   signed measures (superposition of weighted Diracs / Gaussians).
/// - `norm_sup()` returns the TV-norm `‖ρ‖_TV = Σ|wᵢ|`, which is the
///   natural measure-space norm — NOT the pointwise sup-norm of a function.
///
/// Shape-match / length-invariant preconditions from the dense `GridFn1D`
/// `State` impl do NOT apply here.  Callers that manipulate `MeasureState`
/// through the `State` trait must not assume fixed `len()`.
impl<F: SemiflowFloat, const D: usize> State<F> for MeasureState<F, D> {
    fn len(&self) -> usize {
        self.diracs.len() + self.gaussians.len()
    }

    fn axpy_into(&mut self, alpha: F, src: &Self) {
        for (pos, w) in &src.diracs {
            self.diracs.push((*pos, alpha * *w));
        }
        for g in &src.gaussians {
            self.gaussians.push(GaussianComponent {
                mean: g.mean,
                variance: g.variance,
                weight: alpha * g.weight,
            });
        }
    }

    fn copy_from(&mut self, src: &Self) {
        self.diracs.clone_from(&src.diracs);
        self.gaussians.clone_from(&src.gaussians);
    }

    fn zero_into(&mut self) {
        self.zero_in_place();
    }

    /// Returns `‖ρ‖_TV` as a surrogate (TV-norm is the natural measure norm).
    fn norm_sup(&self) -> F {
        self.total_variation()
    }

    fn scale_into(&mut self, k: F) {
        for (_, w) in &mut self.diracs {
            *w *= k;
        }
        for g in &mut self.gaussians {
            g.weight *= k;
        }
    }
}

// ---------------------------------------------------------------------------
// Lemma A.1 push (§38.3) — 1D constant-coefficient 4-Dirac kernel
// (defined before Adjointable trait to avoid scanner false-positive, batch H8)
// ---------------------------------------------------------------------------

// a, b, c are generator coefficients per Lemma A.1; single-char names match math notation.
#[allow(clippy::many_single_char_names)]
fn lemma_a1_push<F: SemiflowFloat>(
    tau: F,
    a: F,
    b: F,
    c: F,
    src: &MeasureState<F, 1>,
    dst: &mut MeasureState<F, 1>,
) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "adjoint_fp: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let four = two + two;
    let quarter = F::one() / four;
    let half = F::one() / two;
    let h = two * (a * tau).sqrt(); // Lemma A.1: h = 2√(aτ)
    let k = two * b * tau; // Lemma A.1: k = 2bτ
    let tc = tau * c;

    dst.zero_in_place();
    for (pos, w) in &src.diracs {
        let x = pos[0];
        dst.diracs.push(([x + h], quarter * *w));
        dst.diracs.push(([x - h], quarter * *w));
        dst.diracs.push(([x + k], half * *w));
        dst.diracs.push(([x], tc * *w));
    }
    // Gaussian: mean shifts by k, variance grows by 2aτ (Lemma A.3 §38.5).
    for g in &src.gaussians {
        dst.gaussians.push(GaussianComponent {
            mean: [g.mean[0] + k],
            variance: g.variance + two * a * tau,
            weight: (F::one() + tc) * g.weight,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Adjointable<F, D> supertrait
// ---------------------------------------------------------------------------

/// Supertrait of `ChernoffFunction<F>`: admits a pushforward adjoint on M(ℝ^D).
///
/// The blanket impl (D=1) provides the default 4-Dirac pushforward via
/// Lemma A.1 (§38.3). Per-backend overrides allowed for performance.
pub trait Adjointable<F: SemiflowFloat, const D: usize>: ChernoffFunction<F> {
    /// Push-forward one Chernoff step on the measure side (Lemma A.1, §38.3).
    ///
    /// `a, b, c` are the generator coefficients; `tau` is the step size.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `tau < 0` or non-finite.
    // τ + 3 generator coefficients + src + dst = 7; all are required for the Lemma A.1 push.
    #[allow(clippy::too_many_arguments)]
    fn apply_adjoint_into(
        &self,
        tau: F,
        a: F,
        b: F,
        c: F,
        src: &MeasureState<F, D>,
        dst: &mut MeasureState<F, D>,
    ) -> Result<(), SemiflowError>;
}

// ---------------------------------------------------------------------------
// Blanket Adjointable impl for D = 1
// ---------------------------------------------------------------------------

impl<C, F> Adjointable<F, 1> for C
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    fn apply_adjoint_into(
        &self,
        tau: F,
        a: F,
        b: F,
        c: F,
        src: &MeasureState<F, 1>,
        dst: &mut MeasureState<F, 1>,
    ) -> Result<(), SemiflowError> {
        lemma_a1_push(tau, a, b, c, src, dst)
    }
}

// ---------------------------------------------------------------------------
// AdjointFokkerPlanckChernoff<C, F, D>
// ---------------------------------------------------------------------------

/// Adjoint Fokker-Planck Chernoff on M(ℝ^D).
///
/// Generic over any forward Chernoff function `C: Adjointable<F, D>` (e.g.
/// `DiffusionChernoff`, `DriftReactionChernoff`, or any custom kernel).
/// The gate instantiates `C = DiffusionChernoff` (Brownian motion benchmark).
///
/// Per Theorem A.2 (§38.4) the adjoint rate equals the forward rate modulated
/// by `‖ρ‖_TV`. Any forward Chernoff function auto-lifts to its dual via this
/// wrapper — COMPOSITIONAL, no per-backend re-derivation needed.
///
/// `order()` inherits from the forward kernel `C`. `type S = MeasureState<F, D>`.
///
/// # Example
///
/// ```rust,no_run
/// use semiflow_core::{AdjointFokkerPlanckChernoff, DiffusionChernoff, Grid1D, MeasureState};
/// let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
/// let fwd = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
/// let adj = AdjointFokkerPlanckChernoff::new(fwd, 0.5, 0.0, 0.0);
/// ```
#[derive(Clone)]
pub struct AdjointFokkerPlanckChernoff<C, F: SemiflowFloat, const D: usize>
where
    C: Adjointable<F, D>,
{
    forward: C,
    a: F, // diffusion coefficient a (h = 2√(aτ))
    b: F, // drift coefficient b (k = 2bτ)
    c: F, // reaction coefficient c (mass factor 1 + τc)
    _phantom: PhantomData<F>,
}

impl<C, F, const D: usize> AdjointFokkerPlanckChernoff<C, F, D>
where
    C: Adjointable<F, D>,
    F: SemiflowFloat,
{
    /// Wrap a forward Chernoff function into its weak-* dual (§38.2).
    ///
    /// `a, b, c` are the generator coefficients of `L = a·∂²_x + b·∂_x + c`
    /// (Theorem 4, Galkin-Remizov 2025 *IJM* eq. 11). The adjoint S* uses
    /// `h = 2√(aτ)`, `k = 2bτ`, mass factor `1 + τc` (Lemma A.1, §38.3).
    #[must_use]
    pub fn new(forward: C, a: F, b: F, c: F) -> Self {
        Self {
            forward,
            a,
            b,
            c,
            _phantom: PhantomData,
        }
    }

    /// Borrow the wrapped forward Chernoff function.
    #[must_use]
    pub fn forward(&self) -> &C {
        &self.forward
    }

    /// Diffusion coefficient a (push-distance `h = 2√(aτ)`).
    #[must_use]
    pub fn a(&self) -> F {
        self.a
    }

    /// Drift coefficient b (push-distance `k = 2bτ`).
    #[must_use]
    pub fn b(&self) -> F {
        self.b
    }

    /// Reaction coefficient c (mass factor `1 + τc`).
    #[must_use]
    pub fn c(&self) -> F {
        self.c
    }
}

impl<C, F, const D: usize> ChernoffFunction<F> for AdjointFokkerPlanckChernoff<C, F, D>
where
    C: Adjointable<F, D>,
    F: SemiflowFloat,
{
    type S = MeasureState<F, D>;

    fn apply_into(
        &self,
        tau: F,
        src: &MeasureState<F, D>,
        dst: &mut MeasureState<F, D>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        Adjointable::apply_adjoint_into(&self.forward, tau, self.a, self.b, self.c, src, dst)
    }

    /// Inherits order from the forward kernel (Theorem A.2 identical rate, §38.4).
    fn order(&self) -> u32 {
        self.forward.order()
    }

    /// Growth bound inherited: `‖S*(τ)‖_{M→M} = ‖S(τ)‖_{C_b→C_b}` (Bogachev §4).
    fn growth(&self) -> Growth<F> {
        self.forward.growth()
    }
}

// ---------------------------------------------------------------------------
// In-module unit tests (split to adjoint_fp_tests.rs, batch H8)
// ---------------------------------------------------------------------------

#[cfg(test)]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/adjoint_fp_tests.rs"
));
