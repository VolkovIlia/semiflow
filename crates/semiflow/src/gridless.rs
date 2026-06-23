//! Gridless particle-ensemble Chernoff evolver — `GridlessChernoff` (v9.0.0, ADR-0155).
//!
//! Applies the 1-D 3-branch kernel (eq. 50.3) per-axis as a sequential splitting
//! sweep over all `d` axes (commuting diagonal-A ⇒ exact splitting; reaction global).
//! `R_P` (d-dimensional product-bin merge, `gridless_reduce`) applied after each axis
//! sub-step keeps the working set to `O(d·P_cap)` — defeating the `3^d` curse.
//!
//! **Narrow ship** — diagonal A, constant scalar coefs, `d ≤ ~10`.  Off-diagonal
//! Cholesky / variable coefs / `d>10` = RESEARCH-TRACK (ADR-0155 §50.7).

use core::marker::PhantomData;

use crate::{
    adjoint_fp::MeasureState,
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    scratch::ScratchPool,
};

pub use crate::gridless_reduce::ParticleReduction;

// ---------------------------------------------------------------------------
// GridlessChernoff<F, D>
// ---------------------------------------------------------------------------

/// Gridless particle-ensemble Chernoff evolver (v9.0.0, ADR-0155, math §50).
///
/// Evolves [`MeasureState<F, D>`] by the per-axis branching sweep (§2.3):
/// axis `j` sub-step: `(x,w) → (x+h_j e_j, ¼w) + (x-h_j e_j, ¼w) + (x+k_j e_j, ½w)`;
/// global reaction `(1+τc)` applied once after all axes; `R_P` cap after each axis.
///
/// `a[D]` — per-axis diffusion (all ≥ 0).  `b[D]` — per-axis drift.  `c` — reaction.
/// Use [`GridlessChernoff::isotropic`] for uniform coefficients.
///
/// ## Example
///
/// ```rust
/// use semiflow::{GridlessChernoff, MeasureState, ParticleReduction};
/// use semiflow::chernoff::ChernoffFunction;
/// use semiflow::scratch::ScratchPool;
///
/// let ev = GridlessChernoff::<f64, 1>::isotropic(
///     0.5, 0.0, 0.0, ParticleReduction::WeightedVoronoi { cap: 64 });
/// let rho0 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
/// let mut rho1 = rho0.clone();
/// let mut pool = ScratchPool::new();
/// ev.apply_into(0.1, &rho0, &mut rho1, &mut pool).unwrap();
/// ```
#[derive(Clone)]
pub struct GridlessChernoff<F: SemiflowFloat, const D: usize> {
    coeff_a: [F; D],
    coeff_b: [F; D],
    coeff_c: F,
    reduction: ParticleReduction,
    _phantom: PhantomData<F>,
}

impl<F: SemiflowFloat, const D: usize> GridlessChernoff<F, D> {
    /// Construct with per-axis `a[D]`, `b[D]`, scalar `c`, and reduction policy.
    /// All `a[j]` must be ≥ 0 (validated at step time).
    #[must_use]
    pub fn new(a: [F; D], b: [F; D], c: F, reduction: ParticleReduction) -> Self {
        Self {
            coeff_a: a,
            coeff_b: b,
            coeff_c: c,
            reduction,
            _phantom: PhantomData,
        }
    }

    /// Convenience constructor for isotropic case (all axes use the same `a` and `b`).
    #[must_use]
    pub fn isotropic(a: F, b: F, c: F, reduction: ParticleReduction) -> Self {
        Self::new([a; D], [b; D], c, reduction)
    }

    /// Per-axis diffusion coefficients.
    #[must_use]
    pub fn a(&self) -> &[F; D] {
        &self.coeff_a
    }

    /// Per-axis drift coefficients.
    #[must_use]
    pub fn b(&self) -> &[F; D] {
        &self.coeff_b
    }

    /// Scalar reaction coefficient.
    #[must_use]
    pub fn c(&self) -> F {
        self.coeff_c
    }

    /// The particle-reduction policy.
    #[must_use]
    pub fn reduction(&self) -> &ParticleReduction {
        &self.reduction
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat, const D: usize> ChernoffFunction<F> for GridlessChernoff<F, D> {
    type S = MeasureState<F, D>;

    /// One Chernoff step: per-axis sweep (§2.3) + global reaction + reduction per axis.
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if `tau < 0` / non-finite, any `a[j] < 0`,
    /// or `WeightedVoronoi.cap == 0`.
    fn apply_into(
        &self,
        tau: F,
        src: &MeasureState<F, D>,
        dst: &mut MeasureState<F, D>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        validate_tau_and_a(tau, &self.coeff_a)?;
        push_forward_sweep(
            tau,
            &self.coeff_a,
            &self.coeff_b,
            self.coeff_c,
            src,
            dst,
            &self.reduction,
        )
    }

    /// Order 2 — per-axis kernel + commuting-axes exact splitting for diagonal A.
    fn order(&self) -> u32 {
        2
    }

    /// Growth: `e^{|c|τ}` (one-step mass factor `1+τc`).
    fn growth(&self) -> Growth<F> {
        Growth::new(F::one(), self.coeff_c.abs())
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate `tau >= 0` and all `a_j >= 0`. Returns `DomainViolation` on failure.
fn validate_tau_and_a<F: SemiflowFloat, const D: usize>(
    tau: F,
    a: &[F; D],
) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "gridless: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    for &aj in a {
        if aj < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "gridless: diffusion coefficient a_j must be >= 0",
                value: aj.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core push-forward sweep (sequential per-axis, §2.3)
// ---------------------------------------------------------------------------

/// Sequential per-axis push-forward: axis 0..D-1, then global reaction `(1+τc)`.
///
/// `R_P` is applied after each axis sub-step (bounds working set to `3·P_cap`).
#[allow(clippy::too_many_arguments)]
fn push_forward_sweep<F: SemiflowFloat, const D: usize>(
    tau: F,
    a: &[F; D],
    b: &[F; D],
    c: F,
    src: &MeasureState<F, D>,
    dst: &mut MeasureState<F, D>,
    reduction: &ParticleReduction,
) -> Result<(), SemiflowError> {
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    *dst = src.clone();
    for j in 0..D {
        let h_j = two * (a[j] * tau).sqrt();
        let k_j = two * b[j] * tau;
        axis_branch_step(dst, j, h_j, k_j);
        reduction.apply(dst)?;
    }
    apply_global_reaction(dst, tau, c);
    Ok(())
}

/// Branch every particle in `ensemble` on axis `j`: 3 children each (c=0 reaction).
fn axis_branch_step<F: SemiflowFloat, const D: usize>(
    ensemble: &mut MeasureState<F, D>,
    axis: usize,
    h: F,
    k: F,
) {
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let quarter = F::one() / (two + two);
    let half = F::one() / two;
    let old: alloc::vec::Vec<([F; D], F)> = ensemble.as_diracs_slice().to_vec();
    ensemble.clear_diracs_with_capacity(old.len() * 3);
    for (pos, w) in old {
        ensemble.push_dirac_raw(shift_axis(pos, axis, h), quarter * w);
        ensemble.push_dirac_raw(shift_axis(pos, axis, -h), quarter * w);
        ensemble.push_dirac_raw(shift_axis(pos, axis, k), half * w);
    }
}

/// Apply global reaction factor `(1+τc)` to all particle weights.
fn apply_global_reaction<F: SemiflowFloat, const D: usize>(
    ensemble: &mut MeasureState<F, D>,
    tau: F,
    c: F,
) {
    if c == F::zero() {
        return;
    }
    let factor = F::one() + tau * c;
    let particles: alloc::vec::Vec<([F; D], F)> = ensemble
        .as_diracs_slice()
        .iter()
        .map(|&(pos, w)| (pos, w * factor))
        .collect();
    ensemble.clear_diracs_with_capacity(particles.len());
    for (pos, w) in particles {
        ensemble.push_dirac_raw(pos, w);
    }
}

/// Return `pos` with axis `j` shifted by `delta` (other axes unchanged).
#[inline]
fn shift_axis<F, const D: usize>(mut pos: [F; D], j: usize, delta: F) -> [F; D]
where
    F: Copy + core::ops::Add<Output = F>,
{
    if j < D {
        pos[j] = pos[j] + delta;
    }
    pos
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "gridless_tests.rs"]
mod tests;
