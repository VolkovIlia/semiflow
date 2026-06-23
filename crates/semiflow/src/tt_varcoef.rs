//! `VarCoefTt` — additive-separable variable-coefficient evolver on [`TtState`].
//!
//! Applies `P₂(τ/2)·k(τ)·P₂(τ/2)` per axis via [`varcoef_axis_step`] on the
//! **mode axis** of each TT core (middle index), leaving bond indices as spectators.
//! This is the carrier-level curse-escape: rank-1 IC stays rank-1 (§52.10d).
//!
//! ## Operator class
//!
//! `L = Σⱼ Lⱼ`, `Lⱼ = ∂_{xⱼ}(aⱼ(xⱼ)·∂_{xⱼ}) + bⱼ(xⱼ)·∂_{xⱼ} + vⱼ(xⱼ)`.
//! Additive-separable (per-axis) only. Non-separable `a(x,y)` is UNREPRESENTABLE.
//!
//! ## Solver-free
//!
//! Uses only: tridiagonal mat-vecs (P₂), 1-D FFT (k(τ)).
//! NO `lu_solve_inplace`, NO `dense_expm`.
//!
//! References: ADR-0178, math.md §52.10.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    tt_core::{tt_round, TtCore},
    tt_chernoff::TtState,
    tt_varcoef_spectral::varcoef_axis_step,
};

// ═══════════════════════════════════════════════════════════════════════════
// §A — VarCoefTt struct and constructor
// ═══════════════════════════════════════════════════════════════════════════

/// Additive-separable variable-coefficient TT evolver (ADR-0178, math §52.10).
///
/// Evolves a [`TtState`] by `exp(τ·L)` where `L = Σⱼ Lⱼ`, each
/// `Lⱼ = ∂_{xⱼ}(aⱼ·∂_{xⱼ}) + bⱼ·∂_{xⱼ} + vⱼ` on a periodic grid.
///
/// ## Proven boundary
/// Order-2 ONLY for ADDITIVE-separable coefficients. Non-separable `a(x,y)`
/// is UNREPRESENTABLE by per-axis arrays.
/// Rank-1 IC → rank-1 output (bond-preserving step, §52.10d).
/// Gate: `G_TT_VARCOEF` (RELEASE-BLOCKING, `slow-tests`).
pub struct VarCoefTt<F: SemiflowFloat> {
    /// Per-axis diffusion `aⱼ(xⱼ)` (length nⱼ each; all entries > 0).
    a_axis: Vec<Vec<F>>,
    /// Per-axis drift `bⱼ(xⱼ)` (length nⱼ each).
    b_axis: Vec<Vec<F>>,
    /// Per-axis reaction `vⱼ(xⱼ)` (length nⱼ or 0 for zero).
    v_axis: Vec<Vec<F>>,
    /// Per-axis domain bounds `(x_min_j, x_max_j)`.
    domain: Vec<(F, F)>,
    /// TT-rounding tolerance applied after each full τ step.
    eps_round: F,
}

impl<F: SemiflowFloat> VarCoefTt<F> {
    /// Construct and validate the evolver.
    ///
    /// # Errors
    /// Returns [`SemiflowError::VarCoefOutOfClass`] if:
    /// - `a_axis.len() == 0` (d must be ≥ 1)
    /// - Shape mismatch between `a_axis`, `b_axis`, `v_axis`, `domain`
    /// - Any `a_axis[j].len() < 2` (n must be ≥ 2)
    /// - Any `a_axis[j][i] ≤ 0` (parabolicity violated)
    pub fn new(
        a_axis: Vec<Vec<F>>,
        b_axis: Vec<Vec<F>>,
        v_axis: Vec<Vec<F>>,
        domain: Vec<(F, F)>,
        eps_round: F,
    ) -> Result<Self, SemiflowError> {
        validate(&a_axis, &b_axis, &v_axis, &domain)?;
        Ok(Self { a_axis, b_axis, v_axis, domain, eps_round })
    }

    /// Number of axes (spatial dimensions).
    pub fn ndim(&self) -> usize { self.a_axis.len() }

    /// Grid spacing for axis `j` (derived from domain and n_j).
    fn axis_dx(&self, j: usize) -> F {
        let n = self.a_axis[j].len();
        axis_dx_from(self.domain[j], n)
    }

    /// Apply one full τ step to `state` using the symmetric Strang palindrome.
    ///
    /// Palindrome: j=0 τ/2, j=1 τ/2, …, j=d-1 τ, …, j=1 τ/2, j=0 τ/2.
    /// Then TT-round at `eps_round`.
    pub fn step(&self, tau: F, state: &mut TtState<F>) {
        let d = self.ndim();
        let half = from_f64::<F>(0.5);
        let half_tau = tau * half;
        // Forward half-sweep: j = 0 .. d-2 at τ/2, j = d-1 at τ
        for j in 0..d {
            let t = if j == d - 1 { tau } else { half_tau };
            let dx = self.axis_dx(j);
            apply_varcoef_core(
                &mut state.cores[j],
                &self.a_axis[j],
                &self.b_axis[j],
                &self.v_axis[j],
                dx, t,
            );
        }
        // Backward half-sweep: j = d-2 .. 0 at τ/2 (j=d-1 already done)
        for j in (0..d.saturating_sub(1)).rev() {
            let dx = self.axis_dx(j);
            apply_varcoef_core(
                &mut state.cores[j],
                &self.a_axis[j],
                &self.b_axis[j],
                &self.v_axis[j],
                dx, half_tau,
            );
        }
        tt_round(&mut state.cores, self.eps_round);
    }

    /// Evolve `state` for `T_final` using `n_steps` equal time steps.
    ///
    /// # Panics
    /// Panics if `n_steps < 1`.
    pub fn evolve(&self, t_final: F, n_steps: usize, state: &mut TtState<F>) {
        assert!(n_steps >= 1, "n_steps must be >= 1");
        let tau = t_final / from_f64(n_steps as f64);
        for _ in 0..n_steps {
            self.step(tau, state);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — Core-level variable-coefficient kernel (the key TT insight)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply `P₂(τ/2)·k(τ)·P₂(τ/2)` to the **mode axis** of `core`.
///
/// `core` has shape `r_left × n × r_right`. For each `(il, ir)` pair, the
/// mode line of length `n` is extracted, stepped via [`varcoef_axis_step`],
/// and written back. Bond indices ride as spectators — rank-preserving.
fn apply_varcoef_core<F: SemiflowFloat>(
    core: &mut TtCore<F>,
    a: &[F],
    b: &[F],
    v: &[F],
    dx: F,
    tau: F,
) {
    let rl = core.r_left;
    let n = core.n;
    let rr = core.r_right;
    let mut line = vec![F::zero(); n];
    for il in 0..rl {
        for ir in 0..rr {
            // Extract mode line: core[il, :, ir]
            for im in 0..n {
                line[im] = core.get(il, im, ir);
            }
            varcoef_axis_step(&mut line, n, dx, a, b, v, tau);
            // Write back
            for im in 0..n {
                core.set(il, im, ir, line[im]);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Grid spacing from domain and n.
fn axis_dx_from<F: SemiflowFloat>(domain: (F, F), n: usize) -> F {
    let (lo, hi) = domain;
    if n <= 1 { return F::one(); }
    (hi - lo) / from_f64(n as f64 - 1.0)
}

/// Validate constructor arguments. Returns `VarCoefOutOfClass` on failure.
fn validate<F: SemiflowFloat>(
    a: &[Vec<F>],
    b: &[Vec<F>],
    v: &[Vec<F>],
    domain: &[(F, F)],
) -> Result<(), SemiflowError> {
    let d = a.len();
    if d == 0 {
        return Err(SemiflowError::VarCoefOutOfClass { detail: "d must be >= 1" });
    }
    if b.len() != d || v.len() != d || domain.len() != d {
        return Err(SemiflowError::VarCoefOutOfClass {
            detail: "a_axis, b_axis, v_axis, domain must all have length d",
        });
    }
    for j in 0..d {
        let nj = a[j].len();
        if nj < 2 {
            return Err(SemiflowError::VarCoefOutOfClass { detail: "n_j must be >= 2" });
        }
        if b[j].len() != nj {
            return Err(SemiflowError::VarCoefOutOfClass {
                detail: "b_axis[j] length must equal a_axis[j] length",
            });
        }
        if !v[j].is_empty() && v[j].len() != nj {
            return Err(SemiflowError::VarCoefOutOfClass {
                detail: "v_axis[j] length must equal a_axis[j] length or be empty",
            });
        }
        for &val in &a[j] {
            if val <= F::zero() {
                return Err(SemiflowError::VarCoefOutOfClass {
                    detail: "a_axis[j][i] must be > 0 (parabolicity)",
                });
            }
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — Unit tests (fast-path: rejection, acceptance, reduction invariant)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use crate::tt_varcoef_spectral::varcoef_axis_step;

    fn flat_a(n: usize, val: f64) -> Vec<Vec<f64>> { vec![vec![val; n]; 2] }
    fn zero_bv(n: usize) -> Vec<Vec<f64>> { vec![vec![0.0; n]; 2] }
    fn domain2() -> Vec<(f64, f64)> { vec![(-5.0, 5.0); 2] }

    #[test]
    fn rejects_d_zero() {
        let r = VarCoefTt::<f64>::new(vec![], vec![], vec![], vec![], 1e-8);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_n_lt_2() {
        let a = vec![vec![1.0_f64]; 2]; // n=1
        let b = zero_bv(1);
        let v = zero_bv(1);
        let r = VarCoefTt::<f64>::new(a, b, v, domain2(), 1e-8);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_a_nonpositive() {
        let mut a = flat_a(4, 1.0);
        a[0][1] = -0.1;
        let r = VarCoefTt::<f64>::new(a, zero_bv(4), zero_bv(4), domain2(), 1e-8);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_shape_mismatch() {
        let a = flat_a(4, 1.0);
        let b = vec![vec![0.0_f64; 4]]; // length 1, not 2
        let v = zero_bv(4);
        let r = VarCoefTt::<f64>::new(a, b, v, domain2(), 1e-8);
        assert!(r.is_err());
    }

    #[test]
    fn accepts_valid() {
        let r = VarCoefTt::<f64>::new(flat_a(4, 0.5), zero_bv(4), zero_bv(4), domain2(), 1e-8);
        assert!(r.is_ok());
    }

    // Reduction invariant: with flat a_j = const, b=0, v=0, apply_varcoef_core
    // equals apply_drift_spectral_axis (≤ 1e-12 diff) on a simple line.
    #[test]
    fn reduction_flat_a_equals_spectral() {
        use crate::tt_drift_spectral::apply_drift_spectral_axis;
        let n = 8usize;
        let dx = (10.0_f64) / (n as f64 - 1.0); // domain [-5,5]
        let a0 = 0.7_f64;
        let a_flat = vec![a0; n];
        let b_zero: Vec<f64> = vec![0.0; n]; // must be length-n, not empty
        let v_zero: Vec<f64> = vec![]; // empty slice = zero reaction
        let tau = 0.02;

        // Reference: pure spectral with const a, no drift
        let init: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.5).cos()).collect();
        let mut spectral_line = init.clone();
        apply_drift_spectral_axis(&mut spectral_line, n, dx, a0, 0.0, tau);

        // varcoef path with flat a (R=0 → P₂=I → pure k(τ))
        let mut varcoef_line = init.clone();
        varcoef_axis_step(&mut varcoef_line, n, dx, &a_flat, &b_zero, &v_zero, tau);

        let max_err = varcoef_line.iter().zip(&spectral_line)
            .map(|(p, q)| (p - q).abs()).fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "reduction invariant violated: max_err={max_err:.3e} (expected < 1e-12)"
        );
    }

    // Smoke: evolve a rank-1 state and verify it stays rank-1 and finite.
    #[test]
    fn evolve_rank1_stays_rank1_and_finite() {
        use crate::tt_chernoff::TtState;
        let n = 8usize;
        let d = 3usize;
        let dx = 10.0_f64 / (n as f64 - 1.0);
        let a_axis: Vec<Vec<f64>> = (0..d)
            .map(|j| (0..n).map(|i| 0.5 + 0.2 * ((i as f64) * dx - 5.0 + j as f64).sin().powi(2))
                .collect())
            .collect();
        let b_axis: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0; n]).collect();
        let v_axis: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0; n]).collect();
        let domain: Vec<(f64, f64)> = vec![(-5.0, 5.0); d];
        let ev = VarCoefTt::new(a_axis, b_axis, v_axis, domain, 1e-8).unwrap();

        let slice: Vec<f64> = (0..n).map(|i| (-(((i as f64) * dx - 5.0).powi(2)) / 2.0).exp()).collect();
        let mut state = TtState::rank1_separable((0..d).map(|_| slice.clone()).collect());

        ev.evolve(0.1, 3, &mut state);
        assert_eq!(state.peak_rank(), 1, "rank-1 IC grew to r={}", state.peak_rank());
        for core in &state.cores {
            assert!(core.data.iter().all(|x| x.is_finite()), "non-finite after evolve");
        }
    }
}
