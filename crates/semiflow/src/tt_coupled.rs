//! `CoupledTtChernoff` — genuine cross-axis TT evolver (v9.1.0 Shift C, §52.9).
//!
//! Additive sibling to `TtChernoff`. Applies per-pair stable pair factors
//! `exp(τ_eff · L_pair)` where `L_pair = c_j·D2_j + c_k·D2_k + 2ρ√(a_j·a_k)·∂_j∂_k`
//! (§52.9 NORMATIVE v9.1.0-s3 round 2; `tt_coupled_pair.rs` for construction).
//!
//! Step (§52.9 NORMATIVE, P3' rotated-factor scheme):
//! (1) diagonal per-axis shift (separable, same as `TtChernoff`);
//! (2) Strang-composed pair sweep — stable `exp(τ_eff·L_pair)` factors applied
//!     ½ forward + ½ reverse for tridiagonal, full-τ for single pair (d=2);
//! (3) reaction scalar `(1+τc)`;
//! (4) TT-rounding.
//!
//! For the constant-coefficient correlated-Gaussian class the pair generators commute
//! (circulant), so the Strang-composed product is **exact** (~1e-14 vs expm, constant-coef).
//!
//! `CouplingTopology::None` → bit-identical to `TtChernoff` (Gate C invariant, §52.3).
//! References: math.md §52.9 NORMATIVE (P3'); ADR-0162; `tt_coupled_pair.rs`;
//! `probe_fix_final.py`; `probe_pair_rank.py` (op-rank ≤ 6).

// Grid/TT indices (usize) cast to f64 for coordinate computations; values ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use alloc::vec::Vec;

use crate::{
    float::SemiflowFloat,
    tt_chernoff::{apply_per_axis_shift_pub, TtState},
    tt_core::tt_round,
    tt_coupled_pair::{pair_sweep_strang, precompute_pair_expsyms},
};

// ═══════════════════════════════════════════════════════════════════════════
// §A — CouplingTopology
// ═══════════════════════════════════════════════════════════════════════════

/// Which pairs of axes to couple with `D1_j⊗D1_k` (mixed-derivative operator).
///
/// - `None` — no coupling. Reduces `CoupledTtChernoff` to `TtChernoff` exactly.
/// - `Tridiagonal(ρ)` — nearest-neighbour pairs `(j, j+1)` for all j (chain coupling).
///   This is the LOCAL topology; analytic peak rank is O(1) independent of d.
/// - `Pairs(pairs)` — arbitrary set of `(j, k, ρ_{jk})` with `j < k`.
///   For all-pairs dense equicorrelation, analytic peak rank ≤ `⌊d/2⌋`.
#[derive(Clone, Debug)]
pub enum CouplingTopology<F: SemiflowFloat> {
    /// No coupling — reduces exactly to the separable `TtChernoff` path.
    None,
    /// Nearest-neighbour chain coupling with uniform correlation `ρ`.
    /// All pairs `(j, j+1, ρ)` for `j ∈ 0..d-1`.
    Tridiagonal(F),
    /// Explicit list of `(j, k, ρ_{jk})` pairs with `j < k`.
    Pairs(Vec<(usize, usize, F)>),
}

impl<F: SemiflowFloat> CouplingTopology<F> {
    /// Extract the list of `(j, k, ρ)` pairs for a given dimension `d`.
    pub fn pairs(&self, d: usize) -> Vec<(usize, usize, F)> {
        match self {
            Self::None => Vec::new(),
            Self::Tridiagonal(rho) => (0..d.saturating_sub(1)).map(|j| (j, j + 1, *rho)).collect(),
            Self::Pairs(ps) => ps.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — CoupledTtChernoff evolver
// ═══════════════════════════════════════════════════════════════════════════

/// Genuine cross-axis TT-Chernoff evolver for correlated diffusion (P3' rotated-factor).
///
/// Extends `TtChernoff` with stable correlated-shift pair factors `exp(τ·L_pair)`,
/// where `L_pair = c_j·D2_j + c_k·D2_k + 2ρ√(a_j·a_k)·∂_j∂_k` (§52.9 NORMATIVE, P3').
/// When `coupling = CouplingTopology::None`, behaviour is bit-identical to
/// `TtChernoff` (Gate C, §52.3 additive-compatibility invariant).
///
/// ## Scope (NORMATIVE — MANDATORY honesty constraint)
///
/// - **v9.1.0 IMPLEMENTED**: constant diagonal A, `b=0` (drift rejected — v9.2.0),
///   scalar reaction c, constant ρ_{jk}, adjacent pairs only (non-adjacent rejected).
///   Pair generators commute (circulant) → Strang product is EXACT (~1e-14 vs `expm`).
///   Rank bounded poly-in-d. SPD threshold: `|ρ|<0.5` interior tridiag d≥4.
/// - **RESEARCH TRACK** (not in scope): variable A, variable ρ(x), non-adjacent pairs,
///   drift b≠0, dense large-d Cholesky — deferred to v9.2.0.
///
/// ## SPD constraint (MANDATORY per §10.12 risk 1)
///
/// Each pair block `B=[[cj,r],[r,ck]]` must be SPD (`det B = cj·ck − r² > 0`).
/// `step()` panics if this is violated — check |ρ| before constructing the evolver.
#[derive(Clone, Debug)]
pub struct CoupledTtChernoff<F: SemiflowFloat> {
    /// Per-axis diffusion coefficients `a_j ≥ 0`.
    pub a: Vec<F>,
    /// Per-axis drift coefficients `b_j`.
    pub b: Vec<F>,
    /// Scalar reaction `c`.
    pub c: F,
    /// Coupling topology — which axis pairs to couple and at what strength.
    pub coupling: CouplingTopology<F>,
    /// Per-axis domain `(x_min_j, x_max_j)`.
    pub domain: Vec<(F, F)>,
    /// TT-rounding tolerance (Frobenius, quasi-optimal per Oseledets 2011).
    pub eps_round: F,
}

impl<F: SemiflowFloat> CoupledTtChernoff<F> {
    /// Construct the evolver.
    ///
    /// # Panics
    /// Panics if lengths differ, if any `b[j]!=0` (drift deferred to v9.2.0),
    /// or if any `Pairs` entry has `k>j+1` (non-adjacent — v9.2.0).
    #[must_use]
    pub fn new(
        a: Vec<F>,
        b: Vec<F>,
        c: F,
        coupling: CouplingTopology<F>,
        domain: Vec<(F, F)>,
        eps_round: F,
    ) -> Self {
        let d = a.len();
        assert_eq!(b.len(), d);
        assert_eq!(domain.len(), d);
        // Guard 1: drift b≠0 not supported (symmetric shift is mean-preserving, not advective).
        // ADR-0162 / §52.9 v9.2.0 deferral.
        for (j, &bj) in b.iter().enumerate() {
            assert!(
                bj == F::zero(),
                "CoupledTtChernoff drift b≠0 is not supported in v9.1.0 \
                 (drift advection unimplemented — see ADR-0162 / §52.9 v9.2.0 deferral); \
                 pass b = 0 (axis {j} has b[{j}] ≠ 0)",
            );
        }
        // Guard 2: non-adjacent pairs (k>j+1) silently freeze axes — reject loud.
        // Adjacent block-disjoint Pairs and Tridiagonal are fully supported.
        // True dense / non-adjacent coupling: v9.2.0.
        if let CouplingTopology::Pairs(ref ps) = coupling {
            for &(j, k, _) in ps {
                let (lo, hi) = if j < k { (j, k) } else { (k, j) };
                assert!(
                    hi == lo + 1,
                    "CoupledTtChernoff non-adjacent pair ({lo},{hi}) with k>j+1 is not \
                     supported in v9.1.0 (only tridiagonal / block-disjoint adjacent pairs; \
                     true dense coupling deferred to v9.2.0)",
                );
            }
        }
        Self {
            a,
            b,
            c,
            coupling,
            domain,
            eps_round,
        }
    }

    /// Number of axes.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.a.len()
    }

    /// Apply one Chernoff step of size `tau` to `state` (in-place).
    ///
    /// Steps (§52.9 NORMATIVE, P3'' spectral-apply scheme):
    /// 1. Diagonal shift sweep for **uncoupled axes only**.
    ///    For `CouplingTopology::None`: full diagonal sweep → Gate C: bit-identical
    ///    to `TtChernoff`.  For tridiagonal: pair factors carry the diffusion.
    /// 2. Strang-composed pair sweep — **spectral apply (R3, solver-free)**:
    ///    `exp(τ·L_pair)·u = ifft2(expsym ⊙ fft2(u))`.  No LU, no `dense_expm`.
    ///    `expsym` is passed in (precomputed once per `evolve`).
    /// 3. Reaction scalar `(1+τc)`.
    /// 4. TT-rounding.
    ///
    /// Prefer `evolve` over calling `step` directly — `evolve` hoists the expsym
    /// build outside the loop (§11.5 mandatory hoist).
    ///
    /// # Panics
    /// Panics if any coupled-pair block is not SPD.
    pub fn step(&self, tau: F, state: &mut TtState<F>) {
        let pairs = self.coupling.pairs(self.ndim());
        // Precompute expsym for this τ (single-step path; hoisted in evolve).
        let (expsym_fwd, expsym_rev) = if pairs.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            precompute_pair_expsyms(tau, &self.a, &self.domain, &pairs, state)
                .expect("CoupledTtChernoff::step — pair-diffusion block not SPD")
        };
        self.step_with_expsym(tau, state, &pairs, &expsym_fwd, &expsym_rev);
    }

    /// Inner step with precomputed expsym slices (hoist-friendly path).
    fn step_with_expsym(
        &self,
        tau: F,
        state: &mut TtState<F>,
        pairs: &[(usize, usize, F)],
        expsym_fwd: &[Vec<F>],
        expsym_rev: &[Vec<F>],
    ) {
        debug_assert_eq!(state.ndim(), self.ndim());
        // Step 1: diagonal shift for uncoupled axes.
        if pairs.is_empty() {
            diagonal_sweep(tau, &self.a, &self.b, &self.domain, state);
        } else {
            let d = self.ndim();
            let mut is_coupled = vec![false; d];
            for &(j, k, _) in pairs {
                is_coupled[j] = true;
                is_coupled[k] = true;
            }
            diagonal_sweep_with_mask(tau, &self.a, &self.b, &self.domain, state, &is_coupled);
        }
        // Step 2: spectral pair sweep (solver-free, §11.3 R3, P3'').
        if !pairs.is_empty() {
            pair_sweep_strang(pairs, expsym_fwd, expsym_rev, state, self.eps_round)
                .expect("CoupledTtChernoff::step — pair SPD violation");
        }
        // Step 3: reaction.
        let reaction = F::one() + tau * self.c;
        for v in &mut state.cores[0].data {
            *v *= reaction;
        }
        // Step 4: TT-rounding.
        tt_round(&mut state.cores, self.eps_round);
    }

    /// Evolve `state` for time `t_final` using `n_steps` Chernoff steps.
    ///
    /// **Hoists** the spectral `expsym` diagonal build outside the step loop (§11.5).
    /// τ is constant within one `evolve` call, so `expsym` is built ONCE and reused.
    ///
    /// # Panics
    /// Panics if `n_steps < 1` or any pair block is not SPD.
    pub fn evolve(&self, t_final: F, n_steps: usize, state: &mut TtState<F>) {
        assert!(n_steps >= 1);
        let tau = t_final / F::from(n_steps).unwrap();
        let pairs = self.coupling.pairs(self.ndim());
        // Build expsym ONCE for this τ (hoist: τ is constant within evolve).
        let (expsym_fwd, expsym_rev) = if pairs.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            precompute_pair_expsyms(tau, &self.a, &self.domain, &pairs, state)
                .expect("CoupledTtChernoff::evolve — pair-diffusion block not SPD")
        };
        // Step loop — expsym reused every step (solver-free, §11.5 mandatory hoist).
        for _ in 0..n_steps {
            self.step_with_expsym(tau, state, &pairs, &expsym_fwd, &expsym_rev);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Diagonal sweep (Step 1: separable part, mirrors TtChernoff::step)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply the per-axis 3-branch Chernoff shift to all modes (same as `TtChernoff`).
fn diagonal_sweep<F: SemiflowFloat>(
    tau: F,
    a: &[F],
    b: &[F],
    domain: &[(F, F)],
    state: &mut TtState<F>,
) {
    let two = F::from(2.0).unwrap();
    for j in 0..state.ndim() {
        let n_j = state.cores[j].n;
        let (xmin, xmax) = domain[j];
        let dx_j = if n_j <= 1 {
            F::one()
        } else {
            (xmax - xmin) / F::from(n_j - 1).unwrap()
        };
        let h_j = two * (a[j] * tau).sqrt();
        let drift_shift = b[j] * tau;
        apply_per_axis_shift_pub(&mut state.cores[j], h_j + drift_shift, dx_j, n_j);
    }
}

/// Apply per-axis shift to UNCOUPLED axes only.
///
/// For coupled axes (`is_coupled[j] = true`): the pair factor already carries the
/// full `a[j]` diffusion, so we skip the diagonal diffusion shift. We still apply
/// drift `b[j]*tau` for coupled axes (drift is not covered by the pair factor).
///
/// For uncoupled axes: apply the full 3-branch shift (diffusion + drift) as normal.
fn diagonal_sweep_with_mask<F: SemiflowFloat>(
    tau: F,
    a: &[F],
    b: &[F],
    domain: &[(F, F)],
    state: &mut TtState<F>,
    is_coupled: &[bool],
) {
    let two = F::from(2.0).unwrap();
    for j in 0..state.ndim() {
        let n_j = state.cores[j].n;
        let (xmin, xmax) = domain[j];
        let dx_j = if n_j <= 1 {
            F::one()
        } else {
            (xmax - xmin) / F::from(n_j - 1).unwrap()
        };
        if is_coupled[j] {
            // Coupled axis: pair factor covers diffusion. Apply only drift (b[j]*tau).
            let drift_shift = b[j] * tau;
            if drift_shift != F::zero() {
                // Drift-only shift: h_j = 0 (no diffusion), shift_total = drift_shift
                apply_per_axis_shift_pub(&mut state.cores[j], drift_shift, dx_j, n_j);
            }
        } else {
            // Uncoupled axis: full 3-branch shift (diffusion + drift)
            let h_j = two * (a[j] * tau).sqrt();
            let drift_shift = b[j] * tau;
            apply_per_axis_shift_pub(&mut state.cores[j], h_j + drift_shift, dx_j, n_j);
        }
    }
}

// §D: coupling sweep replaced by tt_coupled_pair::pair_sweep_strang (P3' rotated factor).
// ADR-0162 Amendment / math.md §52.9 NORMATIVE.

// ═══════════════════════════════════════════════════════════════════════════
// §E — Helpers
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// §F — Inline unit tests (batch H6: moved to tt_coupled_tests.rs)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    include!("tt_coupled_tests.rs");
}
