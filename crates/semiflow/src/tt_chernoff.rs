//! TT-Chernoff evolver — tensor-train Chernoff semigroup (math §50, §52.2 Amd 1).
//!
//! Escapes the exponential curse for linear diagonal-A (Gaussian) diffusion:
//! state `u_T = e^{TL}u_0` stored as TT at cost `O(d·n·r²)`, poly in `d`.
//!
//! # Method (P2, v9.1.0 — cubic-Lagrange band-split shift)
//!
//! Per-axis shift uses `Sₕ = Σ_m c_m(t)·P_{s₀+m}` (w=4, §52.2 Amd 1):
//! each band is a permutation (QTT-rank≤2), 4-band sum rounds to rank 3 constant
//! in L and d (`T_TT_BAND_SHIFT_RANK` / `scripts/tt_band_shift_kit.py`).
//! Convergence O(τ²) to the true semigroup (TRIZ §5.1, §6.2).
//!
//! # Scope (MANDATORY — honesty constraint)
//!
//! - **Implemented**: constant diagonal-A, scalar drift/reaction (Gaussian class).
//!   Rank bounded d/2 (Rohrbach 2022).
//! - **Research track**: off-diagonal A, variable/nonlinear coefs — rank uncapped.
//!
//! # Curse-escape (Gaussian class)
//!
//! Rank-1 IC + local coupling: r=1 (Strang⊗). Dense Cauchy: r≤d/2, O(d³n).
//! Exponential curse n^d escaped. Only r~c^d re-curses (not algebraically bounded).
//!
//! References: Oseledets 2011; Kazeev–Khoromskij 2012; Rohrbach 2022; math §50/§52.

// Grid/TT indices (usize) cast to f64/isize for coordinates; TT sizes ≪ 2^52 and ≪ isize::MAX.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]

extern crate alloc;
use alloc::vec::Vec;

use crate::{
    float::SemiflowFloat,
    tt_core::{tt_round, TtCore},
};

// ═══════════════════════════════════════════════════════════════════════════
// §A — TtState<F>
// ═══════════════════════════════════════════════════════════════════════════

/// Tensor-train state `u(i₁,…,i_d) = G₁[i₁] · G₂[i₂] · … · G_d[i_d]`.
///
/// Each core `G_k` has shape `r_{k-1} × n_k × r_k` (boundary: `r_0 = r_d = 1`).
/// Storage: `O(d · n · r²)` where `r = max_k r_k` is the peak bond rank.
///
/// ## Scope note
/// This type targets the linear diagonal-A Gaussian diffusion class. For that
/// class, `r` is bounded poly-in-d and the exponential curse is escaped.
#[derive(Clone, Debug)]
pub struct TtState<F: SemiflowFloat> {
    /// One core per mode dimension.
    pub cores: Vec<TtCore<F>>,
}

impl<F: SemiflowFloat> TtState<F> {
    /// Number of modes (dimensions).
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.cores.len()
    }

    /// Mode size for axis `j` (number of grid nodes).
    #[must_use]
    pub fn n_j(&self, j: usize) -> usize {
        self.cores[j].n
    }

    /// Peak bond rank (max over all internal bonds).
    #[must_use]
    pub fn peak_rank(&self) -> usize {
        self.cores.iter().map(|c| c.r_right).max().unwrap_or(1)
    }

    /// Total number of stored scalars (working-set size).
    #[must_use]
    pub fn storage_size(&self) -> usize {
        self.cores.iter().map(|c| c.data.len()).sum()
    }

    /// Construct from a rank-1 separable initial condition.
    ///
    /// `u(i₁,…,i_d) = f₁(i₁) · f₂(i₂) · … · f_d(i_d)`, stored exactly as
    /// rank-1 TT (each core is `1 × n_j × 1`, data = the 1D slice `f_j`).
    ///
    /// This is the common initial condition for diagonal-A diffusion and the
    /// lower edge of the TT construction (reduces to Strang⊗ at rank-1).
    #[must_use]
    pub fn rank1_separable(slices: Vec<Vec<F>>) -> Self {
        let cores = slices
            .into_iter()
            .map(|f| {
                let n = f.len();
                let mut core = TtCore::zeros(1, n, 1);
                for (im, &v) in f.iter().enumerate() {
                    core.set(0, im, 0, v);
                }
                core
            })
            .collect();
        Self { cores }
    }

    /// Contract the TT against a separable functional `⟨f, u⟩` where
    /// `f(i₁,…,i_d) = f₁(i₁) · f₂(i₂) · … · f_d(i_d)`.
    ///
    /// Cost: `O(d · n · r²)` — polynomial in d.
    ///
    /// ## Algorithm
    /// Left-to-right contraction:  `η_k = η_{k-1} · (G_k contracted with f_k)`.
    /// Each step: `η_k[j] = Σ_{i, α} η_{k-1}[α] · G_k[α, i, j] · f_k[i]`
    /// → shape `r_{k-1}` → `r_k`.  Final: scalar.
    ///
    /// # Panics
    /// Panics if `functionals.len() != self.ndim()` or any functional length mismatches.
    #[must_use]
    pub fn inner_separable(&self, functionals: &[Vec<F>]) -> F {
        assert_eq!(functionals.len(), self.ndim());
        // η starts as 1×1 identity (left boundary r=1)
        let mut eta: Vec<F> = vec![F::one()];
        for (k, (core, fk)) in self.cores.iter().zip(functionals.iter()).enumerate() {
            assert_eq!(fk.len(), core.n, "functional length mismatch at axis {k}");
            eta = contract_step(core, fk, &eta);
        }
        assert_eq!(eta.len(), 1);
        eta[0]
    }
}

/// One contraction step: `eta_new[j] = Σ_{i,α} eta_old[α] · core[α,i,j] · f[i]`.
fn contract_step<F: SemiflowFloat>(core: &TtCore<F>, f: &[F], eta: &[F]) -> Vec<F> {
    let r_left = core.r_left;
    let r_right = core.r_right;
    debug_assert_eq!(eta.len(), r_left);
    let mut eta_new = vec![F::zero(); r_right];
    for (j, slot) in eta_new.iter_mut().enumerate() {
        let mut s = F::zero();
        for (alpha, &ea) in eta.iter().enumerate() {
            for (i, &fi) in f.iter().enumerate() {
                s += ea * core.get(alpha, i, j) * fi;
            }
        }
        *slot = s;
    }
    eta_new
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — TtChernoff evolver
// ═══════════════════════════════════════════════════════════════════════════

/// TT-Chernoff evolver for linear diagonal-A diffusion (Gaussian class).
///
/// Evolves [`TtState<F>`] by the Chernoff product formula (eq. 50.3 / math §50):
/// for each axis `j`, applies the 3-branch shift:
///   `G_j ← ¼·shift(+h_j)·G_j + ¼·shift(-h_j)·G_j + ½·G_j`
/// where `h_j = 2√(a_j·τ)` (the Chernoff step size on axis `j`).
/// Drift `b_j` shifts the grid points: effective `x → x + b_j·τ`.
/// Reaction `c` scales the whole state: `u → (1 + τ·c)·u`.
///
/// After each full axis sweep: TT-rounding at tolerance `eps_round` compresses
/// bond ranks back to the intrinsic rank of the state.
///
/// ## Scope (MANDATORY)
/// Constant diagonal `a`, `b`, scalar `c`. Variable / off-diagonal A: not supported.
///
/// ## Grid convention
/// Axis `j` has `n_j` uniformly spaced nodes at `x_k = x_min_j + k·dx_j`,
/// `dx_j = (x_max_j - x_min_j) / (n_j - 1)`.
/// The shift-by-`h_j` is implemented as a **periodic index shift** on the core
/// (wrap-around boundary), consistent with the TT-operator rank-1 structure.
/// For smooth, decaying initial conditions the wrap-around error is negligible
/// for sufficiently large domain.
#[derive(Clone, Debug)]
pub struct TtChernoff<F: SemiflowFloat> {
    /// Per-axis diffusion coefficients `a_j` ≥ 0.
    pub a: Vec<F>,
    /// Per-axis drift coefficients `b_j`.
    pub b: Vec<F>,
    /// Scalar reaction `c`.
    pub c: F,
    /// Per-axis domain bounds: `(x_min_j, x_max_j)`.
    pub domain: Vec<(F, F)>,
    /// TT-rounding tolerance.
    pub eps_round: F,
}

impl<F: SemiflowFloat> TtChernoff<F> {
    /// Construct with per-axis `a`, `b`, scalar `c`, domain bounds, rounding eps.
    ///
    /// # Panics
    /// Panics if `b.len()` or `domain.len()` differs from `a.len()`.
    #[must_use]
    pub fn new(a: Vec<F>, b: Vec<F>, c: F, domain: Vec<(F, F)>, eps_round: F) -> Self {
        let d = a.len();
        assert_eq!(b.len(), d);
        assert_eq!(domain.len(), d);
        Self {
            a,
            b,
            c,
            domain,
            eps_round,
        }
    }

    /// Number of axes.
    pub fn ndim(&self) -> usize {
        self.a.len()
    }

    /// Apply one Chernoff step of size `tau` to `state` (in-place).
    ///
    /// 1. For each axis j: apply the 3-branch shift+drift to core j.
    /// 2. Apply global reaction scalar `(1 + tau*c)` to the first core.
    /// 3. TT-round all bonds at `eps_round`.
    pub fn step(&self, tau: F, state: &mut TtState<F>) {
        debug_assert_eq!(state.ndim(), self.ndim());
        for j in 0..self.ndim() {
            let n_j = state.cores[j].n;
            let dx_j = self.dx(j, n_j);
            let h_j = two::<F>() * (self.a[j] * tau).sqrt(); // shift distance
            let drift_shift = self.b[j] * tau; // total drift displacement
            apply_per_axis_shift(&mut state.cores[j], h_j + drift_shift, dx_j, n_j);
        }
        // Global reaction: scale first core
        let reaction = F::one() + tau * self.c;
        for v in &mut state.cores[0].data {
            *v *= reaction;
        }
        // TT-rounding
        tt_round(&mut state.cores, self.eps_round);
    }

    /// Evolve `state` for time `T` using `n_steps` Chernoff steps.
    ///
    /// # Panics
    /// Panics if `n_steps < 1`.
    pub fn evolve(&self, t_final: F, n_steps: usize, state: &mut TtState<F>) {
        assert!(n_steps >= 1);
        let tau = t_final / F::from(n_steps).unwrap();
        for _ in 0..n_steps {
            self.step(tau, state);
        }
    }

    /// Grid spacing for axis `j` with `n` nodes.
    fn dx(&self, j: usize, n: usize) -> F {
        let (xmin, xmax) = self.domain[j];
        if n <= 1 {
            return F::one();
        }
        (xmax - xmin) / F::from(n - 1).unwrap()
    }
}

/// Cubic-Lagrange weights for nodes `{−1,0,1,2}` at abscissa `t = h/dx − ⌊h/dx⌋`.
/// Proved sum-to-1 + degree-≤3 reproduction by `T_TT_BAND_SHIFT_RANK` (§52.2 Amd 1).
#[inline]
fn cubic_band_weights<F: SemiflowFloat>(t: F) -> [F; 4] {
    let t1 = t - F::one();
    let t2 = t - F::from(2.0).unwrap();
    let tp1 = t + F::one();
    let sixth = F::from(1.0 / 6.0).unwrap();
    let half = F::from(0.5).unwrap();
    [
        -(t * t1 * t2) * sixth, // c_{-1}
        (tp1 * t1 * t2) * half, // c_0
        -(tp1 * t * t2) * half, // c_1
        (tp1 * t * t1) * sixth, // c_2
    ]
}

/// 3-branch Chernoff shift via cubic-Lagrange band-split (w=4, §52.2 Amd 1).
///
/// `G_new = ¼·S_{+h}·G + ¼·S_{−h}·G + ½·G`,  where
/// `Sₕ = Σ_{m∈{−1,0,1,2}} c_m(t)·P_{s₀+m}`, `s₀=⌊h/dx⌋`, `t=h/dx−s₀`.
/// Fixed summation order m=−1,0,1,2 for 0-ULP determinism (ADR-0018).
/// No SIMD path — arithmetic-only loop.
#[allow(clippy::similar_names)]
fn apply_per_axis_shift<F: SemiflowFloat>(core: &mut TtCore<F>, h: F, dx: F, n: usize) {
    if dx <= F::zero() || h == F::zero() {
        return;
    }
    let ratio = h / dx;
    let s0 = ratio.floor().to_isize().unwrap_or(0);
    let t_frac = ratio - F::from(s0 as f64).unwrap();
    let wf = cubic_band_weights(t_frac);
    // Negative shift: −h
    let ratio_neg = -ratio;
    let s0_neg = ratio_neg.floor().to_isize().unwrap_or(0);
    let t_neg = ratio_neg - F::from(s0_neg as f64).unwrap();
    let wb = cubic_band_weights(t_neg);
    let w_half = F::from(0.5).unwrap();
    let w_qtr = F::from(0.25).unwrap();
    let r_left = core.r_left;
    let r_right = core.r_right;
    let mut new_data = vec![F::zero(); r_left * n * r_right];
    // Band offsets for +h: m ∈ {−1, 0, 1, 2}
    let fwd_offsets: [isize; 4] = [s0 - 1, s0, s0 + 1, s0 + 2];
    // Band offsets for −h: m ∈ {−1, 0, 1, 2}
    let bwd_offsets: [isize; 4] = [s0_neg - 1, s0_neg, s0_neg + 1, s0_neg + 2];
    let n_i = n as isize;
    for il in 0..r_left {
        for i in 0..n {
            let ii = i as isize;
            for ir in 0..r_right {
                // ½ identity
                let mut v = w_half * core.get(il, i, ir);
                // ¼ S_{+h} — fixed order m = −1,0,1,2
                for (m, &off) in fwd_offsets.iter().enumerate() {
                    let idx = ((ii + off).rem_euclid(n_i)) as usize;
                    v += w_qtr * wf[m] * core.get(il, idx, ir);
                }
                // ¼ S_{−h} — fixed order m = −1,0,1,2
                for (m, &off) in bwd_offsets.iter().enumerate() {
                    let idx = ((ii + off).rem_euclid(n_i)) as usize;
                    v += w_qtr * wb[m] * core.get(il, idx, ir);
                }
                new_data[il * n * r_right + i * r_right + ir] = v;
            }
        }
    }
    core.data = new_data;
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — Helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn two<F: SemiflowFloat>() -> F {
    F::from(2.0).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — pub(crate) re-exports for tt_coupled.rs (additive sibling)
// ═══════════════════════════════════════════════════════════════════════════

/// Re-export for `tt_coupled.rs` diagonal sweep (avoids duplicating the algorithm).
/// Updated to cubic-band signature (P2, math §52.2 Amd 1).
#[inline]
pub(crate) fn apply_per_axis_shift_pub<F: SemiflowFloat>(
    core: &mut TtCore<F>,
    h: F,
    dx: F,
    n: usize,
) {
    apply_per_axis_shift(core, h, dx, n);
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Inline tests — extracted to sibling file per ≤500-line cap.
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[path = "tt_chernoff_tests.rs"]
mod tests;
