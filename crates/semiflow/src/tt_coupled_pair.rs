//! Solver-free spectral pair factor for `CoupledTtChernoff` (v9.1.0 P3'').
//!
//! Replaces the dense LU-Padé `dense_expm` coupling path with a **solver-free**
//! **spectral apply**: `exp(τ·L_pair)·u = ifft2(expsym ⊙ fft2(u))` where
//! `L_pair = c_j·D2_j/dx_j² + c_k·D2_k/dx_k² + 2ρ√(a_j·a_k)·(D1_j/dx_j)⊗(D1_k/dx_k)`
//! and `expsym[m_j,m_k] = exp(τ_eff·symbol(m_j,m_k))` is the precomputed real diagonal.
//!
//! Honouring Theorem-6 R2 ("no linear solver"):
//! - NO `lu_solve_inplace`, NO `dense_expm` on the production path.
//! - Only FFT (fixed unitary) + elementwise exp + Markov band shifts + in-tree SVD.
//!
//! Machine-exact: in-Rust rel-err floor (d=2 self-check): **5.27e-14**; d∈{3,4} gate: ≤1e-12.
//! The numpy single-panel probe (1.2e-15) is a lower d=2 bound, not the Rust floor.
//! Same poly-d TT-op-rank (6).
//!
//! ## Hoist (mandatory §11.5)
//! `expsym` is τ-only. Build it ONCE per `(pair, τ)` before the step loop via
//! `build_pair_expsym`, pass the resulting slice to every call of `apply_pair_factor`.
//!
//! ## SPD scope (MANDATORY per §10.12 risk 1)
//! `det B = c_j·c_k − r² > 0` required. Returns `SemiflowError` if violated.
//!
//! ## No new deps
//! DFT is an in-tree O(n²) direct DFT (`tt_spectral.rs`); trig via `num_traits::Float`
//! already on `SemiflowFloat`. No LAPACK, no FFT crate.
//!
//! References: math.md §52.9 (NORMATIVE R3); ADR-0162; §11.3–11.5;
//! `probe_adjudicate_rotated_shift.py` (R3); `probe_adjudicate_spectral_cost.py`.

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    tt_chernoff::TtState,
    tt_core::TtCore,
    tt_dense_expm::one_sided_jacobi_svd,
    tt_spectral::{apply_spectral_pair_to_panel, pair_expsym_real},
};

// ═══════════════════════════════════════════════════════════════════════════
// §A — SPD check: eigenvalues of the 2×2 pair-diffusion block
// ═══════════════════════════════════════════════════════════════════════════

/// Check SPD and return `(λ₊, λ₋)` of `B = [[cj, r],[r, ck]]`.
///
/// Canonical ordering: `λ₊ ≥ λ₋`. Returns `SemiflowError` if `λ_min ≤ 0`.
///
/// # SPD condition
/// `det B = cj·ck − r² > 0` (both eigenvalues positive). For shared-axis
/// tridiagonal coupling `c_j = a_j/#pairs(j)`: at d≥4 interior axes
/// (2 pairs per axis) this requires `|ρ| < 0.5`.
pub(crate) fn pair_eigen_check<F: SemiflowFloat>(
    cj: F,
    ck: F,
    r: F,
) -> Result<(F, F), SemiflowError> {
    let half = F::from(0.5).unwrap();
    let tr_half = half * (cj + ck);
    let diff_half = half * (cj - ck);
    let disc = (diff_half * diff_half + r * r).sqrt();
    let lam_plus = tr_half + disc;
    let lam_minus = tr_half - disc;
    if lam_minus <= F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "CoupledTtChernoff pair block not SPD: \
                   det(B)=cj*ck-r^2<=0. Reduce |rho| or adjust topology \
                   (tridiagonal d>=4: requires |rho|<0.5; see ss10.12 risk 1)",
            value: 0.0,
        });
    }
    Ok((lam_plus, lam_minus))
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — Dense pair generator (TEST-ONLY reference; not used on production path)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the flat n²×n² matrix for `L_pair = cj·D2_j/dx_j² + ck·D2_k/dx_k² +
/// 2r·(D1_j/dx_j)⊗(D1_k/dx_k)` with periodic boundary conditions.
///
/// Used ONLY by the `d2_exactness_self_check` test (independent dense reference).
/// The production apply path is spectral (`tt_spectral::apply_spectral_pair_to_panel`).
#[cfg(test)]
// inv_dxj2/inv_dxk2 and ij_f/ij_b/ik_f/ik_b are periodic FD stencil names; intentionally similar.
#[allow(clippy::similar_names)]
fn build_l_pair<F: SemiflowFloat>(n: usize, dx_j: F, dx_k: F, cj: F, ck: F, r: F) -> Vec<F> {
    let n2 = n * n;
    let mut l = vec![F::zero(); n2 * n2];
    let two = F::from(2.0).unwrap();
    let inv_dxj2 = F::one() / (dx_j * dx_j);
    let inv_dxk2 = F::one() / (dx_k * dx_k);
    let inv_2dxj = F::one() / (two * dx_j);
    let inv_2dxk = F::one() / (two * dx_k);
    let neg_two = -two;
    let idx = |ij: usize, ik: usize| ij * n + ik;
    for ij in 0..n {
        let ij_f = (ij + 1) % n;
        let ij_b = (ij + n - 1) % n;
        for ik in 0..n {
            let ik_f = (ik + 1) % n;
            let ik_b = (ik + n - 1) % n;
            let row = idx(ij, ik);
            l[row * n2 + idx(ij_f, ik)] = l[row * n2 + idx(ij_f, ik)] + cj * inv_dxj2;
            l[row * n2 + idx(ij_b, ik)] = l[row * n2 + idx(ij_b, ik)] + cj * inv_dxj2;
            l[row * n2 + idx(ij, ik)] = l[row * n2 + idx(ij, ik)] + cj * inv_dxj2 * neg_two;
            l[row * n2 + idx(ij, ik_f)] = l[row * n2 + idx(ij, ik_f)] + ck * inv_dxk2;
            l[row * n2 + idx(ij, ik_b)] = l[row * n2 + idx(ij, ik_b)] + ck * inv_dxk2;
            l[row * n2 + idx(ij, ik)] = l[row * n2 + idx(ij, ik)] + ck * inv_dxk2 * neg_two;
            let coeff = two * r * inv_2dxj * inv_2dxk;
            l[row * n2 + idx(ij_f, ik_f)] = l[row * n2 + idx(ij_f, ik_f)] + coeff;
            l[row * n2 + idx(ij_f, ik_b)] = l[row * n2 + idx(ij_f, ik_b)] - coeff;
            l[row * n2 + idx(ij_b, ik_f)] = l[row * n2 + idx(ij_b, ik_f)] - coeff;
            l[row * n2 + idx(ij_b, ik_b)] = l[row * n2 + idx(ij_b, ik_b)] + coeff;
        }
    }
    l
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Precompute expsym (hoist: build once per (pair, τ))
// ═══════════════════════════════════════════════════════════════════════════

/// Precompute the real spectral diagonal `expsym` for one pair at the given `τ_eff`.
///
/// Call ONCE per pair before the step loop.  Pass the result to every call of
/// `apply_pair_factor` for that pair (τ is constant within `evolve`).
///
/// # Errors
/// Returns `SemiflowError` if the pair block is not SPD.
// j,k,cj,ck,r_cross,tau_eff,dx_j,dx_k,state: all required for spectral pair symbol.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pair_expsym<F: SemiflowFloat>(
    state: &TtState<F>,
    j: usize,
    k: usize,
    cj: F,
    ck: F,
    r_cross: F,
    tau_eff: F,
    dx_j: F,
    dx_k: F,
) -> Result<Vec<F>, SemiflowError> {
    pair_eigen_check(cj, ck, r_cross)?;
    let n_j = state.cores[j].n;
    let n_k = state.cores[k].n;
    Ok(pair_expsym_real(
        n_j, n_k, dx_j, dx_k, cj, ck, r_cross, tau_eff,
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — Apply pair factor to the TT state (spectral, no LU)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply the spectral pair factor `exp(τ_eff · L_pair)` to modes (j, k) of `state`.
///
/// Algorithm (R3, solver-free, §11.3):
/// 1. For each `(n_j × n_k)` slab slice: forward 2D-DFT, elementwise multiply
///    by precomputed `expsym`, inverse 2D-DFT, take real part.
/// 2. Split slab back into two TT cores via one-sided Jacobi truncated SVD.
///
/// NO `dense_expm`, NO `lu_solve_inplace`.  Only DFT + elementwise multiply.
///
/// # `eps_round`
/// Passed through to `one_sided_jacobi_svd` as the SVD truncation threshold
/// (`r = count(σ ≥ eps_round · σ_max)`).  The default from callers is `1e-12`,
/// matching the exactness gate floor for d∈{3,4}.
///
/// # Scope
/// Adjacent pairs only (`k == j+1`). Non-adjacent pairs (SPD-checked only).
///
/// # Errors
/// Returns `SemiflowError` if `expsym` length does not match the slab dimensions.
pub(crate) fn apply_pair_factor<F: SemiflowFloat>(
    state: &mut TtState<F>,
    j: usize,
    k: usize,
    expsym: &[F],
    eps_round: F,
) -> Result<(), SemiflowError> {
    let n_j = state.cores[j].n;
    let n_k = state.cores[k].n;
    if expsym.len() != n_j * n_k {
        return Err(SemiflowError::DomainViolation {
            what: "apply_pair_factor: expsym length != n_j * n_k",
            value: expsym.len() as f64,
        });
    }
    apply_spectral_to_tt_pair(state, j, k, expsym, n_j, n_k, eps_round);
    Ok(())
}

/// Contract TT cores j and k into a slab, apply spectral pair factor, split back.
///
/// The apply is solver-free: per (`n_j×n_k`) slice, apply `apply_spectral_pair_to_panel`
/// (2D DFT + elementwise expsym + inverse DFT + take real).  No `dense_expm`, no LU.
/// `eps_round` is the SVD truncation threshold for the TT-split step.
// All args required for TT spectral apply: state, pairs, domain, expsyms, indices.
#[allow(clippy::too_many_arguments)]
fn apply_spectral_to_tt_pair<F: SemiflowFloat>(
    state: &mut TtState<F>,
    j: usize,
    k: usize,
    expsym: &[F],
    n_j: usize,
    n_k: usize,
    eps_round: F,
) {
    debug_assert_eq!(k, j + 1, "apply_spectral_to_tt_pair: only adjacent pairs");
    let cj = &state.cores[j];
    let ck = &state.cores[k];
    let r_l = cj.r_left;
    let r_m = cj.r_right;
    let r_r = ck.r_right;
    let n2 = n_j * n_k;

    // Phase 1: contract cores j and k into a slab [r_l, n_j, n_k, r_r] row-major.
    let slab = contract_slab(cj, ck, r_l, r_m, r_r, n_j, n_k, n2);

    // Phase 2: apply spectral factor to each (n_j × n_k) panel slice.
    let new_slab = apply_expsym_to_slab(&slab, expsym, r_l, r_r, n_j, n_k, n2);

    // Phase 3: reshape new_slab → matrix [r_l*n_j, n_k*r_r] and split via SVD.
    split_slab_into_cores(state, j, k, &new_slab, r_l, r_r, n_j, n_k, n2, eps_round);
}

/// Contract two adjacent TT cores into a slab `[r_l, n_j, n_k, r_r]` row-major.
// Slab contraction: core, mat, factor geometry all required simultaneously.
#[allow(clippy::too_many_arguments)]
fn contract_slab<F: SemiflowFloat>(
    cj: &TtCore<F>,
    ck: &TtCore<F>,
    r_l: usize,
    r_m: usize,
    r_r: usize,
    n_j: usize,
    n_k: usize,
    n2: usize,
) -> Vec<F> {
    let mut slab = vec![F::zero(); r_l * n2 * r_r];
    for il in 0..r_l {
        for ij in 0..n_j {
            for im in 0..r_m {
                let cj_val = cj.get(il, ij, im);
                if cj_val == F::zero() {
                    continue;
                }
                for ik in 0..n_k {
                    for ir in 0..r_r {
                        let idx = il * n2 * r_r + ij * n_k * r_r + ik * r_r + ir;
                        slab[idx] += cj_val * ck.get(im, ik, ir);
                    }
                }
            }
        }
    }
    slab
}

/// Apply `apply_spectral_pair_to_panel` to every (`n_j` × `n_k`) panel slice in the slab.
// expsym apply: slab geometry + wavenumbers all required simultaneously.
#[allow(clippy::too_many_arguments)]
fn apply_expsym_to_slab<F: SemiflowFloat>(
    slab: &[F],
    expsym: &[F],
    r_l: usize,
    r_r: usize,
    n_j: usize,
    n_k: usize,
    n2: usize,
) -> Vec<F> {
    let mut new_slab = vec![F::zero(); r_l * n2 * r_r];
    let mut panel = vec![F::zero(); n2];
    for il in 0..r_l {
        for ir in 0..r_r {
            for ij in 0..n_j {
                for ik in 0..n_k {
                    panel[ij * n_k + ik] = slab[il * n2 * r_r + ij * n_k * r_r + ik * r_r + ir];
                }
            }
            apply_spectral_pair_to_panel(&mut panel, n_j, n_k, expsym);
            for ij in 0..n_j {
                for ik in 0..n_k {
                    new_slab[il * n2 * r_r + ij * n_k * r_r + ik * r_r + ir] = panel[ij * n_k + ik];
                }
            }
        }
    }
    new_slab
}

/// Reshape `new_slab` `[r_l, n_j, n_k, r_r]` → `[r_l*n_j, n_k*r_r]`, run truncated SVD,
/// write the resulting pair of TT cores back into `state.cores[j]` and `state.cores[k]`.
// Split requires both the slab data and all axis dimensions to form the TT cores.
#[allow(clippy::too_many_arguments)]
fn split_slab_into_cores<F: SemiflowFloat>(
    state: &mut TtState<F>,
    j: usize,
    k: usize,
    new_slab: &[F],
    r_l: usize,
    r_r: usize,
    n_j: usize,
    n_k: usize,
    n2: usize,
    eps_round: F,
) {
    let rows = r_l * n_j;
    let cols = n_k * r_r;
    let mut mat_form = vec![F::zero(); rows * cols];
    for il in 0..r_l {
        for ij in 0..n_j {
            for ik in 0..n_k {
                for ir in 0..r_r {
                    mat_form[(il * n_j + ij) * cols + ik * r_r + ir] =
                        new_slab[il * n2 * r_r + ij * n_k * r_r + ik * r_r + ir];
                }
            }
        }
    }
    // Truncated SVD via one-sided Jacobi (high relative accuracy for all singular values).
    let (u, sv, v) = one_sided_jacobi_svd(&mat_form, rows, cols, eps_round);
    let r_new = sv.len().max(1);

    let mut new_core_j = TtCore::zeros(r_l, n_j, r_new);
    for il in 0..r_l {
        for ij in 0..n_j {
            let row = il * n_j + ij;
            for ir_new in 0..r_new {
                new_core_j.set(il, ij, ir_new, u[row * r_new + ir_new] * sv[ir_new]);
            }
        }
    }
    let mut new_core_k = TtCore::zeros(r_new, n_k, r_r);
    for ir_new in 0..r_new {
        for ik in 0..n_k {
            for ir in 0..r_r {
                new_core_k.set(ir_new, ik, ir, v[(ik * r_r + ir) * r_new + ir_new]);
            }
        }
    }
    state.cores[j] = new_core_j;
    state.cores[k] = new_core_k;
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — Strang-composed pair sweep
// ═══════════════════════════════════════════════════════════════════════════

/// Pair of spectral symbol tables: `(fwd[p], rev[p])` per paired axis.
/// `fwd[p]` = forward half-step (τ/2 for Strang multi-pair, τ for single pair).
/// `rev[p]` = reverse half-step (empty if single pair).
/// Built once per `evolve` by `precompute_pair_expsyms`.
pub(crate) type PairExpsyms<F> = (
    alloc::vec::Vec<alloc::vec::Vec<F>>,
    alloc::vec::Vec<alloc::vec::Vec<F>>,
);

/// Count how many pairs share each axis (for shared-diffusion accounting).
fn count_pairs_per_axis<F>(d: usize, pairs: &[(usize, usize, F)]) -> alloc::vec::Vec<usize> {
    let mut n_pairs_per_axis = vec![0usize; d];
    for &(j, k, _) in pairs {
        n_pairs_per_axis[j] += 1;
        n_pairs_per_axis[k] += 1;
    }
    n_pairs_per_axis
}

pub(crate) fn precompute_pair_expsyms<F: SemiflowFloat>(
    tau: F,
    a: &[F],
    domain: &[(F, F)],
    pairs: &[(usize, usize, F)],
    state: &TtState<F>,
) -> Result<PairExpsyms<F>, SemiflowError> {
    if pairs.is_empty() {
        return Ok((alloc::vec::Vec::new(), alloc::vec::Vec::new()));
    }
    let half = F::from(0.5).unwrap();
    let use_strang = pairs.len() > 1;
    let tau_fwd = if use_strang { tau * half } else { tau };
    let tau_half = tau * half;
    let n_pairs_per_axis = count_pairs_per_axis(a.len(), pairs);
    let mut fwd = alloc::vec::Vec::with_capacity(pairs.len());
    let mut rev = alloc::vec::Vec::new();
    for &(j, k, rho) in pairs {
        let n_j = state.cores[j].n;
        let n_k = state.cores[k].n;
        let dx_j = axis_dx(domain, j, n_j);
        let dx_k = axis_dx(domain, k, n_k);
        let cj = a[j] / F::from(n_pairs_per_axis[j].max(1)).unwrap();
        let ck = a[k] / F::from(n_pairs_per_axis[k].max(1)).unwrap();
        let r_cross = rho * (a[j] * a[k]).sqrt();
        if k == j + 1 {
            fwd.push(build_pair_expsym(
                state, j, k, cj, ck, r_cross, tau_fwd, dx_j, dx_k,
            )?);
        } else {
            // Non-adjacent pairs are rejected at constructor time (CoupledTtChernoff::new
            // guard 2: k>j+1 triggers assert!).  A path reaching here means the pairs
            // list was built bypassing the constructor — reject loudly so coupling is
            // never silently dropped (defense-in-depth against H5 silent-wrong).
            return Err(SemiflowError::UnsupportedOperation {
                what: "precompute_pair_expsyms: non-adjacent pair (k>j+1) — \
                       spectral pair factor only supports adjacent (k==j+1) axes; \
                       true non-adjacent coupling is deferred to v9.2.0 (ADR-0162)",
            });
        }
        if use_strang && k == j + 1 {
            rev.push(build_pair_expsym(
                state, j, k, cj, ck, r_cross, tau_half, dx_j, dx_k,
            )?);
        }
    }
    Ok((fwd, rev))
}

/// Apply the Strang-composed pair sweep using precomputed expsym diagonals.
///
/// `expsym_fwd[p]` and `expsym_rev[p]` are from `precompute_pair_expsyms`.
///
/// Solver-free: no `dense_expm`, no `lu_solve_inplace` on the coupling path.
pub(crate) fn pair_sweep_strang<F: SemiflowFloat>(
    pairs: &[(usize, usize, F)],
    expsym_fwd: &[alloc::vec::Vec<F>],
    expsym_rev: &[alloc::vec::Vec<F>],
    state: &mut TtState<F>,
    eps_round: F,
) -> Result<(), SemiflowError> {
    if pairs.is_empty() {
        return Ok(());
    }
    let use_strang = pairs.len() > 1;
    // Forward chain
    for (p, &(j, k, _)) in pairs.iter().enumerate() {
        if k == j + 1 && !expsym_fwd[p].is_empty() {
            apply_pair_factor(state, j, k, &expsym_fwd[p], eps_round)?;
        }
    }
    // Reverse half-step for Strang symmetrisation
    if use_strang {
        for (p, &(j, k, _)) in pairs.iter().enumerate().rev() {
            if k == j + 1 && p < expsym_rev.len() && !expsym_rev[p].is_empty() {
                apply_pair_factor(state, j, k, &expsym_rev[p], eps_round)?;
            }
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// §F — Helpers + test-visible re-exports
// ═══════════════════════════════════════════════════════════════════════════

fn axis_dx<F: SemiflowFloat>(domain: &[(F, F)], j: usize, n: usize) -> F {
    let (xmin, xmax) = domain[j];
    if n <= 1 {
        return F::one();
    }
    (xmax - xmin) / F::from(n - 1).unwrap()
}

// §E.2: one_sided_jacobi_svd → see tt_dense_expm.rs (imported above).

/// Public (crate-level) alias for `build_l_pair` — used by the d=2 exactness test
/// as the INDEPENDENT dense reference.  NOT used on the production path.
#[cfg(test)]
pub(crate) fn build_l_pair_pub<F: SemiflowFloat>(
    n: usize,
    dx_j: F,
    dx_k: F,
    cj: F,
    ck: F,
    r: F,
) -> Vec<F> {
    build_l_pair(n, dx_j, dx_k, cj, ck, r)
}

/// Public (crate-level) alias for `dense_expm` (from `tt_dense_expm`) — used by
/// the d=2 exactness test as INDEPENDENT reference.  NOT used on production path.
#[cfg(test)]
pub(crate) fn dense_expm_pub<F: SemiflowFloat>(a: &[F], m: usize) -> Vec<F> {
    crate::tt_dense_expm::dense_expm(a, m)
}

// ═══════════════════════════════════════════════════════════════════════════
// §G — Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[path = "tt_coupled_pair_tests.rs"]
mod tests;
