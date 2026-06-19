//! S³ POC — Dense / non-adjacent (all-pairs) coupling curse-escape.
//!
//! Extends `tt_drift_spectral.rs` (ADR-0164) from adjacent-pair coupling to a
//! **fully dense** (all-pairs, `j<k`) symmetric diffusion matrix `D`.  The key
//! insight (TRIZ §1): density lives in the *values* of `D`; bounded TT-rank lives
//! in its *spectral structure*.  A rank-1-dense matrix `D = diag(a) + λ·g·gᵀ` has
//! every off-diagonal entry non-zero AND an off-diagonal block of rank 1 at any cut
//! → TT-rank saturates (not grows) with `d`.
//!
//! ## Mathematics (§2 of design doc)
//!
//! ```text
//! σ(k) = Σ_j D[j,j]·σ_D2(k_j)                        (diagonal diffusion, RE)
//!        − Σ_{j<k} 2·D[j,k]·σ_D1r(k_j)·σ_D1r(k_k)    (ALL-PAIRS cross, RE)
//!        + i·Σ_j b[j]·σ_D1r(k_j)                      (drift, IM; ADR-0164)
//! expsym = exp(τ·σ)  [conjugate-even → real output]
//! ```
//!
//! ## Solver-free (Theorem-6 R2)
//!
//! Operations: d-D DFT → elementwise complex multiply → d-D IDFT → take real.
//! NO `lu_solve_inplace`, NO `dense_expm`, NO triangular solve.
//!
//! ## Scope (PROOF-OF-CONCEPT — constant-coef only)
//!
//! Contract: `contracts/s3-dense-coupling-poc.contract.md`.
//! Design:   `.dev-docs/specs/s3-dense-coupling.md`.
//! ADR:      `docs/adr/0165-dense-coupling-curse-escape.md`.

// Grid/FFT indices (usize) cast to F; all values ≪ 2^52.
// POC module: pub(crate) fns used only by the slow-tests gate.
// usize→u32 casts for n.pow(k): n is a small grid count (≤ 32767) so truncation
// is impossible in practice; doc-markdown lints are suppressed for math notation.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    dead_code
)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use crate::{
    float::{from_f64, SemiflowFloat},
    tt_spectral::{dft_1d_cplx, idft_1d_cplx},
};

// ═══════════════════════════════════════════════════════════════════════════
// §A — Coupling-matrix builders (the ONE new ingredient vs ADR-0164)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a rank-1-dense symmetric coupling matrix `D = diag(a) + λ·g·gᵀ`.
///
/// Every off-diagonal entry `D[i,j] = λ·g_i·g_j ≠ 0` (genuinely dense, all
/// pairs coupled), yet the off-diagonal block has numerical rank 1 across every
/// cut → bounded TT-rank (design §1 / §2).
///
/// `a[j]` = per-axis self-diffusion; `g[j]` = coupling eigenvector (all entries
/// non-zero for full density); `lambda` = coupling strength.
/// Returns row-major `d×d` matrix.
pub(crate) fn rank1_dense_matrix<F: SemiflowFloat>(
    a: &[F],
    g: &[F],
    lambda: F,
) -> Vec<F> {
    let d = a.len();
    debug_assert_eq!(g.len(), d);
    let mut mat = vec![F::zero(); d * d];
    for i in 0..d {
        for j in 0..d {
            let cross = lambda * g[i] * g[j];
            if i == j {
                mat[i * d + j] = a[i] + cross;
            } else {
                mat[i * d + j] = cross;
            }
        }
    }
    mat
}

/// Build a rank-m-dense symmetric matrix `D = diag(a) + Σ_{a<m} λ_a·g_a·g_aᵀ`.
///
/// Used ONLY by the negative-boundary contrast (m=2) — proves the gate is non-
/// vacuous: rank-2-dense explodes while rank-1-dense saturates.
pub(crate) fn rankm_dense_matrix<F: SemiflowFloat>(
    a: &[F],
    factors: &[&[F]],
    lambdas: &[F],
) -> Vec<F> {
    let d = a.len();
    let m = factors.len();
    debug_assert_eq!(lambdas.len(), m);
    let mut mat = vec![F::zero(); d * d];
    for i in 0..d {
        mat[i * d + i] = a[i];
    }
    for q in 0..m {
        let g = factors[q];
        debug_assert_eq!(g.len(), d);
        let lam = lambdas[q];
        for i in 0..d {
            for j in 0..d {
                mat[i * d + j] += lam * g[i] * g[j];
            }
        }
    }
    mat
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — Per-axis spectral symbols (shared helper, mirrors ADR-0164)
// ═══════════════════════════════════════════════════════════════════════════

/// Build `σ_D2[m] = (2·cos ω − 2)/dx²` and `σ_D1r[m] = sin ω / dx`.
fn axis_symbols<F: SemiflowFloat>(n: usize, dx: F) -> (Vec<F>, Vec<F>) {
    let two_pi = from_f64::<F>(core::f64::consts::TAU);
    let two = from_f64::<F>(2.0);
    let nf = from_f64::<F>(n as f64);
    let sym_d2: Vec<F> = (0..n)
        .map(|m| {
            let omega = two_pi * from_f64::<F>(m as f64) / nf;
            (two * omega.cos() - two) / (dx * dx)
        })
        .collect();
    let sym_d1r: Vec<F> = (0..n)
        .map(|m| {
            let omega = two_pi * from_f64::<F>(m as f64) / nf;
            omega.sin() / dx
        })
        .collect();
    (sym_d2, sym_d1r)
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Full d-D complex expsym (all-pairs coupling; the mathematical core)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the COMPLEX d-D expsym for `exp(τ·σ(k))` over the full `n^d` Fourier grid.
///
/// `σ(m₀..m_{d-1}) = Σ_j D[j,j]·σ_D2(m_j)`          (diagonal diffusion, RE)
///                 `− Σ_{j<k} 2·D[j,k]·σ_D1r(m_j)·σ_D1r(m_k)` (ALL-PAIRS cross, RE)
///                 `+ i·Σ_j b[j]·σ_D1r(m_j)`           (drift, IM)
///
/// Returns interleaved `(re,im)` of length `2·n^d`.
///
/// Adjacency-agnostic: the pair sum runs over ALL `(j,k)` with `j<k`.  This is
/// exactly what v9.1 refused for non-adjacent coupling; the spectral symbol
/// makes adjacency irrelevant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_expsym_nd<F: SemiflowFloat>(
    n: usize,
    d: usize,
    dx: F,
    d_mat: &[F],
    b: &[F],
    tau: F,
) -> Vec<F> {
    debug_assert_eq!(d_mat.len(), d * d);
    debug_assert_eq!(b.len(), d);

    let two = from_f64::<F>(2.0);
    let (sym_d2, sym_d1r) = axis_symbols(n, dx);

    let nd = n.pow(d as u32);
    let mut out = vec![F::zero(); 2 * nd];

    for flat in 0..nd {
        // Decode multi-index: modes[j] is the Fourier mode on axis j.
        let mut f = flat;
        let mut modes = vec![0usize; d];
        for j in (0..d).rev() {
            modes[j] = f % n;
            f /= n;
        }

        // Build real and imaginary parts of the symbol.
        let sym_re = compute_sym_re(&modes, d, &sym_d2, &sym_d1r, d_mat, two);
        let sym_im = compute_sym_im(&modes, d, &sym_d1r, b);

        // expsym = exp(τ·sym_re) · (cos(τ·sym_im) + i·sin(τ·sym_im))
        let exp_re = (tau * sym_re).exp();
        let phase = tau * sym_im;
        out[2 * flat] = exp_re * phase.cos();
        out[2 * flat + 1] = exp_re * phase.sin();
    }
    out
}

/// Compute RE part of symbol at one mode: diagonal diffusion + ALL-PAIRS cross.
// Comments use math notation (Sigma, sigma_D2); clippy doc-markdown is suppressed.
fn compute_sym_re<F: SemiflowFloat>(
    modes: &[usize],
    d: usize,
    sym_d2: &[F],
    sym_d1r: &[F],
    d_mat: &[F],
    two: F,
) -> F {
    let mut re = F::zero();
    // Diagonal diffusion: sum_j D[j,j] * sd2(m_j)
    for j in 0..d {
        re += d_mat[j * d + j] * sym_d2[modes[j]];
    }
    // All-pairs cross: -sum_{j<k} 2*D[j,k]*sd1r(m_j)*sd1r(m_k)
    for j in 0..d {
        for k in (j + 1)..d {
            re -= two * d_mat[j * d + k] * sym_d1r[modes[j]] * sym_d1r[modes[k]];
        }
    }
    re
}

/// Compute IM part of symbol at one mode: drift sum_j b[j]*sd1r(m_j).
fn compute_sym_im<F: SemiflowFloat>(
    modes: &[usize],
    d: usize,
    sym_d1r: &[F],
    b: &[F],
) -> F {
    let mut im = F::zero();
    for j in 0..d {
        im += b[j] * sym_d1r[modes[j]];
    }
    im
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — d-D FFT/IFFT helpers (sequential 1-D DFTs along each axis)
// ═══════════════════════════════════════════════════════════════════════════

/// Forward d-D DFT: real flat `n^d` → complex interleaved `2·n^d`.
fn fft_nd_real<F: SemiflowFloat>(u: &[F], n: usize, d: usize) -> Vec<F> {
    // Embed real as complex interleaved.
    let mut cplx: Vec<F> = u.iter().flat_map(|&v| [v, F::zero()]).collect();
    apply_fft_nd_inplace(&mut cplx, n, d, false);
    cplx
}

/// Inverse d-D DFT: complex interleaved → real (returns max|imag|).
fn ifft_nd<F: SemiflowFloat>(cplx: &[F], n: usize, d: usize) -> (Vec<F>, F) {
    let nd = n.pow(d as u32);
    let mut cur = cplx.to_vec();
    apply_fft_nd_inplace(&mut cur, n, d, true);
    collect_real(&cur, nd)
}

/// Apply sequential 1-D DFTs in-place along all axes.
fn apply_fft_nd_inplace<F: SemiflowFloat>(cplx: &mut Vec<F>, n: usize, d: usize, inverse: bool) {
    for j in 0..d {
        let stride = n.pow((d - 1 - j) as u32);
        let n_before = n.pow(j as u32);
        let mut next = cplx.clone();
        for ib in 0..n_before {
            for ia in 0..stride {
                let line = extract_line(cplx, n, stride, ib, ia);
                let transformed = if inverse { idft_1d_cplx(&line) } else { dft_1d_cplx(&line) };
                store_line(&mut next, &transformed, n, stride, ib, ia);
            }
        }
        *cplx = next;
    }
}

/// Extract one 1-D line (complex interleaved) along an axis.
fn extract_line<F: SemiflowFloat>(
    cplx: &[F],
    n: usize,
    stride: usize,
    ib: usize,
    ia: usize,
) -> Vec<F> {
    (0..n)
        .flat_map(|k| {
            let idx = ib * n * stride + k * stride + ia;
            [cplx[2 * idx], cplx[2 * idx + 1]]
        })
        .collect()
}

/// Store one transformed 1-D line back into the buffer.
fn store_line<F: SemiflowFloat>(
    buf: &mut [F],
    line: &[F],
    n: usize,
    stride: usize,
    ib: usize,
    ia: usize,
) {
    for k in 0..n {
        let idx = ib * n * stride + k * stride + ia;
        buf[2 * idx] = line[2 * k];
        buf[2 * idx + 1] = line[2 * k + 1];
    }
}

/// Collect real part and track max|imag| residue.
fn collect_real<F: SemiflowFloat>(cplx: &[F], nd: usize) -> (Vec<F>, F) {
    let mut out = vec![F::zero(); nd];
    let mut max_imag = F::zero();
    for i in 0..nd {
        out[i] = cplx[2 * i];
        let im = cplx[2 * i + 1].abs();
        if im > max_imag {
            max_imag = im;
        }
    }
    (out, max_imag)
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — Dense-coupling d-D evolver (solver-free, Theorem-6 R2)
// ═══════════════════════════════════════════════════════════════════════════

/// Evolve `u0` (flat `n^d` real) by the full d-D complex spectral symbol.
///
/// Algorithm: d-D DFT → elementwise COMPLEX multiply by [`dense_expsym_nd`] → d-D IDFT
/// → take real.
///
/// NO `lu_solve_inplace`, NO `dense_expm`, NO triangular solve (Theorem-6 R2).
/// Imaginary residue < 1e-10 (asserted by gate assert 3).
///
/// Returns `(evolved flat n^d real state, max |imag residue|)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_coupling_evolve<F: SemiflowFloat>(
    u0: &[F],
    n: usize,
    d: usize,
    dx: F,
    d_mat: &[F],
    b: &[F],
    tau: F,
) -> (Vec<F>, F) {
    // Step 1: forward d-D DFT.
    let mut cplx = fft_nd_real(u0, n, d);
    // Step 2: build full d-D complex expsym.
    let expsym = dense_expsym_nd(n, d, dx, d_mat, b, tau);
    // Step 3: elementwise complex multiply: (a+ib)·(c+id) = (ac−bd) + i(ad+bc).
    let nd = u0.len();
    for i in 0..nd {
        let (fr, fi) = (cplx[2 * i], cplx[2 * i + 1]);
        let (er, ei) = (expsym[2 * i], expsym[2 * i + 1]);
        cplx[2 * i] = fr * er - fi * ei;
        cplx[2 * i + 1] = fr * ei + fi * er;
    }
    // Step 4: inverse d-D DFT, take real.
    ifft_nd(&cplx, n, d)
}

// ═══════════════════════════════════════════════════════════════════════════
// §F — Unit tests (fast; normative reduction invariants from contract §1.4)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// §1.4(b): D = diag(a), b = 0 → symbol equals separable diffusion.
    ///
    /// With no off-diagonal coupling and no drift the cross sum vanishes and the
    /// imaginary part is zero. The real part is the separable diffusion symbol.
    #[test]
    fn diagonal_d_zero_b_is_separable() {
        let n = 7usize;
        let d = 3usize;
        let dx = 1.0_f64 / n as f64;
        let a_diag = [0.5f64, 0.7, 0.4];
        let b_zero = [0.0f64, 0.0, 0.0];
        let tau = 0.02f64;

        // Dense path with D = diag(a), b = 0.
        let d_mat = rank1_dense_matrix(&a_diag, &[0.0f64, 0.0, 0.0], 0.0);
        let es_dense = dense_expsym_nd(n, d, dx, &d_mat, &b_zero, tau);

        // Reference: compute separable symbol manually.
        let two_pi = core::f64::consts::TAU;
        let sym_d2: Vec<f64> = (0..n)
            .map(|m| (2.0 * (two_pi * m as f64 / n as f64).cos() - 2.0) / (dx * dx))
            .collect();
        let nd = n.pow(d as u32);
        for flat in 0..nd {
            let mut f = flat;
            let mut sym_re = 0.0f64;
            for j in (0..d).rev() {
                let mj = f % n;
                sym_re += a_diag[j] * sym_d2[mj];
                f /= n;
            }
            let expected_re = (tau * sym_re).exp();
            let re = es_dense[2 * flat];
            let im = es_dense[2 * flat + 1];
            assert!(
                im.abs() < 1e-14,
                "im nonzero at flat={flat}: {im:.3e}"
            );
            assert!(
                (re - expected_re).abs() < 1e-12,
                "re mismatch at flat={flat}: got {re:.10}, expected {expected_re:.10}"
            );
        }
    }

    /// §1.4(a): tridiagonal D → dense expsym must equal adjacent-only symbol bit-for-bit.
    ///
    /// When D\[j,k\] = 0 for |j-k| > 1 the pair sum reduces to adjacent pairs only.
    #[test]
    fn tridiagonal_d_equals_adjacent_symbol() {
        let n = 5usize;
        let d = 3usize;
        let dx = 1.0_f64 / n as f64;
        let tau = 0.02f64;
        let rho = 0.15f64;
        let a_val = 0.5f64;
        let b_vals = [0.6f64, 0.7, 0.8];

        let d_diag = [a_val, a_val, a_val];
        let mut d_tri = rank1_dense_matrix(&d_diag, &[0.0f64, 0.0, 0.0], 0.0);
        d_tri[1] = rho; d_tri[d] = rho; d_tri[d + 2] = rho; d_tri[2 * d + 1] = rho;

        let es_dense = dense_expsym_nd(n, d, dx, &d_tri, &b_vals, tau);
        check_adjacent_only_symbol(&es_dense, n, d, dx, tau, rho, a_val, &b_vals);
    }

    /// Compare expsym against the adjacent-only reference formula (d=3 case).
    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
    fn check_adjacent_only_symbol(
        es_dense: &[f64], n: usize, d: usize, dx: f64, tau: f64, rho: f64, a: f64, b: &[f64],
    ) {
        let two_pi = core::f64::consts::TAU;
        let nf = n as f64;
        let sd2: Vec<f64> = (0..n)
            .map(|m| (2.0 * (two_pi * m as f64 / nf).cos() - 2.0) / (dx * dx))
            .collect();
        let sd1r: Vec<f64> = (0..n).map(|m| (two_pi * m as f64 / nf).sin() / dx).collect();
        let nd = n.pow(d as u32);
        for flat in 0..nd {
            let mut f = flat;
            let mut modes = [0usize; 3];
            for j in (0..d).rev() { modes[j] = f % n; f /= n; }
            let mut sym_re: f64 = (0..d).map(|j| a * sd2[modes[j]]).sum();
            sym_re -= 2.0 * rho * sd1r[modes[0]] * sd1r[modes[1]];
            sym_re -= 2.0 * rho * sd1r[modes[1]] * sd1r[modes[2]];
            let sym_im: f64 = b.iter().zip(modes.iter()).map(|(&bj, &m)| bj * sd1r[m]).sum();
            let exp_re = (tau * sym_re).exp();
            let phase = tau * sym_im;
            let (ere, eim) = (exp_re * phase.cos(), exp_re * phase.sin());
            assert_eq!(es_dense[2*flat].to_bits(), ere.to_bits(),
                "tridiag re mismatch flat={flat}: got {:.12}, exp {ere:.12}", es_dense[2*flat]);
            assert_eq!(es_dense[2*flat+1].to_bits(), eim.to_bits(),
                "tridiag im mismatch flat={flat}: got {:.12}, exp {eim:.12}", es_dense[2*flat+1]);
        }
    }

    /// Smoke test: rank1_dense_matrix has non-zero off-diagonals.
    #[test]
    fn rank1_dense_nonzero_offdiag() {
        let d = 4usize;
        let a: Vec<f64> = vec![0.5; d];
        let g: Vec<f64> = (0..d)
            .map(|k| (k as f64 * 0.3 + 0.5).cos() * 0.6)
            .collect();
        let mat = rank1_dense_matrix(&a, &g, 0.25);
        let n_offdiag = (0..d)
            .flat_map(|i| (0..d).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j && mat[i * d + j].abs() > 1e-14)
            .count();
        assert_eq!(n_offdiag, d * (d - 1), "all off-diag should be non-zero");
    }

    /// Round-trip: fft_nd_real → ifft_nd recovers original.
    #[test]
    fn fft_nd_roundtrip() {
        let n = 5usize;
        let d = 3usize;
        let nd = n.pow(d as u32);
        let u0: Vec<f64> = (0..nd).map(|i| ((i as f64) * 0.37 + 0.1).sin()).collect();
        let cplx = fft_nd_real(&u0, n, d);
        let (recovered, max_imag) = ifft_nd(&cplx, n, d);
        let max_err = u0.iter().zip(recovered.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_err < 1e-12, "round-trip err={max_err:.3e}");
        assert!(max_imag < 1e-12, "round-trip max_imag={max_imag:.3e}");
    }
}
