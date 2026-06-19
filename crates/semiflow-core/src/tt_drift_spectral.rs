//! S³ POC — Complex-spectral drift pair factor (`DriftSpectralPairFactor`).
//!
//! Extends `tt_spectral.rs` (§11.3 real spectral pair-factor) by widening the
//! Fourier symbol from **real** to **complex**: the diffusion is the real part,
//! the drift is the **imaginary** part.  The conjugate-even symmetry of the symbol
//! (for real `a, b`) guarantees that `ifft(expsym ⊙ fft(real))` is real, so the
//! operator is rank-O(1) and solver-free while fully carrying advection.
//!
//! ## New mathematics (§11.3 extension, ADR-0164)
//!
//! ```text
//! symbol(mj,mk) = cj·σ_D2(mj) + ck·σ_D2(mk)          (real, diffusion)
//!               + i·bj·σ_D1r(mj) + i·bk·σ_D1r(mk)    (imaginary, DRIFT)
//!               − 2r·σ_D1r(mj)·σ_D1r(mk)              (real, cross; i·i=−1)
//!
//! σ_D2(m) = (2·cos ω − 2)/dx²,  σ_D1r(m) = sin ω / dx,  ω = 2π·m/n
//!
//! expsym(mj,mk) = exp(τ_eff · symbol(mj,mk))   [COMPLEX]
//!   = exp(τ·Re sym)·(cos(τ·Im sym) + i·sin(τ·Im sym))
//! ```
//!
//! **bj=bk=0 reduces exactly to `pair_expsym_real`** (0 ULP, see reduction test).
//!
//! ## Solver-free (Theorem-6 R2)
//!
//! Operations: DFT, elementwise complex multiply, IDFT, take real.
//! NO `lu_solve_inplace`, NO `dense_expm`, NO triangular solve.
//!
//! ## Scope (PROOF-OF-CONCEPT, constant-coef only)
//!
//! Ref: `contracts/s3-drift-spectral-poc.contract.md`,
//!      `.dev-docs/specs/s3-triz-general-curse-escape.md`.

// Grid/FFT indices (usize) cast to F for angle computation; all values ≪ 2^52.
// POC module: pub(crate) fns are used only by the slow-tests gate, not the lib.
#![allow(clippy::cast_precision_loss, dead_code)]
#![cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]

extern crate alloc;
use alloc::{vec, vec::Vec};

use crate::{
    float::{from_f64, SemiflowFloat},
    tt_spectral::{dft_1d_cplx, dft_1d_real_to_cplx, idft_1d_cplx},
};

// ═══════════════════════════════════════════════════════════════════════════
// §A — Complex symbol builder (the ONE mathematical novelty)
// ═══════════════════════════════════════════════════════════════════════════

/// Build per-axis spectral symbols (D2=diffusion, D1r=sin/dx for drift/cross).
///
/// `sym_d2[m] = (2·cos ω − 2)/dx²`,  `sym_d1r[m] = sin ω / dx`,  `ω = 2π·m/n`.
fn axis_symbols_drift<F: SemiflowFloat>(n: usize, dx: F) -> (Vec<F>, Vec<F>) {
    let two_pi = from_f64::<F>(core::f64::consts::TAU);
    let two = from_f64::<F>(2.0);
    let sym_d2: Vec<F> = (0..n)
        .map(|m| {
            let omega = two_pi * from_f64::<F>(m as f64) / from_f64::<F>(n as f64);
            (two * omega.cos() - two) / (dx * dx)
        })
        .collect();
    let sym_d1r: Vec<F> = (0..n)
        .map(|m| {
            let omega = two_pi * from_f64::<F>(m as f64) / from_f64::<F>(n as f64);
            omega.sin() / dx
        })
        .collect();
    (sym_d2, sym_d1r)
}

/// Build the COMPLEX expsym diagonal for `exp(τ·(L_diff + L_drift + L_cross))`
/// over a single `(j,k)` pair panel.  Returns interleaved `(re,im)` of length `2·n_j·n_k`.
///
/// ```text
/// symbol(mj,mk) = cj·σ_D2(mj) + ck·σ_D2(mk)          re part
///               + i·(bj·σ_D1r(mj) + bk·σ_D1r(mk))    im part (DRIFT)
///               − 2r·σ_D1r(mj)·σ_D1r(mk)              re part (cross)
///
/// expsym = exp(τ·Re)·cos(τ·Im) + i·exp(τ·Re)·sin(τ·Im)
/// ```
///
/// With `bj=bk=0` the imaginary part is zero and the real part equals
/// `pair_expsym_real(..)`  to 0 ULP (reduction invariant, §1.4 of contract).
///
/// # Arguments
/// * `n_j, n_k` — grid sizes on axes j, k
/// * `dx_j, dx_k` — grid spacings
/// * `cj, ck` — diffusion coefficients (`a_j`/#pairs(j), same as `v9.1`)
/// * `bj, bk` — DRIFT coefficients (NEW; zero ⇒ reduces to `pair_expsym_real`)
/// * `r_cross` — cross-diffusion coupling coefficient
/// * `tau_eff` — effective time step
#[allow(clippy::too_many_arguments)]
pub(crate) fn drift_pair_expsym_cplx<F: SemiflowFloat>(
    n_j: usize,
    n_k: usize,
    dx_j: F,
    dx_k: F,
    cj: F,
    ck: F,
    bj: F,
    bk: F,
    r_cross: F,
    tau_eff: F,
) -> Vec<F> {
    let (sym_d2_j, sym_d1r_j) = axis_symbols_drift(n_j, dx_j);
    let (sym_d2_k, sym_d1r_k) = axis_symbols_drift(n_k, dx_k);
    let two = from_f64::<F>(2.0);

    // Output: interleaved (re, im), length 2·n_j·n_k.
    let mut out = vec![F::zero(); 2 * n_j * n_k];
    for mj in 0..n_j {
        for mk in 0..n_k {
            // Real part: diffusion + cross (i·σ_D1·i·σ_D1 = −σ_D1·σ_D1).
            let sym_re = cj * sym_d2_j[mj] + ck * sym_d2_k[mk]
                - two * r_cross * sym_d1r_j[mj] * sym_d1r_k[mk];
            // Imaginary part: drift (i·b·σ_D1r on each axis).
            let sym_im = bj * sym_d1r_j[mj] + bk * sym_d1r_k[mk];
            // expsym = exp(τ·(sym_re + i·sym_im)) = e^(τ·sym_re)·e^(i·τ·sym_im)
            let exp_re = (tau_eff * sym_re).exp();
            let phase = tau_eff * sym_im;
            let idx = mj * n_k + mk;
            out[2 * idx] = exp_re * phase.cos();
            out[2 * idx + 1] = exp_re * phase.sin();
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — 2-D FFT helpers (complex-multiply version)
// ═══════════════════════════════════════════════════════════════════════════

/// Forward 2-D DFT of a real `n_j × n_k` row-major panel (same layout as `tt_spectral`).
///
/// Returns complex interleaved `[mj·n_k·2 + mk·2 + (0=re|1=im)]`.
fn fft2_real_drift<F: SemiflowFloat>(panel: &[F], n_j: usize, n_k: usize) -> Vec<F> {
    // Stage 1: DFT along axis j for each k column.
    let mut cplx_j = vec![F::zero(); n_j * n_k * 2];
    for ik in 0..n_k {
        let col_real: Vec<F> = (0..n_j).map(|ij| panel[ij * n_k + ik]).collect();
        let col_cplx = dft_1d_real_to_cplx(&col_real);
        for mj in 0..n_j {
            cplx_j[mj * n_k * 2 + ik * 2] = col_cplx[2 * mj];
            cplx_j[mj * n_k * 2 + ik * 2 + 1] = col_cplx[2 * mj + 1];
        }
    }
    // Stage 2: DFT along axis k for each mj row.
    let mut cplx_jk = vec![F::zero(); n_j * n_k * 2];
    for mj in 0..n_j {
        let row: Vec<F> = (0..n_k)
            .flat_map(|ik| {
                [
                    cplx_j[mj * n_k * 2 + ik * 2],
                    cplx_j[mj * n_k * 2 + ik * 2 + 1],
                ]
            })
            .collect();
        let row_f = dft_1d_cplx(&row);
        for mk in 0..n_k {
            cplx_jk[mj * n_k * 2 + mk * 2] = row_f[2 * mk];
            cplx_jk[mj * n_k * 2 + mk * 2 + 1] = row_f[2 * mk + 1];
        }
    }
    cplx_jk
}

/// Inverse 2-D DFT; write real part into `panel`, return max |imag residue|.
fn ifft2_write_real_drift<F: SemiflowFloat>(
    cplx: &[F],
    panel: &mut [F],
    n_j: usize,
    n_k: usize,
) -> F {
    // Stage 4: IDFT along axis k for each mj row.
    let mut cplx_j2 = vec![F::zero(); n_j * n_k * 2];
    for mj in 0..n_j {
        let row: Vec<F> = (0..n_k)
            .flat_map(|mk| [cplx[mj * n_k * 2 + mk * 2], cplx[mj * n_k * 2 + mk * 2 + 1]])
            .collect();
        let row_inv = idft_1d_cplx(&row);
        for ik in 0..n_k {
            cplx_j2[mj * n_k * 2 + ik * 2] = row_inv[2 * ik];
            cplx_j2[mj * n_k * 2 + ik * 2 + 1] = row_inv[2 * ik + 1];
        }
    }
    // Stage 5: IDFT along axis j, collect real part and track imag residue.
    let mut max_imag = F::zero();
    for ik in 0..n_k {
        let col: Vec<F> = (0..n_j)
            .flat_map(|mj| {
                [
                    cplx_j2[mj * n_k * 2 + ik * 2],
                    cplx_j2[mj * n_k * 2 + ik * 2 + 1],
                ]
            })
            .collect();
        let col_inv = idft_1d_cplx(&col);
        for ij in 0..n_j {
            panel[ij * n_k + ik] = col_inv[2 * ij];
            let im_abs = col_inv[2 * ij + 1].abs();
            if im_abs > max_imag {
                max_imag = im_abs;
            }
        }
    }
    max_imag
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Complex-expsym panel apply (contract §1.2)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply `exp(τ·L_pair)` (with drift) to a flat real `n_j×n_k` panel via complex spectral.
///
/// Algorithm: fft2 → elementwise COMPLEX multiply by `expsym_cplx` → ifft2 → take real.
/// Returns `max |imag residue|` (must be < 1e-12 per contract assert 3).
///
/// NO `lu_solve_inplace`, NO `dense_expm`, NO triangular solve (Theorem-6 R2).
pub(crate) fn apply_drift_spectral_pair_to_panel<F: SemiflowFloat>(
    panel: &mut [F],
    n_j: usize,
    n_k: usize,
    expsym_cplx: &[F],
) -> F {
    debug_assert_eq!(panel.len(), n_j * n_k);
    debug_assert_eq!(expsym_cplx.len(), 2 * n_j * n_k);

    // Forward 2-D FFT.
    let mut cplx = fft2_real_drift(panel, n_j, n_k);

    // Elementwise COMPLEX multiply: (a+ib)·(c+id) = (ac−bd) + i(ad+bc).
    for mj in 0..n_j {
        for mk in 0..n_k {
            let idx = mj * n_k + mk;
            let fre = cplx[2 * idx];
            let fim = cplx[2 * idx + 1];
            let ere = expsym_cplx[2 * idx];
            let eim = expsym_cplx[2 * idx + 1];
            cplx[2 * idx] = fre * ere - fim * eim;
            cplx[2 * idx + 1] = fre * eim + fim * ere;
        }
    }

    // Inverse 2-D FFT → real output; return imag residue.
    ifft2_write_real_drift(&cplx, panel, n_j, n_k)
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — Separable 1-D drift apply (contract §1.3)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply per-axis `exp(τ·(a·∂² + b·∂))` to a 1-D real line via complex spectral symbol.
///
/// The symbol is `a·σ_D2(m) + i·b·σ_D1r(m)` — same math as the pair factor but 1-D.
/// Proves rank-1 / O(d·n) curse-escape for the separable case (Probe B in design doc).
/// Returns `max |imag residue|`.
pub(crate) fn apply_drift_spectral_axis<F: SemiflowFloat>(
    line: &mut [F],
    n: usize,
    dx: F,
    a: F,
    b: F,
    tau: F,
) -> F {
    debug_assert_eq!(line.len(), n);
    let (sym_d2, sym_d1r) = axis_symbols_drift(n, dx);

    // Forward DFT.
    let mut cplx = dft_1d_real_to_cplx(line);

    // Complex multiply: cplx[m] *= exp(τ·(a·d2 + i·b·d1r)).
    for m in 0..n {
        let sym_re = a * sym_d2[m];
        let sym_im = b * sym_d1r[m];
        let exp_re = (tau * sym_re).exp();
        let phase = tau * sym_im;
        let fre = cplx[2 * m];
        let fim = cplx[2 * m + 1];
        let ere = exp_re * phase.cos();
        let eim = exp_re * phase.sin();
        cplx[2 * m] = fre * ere - fim * eim;
        cplx[2 * m + 1] = fre * eim + fim * ere;
    }

    // Inverse DFT → real.
    let recovered = idft_1d_cplx(&cplx);
    let mut max_imag = F::zero();
    for i in 0..n {
        line[i] = recovered[2 * i];
        let im_abs = recovered[2 * i + 1].abs();
        if im_abs > max_imag {
            max_imag = im_abs;
        }
    }
    max_imag
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — Unit tests (run in test-fast; normative per contract §1.4)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tt_spectral::pair_expsym_real;

    // ── Reduction invariant (§1.4, contract NORMATIVE) ───────────────────
    // With bj=bk=0 the complex expsym MUST equal pair_expsym_real to 0 ULP
    // (re-part) and the imaginary part must be exactly zero.
    // This proves the new code is a faithful SUPERSET, not a relabel.
    #[test]
    fn reduction_to_real_expsym_zero_ulp() {
        let (n_j, n_k) = (7usize, 9usize);
        let dx_j = 1.0f64 / (n_j as f64 - 1.0);
        let dx_k = 1.0f64 / (n_k as f64 - 1.0);
        let (cj, ck) = (0.8f64, 0.6f64);
        let r = 0.4f64 * (cj * ck).sqrt();
        let tau = 0.35 * dx_j * dx_j;

        // Reference: the existing real expsym.
        let real_es = pair_expsym_real(n_j, n_k, dx_j, dx_k, cj, ck, r, tau);

        // New: complex expsym with bj=bk=0.
        let cplx_es = drift_pair_expsym_cplx(n_j, n_k, dx_j, dx_k, cj, ck, 0.0, 0.0, r, tau);

        assert_eq!(cplx_es.len(), 2 * n_j * n_k);
        for idx in 0..(n_j * n_k) {
            let re = cplx_es[2 * idx];
            let im = cplx_es[2 * idx + 1];
            let expected = real_es[idx];
            // 0 ULP: bit-identical (no floating-point difference allowed).
            assert_eq!(
                re.to_bits(),
                expected.to_bits(),
                "re mismatch at idx={idx}: got {re}, expected {expected}"
            );
            // im must be exactly zero in value (sym_im = 0 when bj=bk=0).
            // Accept ±0.0 (both are exact zero; only the sign bit differs).
            assert!(
                im == 0.0,
                "im nonzero at idx={idx}: {im:.3e}"
            );
        }
    }

    // ── 1-D axis apply: round-trip at b=0 matches simple diffusion ───────
    // With b=0 the axis apply must reproduce the 1-D diffusion (real expsym).
    #[test]
    fn axis_apply_zero_drift_matches_real() {
        let n = 11usize;
        let dx = 1.0f64 / (n as f64 - 1.0);
        let a = 0.7f64;
        let tau = 0.35 * dx * dx;

        let mut line: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37 + 0.1).sin()).collect();
        let line_orig = line.clone();

        // Apply once with b=0.
        let imag_res = apply_drift_spectral_axis(&mut line, n, dx, a, 0.0, tau);
        assert!(
            imag_res < 1e-13,
            "imag residue with b=0: {imag_res:.3e} (expected <1e-13)"
        );

        // Apply in reverse (with tau → -tau) should recover original.
        let imag_res2 = apply_drift_spectral_axis(&mut line, n, dx, a, 0.0, -tau);
        assert!(imag_res2 < 1e-13, "reverse imag residue: {imag_res2:.3e}");
        let max_err = line
            .iter()
            .zip(line_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-12,
            "round-trip error with b=0: {max_err:.3e} (expected <1e-12)"
        );
    }

    // ── 2-D apply: identity (expsym = ones) preserves panel ──────────────
    #[test]
    fn drift_apply_identity_expsym() {
        let (n_j, n_k) = (5usize, 6usize);
        let panel_orig: Vec<f64> = (0..n_j * n_k)
            .map(|i| ((i as f64) * 0.17 + 0.3).sin())
            .collect();
        let mut panel = panel_orig.clone();
        // Identity expsym: re=1, im=0 for all modes.
        let mut expsym = vec![0.0f64; 2 * n_j * n_k];
        for idx in 0..(n_j * n_k) {
            expsym[2 * idx] = 1.0;
        }
        let imag_res = apply_drift_spectral_pair_to_panel(&mut panel, n_j, n_k, &expsym);
        assert!(imag_res < 1e-11, "identity imag residue: {imag_res:.3e}");
        for i in 0..n_j * n_k {
            let err = (panel[i] - panel_orig[i]).abs();
            assert!(err < 1e-11, "identity err at {i}: {err:.3e}");
        }
    }

    // ── Drift is genuinely sub-grid (contract assert 2 pre-check) ────────
    // At the gate parameters b·τ/dx must be non-integer with frac > 0.05.
    #[test]
    fn drift_parameters_sub_grid() {
        let n = 7usize;
        let dx = 1.0f64 / (n as f64 - 1.0);
        let tau = 0.35 * dx * dx;
        let b = 1.3f64;
        let frac = (b * tau / dx).fract().abs();
        assert!(
            frac > 0.05,
            "b·τ/dx is too close to an integer; frac={frac:.4} (need >0.05)"
        );
    }
}
