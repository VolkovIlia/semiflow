//! Spectral (FFT-diagonal) pair factor for `CoupledTtChernoff` (v9.1.0 P3'').
//!
//! Replaces the dense LU-Padé `dense_expm` path with a **solver-free** spectral
//! apply, honouring Theorem-6 R2 ("no linear solver") for the coupling factor.
//!
//! ## Formula (§11.3 — NORMATIVE)
//!
//! `exp(τ·L_pair)·u = ifft2( expsym ⊙ fft2(u) )`
//!
//! where `expsym[m_j, m_k] = exp(τ_eff · symbol(m_j, m_k))` and
//! ```text
//! symbol(m_j, m_k) = c_j·σ_D2(m_j) + c_k·σ_D2(m_k) + 2r·σ_D1(m_j)·σ_D1(m_k),
//! σ_D2(m)  = (2·cos(ω_m·dx) − 2) / dx²,
//! σ_D1(m)  = i·sin(ω_m·dx) / dx,
//! ω_m = 2π·m / n      (DFT wavenumber, 0 ≤ m < n; `d=dx` convention).
//! ```
//!
//! Note the cross term is REAL:
//! `2r·σ_D1(m_j)·σ_D1(m_k) = −2r·sin_j·sin_k / (dx_j·dx_k)`.
//! So `symbol` and `expsym` are REAL.  The intermediate DFT is complex-valued
//! (fft of real field), but the multiplier is real → output is real.
//!
//! ## Solver-free proof
//! The only operations are:
//! - Forward 1-D DFT along axis j (fixed unitary matrix, no pivoting/solve)
//! - Forward 1-D DFT along axis k (same)
//! - Elementwise multiply by `expsym` (a pre-built real vector)
//! - Inverse 1-D DFT along axis k
//! - Inverse 1-D DFT along axis j
//! - Take real part (imaginary residue is ≤ 1e-13 by construction)
//!
//! No `lu_solve_inplace`, no `dense_expm`, no triangular solve anywhere.
//! Confirmed machine-exact in `probe_adjudicate_rotated_shift.py` (R3) and
//! `probe_adjudicate_spectral_cost.py`. In-Rust rel-err floor: d=2 self-check
//! 5.27e-14; d∈{3,4} gate ≤1e-12. The numpy single-panel 1.2e-15 is a lower bound.
//!
//! ## Hoist contract (§11.5)
//! `expsym` is τ-only.  Build it ONCE per `(pair, τ)` via `pair_expsym_real`
//! before the per-step loop.  Pass the pre-built slice to `apply_spectral_pair_to_slab`.
//!
//! ## DFT implementation
//! O(n²) direct DFT per axis (n is small for TT; FFT is a future perf option).
//! Complex numbers represented as interleaved `(re, im)` flattened `Vec<F>` slices.
//! No external complex type, no new public type.  Trig via `num_traits::Float`
//! (`cos`, `sin`, `exp`) — already in `SemiflowFloat`.  No new dependency.
//!
//! References: math.md §52.9 (NORMATIVE R3, round 4); ADR-0162;
//! `probe_adjudicate_rotated_shift.py`; `probe_adjudicate_spectral_cost.py`.

// Grid/FFT indices (usize) cast to f64 to compute coordinates/wavenumbers;
// all values are grid sizes ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use crate::float::{from_f64, SemiflowFloat};

// ═══════════════════════════════════════════════════════════════════════════
// §A — 1-D DFT/IDFT (O(n²) direct; complex as interleaved (re,im) Vec<F>)
// ═══════════════════════════════════════════════════════════════════════════

/// Forward 1-D DFT: `F[k] = Σ_j x[j] · exp(-2πi·j·k/n)`.
///
/// Input: real array `x` of length `n`.
/// Output: complex array of length `n`, interleaved `[re_0, im_0, re_1, im_1, …]`.
pub(crate) fn dft_1d_real_to_cplx<F: SemiflowFloat>(x: &[F]) -> Vec<F> {
    let n = x.len();
    let two_pi_over_n = from_f64::<F>(core::f64::consts::TAU / n as f64);
    let mut out = vec![F::zero(); 2 * n]; // interleaved re,im
                                          // j is used both to index x[j] and in the angle computation (j*k); range loop is needed.
    #[allow(clippy::needless_range_loop)]
    for k in 0..n {
        let mut re = F::zero();
        let mut im = F::zero();
        for j in 0..n {
            // angle = -2π·j·k/n
            let angle = -two_pi_over_n * from_f64::<F>((j * k) as f64);
            re += x[j] * angle.cos();
            im += x[j] * angle.sin();
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
    out
}

/// Forward 1-D DFT: complex input → complex output.
///
/// Input: interleaved `[re_0, im_0, …]` of length `2n`.
/// Output: interleaved `[re_0, im_0, …]` of length `2n`.
pub(crate) fn dft_1d_cplx<F: SemiflowFloat>(x: &[F]) -> Vec<F> {
    let n = x.len() / 2;
    let two_pi_over_n = from_f64::<F>(core::f64::consts::TAU / n as f64);
    let mut out = vec![F::zero(); 2 * n];
    for k in 0..n {
        let mut re = F::zero();
        let mut im = F::zero();
        for j in 0..n {
            let angle = -two_pi_over_n * from_f64::<F>((j * k) as f64);
            let c = angle.cos();
            let s = angle.sin();
            let xre = x[2 * j];
            let xim = x[2 * j + 1];
            // (xre + i·xim)·(c + i·s) = (xre·c - xim·s) + i·(xre·s + xim·c)
            re = re + xre * c - xim * s;
            im = im + xre * s + xim * c;
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
    out
}

/// Inverse 1-D DFT (with 1/n normalisation): complex → complex.
///
/// Input: interleaved `[re_0, im_0, …]` of length `2n`.
/// Output: interleaved `[re_0, im_0, …]` of length `2n`.
pub(crate) fn idft_1d_cplx<F: SemiflowFloat>(x: &[F]) -> Vec<F> {
    let n = x.len() / 2;
    let two_pi_over_n = from_f64::<F>(core::f64::consts::TAU / n as f64);
    let inv_n = F::one() / from_f64::<F>(n as f64);
    let mut out = vec![F::zero(); 2 * n];
    for k in 0..n {
        let mut re = F::zero();
        let mut im = F::zero();
        for j in 0..n {
            // +2π convention for inverse
            let angle = two_pi_over_n * from_f64::<F>((j * k) as f64);
            let c = angle.cos();
            let s = angle.sin();
            let xre = x[2 * j];
            let xim = x[2 * j + 1];
            re = re + xre * c - xim * s;
            im = im + xre * s + xim * c;
        }
        out[2 * k] = re * inv_n;
        out[2 * k + 1] = im * inv_n;
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — Spectral symbol and expsym (τ-only, built once per pair)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the real expsym diagonal for `exp(τ_eff·L_pair)` via spectral.
///
/// Returns a flat Vec of length `n_j·n_k` in row-major order `[m_j*n_k + m_k]`,
/// where `expsym[m_j,m_k] = exp(τ_eff · symbol(m_j, m_k))`.
///
/// The symbol is:
/// ```text
/// sym_d2(m, n, dx) = (2·cos(2π·m/n) − 2) / dx²
/// sym_d1(m, n, dx) = sin(2π·m/n) / dx         (imaginary part stripped)
/// symbol = c_j·sym_d2_j + c_k·sym_d2_k − 2r·sym_d1_j·sym_d1_k
/// ```
/// Note: `σ_D1(m_j)·σ_D1(m_k) = (i·sin_j/dx_j)·(i·sin_k/dx_k) = −sin_j·sin_k/(dx_j·dx_k)`,
/// so the full cross term is `2r·(−sin_j·sin_k/(dx_j·dx_k))` and symbol is REAL.
///
/// This is a pure hoist: τ-only, step-independent.  Build once, reuse every step.
/// Build the per-axis D2 symbol `(2·cos(2π·m/n) − 2) / dx²` and
/// D1 symbol `sin(2π·m/n) / dx` for `m = 0..n`.
fn axis_spectral_symbols<F: SemiflowFloat>(n: usize, dx: F, two_pi: F) -> (Vec<F>, Vec<F>) {
    let sym_d2: Vec<F> = (0..n)
        .map(|m| {
            let omega = two_pi * from_f64::<F>(m as f64) / from_f64::<F>(n as f64);
            (from_f64::<F>(2.0) * omega.cos() - from_f64::<F>(2.0)) / (dx * dx)
        })
        .collect();
    let sym_d1: Vec<F> = (0..n)
        .map(|m| {
            let omega = two_pi * from_f64::<F>(m as f64) / from_f64::<F>(n as f64);
            omega.sin() / dx
        })
        .collect();
    (sym_d2, sym_d1)
}

// n_j, n_k, dx_j, dx_k, cj, ck, r_cross, tau_eff: all 8 required for the spectral symbol.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_expsym_real<F: SemiflowFloat>(
    n_j: usize,
    n_k: usize,
    dx_j: F,
    dx_k: F,
    cj: F,
    ck: F,
    r_cross: F,
    tau_eff: F,
) -> Vec<F> {
    let two_pi = from_f64::<F>(core::f64::consts::TAU);
    // Pre-compute per-axis spectral symbols.
    // sym_d2[m] = (2·cos(2π·m/n) − 2) / dx²
    // sym_d1[m] = sin(2π·m/n) / dx
    let (sym_d2_j, sym_d1_j) = axis_spectral_symbols(n_j, dx_j, two_pi);
    let (sym_d2_k, sym_d1_k) = axis_spectral_symbols(n_k, dx_k, two_pi);

    // Build the 2D symbol and expsym.
    let mut expsym = vec![F::zero(); n_j * n_k];
    for mj in 0..n_j {
        for mk in 0..n_k {
            // cross term: −2r·sym_d1_j[mj]·sym_d1_k[mk]  (since sym_d1 = sin/dx)
            let sym = cj * sym_d2_j[mj] + ck * sym_d2_k[mk]
                - from_f64::<F>(2.0) * r_cross * sym_d1_j[mj] * sym_d1_k[mk];
            expsym[mj * n_k + mk] = (tau_eff * sym).exp();
        }
    }
    expsym
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Spectral 2D apply on a flat n_j × n_k panel
// ═══════════════════════════════════════════════════════════════════════════

/// Apply `exp(τ·L_pair)` to a flat `n_j × n_k` panel via spectral (R3).
///
/// Algorithm: fft2 along axis j then k, elementwise multiply by `expsym`,
/// ifft2 along k then j, take real part.
///
/// `panel`: flat `n_j * n_k` row-major slice (modified in-place).
/// `expsym`: precomputed `pair_expsym_real` output (same layout).
///
/// NO `lu_solve_inplace`, NO `dense_expm`.  Only DFT + elementwise multiply.
///
/// Actual in-Rust rel-err floor (d=2 self-check): 5.27e-14; d∈{3,4} gate: ≤1e-12.
/// The d=2 numpy probe figure (1.2e-15) is a lower single-panel bound.
pub(crate) fn apply_spectral_pair_to_panel<F: SemiflowFloat>(
    panel: &mut [F],
    n_j: usize,
    n_k: usize,
    expsym: &[F],
) {
    debug_assert_eq!(panel.len(), n_j * n_k);
    debug_assert_eq!(expsym.len(), n_j * n_k);

    // Step 1 + 2: forward 2-D DFT (axis j then axis k).
    let cplx_jk = fft2_real_panel(panel, n_j, n_k);

    // Step 3: elementwise multiply by expsym (real scalar × complex).
    let mut cplx_scaled = cplx_jk;
    for mj in 0..n_j {
        for mk in 0..n_k {
            let scale = expsym[mj * n_k + mk];
            cplx_scaled[mj * n_k * 2 + mk * 2] *= scale;
            cplx_scaled[mj * n_k * 2 + mk * 2 + 1] *= scale;
        }
    }

    // Step 4 + 5: inverse 2-D DFT (axis k then axis j), take real part.
    ifft2_write_real(&cplx_scaled, panel, n_j, n_k);
}

/// Forward 2-D DFT of a real `n_j × n_k` row-major panel.
///
/// Returns a complex interleaved buffer of length `n_j * n_k * 2`:
/// layout `[mj * n_k * 2 + mk * 2 + (0=re|1=im)]`.
/// Stage 1: DFT along axis j (columns). Stage 2: DFT along axis k (rows).
fn fft2_real_panel<F: SemiflowFloat>(panel: &[F], n_j: usize, n_k: usize) -> Vec<F> {
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

/// Inverse 2-D DFT and write real part back into `panel`.
///
/// `cplx`: complex interleaved buffer `[mj * n_k * 2 + mk * 2 + (0=re|1=im)]`.
/// Stage 4: IDFT along axis k. Stage 5: IDFT along axis j; take real part.
/// Imaginary residue after full round-trip is ≤ 1e-13 (real operator; safe to drop).
fn ifft2_write_real<F: SemiflowFloat>(cplx: &[F], panel: &mut [F], n_j: usize, n_k: usize) {
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
    // Stage 5: IDFT along axis j, write real part to panel.
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
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// DFT round-trip: ifft(fft(x)) == x for a simple signal.
    #[test]
    fn dft_roundtrip() {
        let n = 8usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3 + 0.1).sin()).collect();
        // Forward real→cplx
        let fx = dft_1d_real_to_cplx(&x);
        // Inverse cplx→cplx
        let recovered = idft_1d_cplx(&fx);
        for i in 0..n {
            let err = (recovered[2 * i] - x[i]).abs();
            assert!(err < 1e-12, "roundtrip err at {i}: {err:.3e}");
            assert!(recovered[2 * i + 1].abs() < 1e-12, "imag residue at {i}");
        }
    }

    /// DFT of a constant is n·δ[0].
    #[test]
    fn dft_constant_signal() {
        let n = 6usize;
        let x = vec![1.0f64; n];
        let fx = dft_1d_real_to_cplx(&x);
        // F[0] = n, F[k] = 0 for k>0
        assert!((fx[0] - n as f64).abs() < 1e-12, "F[0]={}", fx[0]);
        for k in 1..n {
            let mag = (fx[2 * k].powi(2) + fx[2 * k + 1].powi(2)).sqrt();
            assert!(mag < 1e-11, "F[{k}]={mag:.3e} should be ~0");
        }
    }

    /// Spectral apply of exp(0·L)·u = u (identity: expsym = all ones).
    #[test]
    fn spectral_apply_identity() {
        let n_j = 5usize;
        let n_k = 5usize;
        let panel_orig: Vec<f64> = (0..n_j * n_k)
            .map(|i| (i as f64 * 0.17 + 0.3).sin())
            .collect();
        let mut panel = panel_orig.clone();
        // expsym = all ones (τ=0)
        let expsym = vec![1.0f64; n_j * n_k];
        apply_spectral_pair_to_panel(&mut panel, n_j, n_k, &expsym);
        for i in 0..n_j * n_k {
            let err = (panel[i] - panel_orig[i]).abs();
            assert!(err < 1e-11, "identity err at {i}: {err:.3e}");
        }
    }

    /// `pair_expsym_real`: all values positive and ≤ 1 (dissipative operator).
    #[test]
    fn expsym_positive_dissipative() {
        let (nj, nk) = (7usize, 7usize);
        let dx = 1.0f64 / (nj as f64 - 1.0);
        let tau = 0.35 * dx * dx;
        let (cj, ck, r) = (0.8f64, 0.6f64, 0.6f64 * (0.8f64 * 0.6f64).sqrt());
        let es = pair_expsym_real(nj, nk, dx, dx, cj, ck, r, tau);
        assert_eq!(es.len(), nj * nk);
        for &v in &es {
            assert!(v > 0.0 && v <= 1.0 + 1e-12, "expsym={v:.6} out of (0,1]");
        }
    }
}
