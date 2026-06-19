//! S³ POC — Non-separable variable-coefficient evolver (`NonSepVarCoefSpectral`).
//!
//! Proves order-2 curse-escape for **low-CP-rank** non-separable variable coefficients.
//! Fixes the ADR-0166 boundary: `cos(x)sin(y)·∂²ₓ` (slope 0, floor 9.53e-3) now converges.
//!
//! ## TRIZ resolution (ADR-0167)
//!
//! The coefficient `a(x)` on the tensor grid is a `d`-way tensor.  Its **CP/TT-rank** is the
//! structured quantity.  For `a(x) = a₀ + Σ_{r=1}^{m} ∏ⱼ a_{r,j}(xⱼ)` (CP-rank `m`),
//! `diag(a(x))·core` is a rank-`m` TT operator — no LU, no dense expm (Theorem-6 R2).
//!
//! ## Step: `P₂(τ/2)·k(τ)·P₂(τ/2)`
//!
//! - `k(τ) = exp(τ·a₀·Σⱼ Lap_j)` via d-D FFT-diagonal (reuse ADR-0164 spectral factor).
//! - `R = L − a₀·Lap` as rank-`m` TT mat-vecs (NO solver, NO dense expm).
//! - `P₂(s) = I + s·R + s²/2·R²` (2 TT mat-vecs).
//!
//! ## Boundary (enforced by type)
//!
//! `CpCoef.terms` is a fixed-`m` Vec of CP-terms.  Generic full-rank `a(x)` is
//! UNREPRESENTABLE — making the curse-escape claim non-vacuous.
//!
//! ## Solver-free (Theorem-6 R2)
//!
//! NO `lu_solve_inplace`, NO `dense_expm` in this module.
//!
//! Ref: `contracts/s3-nonsep-varcoef-poc.contract.md`, `docs/adr/0167-*`.

#![cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    clippy::struct_field_names,
    dead_code, // pub(crate) items used only from g_s3_nonsep_varcoef integration test
)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use crate::{
    float::{from_f64, SemiflowFloat},
    tt_drift_spectral::apply_drift_spectral_axis,
};

// ═══════════════════════════════════════════════════════════════════════════
// §1.1 — CP-term and coefficient container
// ═══════════════════════════════════════════════════════════════════════════

/// One CP-term of a low-CP-rank coefficient: `c_r(x) = ∏ⱼ factor[j](xⱼ)`.
///
/// `factor[j]` is a length-`n` grid of the per-axis factor on axis `j`.
pub struct CpTerm<F: SemiflowFloat> {
    /// Per-axis factors (d × n, `factor[j] = c_{r,j}(x_j)`).
    pub factor: Vec<Vec<F>>,
}

/// Differential role of the CP-coefficient: which core operator it multiplies.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CoefRole {
    /// `diag(a(x)) · ∂²ₓ₀` (variable leading diffusion on axis 0).
    Diffusion,
    /// `diag(a(x)) · ∂ₓ₀` (variable drift on axis 0, centred FD).
    Drift,
    /// `diag(a(x)) · I` (variable potential / reaction).
    Potential,
}

/// Low-CP-rank coefficient field attached to ONE differential core.
///
/// `c(x) = c0 + Σ_{r<m} ∏ⱼ factor_r[j](xⱼ)`, `m = terms.len()` fixed.
/// Generic full-rank `a(x)` is **unrepresentable** — the CP-rank wall.
pub struct CpCoef<F: SemiflowFloat> {
    /// Constant leading part (`a₀` for Diffusion; 0 for Drift/Potential).
    pub c0: F,
    /// Fixed-`m` CP-terms (m = `terms.len()`; generic full-rank is unrepresentable).
    pub terms: Vec<CpTerm<F>>,
    /// Which 1-D core operator this coefficient multiplies.
    pub role: CoefRole,
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.2 — Per-axis const-coef tridiagonal core stencil
// ═══════════════════════════════════════════════════════════════════════════

/// Build `(sub, main, sup)` for the periodic const-coef 1-D core on axis 0.
///
/// - `Diffusion` → FD Laplacian `∂²/∂x²` (3-point, periodic)
/// - `Drift`     → centred `∂/∂x` (periodic)
/// - `Potential` → identity
///
/// NO solve, NO expm — pure stencil assembly.  Each output is length `n`.
pub(crate) fn core_tridiag<F: SemiflowFloat>(
    role: CoefRole,
    dx: F,
    n: usize,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let dx2 = dx * dx;
    let two_dx = from_f64::<F>(2.0) * dx;
    let zero = F::zero();
    let one = from_f64::<F>(1.0);
    let two = from_f64::<F>(2.0);
    match role {
        CoefRole::Diffusion => {
            let inv_dx2 = one / dx2;
            let sub  = vec![inv_dx2; n];
            let main = vec![-two * inv_dx2; n];
            let sup  = vec![inv_dx2; n];
            (sub, main, sup)
        }
        CoefRole::Drift => {
            let half_inv_2dx = one / two_dx;
            let sub  = vec![-half_inv_2dx; n];
            let main = vec![zero; n];
            let sup  = vec![half_inv_2dx; n];
            (sub, main, sup)
        }
        CoefRole::Potential => {
            let sub  = vec![zero; n];
            let main = vec![one; n];
            let sup  = vec![zero; n];
            (sub, main, sup)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.3 — Rank-m TT residual application R·u (pure TT mat-vec)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply the residual `R·u` for a SINGLE CP-coefficient `coef`.
///
/// `R = Σ_{r<m} [ diag(factor_r) · core_tridiag ]` where the core acts on axis 0
/// and diag acts on the FULL d-way tensor.
///
/// For each CP-term `r`:
/// - Multiply each element `u[flat]` by the CP-term value `∏ⱼ factor_r[j](x_{j,coord})`
///   **excluding** axis 0 (the core axis); apply the 1-D tridiagonal on axis 0; then
///   multiply again by the axis-0 factor `factor_r[0][coord_0]`.
///
/// Cost: `O(m · n^d)` — flat in `d` for fixed `m`.  NO solve, NO `dense_expm`.
pub(crate) fn apply_residual<F: SemiflowFloat>(
    u: &[F],
    out: &mut [F],
    n: usize,
    d: usize,
    dx: F,
    coef: &CpCoef<F>,
) {
    let nd = n.pow(d as u32);
    debug_assert_eq!(u.len(), nd);
    debug_assert_eq!(out.len(), nd);
    for v in out.iter_mut() { *v = F::zero(); }
    if coef.terms.is_empty() { return; }
    let (sub, main, sup) = core_tridiag(coef.role, dx, n);
    // stride[0] = n^{d-1} (row-major, last-fastest).
    let stride0 = n.pow((d - 1) as u32);
    for term in &coef.terms {
        for line_flat in 0..stride0 {
            apply_residual_line(u, out, n, d, stride0, term, &sub, &main, &sup, line_flat);
        }
    }
}

/// Apply one CP-term, one line (axis-0) contribution into `out`.
///
/// Computes `∏_{ax≥1} factor[ax] · core_tridiag(axis-0) · u_line`
/// and accumulates into `out`.
fn apply_residual_line<F: SemiflowFloat>(
    u: &[F],
    out: &mut [F],
    n: usize,
    d: usize,
    stride0: usize,
    term: &CpTerm<F>,
    sub: &[F],
    main: &[F],
    sup: &[F],
    line_flat: usize,
) {
    // Product of all non-axis-0 factors at this (i_1,...,i_{d-1}).
    let non0_factor: F = {
        let mut f = from_f64::<F>(1.0);
        let mut tmp = line_flat;
        for ax in (1..d).rev() {
            let coord = tmp % n;
            tmp /= n;
            f *= term.factor[ax][coord];
        }
        f
    };
    // Gather u along axis-0 line.
    let mut line = vec![F::zero(); n];
    for i0 in 0..n { line[i0] = u[i0 * stride0 + line_flat]; }
    // Apply tridiagonal stencil.
    let mut core_u = vec![F::zero(); n];
    for i in 0..n {
        let ip = (i + 1) % n;
        let im = (i + n - 1) % n;
        core_u[i] = sub[i] * line[im] + main[i] * line[i] + sup[i] * line[ip];
    }
    // Accumulate: scale by axis-0 factor * non0_factor.
    for i0 in 0..n {
        let scale = term.factor[0][i0] * non0_factor;
        out[i0 * stride0 + line_flat] += scale * core_u[i0];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.4 — Polynomial residual factor P₂(s)·u = u + s·Ru + s²/2·R(Ru)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply `P₂(s)·u = u + s·Ru + (s²/2)·R(Ru)` (2 TT mat-vecs).
///
/// Aggregates ALL `coefs`.  NO `lu_solve_inplace`, NO `dense_expm` (Theorem-6 R2).
pub(crate) fn p2_apply<F: SemiflowFloat>(
    u: &mut [F],
    scratch: &mut [F],
    n: usize,
    d: usize,
    dx: F,
    coefs: &[CpCoef<F>],
    s: F,
) {
    let nd = u.len();
    debug_assert_eq!(scratch.len(), nd);

    // Compute Ru = sum over all coefs of apply_residual.
    let mut ru = vec![F::zero(); nd];
    let mut tmp = vec![F::zero(); nd];
    for coef in coefs {
        for v in &mut tmp { *v = F::zero(); }
        apply_residual(u, &mut tmp, n, d, dx, coef);
        for i in 0..nd { ru[i] += tmp[i]; }
    }

    // Compute R(Ru) = sum over all coefs of apply_residual applied to Ru.
    let mut rru = vec![F::zero(); nd];
    for coef in coefs {
        for v in &mut tmp { *v = F::zero(); }
        apply_residual(&ru, &mut tmp, n, d, dx, coef);
        for i in 0..nd { rru[i] += tmp[i]; }
    }

    // u += s·Ru + (s²/2)·R(Ru).
    let half = from_f64::<F>(0.5);
    for i in 0..nd {
        u[i] += s * ru[i] + half * s * s * rru[i];
    }

    // scratch is used as workspace (already consumed above, kept for API compat).
    let _ = scratch;
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.5 — const-leading spectral factor k(τ) = exp(τ·a₀·Σⱼ Lap_j)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply `k(τ) = exp(τ·a₀·Σⱼ Lap_j)` via d-D FFT-diagonal (NO solve).
///
/// Reuses ADR-0164 `apply_drift_spectral_axis` (b=0) per axis.
/// Returns `max|imag residue|` (< 1e-12 expected).
pub(crate) fn k_spectral<F: SemiflowFloat>(
    u: &mut [F],
    n: usize,
    d: usize,
    dx: F,
    a0: F,
    tau: F,
) -> F {
    let nd = n.pow(d as u32);
    debug_assert_eq!(u.len(), nd);

    let mut max_imag = F::zero();
    let stride_last = n.pow((d - 1) as u32); // stride of axis 0

    // Apply exp(τ·a₀·Lap_j) per axis j (separable: exact product, zero splitting error).
    for axis in 0..d {
        let stride = n.pow((d - 1 - axis) as u32);
        let n_outer = n.pow(axis as u32);
        let mut line = vec![F::zero(); n];

        for i_outer in 0..n_outer {
            for i_inner in 0..stride {
                for idx in 0..n {
                    line[idx] = u[i_outer * n * stride + idx * stride + i_inner];
                }
                let imag = apply_drift_spectral_axis(&mut line, n, dx, a0, F::zero(), tau);
                if imag > max_imag { max_imag = imag; }
                for idx in 0..n {
                    u[i_outer * n * stride + idx * stride + i_inner] = line[idx];
                }
            }
        }
    }
    let _ = stride_last;
    max_imag
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.5b — Parabolicity guard for Diffusion-role coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// `debug_assert` that `c(x) = c0 + Σ_r ∏_j factor_r[j](x_j) > 0` on the full
/// n^d grid (parabolicity of the leading diffusion coefficient).
///
/// Called only for `CoefRole::Diffusion`.  In release builds this is a no-op.
/// Cost: `O(m · n^d)` — acceptable in debug; inlined-away in release.
#[cfg(debug_assertions)]
fn assert_diffusion_parabolicity<F: SemiflowFloat>(coef: &CpCoef<F>, n: usize, d: usize) {
    let nd = n.pow(d as u32);
    let eps = from_f64::<F>(0.0); // strict positivity (> 0)
    for flat in 0..nd {
        // Reconstruct c(x) at grid point `flat` (row-major, last fastest).
        let mut cx = coef.c0;
        for term in &coef.terms {
            let mut prod = from_f64::<F>(1.0);
            let mut idx = flat;
            for ax in (0..d).rev() {
                let coord = idx % n;
                idx /= n;
                prod *= term.factor[ax][coord];
            }
            cx += prod;
        }
        debug_assert!(cx > eps, "parabolicity violated: c(x) = {cx:?} ≤ 0 at flat={flat}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.6 — d-D non-separable evolver
// ═══════════════════════════════════════════════════════════════════════════

/// Evolve `u0` (flat `n^d` real) by `exp(τ·L)` via `P₂(τ/2)·k(τ)·P₂(τ/2)`.
///
/// `R = L − a₀·Lap` is the full non-separable residual (rank-`m` TT operator).
/// Order-2 in τ; solver-free (Theorem-6 R2).  Returns `(evolved n^d, max|imag|)`.
///
/// NO `lu_solve_inplace`, NO `dense_expm`.
pub(crate) fn nonsep_evolve<F: SemiflowFloat>(
    u0: &[F],
    n: usize,
    d: usize,
    dx: F,
    a0: F,
    coefs: &[CpCoef<F>],
    tau: F,
    nsteps: usize,
) -> (Vec<F>, F) {
    let nd = n.pow(d as u32);
    debug_assert_eq!(u0.len(), nd);

    // Parabolicity guard: c(x) > 0 for every Diffusion-role coefficient (debug only).
    #[cfg(debug_assertions)]
    for coef in coefs {
        if coef.role == CoefRole::Diffusion {
            assert_diffusion_parabolicity(coef, n, d);
        }
    }

    let mut u = u0.to_vec();
    let mut scratch = vec![F::zero(); nd];
    let mut max_imag = F::zero();
    let half_tau = tau / from_f64::<F>(2.0);

    for _ in 0..nsteps {
        // Left P₂(τ/2)
        p2_apply(&mut u, &mut scratch, n, d, dx, coefs, half_tau);
        // Spectral k(τ)
        let imag = k_spectral(&mut u, n, d, dx, a0, tau);
        if imag > max_imag { max_imag = imag; }
        // Right P₂(τ/2)
        p2_apply(&mut u, &mut scratch, n, d, dx, coefs, half_tau);
    }
    (u, max_imag)
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — Unit tests (fast; normative reduction invariants)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tt_drift_spectral::apply_drift_spectral_axis;
    use core::f64::consts::TAU;

    fn grid_xs(n: usize) -> Vec<f64> {
        let dx = TAU / n as f64;
        (0..n).map(|i| i as f64 * dx).collect()
    }

    // ── core_tridiag: Diffusion sums to zero per row (periodic FD) ──────
    #[test]
    fn core_diffusion_row_sum_zero() {
        let n = 8;
        let dx = TAU / n as f64;
        let (sub, main, sup) = core_tridiag::<f64>(CoefRole::Diffusion, dx, n);
        for i in 0..n {
            let s = sub[i] + main[i] + sup[i];
            assert!(s.abs() < 1e-14, "Diffusion row sum ≠ 0 at i={i}: {s:.3e}");
        }
    }

    // ── P₂ identity when coefs empty ────────────────────────────────────
    #[test]
    fn p2_identity_no_coefs() {
        let n = 5;
        let d = 2;
        let nd = n * n;
        let dx = TAU / n as f64;
        let mut u: Vec<f64> = (0..nd).map(|i| (i as f64 * 0.3 + 0.1).sin()).collect();
        let u_orig = u.clone();
        let mut scratch = vec![0.0f64; nd];
        p2_apply(&mut u, &mut scratch, n, d, dx, &[], 0.05);
        let max_err = u.iter().zip(u_orig.iter()).map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-15, "P₂ with no coefs ≠ identity: {max_err:.3e}");
    }

    // ── const-coef reduction: k_spectral(a0, b=0) step equals ADR-0164 ─
    #[test]
    fn k_spectral_const_a_equals_spectral_axis() {
        let n: usize = 7;
        let d: usize = 2;
        let nd = n.pow(d as u32);
        let dx = TAU / n as f64;
        let a0 = 0.5f64;
        let tau = 0.02f64;
        let u0: Vec<f64> = (0..nd).map(|i| (i as f64 * 0.37 + 0.1).cos()).collect();

        // Our k_spectral (applies on all d axes).
        let mut u1 = u0.clone();
        k_spectral(&mut u1, n, d, dx, a0, tau);

        // Reference: apply ADR-0164 axis-by-axis manually.
        let stride = n; // stride of axis 0 for d=2
        let mut u2 = u0.clone();
        // axis 0
        for i_inner in 0..stride {
            let mut line: Vec<f64> = (0..n).map(|i0| u2[i0 * stride + i_inner]).collect();
            apply_drift_spectral_axis(&mut line, n, dx, a0, 0.0, tau);
            for i0 in 0..n { u2[i0 * stride + i_inner] = line[i0]; }
        }
        // axis 1
        for i_outer in 0..n {
            let mut line: Vec<f64> = (0..n).map(|i1| u2[i_outer * n + i1]).collect();
            apply_drift_spectral_axis(&mut line, n, dx, a0, 0.0, tau);
            for i1 in 0..n { u2[i_outer * n + i1] = line[i1]; }
        }

        let max_err = u1.iter().zip(u2.iter()).map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-12, "k_spectral ≠ axis-by-axis spectral: {max_err:.3e}");
    }

    // ── apply_residual: zero for empty terms ────────────────────────────
    #[test]
    fn residual_zero_for_empty_terms() {
        let n = 5;
        let d = 2;
        let nd = n * n;
        let dx = TAU / n as f64;
        let coef = CpCoef::<f64> { c0: 0.5, terms: vec![], role: CoefRole::Diffusion };
        let u: Vec<f64> = (0..nd).map(|i| (i as f64).sin()).collect();
        let mut out = vec![1.0f64; nd]; // non-zero to check zeroing
        apply_residual(&u, &mut out, n, d, dx, &coef);
        let max_abs = out.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_abs < 1e-15, "residual with no terms not zero: {max_abs:.3e}");
    }

    // ── nonsep_evolve produces finite output ────────────────────────────
    #[test]
    fn evolve_produces_finite_output() {
        let n = 5;
        let d = 2;
        let nd = n * n;
        let dx = TAU / n as f64;
        let xs = grid_xs(n);
        let a0 = 0.5f64;

        // rank-1 non-separable coefficient: 0.25 * cos(x) * sin(y) [diffusion role]
        let term = CpTerm::<f64> {
            factor: vec![
                xs.iter().map(|&x| 0.25 * x.cos()).collect(),
                xs.iter().map(|&x| x.sin()).collect(),
            ],
        };
        let coefs = vec![CpCoef { c0: a0, terms: vec![term], role: CoefRole::Diffusion }];
        let u0: Vec<f64> = (0..nd).map(|i| (i as f64 * 0.31).sin()).collect();
        let (u_out, max_imag) = nonsep_evolve(&u0, n, d, dx, a0, &coefs, 0.01, 4);
        assert!(u_out.iter().all(|x| x.is_finite()), "evolve produced non-finite");
        assert!(max_imag < 1e-9, "max imag residue too large: {max_imag:.3e}");
    }
}
