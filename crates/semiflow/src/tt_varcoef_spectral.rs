//! S³ POC — Variable-coefficient additive-split evolver (`VarCoefSplitSpectral`).
//!
//! Proves order-2 curse-escape for **additive-separable** variable coefficients:
//! `L = Σⱼ Lⱼ`,  `Lⱼ = ∂_{xⱼ}(aⱼ(xⱼ)·∂_{xⱼ}) + bⱼ(xⱼ)·∂_{xⱼ} + vⱼ(xⱼ)`.
//!
//! ## Two-layer design (ADR-0166)
//!
//! - **Layer-1 (inter-axis):** additive generators on disjoint axes commute
//!   (`[Lⱼ,Lₖ]=0`), so `exp(τΣLⱼ)=∏ⱼexp(τLⱼ)` EXACTLY.  Each factor is
//!   rank-1 TT-operator (identity on other axes).  ZERO inter-axis splitting error.
//! - **Layer-2 (intra-axis):** 1-D variable-coef `exp(τLⱼ)` approximated by
//!   `P₂(τ/2)·k(τ)·P₂(τ/2)`, where `k(τ)=exp(τ·a₀·Lap)` is the const-coef
//!   spectral factor (ADR-0164) and `P₂(s)=I+s·R+s²/2·R²` is the 2nd-order
//!   polynomial Chernoff factor for `exp(s·R)`, `R=Lⱼ−a₀·Lap_fd`.  Order-2 in τ.
//!
//! ## Boundary (enforced by type)
//!
//! Non-separable `a(x,y)` is UNREPRESENTABLE: `AxisCoef` only stores per-axis
//! arrays.  Asserts 4 + multiplier-rank explosion prove the scheme does NOT
//! converge off the additive class (wrong-operator floor).
//!
//! ## Solver-free (Theorem-6 R2)
//!
//! Evolver uses: 1-D FFT/IDFT (`tt_spectral`), tridiagonal mat-vecs (2 per step).
//! NO `lu_solve_inplace`, NO `dense_expm`.
//!
//! Ref: `contracts/s3-variable-coef-poc.contract.md`, `.dev-docs/specs/s3-variable-coef.md`.

#![cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    clippy::struct_field_names,
    dead_code
)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use crate::{
    float::{from_f64, SemiflowFloat},
    tt_drift_spectral::apply_drift_spectral_axis,
};

// ═══════════════════════════════════════════════════════════════════════════
// §1.1 — Per-axis coefficient container (boundary enforced by type)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-axis ADDITIVE-separable coefficient container.
///
/// Non-separable `a(x,y)` is UNREPRESENTABLE by construction — per-axis
/// arrays cannot encode a full 2-D function, which is the class boundary.
///
/// Fields:
/// - `a_axis[j]`: length-`n` grid of leading diffusion on axis `j` (must be `> 0`).
/// - `b_axis[j]`: length-`n` drift on axis `j`.
/// - `v_axis[j]`: length-`n` reaction on axis `j` (empty slice ⇒ zero).
pub struct AxisCoef<F: SemiflowFloat> {
    /// Per-axis diffusion coefficients `a_j(x_j)` (d × n, each entry > 0).
    pub a_axis: Vec<Vec<F>>,
    /// Per-axis drift coefficients `b_j(x_j)` (d × n).
    pub b_axis: Vec<Vec<F>>,
    /// Per-axis reaction `v_j(x_j)` (d × n; empty row ⇒ zero).
    pub v_axis: Vec<Vec<F>>,
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.2 — Tridiagonal residual operator R = L_j − a0·Lap_fd
// ═══════════════════════════════════════════════════════════════════════════

/// Build residual `R = L_j − a0·Lap_fd` as 3 periodic diagonals.
///
/// `L_j` is the divergence-form FD generator:
/// `L_j[i] = (a_{i+1/2}(u_{i+1}−u_i) − a_{i−1/2}(u_i−u_{i−1}))/dx²
///           + b_i(u_{i+1}−u_{i−1})/(2dx) + v_i·u_i`
/// where `a_{i+1/2} = (a[i]+a[(i+1)%n])/2`.
///
/// Returns `(sub, main, sup)`, each length `n`, for the periodic tridiagonal.
/// NO solve, NO expm — pure coefficient assembly.
pub(crate) fn residual_tridiag<F: SemiflowFloat>(
    a: &[F],
    b: &[F],
    v: &[F],
    dx: F,
    a0: F,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let n = a.len();
    let dx2 = dx * dx;
    let two_dx = from_f64::<F>(2.0) * dx;
    let half = from_f64::<F>(0.5);
    let two = from_f64::<F>(2.0);
    let mut lower = vec![F::zero(); n];
    let mut center = vec![F::zero(); n];
    let mut upper = vec![F::zero(); n];

    for i in 0..n {
        let ip = (i + 1) % n;
        let im = (i + n - 1) % n;
        // half-point diffusion: a_{i±1/2} = (a[i] + a[i±1]) / 2
        let ahp = (a[i] + a[ip]) * half; // a_{i+1/2}
        let ahm = (a[i] + a[im]) * half; // a_{i-1/2}
                                         // L_j row entries (divergence-form FD)
        let lj_up = ahp / dx2 + b[i] / two_dx;
        let lj_mid = -(ahp + ahm) / dx2 + (if v.is_empty() { F::zero() } else { v[i] });
        let lj_lo = ahm / dx2 - b[i] / two_dx;
        // Subtract a0·Lap_fd (periodic FD Laplacian with coef=1):
        //   Lap_fd: lower = 1/dx², center = -2/dx², upper = 1/dx²
        lower[i] = lj_lo - a0 / dx2;
        center[i] = lj_mid + two * a0 / dx2; // lj_mid - a0*(-2/dx²) = lj_mid + 2a0/dx²
        upper[i] = lj_up - a0 / dx2;
    }

    (lower, center, upper)
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.3 — Polynomial residual factor P₂(s) = I + s·R + s²/2·R²
// ═══════════════════════════════════════════════════════════════════════════

/// Apply `P₂(s)·u = u + s·(Ru) + (s²/2)·R(Ru)` for periodic tridiagonal `R`.
///
/// 2 tridiagonal mat-vecs.  ZERO `lu_solve_inplace`, ZERO `dense_expm`.
pub(crate) fn p2_apply_tridiag<F: SemiflowFloat>(
    line: &mut [F],
    sub: &[F],
    main: &[F],
    sup: &[F],
    s: F,
) {
    let n = line.len();
    // First mat-vec: ru = R · u
    let mut ru = vec![F::zero(); n];
    for i in 0..n {
        let ip = (i + 1) % n;
        let im = (i + n - 1) % n;
        ru[i] = sub[i] * line[im] + main[i] * line[i] + sup[i] * line[ip];
    }
    // Second mat-vec: rru = R · (R·u)
    let mut rru = vec![F::zero(); n];
    for i in 0..n {
        let ip = (i + 1) % n;
        let im = (i + n - 1) % n;
        rru[i] = sub[i] * ru[im] + main[i] * ru[i] + sup[i] * ru[ip];
    }
    // u += s·(Ru) + (s²/2)·R(Ru)
    let half = from_f64::<F>(0.5);
    for i in 0..n {
        line[i] = line[i] + s * ru[i] + half * s * s * rru[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.4 — 1-D variable-coef Chernoff factor
// ═══════════════════════════════════════════════════════════════════════════

/// One 1-D variable-coef step: `P₂(τ/2)·k(τ)·P₂(τ/2)`.
///
/// `k(τ)=exp(τ·a0·Lap)` via 1-D spectral (ADR-0164 `apply_drift_spectral_axis`
/// with `b=0`). The residual `R` diagonals are built from `a`, `b`, `v`.
/// Returns `max|imag residue|` from the spectral factor (< 1e-12 expected).
pub(crate) fn varcoef_axis_step<F: SemiflowFloat>(
    line: &mut [F],
    n: usize,
    dx: F,
    a: &[F],
    b: &[F],
    v: &[F],
    tau: F,
) -> F {
    debug_assert_eq!(line.len(), n);
    // mean leading diffusion coefficient
    let a0 = a.iter().copied().fold(F::zero(), |acc, x| acc + x) / from_f64(n as f64);
    let half_tau = tau / from_f64(2.0);
    let (r_lower, r_center, r_upper) = residual_tridiag(a, b, v, dx, a0);
    // Left P₂(τ/2)
    p2_apply_tridiag(line, &r_lower, &r_center, &r_upper, half_tau);
    // const-coef spectral factor k(τ) = exp(τ·a0·Lap), drift b=0
    let imag = apply_drift_spectral_axis(line, n, dx, a0, F::zero(), tau);
    // Right P₂(τ/2)
    p2_apply_tridiag(line, &r_lower, &r_center, &r_upper, half_tau);
    imag
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.5 — d-D additive-split evolver (per-axis Strang)
// ═══════════════════════════════════════════════════════════════════════════

/// Evolve flat `n^d` real state by `exp(τ·L)`, `L = Σⱼ Lⱼ` (additive-separable).
///
/// Symmetric per-axis Strang: forward sweep `j=0..d`, backward `j=d-1..0`,
/// each axis a `varcoef_axis_step` at half-step `τ/2` (forward) and `τ/2`
/// (backward), giving one full `τ` step.  Combined palindrome is:
/// `(j=0: τ/2)(j=1: τ/2)…(j=d-1: τ)…(j=1: τ/2)(j=0: τ/2)`.
///
/// Returns `(evolved n^d vec, max|imag residue| over all axis steps)`.
/// NO `lu_solve_inplace`, NO `dense_expm` (Theorem-6 R2).
pub(crate) fn varcoef_evolve<F: SemiflowFloat>(
    u0: &[F],
    n: usize,
    d: usize,
    dx: F,
    coef: &AxisCoef<F>,
    tau: F,
    nsteps: usize,
) -> (Vec<F>, F) {
    let nd = n.pow(d as u32);
    debug_assert_eq!(u0.len(), nd);
    let mut u = u0.to_vec();
    let half_tau = tau / from_f64(2.0);
    let mut max_imag = F::zero();

    for _ in 0..nsteps {
        // Forward half-sweep: j=0..d-1 with τ/2, then j=d-1 with τ.
        for j in 0..d {
            let step_tau = if j == d - 1 { tau } else { half_tau };
            let imag = apply_axis_sweep(&mut u, n, d, j, dx, coef, step_tau);
            if imag > max_imag {
                max_imag = imag;
            }
        }
        // Backward half-sweep: j=d-2..0 with τ/2 (j=d-1 already done).
        for j in (0..d - 1).rev() {
            let imag = apply_axis_sweep(&mut u, n, d, j, dx, coef, half_tau);
            if imag > max_imag {
                max_imag = imag;
            }
        }
    }
    (u, max_imag)
}

// ═══════════════════════════════════════════════════════════════════════════
// §1.5a — Per-axis sweep helper
// ═══════════════════════════════════════════════════════════════════════════

/// Apply `varcoef_axis_step` to every 1-D line along axis `j` of a `n^d` tensor.
///
/// The tensor is stored in standard row-major (last index fastest):
/// `idx(i_0,…,i_{d-1}) = Σ_j i_j · n^{d-1-j}`.
/// Stride along axis `j` is `n^{d-1-j}`.  Returns max|imag| over all lines.
fn apply_axis_sweep<F: SemiflowFloat>(
    u: &mut [F],
    n: usize,
    d: usize,
    axis: usize,
    dx: F,
    coef: &AxisCoef<F>,
    tau: F,
) -> F {
    let stride = n.pow((d - 1 - axis) as u32);
    let n_outer = n.pow(axis as u32);
    let a_coef = &coef.a_axis[axis];
    let b_coef = &coef.b_axis[axis];
    let v_coef = &coef.v_axis[axis];
    let mut max_imag = F::zero();
    let mut line = vec![F::zero(); n];

    for i_outer in 0..n_outer {
        for i_inner in 0..stride {
            // Extract line along axis `axis`
            for idx in 0..n {
                line[idx] = u[i_outer * n * stride + idx * stride + i_inner];
            }
            let imag = varcoef_axis_step(&mut line, n, dx, a_coef, b_coef, v_coef, tau);
            if imag > max_imag {
                max_imag = imag;
            }
            // Write back
            for idx in 0..n {
                u[i_outer * n * stride + idx * stride + i_inner] = line[idx];
            }
        }
    }
    max_imag
}

// ═══════════════════════════════════════════════════════════════════════════
// §F — Public wrapper type (ADR-0169 boundary-as-type promotion)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "s3-poc")]
/// Order-2 additive-separable variable-coefficient evolver (S³ POC).
///
/// Converges at order 2 for operators `L = Σⱼ Lⱼ` where each `Lⱼ` depends
/// only on axis `j`.  Non-separable `a(x,y)` is unrepresentable by [`AxisCoef`].
///
/// ## Proven boundary
/// Order-2 ONLY for ADDITIVE-separable coefficients `L = Σⱼ Lⱼ`. Non-separable
/// `a(x,y)` is UNREPRESENTABLE by [`AxisCoef`] (per-axis arrays only) — for
/// low-CP-rank non-separable coefficients use [`crate::S3NonSepVarCoefEvolver`]. Proof:
/// `g_s3_varcoef_spectral` (RELEASE-BLOCKING, `slow-tests`), slope ≤ −1.9. The
/// scheme does NOT converge off the additive class (wrong-operator floor). See
/// ADR-0166, math.md §53.3.
pub struct S3VarCoefEvolver<F: SemiflowFloat> {
    /// Grid size (same on every axis).
    n: usize,
    /// Number of spatial dimensions.
    d: usize,
    /// Grid spacing.
    dx: F,
    /// Per-axis coefficient container.
    coef: AxisCoef<F>,
}

#[cfg(feature = "s3-poc")]
impl<F: SemiflowFloat> S3VarCoefEvolver<F> {
    /// Construct the evolver; validates per-axis shapes and parabolicity.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if any `a_axis[j]` has
    /// length != `n`, any `a_axis[j][i] ≤ 0`, or shape mismatches.
    pub fn new(n: usize, d: usize, dx: F, coef: AxisCoef<F>) -> Result<Self, crate::SemiflowError> {
        validate_varcoef(n, d, dx, &coef)?;
        Ok(Self { n, d, dx, coef })
    }

    /// Evolve state `u0` (flat `n^d`) for `nsteps` time steps of size `tau`.
    ///
    /// Returns `(evolved_state, max_imag_residue)`.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if `u0.len() != n^d`.
    pub fn evolve(
        &self,
        u0: &[F],
        tau: F,
        nsteps: usize,
    ) -> Result<(Vec<F>, F), crate::SemiflowError> {
        let nd = self.n.pow(self.d as u32);
        if u0.len() != nd {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "u0 length must equal n^d",
            });
        }
        Ok(varcoef_evolve(
            u0, self.n, self.d, self.dx, &self.coef, tau, nsteps,
        ))
    }
}

#[cfg(feature = "s3-poc")]
fn validate_varcoef<F: SemiflowFloat>(
    n: usize,
    d: usize,
    dx: F,
    coef: &AxisCoef<F>,
) -> Result<(), crate::SemiflowError> {
    if coef.a_axis.len() != d || coef.b_axis.len() != d || coef.v_axis.len() != d {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "AxisCoef arrays must have length d",
        });
    }
    if n < 2 {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "n must be >= 2",
        });
    }
    if dx <= F::zero() {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "dx must be > 0",
        });
    }
    for (j, a_j) in coef.a_axis.iter().enumerate() {
        if a_j.len() != n {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "a_axis[j] must have length n",
            });
        }
        for &v in a_j {
            if v <= F::zero() {
                return Err(crate::SemiflowError::S3OutOfClass {
                    detail: "a_axis[j][i] must be > 0 (parabolicity)",
                });
            }
        }
        let _ = j;
    }
    Ok(())
}

// §E — Unit tests (fast; normative per contract §1.6)
#[cfg(test)]
mod tests {
    include!("tt_varcoef_spectral_tests_mod.rs");
}

#[cfg(all(test, feature = "s3-poc"))]
mod tests_s3 {
    include!("tt_varcoef_spectral_tests_s3_mod.rs");
}
