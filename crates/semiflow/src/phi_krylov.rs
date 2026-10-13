//! Adaptive Krylov (Arnoldi) φ-combination for any generator (ADR-0211, math §68).
//!
//! The Taylor φ path (§62.4, ADR-0210) needs `Θ(τ‖G‖)` generator applications —
//! fine for `τ‖G‖ ≲ 10²`, costly for a stiff non-symmetric generator (a drifted
//! Fokker–Planck operator at `τ‖G‖ = 4·10³` costs 33 960 per combination, already
//! halved by the ADR-0210 shift). This path runs Arnoldi on the SAME η-scaled
//! augmented operator `B̃_η = [[τG, ηW], [0, J_p]]` and time-steps
//! `y(1) = e^{B̃_η}y(0)` with the adaptive substep control of Expokit's `expv`
//! (Sidje, ACM TOMS 24 (1998)): Krylov dimension `m`, local error from the `φ₁`/`φ₂`
//! corrector terms of the `(m+2)`-augmented Hessenberg exponential, step rejection
//! reuses the basis. For a fixed `m` each substep covers a bounded share of
//! `τ‖G‖`, so the application count still grows with `τ‖G‖` (4–5× per decade on the
//! gate operator) but from a far smaller constant: 1 023 against 33 960 at
//! `4·10³`. All control is pure arithmetic on deterministic quantities: the result
//! is reproducible bit for bit.

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    general_operator::GeneralOperator,
    generator_action::{CsrGenerator, GeneratorAction},
    phi_action::{eta_scaled_columns, validate_combination},
    phi_action_helpers::Coupling,
    scratch::ScratchPool,
};

/// Options of [`phi_combination_krylov`]; build with `default()` and `with_*`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct KrylovPhiOptions {
    /// Target accuracy relative to `‖y(0)‖₂` of the augmented start vector,
    /// `∈ (0, 1)`. Default `1e−10`.
    pub tol: f64,
    /// Krylov dimension per substep (clamped to `[2, n + p]`). Default 30.
    pub m: usize,
    /// Substep cap; exceeded → `ConvergenceFailed`. Default 100 000.
    pub max_substeps: usize,
}

impl Default for KrylovPhiOptions {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            m: 30,
            max_substeps: 100_000,
        }
    }
}

impl KrylovPhiOptions {
    /// Set the tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Set the Krylov dimension.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }
    /// Set the substep cap.
    #[must_use]
    pub fn with_max_substeps(mut self, max_substeps: usize) -> Self {
        self.max_substeps = max_substeps;
        self
    }
}

/// Diagnostics of one Krylov φ evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct KrylovPhiReport {
    /// Generator applications (Arnoldi steps plus one per substep for the error estimate).
    pub matvecs: usize,
    /// Accepted substeps.
    pub substeps: usize,
    /// Rejected substep sizes (the basis is reused).
    pub rejected: usize,
    /// Expokit's accumulated estimate `Σ max(errᵢ, ‖B̃_η‖·ε·‖yᵢ‖₂) / ‖y(0)‖₂` over the
    /// substeps (`errᵢ` the accepted local estimate, §68.2; a happy breakdown adds
    /// `β·h_{j+1,j}·Δt`). An estimate, not a bound: the round-off term dominates once
    /// `τ‖G‖·ε` approaches `tol` and can exceed `tol` by orders of magnitude.
    pub est_error: f64,
}

/// `out ← Σ_{k=0}^{p} τ^k φ_k(τG) w_k` (`p = w.len() − 1 ≤ 3`, the semantics of
/// [`crate::phi_action::phi_combination`]) by adaptive Arnoldi.
///
/// # Errors
/// The `DomainViolation`s of `phi_combination` (shapes, non-finite input, `τ < 0`,
/// overflow of `τ^k‖w_k‖`); `DomainViolation` for `tol ∉ (0, 1)`; `Unsupported`
/// for `F` other than `f32` / `f64` (forward-mode `Dual`);
/// `ConvergenceFailed` when the substep cap or the rejection cap (10 per substep)
/// is hit.
pub fn phi_combination_krylov<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    tau: F,
    w: &[&[F]],
    out: &mut [F],
    opts: KrylovPhiOptions,
    scratch: &mut ScratchPool<F>,
) -> Result<KrylovPhiReport, SemiflowError> {
    let n = op.dim();
    validate_combination(n, tau, w, out.len())?;
    // Arnoldi coefficients are f64 scalars: a forward-mode `Dual` (of f32 or f64)
    // would silently lose its tangent.
    if !crate::float::is_plain_ieee::<F>() {
        return Err(SemiflowError::Unsupported {
            feature: "phi_combination_krylov: forward-mode Dual (only f32 / f64)",
        });
    }
    if !(opts.tol > 0.0 && opts.tol < 1.0) {
        return Err(SemiflowError::DomainViolation {
            what: "phi_combination_krylov: tol must be in (0, 1)",
            value: opts.tol,
        });
    }
    let degree = w.len() - 1;
    let (cols, gains, inv_eta) = eta_scaled_columns(w, tau)?;
    let coupling = Coupling {
        cols: &cols[..degree],
        coef: &gains[..degree],
    };
    let mut y = scratch.take_vec(n + degree);
    y.fill(F::zero());
    y[..n].copy_from_slice(w[0]);
    if degree > 0 {
        y[n + degree - 1] = inv_eta;
    }
    let anorm =
        tau.to_f64().unwrap_or(f64::NAN) * op.norm_bound().to_f64().unwrap_or(f64::NAN) + 2.0;
    let aug = crate::phi_krylov_arnoldi::Augmented {
        op,
        coupling: &coupling,
        tau,
        n,
    };
    let result = crate::phi_krylov_arnoldi::expv(&aug, anorm, &mut y, opts, scratch);
    out.copy_from_slice(&y[..n]);
    scratch.return_vec(y);
    result
}

/// `out ← e^{−τA}v` for a general CSR operator `A` by adaptive Arnoldi
/// (`G = −A`; non-symmetric `A` welcome). Far fewer applications than the Taylor
/// `expmv` at large `τ‖A‖` (module docs; 2.3–12× faster in wall clock on a
/// tridiagonal stencil at `τ‖A‖ = 4·10³ … 4·10⁵`).
///
/// # Errors
/// As [`phi_combination_krylov`]; `DomainViolation` for a non-finite norm bound.
pub fn expmv_krylov<F: SemiflowFloat>(
    op: &GeneralOperator<F>,
    tau: F,
    v: &[F],
    out: &mut [F],
    opts: KrylovPhiOptions,
    scratch: &mut ScratchPool<F>,
) -> Result<KrylovPhiReport, SemiflowError> {
    let gen = CsrGenerator::from_general(op, None)?;
    phi_combination_krylov(&gen, tau, &[v], out, opts, scratch)
}

/// Which evaluator the φ consumers (ETDRK4) use.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum PhiSolver {
    /// Augmented Taylor sweeps (§62.4, ADR-0210): `Θ(τ‖G‖)`, exact counts. Default.
    #[default]
    Taylor,
    /// Adaptive Arnoldi (this module): far fewer applications for stiff `G`.
    Krylov(KrylovPhiOptions),
}

/// Dispatch one combination to the chosen evaluator.
///
/// # Errors
/// As the chosen evaluator.
pub(crate) fn combine<F: SemiflowFloat, Op: GeneratorAction<F>>(
    solver: &PhiSolver,
    op: &Op,
    tau: F,
    w: &[&[F]],
    out: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    match solver {
        PhiSolver::Taylor => crate::phi_action::phi_combination(op, tau, w, out, scratch),
        PhiSolver::Krylov(opts) => {
            phi_combination_krylov(op, tau, w, out, *opts, scratch).map(|_| ())
        }
    }
}

/// `‖x‖₂` in `f64`.
pub(crate) fn norm2<F: SemiflowFloat>(x: &[F]) -> f64 {
    libm::sqrt(dot(x, x))
}

/// `⟨x, y⟩` in `f64`, four interleaved partial sums (fixed order: deterministic).
pub(crate) fn dot<F: SemiflowFloat>(x: &[F], y: &[F]) -> f64 {
    let g = |a: F| a.to_f64().unwrap_or(f64::NAN);
    let mut acc = [0.0_f64; 4];
    let (xc, yc) = (x.chunks_exact(4), y.chunks_exact(4));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    for (a, b) in xc.zip(yc) {
        for l in 0..4 {
            acc[l] += g(a[l]) * g(b[l]);
        }
    }
    let tail: f64 = xr.iter().zip(yr).map(|(&a, &b)| g(a) * g(b)).sum();
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + tail
}

/// `y ← y + a·x`.
pub(crate) fn axpy<F: SemiflowFloat>(y: &mut [F], a: f64, x: &[F]) {
    let a: F = from_f64(a);
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[cfg(test)]
#[path = "phi_krylov_tests.rs"]
mod tests;
