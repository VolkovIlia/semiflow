//! Dependency-free preconditioned CG solver for `(I + Δt·Â) x = b` (§59, ADR-0190).
//!
//! ## Design
//!
//! [`pcg_shifted`] solves the shifted SPD system `S x = b` where `S = I + Δt·Â` by
//! preconditioned Conjugate Gradient, reusing `SymmetricLinearOp::apply_into_slice`.
//! No new dependency; no heap allocation inside the CG loop (scratch pool, §ADR-0041).
//!
//! [`implicit_euler_action`] builds the preconditioner ONCE per `(Â, Δt)` and loops
//! `n_steps` PCG solves to compute `(I + Δt·Â)^{-n_steps} · w0` (backward-Euler, §59.1).
//!
//! **SPD guarantee (§59.3):** `σ(I + Δt·Â) ⊂ [1, 1+Δt·λ_max] ⊂ (0,∞)` for every
//! `Δt > 0`, even when `Â` is singular — CG cannot break down.

use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    scratch::ScratchPool,
    symmetric_operator::SymmetricLinearOp,
};

// ── Preconditioner trait ──────────────────────────────────────────────────────

/// Applied once per CG iteration: `z ← P⁻¹ r` (§59.2).
///
/// Built once per `(Â, Δt)` and reused across all sub-steps and channels.
pub(crate) trait Preconditioner<F: SemiflowFloat> {
    fn apply(&self, r: &[F], z: &mut [F]);
}

// ── Jacobi preconditioner ─────────────────────────────────────────────────────

/// Diagonal preconditioner `P = diag(I + Δt·Â)` (§59.2 v1 default).
///
/// Built by N unit-vector matvecs; total cost = O(nnz).  For a tridiagonal
/// N=400 operator that is ~1200 multiply-adds — negligible versus `n_steps` solves.
pub(crate) struct Jacobi<F> {
    inv_diag: Vec<F>,
}

impl<F: SemiflowFloat> Jacobi<F> {
    /// Probe each standard basis vector to read diagonal entries of `Â`.
    ///
    /// `inv_diag[i] = 1 / (1 + Δt · Â[i,i])`.  If a pivot is non-positive
    /// (unexpected for SPD `S`) it is silently replaced by `1` (§59.2 fallback).
    pub(crate) fn build(
        op: &dyn SymmetricLinearOp<F>,
        dt: F,
        scratch: &mut ScratchPool<F>,
    ) -> Self {
        let n = op.n();
        let tiny = F::from(1e-300_f64).unwrap();
        let mut unit = scratch.take_vec(n);
        let mut col  = scratch.take_vec(n);
        let mut inv_diag = Vec::with_capacity(n);
        for i in 0..n {
            unit[i] = F::one();
            op.apply_into_slice(&unit, &mut col); // col = A·e_i
            let s_ii = F::one() + dt * col[i];
            let safe  = if s_ii > tiny { s_ii } else { F::one() };
            inv_diag.push(F::one() / safe);
            unit[i] = F::zero();
        }
        scratch.return_vec(unit);
        scratch.return_vec(col);
        Self { inv_diag }
    }
}

impl<F: SemiflowFloat> Preconditioner<F> for Jacobi<F> {
    fn apply(&self, r: &[F], z: &mut [F]) {
        for (zi, (ri, &pi)) in z.iter_mut().zip(r.iter().zip(self.inv_diag.iter())) {
            *zi = *ri * pi;
        }
    }
}

// ── BLAS-like helpers (no heap, all slice-based) ──────────────────────────────

fn vec_norm_sq<F: SemiflowFloat>(v: &[F]) -> F {
    v.iter().fold(F::zero(), |s, &x| s + x * x)
}

fn vec_dot<F: SemiflowFloat>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b.iter()).fold(F::zero(), |s, (&ai, &bi)| s + ai * bi)
}

/// `r ← b − (x + dt · A·x)` = `b − S·x`, using `sp` as temporary for `A·x`.
fn compute_residual<F: SemiflowFloat>(
    op: &dyn SymmetricLinearOp<F>,
    dt: F,
    b: &[F],
    x: &[F],
    sp: &mut [F],
    r: &mut [F],
) {
    op.apply_into_slice(x, sp);                    // sp = A·x
    for i in 0..x.len() {
        r[i] = b[i] - x[i] - dt * sp[i];          // r = b - S·x
    }
}

/// `sp ← p + dt · A·p` = `S·p` (one matvec).
fn shifted_matvec<F: SemiflowFloat>(
    op: &dyn SymmetricLinearOp<F>,
    dt: F,
    p: &[F],
    sp: &mut [F],
) {
    op.apply_into_slice(p, sp);                    // sp = A·p
    for i in 0..p.len() {
        sp[i] = p[i] + dt * sp[i];                // sp = S·p
    }
}

// ── PCG solver ────────────────────────────────────────────────────────────────

/// Solve `(I + dt·op) x = b` by preconditioned CG (§59.4).
///
/// `x` on entry: warm start (caller sets `x ← b`).
/// `x` on exit: solution or last iterate on failure.
///
/// Returns `Ok(iters)` on convergence; `Err(ConvergenceFailed)` otherwise.
/// Borrows four scratch vectors and releases them before returning — no allocation.
pub(crate) fn pcg_shifted<F: SemiflowFloat>(
    op: &dyn SymmetricLinearOp<F>,
    dt: F,
    b: &[F],
    x: &mut [F],
    precond: &dyn Preconditioner<F>,
    tol_cg: F,
    max_iter: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<usize, SemiflowError> {
    let n = op.n();
    let b_norm_sq = vec_norm_sq(b);
    let tiny = F::from(1e-300_f64).unwrap();
    if b_norm_sq < tiny {
        for xi in x.iter_mut() { *xi = F::zero(); }
        return Ok(0);
    }
    let tol_abs_sq = tol_cg * tol_cg * b_norm_sq;
    let mut r  = scratch.take_vec(n);
    let mut z  = scratch.take_vec(n);
    let mut p  = scratch.take_vec(n);
    let mut sp = scratch.take_vec(n);
    compute_residual(op, dt, b, x, &mut sp, &mut r);
    precond.apply(&r, &mut z);
    p.copy_from_slice(&z);
    let rz = vec_dot(&r, &z);
    let result = cg_loop(op, dt, x, tol_abs_sq, max_iter, precond,
                         &mut r, &mut z, &mut p, &mut sp, rz);
    scratch.return_vec(r);
    scratch.return_vec(z);
    scratch.return_vec(p);
    scratch.return_vec(sp);
    result
}

/// Inner CG iterate loop (extracted to keep `pcg_shifted` ≤ 50 lines).
fn cg_loop<F: SemiflowFloat>(
    op: &dyn SymmetricLinearOp<F>,
    dt: F,
    x: &mut [F],
    tol_abs_sq: F,
    max_iter: usize,
    precond: &dyn Preconditioner<F>,
    r: &mut Vec<F>,
    z: &mut Vec<F>,
    p: &mut Vec<F>,
    sp: &mut Vec<F>,
    mut rz: F,
) -> Result<usize, SemiflowError> {
    let tiny = F::from(1e-300_f64).unwrap();
    let mut last_r_sq = F::from(f64::INFINITY).unwrap_or(F::zero());
    for iter in 0..max_iter {
        shifted_matvec(op, dt, p, sp);             // sp = S·p
        let psp = vec_dot(p, sp);
        if psp <= F::zero() { break; }             // SPD breakdown (unexpected)
        let alpha = rz / psp;
        for (xi, &pi)  in x.iter_mut().zip(p.iter()) { *xi += alpha * pi; }
        for (ri, &si)  in r.iter_mut().zip(sp.iter()) { *ri -= alpha * si; }
        last_r_sq = vec_norm_sq(r);
        if last_r_sq <= tol_abs_sq { return Ok(iter + 1); }
        precond.apply(r, z);
        let rz_new = vec_dot(r, z);
        if rz_new.abs() < tiny { return Ok(iter + 1); }   // stagnated
        let beta = rz_new / rz;
        for (pi, &zi) in p.iter_mut().zip(z.iter()) { *pi = zi + beta * *pi; }
        rz = rz_new;
    }
    let last_residual = last_r_sq.sqrt().to_f64().unwrap_or(f64::NAN);
    Err(SemiflowError::ConvergenceFailed { last_residual, max_iter })
}

// ── Implicit-Euler action ─────────────────────────────────────────────────────

/// `dst ← (I + Δt·op)^{-n_steps} · src` (backward-Euler, §59.1).
///
/// Builds the Jacobi preconditioner once for the fixed `Δt = tau / n_steps`,
/// then loops `n_steps` PCG solves with warm start `x₀ ← u_k` each step.
///
/// # Errors
/// - [`SemiflowError::DomainViolation`] if `n_steps < 1`.
/// - [`SemiflowError::ConvergenceFailed`] if CG stalls within `max_iter`.
pub(crate) fn implicit_euler_action<F: SemiflowFloat>(
    op: &dyn SymmetricLinearOp<F>,
    src: &[F],
    dst: &mut [F],
    tau: F,
    n_steps: usize,
    tol: F,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if n_steps < 1 {
        return Err(SemiflowError::DomainViolation {
            what: "implicit_euler_action: n_steps must be >= 1",
            value: 0.0,
        });
    }
    let n      = op.n();
    let dt     = tau / F::from(n_steps as f64).unwrap();
    let tol_cg = tol.max(F::from(1e-12_f64).unwrap());
    let max_it = compute_max_iter(op, dt, n);
    let precond = Jacobi::build(op, dt, scratch);
    run_substeps(op, src, dst, dt, n_steps, tol_cg, max_it, &precond, scratch)
}

/// `max_iter = min(N, ceil(4·√(1 + Δt·λ_max)))` (§59.4, CG iteration budget).
fn compute_max_iter<F: SemiflowFloat>(op: &dyn SymmetricLinearOp<F>, dt: F, n: usize) -> usize {
    let lam  = op.lambda_max_bound().to_f64().unwrap_or(1.0);
    let dt_f = dt.to_f64().unwrap_or(0.0);
    let raw  = (4.0_f64 * (1.0 + dt_f * lam).sqrt()).ceil() as usize;
    n.min(raw).max(1)
}

/// Loop `n_steps` backward-Euler sub-steps; each solves `S · u_{k+1} = u_k`.
fn run_substeps<F: SemiflowFloat>(
    op: &dyn SymmetricLinearOp<F>,
    src: &[F],
    dst: &mut [F],
    dt: F,
    n_steps: usize,
    tol_cg: F,
    max_iter: usize,
    precond: &Jacobi<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = op.n();
    let mut u = scratch.take_vec(n);       // u_k — current iterate
    u.copy_from_slice(src);
    for _ in 0..n_steps {
        // x and u are independent scratch Vecs (separate pool allocations) — no aliasing.
        let mut x = scratch.take_vec(n);
        x.copy_from_slice(&u);             // warm start x₀ = b = u_k
        pcg_shifted(op, dt, &u, &mut x, precond, tol_cg, max_iter, scratch)?;
        core::mem::swap(&mut u, &mut x);   // u = solution; x = old u (returned to pool)
        scratch.return_vec(x);
    }
    dst.copy_from_slice(&u);
    scratch.return_vec(u);
    Ok(())
}
