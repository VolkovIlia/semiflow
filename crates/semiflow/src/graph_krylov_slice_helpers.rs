// Slice-based `graph_expmv_krylov` — included at module scope of `graph_krylov.rs`.
//
// Works with any `SymmetricLinearOp<F>`; no `GraphSignal` or `Graph` required.
// Private helpers (chebyshev_series, lanczos_step_inner, etc.) are in scope
// because this file is `include!`d, not a separate module.

// ── Chebyshev branch (ADR-0205: one expansion, degree ~ √z) ──────────────────

/// `(s, m)` of the Chebyshev path: `s` equal substeps of degree `m`.
///
/// `s = 1` unless the single expansion would exceed `MAX_CHEB_DEGREE`
/// (`z ≳ 1.5·10¹⁰`), in which case `z` is halved until it fits (each substep then
/// gets `tol/s`). Returns `(1, 0)` for `z = 0` (`L = 0` or `τ = 0`: identity).
#[allow(clippy::cast_precision_loss)]
fn chebyshev_schedule(z: f64, tol: f64) -> (u32, usize) {
    if z.is_nan() || z <= 0.0 {
        return (1, 0);
    }
    let mut s = 1_u32;
    loop {
        let m = exp_chebyshev_degree(z / f64::from(s), tol / f64::from(s));
        if m < MAX_CHEB_DEGREE || s >= (1 << 30) {
            return (s, m);
        }
        s *= 2;
    }
}

/// `out ← e^{−τA}·v` by the Chebyshev series of `e^{−z(1+x)}`, `B = (2/λ)A − I`.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn expmv_chebyshev<F: SemiflowFloat, Op: SymmetricLinearOp<F> + ?Sized>(
    op: &Op,
    lambda_max: F,
    tau: F,
    v: &[F],
    out: &mut [F],
    tol: F,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = op.n();
    let z = (tau * lambda_max).to_f64().unwrap_or(f64::NAN) * 0.5;
    let tol_f = tol.to_f64().unwrap_or(1e-10);
    let (s, m) = chebyshev_schedule(z, tol_f);
    if m == 0 {
        out[..n].copy_from_slice(&v[..n]);
        return Ok(());
    }
    let mut coeffs_f64 = Vec::new();
    exp_chebyshev_coefficients(z / f64::from(s), tol_f / f64::from(s), &mut coeffs_f64);
    let mut coeffs = scratch.take_vec(coeffs_f64.len());
    for (c, &a) in coeffs.iter_mut().zip(&coeffs_f64) {
        *c = F::from(a).unwrap_or_else(F::zero);
    }
    let scale = F::from(2.0_f64).unwrap() / lambda_max;
    let mut bufs = [(); 4].map(|()| scratch.take_vec(n));
    let mut current = scratch.take_vec(n);
    current.copy_from_slice(&v[..n]);
    for _ in 0..s {
        let [t_prev, t_curr, spmv, result] = &mut bufs;
        chebyshev_series(op, &coeffs, scale, &current, t_prev, t_curr, spmv, result);
        core::mem::swap(&mut current, result);
    }
    let finite = current.iter().all(|x| x.is_finite());
    out[..n].copy_from_slice(&current);
    scratch.return_vec(current);
    scratch.return_vec(coeffs);
    for b in bufs {
        scratch.return_vec(b);
    }
    if finite {
        Ok(())
    } else {
        Err(SemiflowError::DomainViolation {
            what: "Chebyshev expmv: non-finite output (non-finite input or lambda_max bound)",
            value: z,
        })
    }
}

// ── Lanczos branch (ADR-0205: Hochbruck–Lubich a-priori schedule) ────────────

/// `out ← e^{−τA}·v` by `s` Lanczos steps of dimension `m` (see `lanczos_schedule`).
#[allow(clippy::too_many_arguments)]
fn expmv_lanczos<F: SemiflowFloat, Op: SymmetricLinearOp<F> + ?Sized>(
    op: &Op,
    lambda_max: F,
    tau: F,
    v: &[F],
    out: &mut [F],
    m_max: usize,
    tol: F,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = op.n();
    let (s, m) = lanczos_schedule(
        lambda_max.to_f64().unwrap_or(f64::NAN),
        tau.to_f64().unwrap_or(f64::NAN),
        tol.to_f64().unwrap_or(1e-10),
        m_max,
        n,
    );
    if s == u64::MAX {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "Lanczos expmv: tau*lambda_max needs more than 1e15 substeps at this m_max",
            value: (tau * lambda_max).to_f64().unwrap_or(f64::NAN),
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let step_tau = tau / F::from(s as f64).unwrap();
    let mut current = scratch.take_vec(n);
    let mut next = scratch.take_vec(n);
    current.copy_from_slice(&v[..n]);
    for _ in 0..s {
        lanczos_step_inner(op, &current, &mut next, step_tau, m, scratch)?;
        core::mem::swap(&mut current, &mut next);
    }
    out[..n].copy_from_slice(&current);
    scratch.return_vec(current);
    scratch.return_vec(next);
    Ok(())
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute `e^{−τA} · v` into `out` for any [`SymmetricLinearOp`].
///
/// Both `v` and `out` must have length `op.n()`.
///
/// # Errors
///
/// [`SemiflowError::DomainViolation`] if `tau` is negative or not finite, or the
/// operator's `lambda_max_bound()` is non-finite or negative (issue #44).
///
/// # Panics
///
/// Panics only if `F` cannot represent the constants `2.0` or `0.0` (impossible
/// for all standard IEEE-754 float types).
#[allow(clippy::too_many_arguments)]
pub fn graph_expmv_krylov<F, Op>(
    op: &Op,
    tau: F,
    v: &[F],
    out: &mut [F],
    path: KrylovPath,
    tol: F,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Op: SymmetricLinearOp<F>,
{
    validate_tau(tau)?;
    let lambda_max = op.lambda_max_bound();
    check_schedule_arg(lambda_max, tau)?;
    match path {
        KrylovPath::Chebyshev => expmv_chebyshev(op, lambda_max, tau, v, out, tol, scratch),
        KrylovPath::Lanczos { m_max } => {
            expmv_lanczos(op, lambda_max, tau, v, out, m_max, tol, scratch)
        }
        KrylovPath::ImplicitEuler { n_steps, cg_max_iter } => implicit_euler_action(
            op as &dyn SymmetricLinearOp<F>,
            v,
            out,
            tau,
            n_steps,
            tol,
            cg_max_iter,
            scratch,
        ),
    }
}
