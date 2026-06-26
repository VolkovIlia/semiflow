// Slice-based `graph_expmv_krylov` — included at module scope of `graph_krylov.rs`.
//
// Works with any `SymmetricLinearOp<F>`; no `GraphSignal` or `Graph` required.
// Private helpers (chebyshev_accumulate, lanczos_step_inner, etc.) are in scope
// because this file is `include!`d, not a separate module.

// ── Chebyshev branch ─────────────────────────────────────────────────────────

fn expmv_chebyshev<F: SemiflowFloat, Op: SymmetricLinearOp<F>>(
    op: &Op,
    tau: F,
    v: &[F],
    out: &mut [F],
    tol: F,
    scratch: &mut ScratchPool<F>,
) {
    let n = op.n();
    let lambda_max = op.lambda_max_bound();
    let z = tau * lambda_max / F::from(2.0_f64).unwrap();
    let m = chebyshev_degree(z, tol);
    let em_z = (-z).exp();
    let scale = F::from(2.0_f64).unwrap() / lambda_max;
    let two = F::from(2.0_f64).unwrap();

    let mut t_prev = scratch.take_vec(n);
    let mut t_curr = scratch.take_vec(n);
    let mut spmv   = scratch.take_vec(n);
    let mut result = scratch.take_vec(n);

    t_curr.copy_from_slice(v);
    let c0 = em_z * bessel_i_k(0, z);
    for i in 0..n { result[i] = c0 * v[i]; }
    if m >= 1 {
        chebyshev_accumulate(op, v, &mut t_prev, &mut t_curr, &mut spmv, &mut result, n, m, scale, two, z, em_z);
    }
    out[..n].copy_from_slice(&result);

    scratch.return_vec(t_prev);
    scratch.return_vec(t_curr);
    scratch.return_vec(spmv);
    scratch.return_vec(result);
}

// ── Lanczos branch ────────────────────────────────────────────────────────────

fn expmv_lanczos<F: SemiflowFloat, Op: SymmetricLinearOp<F>>(
    op: &Op,
    tau: F,
    v: &[F],
    out: &mut [F],
    m_max: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = op.n();
    let lambda_max = op.lambda_max_bound();
    let (s, m) = lanczos_select_s_m(lambda_max, tau);
    let m = (m as usize).min(m_max);
    let step_tau = tau / F::from(f64::from(s)).unwrap();

    let mut current = scratch.take_vec(n);
    let mut next    = scratch.take_vec(n);
    current.copy_from_slice(v);
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
/// [`SemiflowError::DomainViolation`] if `tau` is negative or not finite.
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
    match path {
        KrylovPath::Chebyshev => {
            expmv_chebyshev(op, tau, v, out, tol, scratch);
        }
        KrylovPath::Lanczos { m_max } => {
            expmv_lanczos(op, tau, v, out, m_max, scratch)?;
        }
    }
    Ok(())
}
