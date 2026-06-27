// Private helpers for `graph_krylov.rs` — included via `include!` at module scope.
//
// All functions live in the `graph_krylov` module (not a child module), so
// all items visible in `graph_krylov.rs` are directly in scope here.

// ── Chebyshev substep infrastructure ─────────────────────────────────────────

/// Number of Chebyshev substeps so every substep's `z_sub ≤ Z_SAFE`.
///
/// Returns 1 for non-stiff operators.  For stiff ones returns `⌈z_total / Z_SAFE⌉`.
fn cheb_substep_count<F: SemiflowFloat>(z_total: F) -> u32 {
    let z_f64 = z_total.to_f64().unwrap_or(0.0);
    if !z_f64.is_finite() || z_f64 <= Z_SAFE {
        return 1;
    }
    let s_f64 = (z_f64 / Z_SAFE).ceil();
    if s_f64 > f64::from(u32::MAX) {
        return u32::MAX;
    }
    // s_f64 ∈ [2, u32::MAX] after the guards above — cast is exact.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    { s_f64 as u32 }
}

/// One Chebyshev substep: writes `e^{-step_tau·op} · current` into `result`.
///
/// Errors with [`SemiflowError::DomainViolation`] if the output is non-finite.
/// Under normal operation (`z_sub ≤ Z_SAFE`) this guard never fires.
#[allow(clippy::too_many_arguments)]
fn chebyshev_step<F: SemiflowFloat>(
    op: &impl SymmetricLinearOp<F>,
    current: &[F],
    t_prev: &mut Vec<F>,
    t_curr: &mut Vec<F>,
    spmv: &mut [F],
    result: &mut [F],
    n: usize,
    m: usize,
    scale: F,
    two: F,
    z_sub: F,
    em_z: F,
    z_total: F,
) -> Result<(), SemiflowError> {
    t_curr.copy_from_slice(current);
    let c0 = em_z * bessel_i_k(0, z_sub);
    for i in 0..n { result[i] = c0 * current[i]; }
    if m >= 1 {
        chebyshev_accumulate(
            op, current, t_prev, t_curr, spmv, result,
            n, m, scale, two, z_sub, em_z,
        );
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(SemiflowError::DomainViolation {
            what: "Chebyshev step: non-finite output (NaN/Inf) — \
                   this is a semiflow bug; please report with z_total",
            value: z_total.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// ── Chebyshev term accumulation ───────────────────────────────────────────────

/// Accumulate Chebyshev terms `T_k` for k=1..=m into `result` (called only when m ≥ 1).
///
/// Mutates the three Chebyshev work vectors and the `result` accumulator in place.
#[allow(clippy::too_many_arguments)]
fn chebyshev_accumulate<F: SemiflowFloat>(
    op: &impl SymmetricLinearOp<F>,
    src_v: &[F],
    t_prev: &mut Vec<F>,
    t_curr: &mut Vec<F>,
    spmv: &mut [F],
    result: &mut [F],
    n: usize,
    m: usize,
    scale: F,
    two: F,
    z: F,
    em_z: F,
) {
    // k=1: SpMV; T_1(B)v = (2/λ)·L·v − v
    op.apply_into_slice(src_v, spmv);
    t_prev.copy_from_slice(src_v); // t_prev = T_0 = v
    for i in 0..n { t_curr[i] = scale * spmv[i] - src_v[i]; }
    let c1 = -two * em_z * bessel_i_k(1, z);
    for i in 0..n { result[i] += c1 * t_curr[i]; }
    // k=2..=m: T_{k+1} = 2B·T_k − T_{k-1} = 2·scale·L·T_k − 2·T_k − T_{k-1}
    for k in 2..=m {
        op.apply_into_slice(t_curr, spmv);
        // Compute T_{k+1} in-place into t_prev (T_{k-1} slot)
        for i in 0..n {
            t_prev[i] = two * scale * spmv[i] - two * t_curr[i] - t_prev[i];
        }
        let sign = if k % 2 == 0 { F::one() } else { -F::one() };
        let ck = two * em_z * sign * bessel_i_k(k, z);
        for i in 0..n { result[i] += ck * t_prev[i]; }
        core::mem::swap(t_prev, t_curr); // advance: t_curr = T_{k+1}
    }
}

/// Run the Lanczos three-term recurrence for up to `m` steps.
///
/// Fills `alpha[0..m]` and `beta[1..m]` (tridiagonal coefficients), stores
/// orthonormal Krylov basis into `q_basis` (column-major, stride `n`), and
/// returns `m_actual ≤ m` (early exits when an invariant subspace is found).
#[allow(clippy::too_many_arguments)]
fn lanczos_iterate<F: SemiflowFloat>(
    op: &impl SymmetricLinearOp<F>,
    q_curr: &mut [F],
    q_prev: &mut [F],
    z_buf: &mut [F],
    q_basis: &mut [F],
    alpha: &mut [F; MAX_LANCZOS_DIM],
    beta: &mut [F; MAX_LANCZOS_DIM],
    n: usize,
    m: usize,
) -> usize {
    let mut m_actual = 0usize;
    for k in 0..m {
        op.apply_into_slice(q_curr, z_buf);
        alpha[k] = q_curr.iter().zip(z_buf.iter()).map(|(&a, &b)| a * b).fold(F::zero(), |s, x| s + x);
        for i in 0..n { z_buf[i] = z_buf[i] - alpha[k] * q_curr[i] - beta[k] * q_prev[i]; }
        let bk1 = z_buf.iter().map(|&x| x * x).fold(F::zero(), |s, x| s + x).sqrt();
        m_actual = k + 1;
        if bk1 < F::from(1e-14_f64).unwrap() { break; }
        beta[k + 1] = bk1;
        let inv_b = F::one() / bk1;
        for z in z_buf.iter_mut() { *z *= inv_b; }
        if k + 1 < m { q_basis[(k + 1) * n..(k + 2) * n].copy_from_slice(z_buf); }
        q_prev.copy_from_slice(q_curr);
        q_curr.copy_from_slice(z_buf);
    }
    m_actual
}
