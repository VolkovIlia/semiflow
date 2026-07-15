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
        // k+1 == beta.len() on the final iteration (k = MAX_LANCZOS_DIM-1): skip the
        // write — this slot is only ever read in the *next* iteration which does not
        // exist when k == m-1.  Skipping is safe and correct; it fixes the OOB panic
        // reported in fix/lanczos-stiff-oob.
        if k + 1 < beta.len() { beta[k + 1] = bk1; }
        let inv_b = F::one() / bk1;
        for z in z_buf.iter_mut() { *z *= inv_b; }
        if k + 1 < m { q_basis[(k + 1) * n..(k + 2) * n].copy_from_slice(z_buf); }
        q_prev.copy_from_slice(q_curr);
        q_curr.copy_from_slice(z_buf);
    }
    m_actual
}

// ── Modified Bessel I_k(z) — no_std power series ─────────────────────────────
//
// I_k(z) = Σ_{m=0}^∞ (z/2)^{2m+k} / (m! · (m+k)!)
// Term recurrence: term_{m+1} = term_m · (z/2)² / ((m+1)(m+k+1))

fn bessel_i_k<F: SemiflowFloat>(k: usize, z: F) -> F {
    if z < F::from(1e-300_f64).unwrap() {
        return if k == 0 { F::one() } else { F::zero() };
    }
    let hz = z / F::from(2.0_f64).unwrap();
    let hz2 = hz * hz;
    // Leading term (z/2)^k / k!
    // Loop indices are Bessel series indices bounded by degree (≤ 200) — precision loss impossible.
    #[allow(clippy::cast_precision_loss)]
    let mut term = {
        let mut t = F::one();
        for i in 1..=(k as u64) {
            t = t * hz / F::from(i as f64).unwrap();
        }
        t
    };
    let mut sum = term;
    #[allow(clippy::cast_precision_loss)]
    for m in 0u64..1000 {
        term = term * hz2
            / (F::from((m + 1) as f64).unwrap() * F::from((m + 1 + k as u64) as f64).unwrap());
        let next = sum + term;
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

// ── Implicit-Euler action bridging GraphSignal → slice → GraphSignal ─────────

/// Bridge `implicit_euler_action` (slice API) to the `GraphSignal` domain.
///
/// Writes to `dst` via `zero_into` + `axpy_into_slice` — the same pattern used
/// by `chebyshev_action` and `lanczos_action`, so no `values_mut` is needed.
// 8 args by necessity — op/src/dst/tau/n_steps/tol/cg_max_iter/scratch.
#[allow(clippy::too_many_arguments)]
fn implicit_euler_gk_action<F: SemiflowFloat>(
    op: &impl SymmetricLinearOp<F>,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    tau: F,
    n_steps: usize,
    tol: F,
    cg_max_iter: Option<usize>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = src.len();
    let mut out = scratch.take_vec(n);
    implicit_euler_action(
        op as &dyn SymmetricLinearOp<F>,
        src.values(),
        &mut out,
        tau,
        n_steps,
        tol,
        cg_max_iter,
        scratch,
    )?;
    dst.zero_into();
    dst.axpy_into_slice(F::one(), &out);
    scratch.return_vec(out);
    Ok(())
}

// ── Chebyshev and Lanczos semigroup actions ───────────────────────────────────
//
// Moved here from graph_krylov.rs to keep that file within the 500-line budget.

// 7 args by necessity — 4 op-state vars + tau/lambda_max/tol/scratch.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn chebyshev_action<F: SemiflowFloat>(
    op: &impl SymmetricLinearOp<F>,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    tau: F,
    lambda_max: F,
    tol: F,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = src.len();
    let z_total = tau * lambda_max / F::from(2.0_f64).unwrap();
    let s = cheb_substep_count(z_total);
    let step_tau = tau / F::from(f64::from(s)).unwrap(); // f64::from(u32) exact
    let z_sub = step_tau * lambda_max / F::from(2.0_f64).unwrap();
    let m = chebyshev_degree(z_sub, tol);
    let em_z = (-z_sub).exp();
    let scale = F::from(2.0_f64).unwrap() / lambda_max;
    let two = F::from(2.0_f64).unwrap();
    let mut t_prev = scratch.take_vec(n);
    let mut t_curr = scratch.take_vec(n);
    let mut spmv = scratch.take_vec(n);
    let mut result = scratch.take_vec(n);
    let mut current = scratch.take_vec(n);
    current.copy_from_slice(src.values());
    for _ in 0..s {
        chebyshev_step(
            op,
            &current,
            &mut t_prev,
            &mut t_curr,
            &mut spmv,
            &mut result,
            n,
            m,
            scale,
            two,
            z_sub,
            em_z,
            z_total,
        )?;
        core::mem::swap(&mut current, &mut result);
    }
    dst.zero_into();
    dst.axpy_into_slice(F::one(), &current);
    scratch.return_vec(t_prev);
    scratch.return_vec(t_curr);
    scratch.return_vec(spmv);
    scratch.return_vec(result);
    scratch.return_vec(current);
    Ok(())
}

/// One Lanczos step: `dst ≈ e^{-tau·A} · src` using m Krylov iterations.
#[allow(clippy::too_many_lines)]
fn lanczos_step_inner<F: SemiflowFloat>(
    op: &impl SymmetricLinearOp<F>,
    src: &[F],
    dst: &mut [F],
    tau: F,
    m: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = src.len();
    let m = m.min(n).min(MAX_LANCZOS_DIM);
    let v_norm = src
        .iter()
        .map(|&x| x * x)
        .fold(F::zero(), |a, x| a + x)
        .sqrt();
    if v_norm < F::from(1e-300_f64).unwrap() {
        for x in dst.iter_mut() {
            *x = F::zero();
        }
        return Ok(());
    }
    let mut q_basis = scratch.take_vec(m * n);
    let mut q_prev = scratch.take_vec(n);
    let mut q_curr = scratch.take_vec(n);
    let mut z_buf = scratch.take_vec(n);
    let mut alpha = [F::zero(); MAX_LANCZOS_DIM];
    let mut beta = [F::zero(); MAX_LANCZOS_DIM];

    // q_1 = v / ‖v‖; store as first basis column
    let inv_v = F::one() / v_norm;
    for i in 0..n {
        q_curr[i] = src[i] * inv_v;
    }
    q_basis[0..n].copy_from_slice(&q_curr);

    let m_actual = lanczos_iterate(
        op,
        &mut q_curr,
        &mut q_prev,
        &mut z_buf,
        &mut q_basis,
        &mut alpha,
        &mut beta,
        n,
        m,
    );

    // Reconstruct dst = Q_m · e^{-τ T_m} · (‖v‖ e_1)
    let exp_t = build_exp_tridiag(&alpha, &beta, tau, m_actual)?;
    for x in dst.iter_mut() {
        *x = F::zero();
    }
    for k in 0..m_actual {
        let coeff = v_norm * exp_t[k][0];
        let qk = &q_basis[k * n..(k + 1) * n];
        for i in 0..n {
            dst[i] += coeff * qk[i];
        }
    }

    scratch.return_vec(q_basis);
    scratch.return_vec(q_prev);
    scratch.return_vec(q_curr);
    scratch.return_vec(z_buf);
    Ok(())
}
