// Private helpers for `graph_krylov.rs` — included via `include!` at module scope.
//
// All functions live in the `graph_krylov` module (not a child module), so
// all items visible in `graph_krylov.rs` are directly in scope here.

// ── Chebyshev series ──────────────────────────────────────────────────────────
//
// e^{-τA}v = Σ_{k=0}^m a_k · T_k(B)v,  B = (2/λ_max)·A − I,  z = τλ_max/2,
// a_0 = e^{-z}I_0(z), a_k = 2(−1)^k e^{-z}I_k(z)  (`crate::cheb_coeffs`).
// Recurrence: T_{k+1}(B)v = 2·B·T_k(B)v − T_{k-1}(B)v.

/// `result ← Σₖ aₖ Tₖ(B)·v` with `B = scale·A − I` (forward three-term recurrence).
#[allow(clippy::too_many_arguments)]
fn chebyshev_series<F: SemiflowFloat, Op: SymmetricLinearOp<F> + ?Sized>(
    op: &Op,
    a: &[F],
    scale: F,
    v: &[F],
    t_prev: &mut [F],
    t_curr: &mut [F],
    spmv: &mut [F],
    result: &mut [F],
) {
    let two = F::one() + F::one();
    for (r, &x) in result.iter_mut().zip(v) {
        *r = a[0] * x;
    }
    if a.len() < 2 {
        return;
    }
    // k=1: T_1(B)v = scale·A·v − v
    op.apply_into_slice(v, spmv);
    t_prev.copy_from_slice(v);
    for ((t, &s), &x) in t_curr.iter_mut().zip(spmv.iter()).zip(v) {
        *t = scale * s - x;
    }
    for (r, &t) in result.iter_mut().zip(t_curr.iter()) {
        *r += a[1] * t;
    }
    let (mut t_prev, mut t_curr) = (t_prev, t_curr);
    for &ak in &a[2..] {
        op.apply_into_slice(t_curr, spmv);
        // T_{k+1} into the T_{k-1} slot: 2·scale·A·T_k − 2·T_k − T_{k-1}
        for ((p, &s), &c) in t_prev.iter_mut().zip(spmv.iter()).zip(t_curr.iter()) {
            *p = two * scale * s - two * c - *p;
        }
        for (r, &p) in result.iter_mut().zip(t_prev.iter()) {
            *r += ak * p;
        }
        core::mem::swap(&mut t_prev, &mut t_curr);
    }
}

// ── Lanczos ───────────────────────────────────────────────────────────────────

/// Run the Lanczos three-term recurrence for up to `m` steps.
///
/// Fills `alpha[0..m]` and `beta[1..m]` (tridiagonal coefficients), stores
/// orthonormal Krylov basis into `q_basis` (column-major, stride `n`), and
/// returns `m_actual ≤ m` (early exits when an invariant subspace is found).
#[allow(clippy::too_many_arguments)]
fn lanczos_iterate<F: SemiflowFloat, Op: SymmetricLinearOp<F> + ?Sized>(
    op: &Op,
    q_curr: &mut [F],
    q_prev: &mut [F],
    z_buf: &mut [F],
    q_basis: &mut [F],
    alpha: &mut [F],
    beta: &mut [F],
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

/// One Lanczos step: `dst ≈ e^{−τA}·src` from an `m`-dimensional Krylov space.
///
/// `e^{−τT_m}e₁` comes from the eigen-decomposition of the tridiagonal `T_m`
/// (`lanczos_sched::tridiag_exp_e1`, in `f64`); the old dense Padé-13 on a
/// zero-padded 18×18 matrix cost more than the `m` mat-vecs it served.
fn lanczos_step_inner<F: SemiflowFloat, Op: SymmetricLinearOp<F> + ?Sized>(
    op: &Op,
    src: &[F],
    dst: &mut [F],
    tau: F,
    m: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = src.len();
    let m = m.min(n).max(1);
    let v_norm = src.iter().map(|&x| x * x).fold(F::zero(), |a, x| a + x).sqrt();
    if v_norm < F::from(1e-300_f64).unwrap() {
        dst.fill(F::zero());
        return Ok(());
    }
    let mut q_basis = scratch.take_vec(m * n);
    let mut q_prev = scratch.take_vec(n);
    let mut q_curr = scratch.take_vec(n);
    let mut z_buf = scratch.take_vec(n);
    let mut alpha = scratch.take_vec(m);
    let mut beta = scratch.take_vec(m + 1);

    let inv_v = F::one() / v_norm;
    for (q, &x) in q_curr.iter_mut().zip(src) {
        *q = x * inv_v;
    }
    q_basis[0..n].copy_from_slice(&q_curr);
    let m_actual = lanczos_iterate(
        op, &mut q_curr, &mut q_prev, &mut z_buf, &mut q_basis, &mut alpha, &mut beta, n, m,
    );
    let to64 = |x: F| x.to_f64().unwrap_or(f64::NAN);
    let a64: Vec<f64> = alpha[..m_actual].iter().map(|&x| to64(x)).collect();
    let b64: Vec<f64> = beta[1..m_actual].iter().map(|&x| to64(x)).collect();
    let y = crate::lanczos_sched::tridiag_exp_e1(&a64, &b64, to64(tau));
    if let Some(y) = &y {
        combine_basis(dst, &q_basis, y, v_norm, n);
    }
    for b in [q_basis, q_prev, q_curr, z_buf, alpha, beta] {
        scratch.return_vec(b);
    }
    y.map(|_| ()).ok_or(SemiflowError::DomainViolation {
        what: "Lanczos: tridiagonal eigen-decomposition did not converge",
        value: to64(tau),
    })
}

/// `dst = ‖v‖·Σ_k y_k q_k` over the stored Lanczos basis (row `k` of `q_basis`).
fn combine_basis<F: SemiflowFloat>(dst: &mut [F], q_basis: &[F], y: &[f64], v_norm: F, n: usize) {
    dst.fill(F::zero());
    for (k, &yk) in y.iter().enumerate() {
        let coeff = v_norm * F::from(yk).unwrap_or_else(F::zero);
        for (d, &q) in dst.iter_mut().zip(&q_basis[k * n..(k + 1) * n]) {
            *d += coeff * q;
        }
    }
}

// ── GraphSignal bridges (ChernoffFunction impl) ──────────────────────────────

/// Run a slice kernel on `src` and write the result into `dst`.
fn via_slices<F: SemiflowFloat>(
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
    kernel: impl FnOnce(&[F], &mut [F], &mut ScratchPool<F>) -> Result<(), SemiflowError>,
) -> Result<(), SemiflowError> {
    let mut out = scratch.take_vec(src.len());
    let r = kernel(src.values(), &mut out, scratch);
    if r.is_ok() {
        dst.zero_into();
        dst.axpy_into_slice(F::one(), &out);
    }
    scratch.return_vec(out);
    r
}
