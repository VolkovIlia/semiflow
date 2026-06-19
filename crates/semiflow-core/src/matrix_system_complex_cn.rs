// Included into `matrix_system_complex.rs` via `include!`.
// Contains the block-CN diffusion step and its helpers.
// DO NOT declare a module here — this file is included at file scope.

/// Context for the block-CN diffusion step (reduces argument count).
struct CnCtx<'a, C: SemiflowComplex, const M: usize> {
    ht: C,
    inv_dx2: C,
    inv_2dx: C,
    a_at_k: &'a [[[C; M]; M]],
    b_at_k: &'a [[[C; M]; M]],
    n: usize,
}

impl<C: SemiflowComplex, const M: usize> CnCtx<'_, C, M> {
    /// LHS blocks `(lower, center, upper)` at grid point `k`.
    #[allow(clippy::type_complexity)]
    fn lhs_blocks(&self, k: usize) -> ([[C; M]; M], [[C; M]; M], [[C; M]; M]) {
        let (a, b) = (&self.a_at_k[k], &self.b_at_k[k]);
        let mut lower = [[C::zero(); M]; M];
        let mut center = [[C::zero(); M]; M];
        let mut upper = [[C::zero(); M]; M];
        for i in 0..M {
            center[i][i] = C::one();
            for j in 0..M {
                let ca = a[i][j] * self.inv_dx2;
                let cb = b[i][j] * self.inv_2dx;
                lower[i][j] = -(self.ht * (ca - cb));
                center[i][j] += self.ht * (ca + ca);
                upper[i][j] = -(self.ht * (ca + cb));
            }
        }
        (lower, center, upper)
    }

    /// RHS (explicit part): `(I + τ/2 L^h) · u_in` at point `k`.
    fn rhs_at(&self, k: usize, u: &[C]) -> [C; M] {
        let (a, b) = (&self.a_at_k[k], &self.b_at_k[k]);
        let uk = read_point::<C, M>(u, k);
        let prev = if k == 0 { uk } else { read_point::<C, M>(u, k - 1) };
        let next = if k == self.n - 1 { uk } else { read_point::<C, M>(u, k + 1) };
        let mut out = uk;
        for i in 0..M {
            for j in 0..M {
                let ca = a[i][j] * self.inv_dx2;
                let cb = b[i][j] * self.inv_2dx;
                out[i] += self.ht * ((ca - cb) * prev[j] - (ca + ca) * uk[j] + (ca + cb) * next[j]);
            }
        }
        out
    }
}

/// Read M-component point at index `k` from flat slice.
fn read_point<C: SemiflowComplex, const M: usize>(u: &[C], k: usize) -> [C; M] {
    let mut v = [C::zero(); M];
    v.copy_from_slice(&u[k * M..k * M + M]);
    v
}

#[allow(clippy::too_many_arguments)]
fn complex_block_cn_diff_step<C: SemiflowComplex, const M: usize>(
    tau: C::Real,
    a_at_k: &[[[C; M]; M]],
    b_at_k: &[[[C; M]; M]],
    u_in: &[C],
    u_out: &mut [C],
    n: usize,
    dx: C::Real,
) -> Result<(), SemiflowError> {
    let two = C::Real::one() + C::Real::one();
    let ctx = CnCtx {
        ht: C::from_real(tau / two),
        inv_dx2: C::from_real(C::Real::one() / (dx * dx)),
        inv_2dx: C::from_real(C::Real::one() / (two * dx)),
        a_at_k,
        b_at_k,
        n,
    };
    let (d_prime, rhs_prime) = thomas_forward::<C, M>(&ctx, u_in)?;
    thomas_backward::<C, M>(&d_prime, &rhs_prime, u_out, n);
    Ok(())
}

/// Forward Thomas sweep; returns `(d_prime, rhs_prime)`.
#[allow(clippy::type_complexity)]
fn thomas_forward<C: SemiflowComplex, const M: usize>(
    ctx: &CnCtx<'_, C, M>,
    u_in: &[C],
) -> Result<(Vec<[[C; M]; M]>, Vec<[C; M]>), SemiflowError> {
    let n = ctx.n;
    let mut d_prime: Vec<[[C; M]; M]> = vec![[[C::zero(); M]; M]; n];
    let mut rhs_prime: Vec<[C; M]> = vec![[C::zero(); M]; n];
    {
        let (_sub, diag, sup) = ctx.lhs_blocks(0);
        let rhs0 = ctx.rhs_at(0, u_in);
        let inv_d = cblock_inv::<C, M>(&diag)?;
        d_prime[0] = cblock_mul::<C, M>(&inv_d, &sup);
        rhs_prime[0] = cblock_vec::<C, M>(&inv_d, &rhs0);
    }
    for k in 1..n {
        let (lower_k, diag_k, upper_k) = ctx.lhs_blocks(k);
        let rk = ctx.rhs_at(k, u_in);
        let elim_diag = cblock_mul::<C, M>(&lower_k, &d_prime[k - 1]);
        let mut nd = diag_k;
        for i in 0..M { for j in 0..M { nd[i][j] -= elim_diag[i][j]; } }
        let elim_rhs = cblock_vec::<C, M>(&lower_k, &rhs_prime[k - 1]);
        let mut nr = rk;
        for i in 0..M { nr[i] -= elim_rhs[i]; }
        let inv_nd = cblock_inv::<C, M>(&nd)?;
        if k < n - 1 { d_prime[k] = cblock_mul::<C, M>(&inv_nd, &upper_k); }
        rhs_prime[k] = cblock_vec::<C, M>(&inv_nd, &nr);
    }
    Ok((d_prime, rhs_prime))
}

/// Back substitution after Thomas forward sweep.
fn thomas_backward<C: SemiflowComplex, const M: usize>(
    d_prime: &[[[C; M]; M]],
    rhs_prime: &[[C; M]],
    u_out: &mut [C],
    n: usize,
) {
    let last = rhs_prime[n - 1];
    for i in 0..M { u_out[(n - 1) * M + i] = last[i]; }
    for k in (0..n - 1).rev() {
        let xnext = read_point::<C, M>(u_out, k + 1);
        let sub_x = cblock_vec::<C, M>(&d_prime[k], &xnext);
        for i in 0..M { u_out[k * M + i] = rhs_prime[k][i] - sub_x[i]; }
    }
}

#[inline]
fn cblock_mul<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
    b: &[[C; M]; M],
) -> [[C; M]; M] {
    let mut c = [[C::zero(); M]; M];
    for i in 0..M {
        for kk in 0..M {
            for j in 0..M {
                c[i][j] += a[i][kk] * b[kk][j];
            }
        }
    }
    c
}

#[inline]
fn cblock_vec<C: SemiflowComplex, const M: usize>(a: &[[C; M]; M], v: &[C; M]) -> [C; M] {
    let mut out = [C::zero(); M];
    for i in 0..M {
        for j in 0..M {
            out[i] += a[i][j] * v[j];
        }
    }
    out
}

fn cblock_inv<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
) -> Result<[[C; M]; M], SemiflowError> {
    crate::matrix_pade_complex::cmat_inv_complex_dispatch::<C, M>(a)
}
