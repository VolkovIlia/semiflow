//! Zero-fill incomplete Cholesky `IC(0)` preconditioner (ADR-0202, §62.2.b).
//!
//! `P = L̃L̃ᵀ` with `L̃` on exactly the lower pattern of `S = λ·diag(m) + A`
//! (Meijerink–van der Vorst 1977, Math. Comp. 31:148). `build` returns `None` on a
//! non-positive (or non-finite) pivot; the caller then falls back to Jacobi
//! (§59.2 rule) and reports it. Only `+ − × ÷` and `sqrt`, in fixed row order:
//! bit-identical `std`/`no_std` (ADR-0200).

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::{float::SemiflowFloat, pcg::Preconditioner};

/// Lower factor `L̃` in CSR; the diagonal is the LAST entry of each row.
#[derive(Clone)]
pub(crate) struct Ic0<F> {
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    vals: Vec<F>,
}

/// `Σ_k a[k]·b[k]` over matching columns of two sorted index/value lists.
fn sparse_dot<F: SemiflowFloat>(ca: &[u32], va: &[F], cb: &[u32], vb: &[F]) -> F {
    let (mut p, mut q, mut acc) = (0, 0, F::zero());
    while p < ca.len() && q < cb.len() {
        match ca[p].cmp(&cb[q]) {
            Ordering::Less => p += 1,
            Ordering::Greater => q += 1,
            Ordering::Equal => {
                acc += va[p] * vb[q];
                p += 1;
                q += 1;
            }
        }
    }
    acc
}

impl<F: SemiflowFloat> Ic0<F> {
    /// Factor `S = A + diag(shift)` (`A` given as CSR, sorted columns).
    ///
    /// Returns `None` if any pivot is `≤ 0` or non-finite.
    pub(crate) fn build(
        n: usize,
        row_ptr: &[usize],
        col_idx: &[u32],
        vals: &[F],
        shift: &[F],
    ) -> Option<Self> {
        let mut lp = Vec::with_capacity(n + 1);
        let mut lc: Vec<u32> = Vec::with_capacity(vals.len());
        let mut lv: Vec<F> = Vec::with_capacity(vals.len());
        lp.push(0_usize);
        for i in 0..n {
            let start = lc.len();
            let mut a_ii = shift[i];
            for k in row_ptr[i]..row_ptr[i + 1] {
                let j = col_idx[k] as usize;
                if j == i {
                    a_ii += vals[k];
                } else if j < i {
                    let (cj, vj) = (&lc[lp[j]..lp[j + 1] - 1], &lv[lp[j]..lp[j + 1] - 1]);
                    let dot = sparse_dot(&lc[start..], &lv[start..], cj, vj);
                    let l_jj = lv[lp[j + 1] - 1];
                    lc.push(col_idx[k]);
                    lv.push((vals[k] - dot) / l_jj);
                }
            }
            let sq: F = lv[start..].iter().fold(F::zero(), |s, &l| s + l * l);
            let pivot = a_ii - sq;
            if !(pivot > F::zero() && pivot.is_finite()) {
                return None;
            }
            #[allow(clippy::cast_possible_truncation)]
            lc.push(i as u32);
            lv.push(pivot.sqrt());
            lp.push(lc.len());
        }
        Some(Self {
            row_ptr: lp,
            col_idx: lc,
            vals: lv,
        })
    }
}

impl<F: SemiflowFloat> Preconditioner<F> for Ic0<F> {
    /// `z ← (L̃L̃ᵀ)⁻¹ r`: forward sweep `L̃y = r`, then backward sweep `L̃ᵀz = y`.
    fn apply(&self, rhs: &[F], out: &mut [F]) {
        let len = rhs.len();
        out.copy_from_slice(rhs);
        for i in 0..len {
            let (first, diag) = (self.row_ptr[i], self.row_ptr[i + 1] - 1);
            let mut acc = out[i];
            for k in first..diag {
                acc -= self.vals[k] * out[self.col_idx[k] as usize];
            }
            out[i] = acc / self.vals[diag];
        }
        for i in (0..len).rev() {
            let (first, diag) = (self.row_ptr[i], self.row_ptr[i + 1] - 1);
            let zi = out[i] / self.vals[diag];
            out[i] = zi;
            for k in first..diag {
                out[self.col_idx[k] as usize] -= self.vals[k] * zi;
            }
        }
    }
}
