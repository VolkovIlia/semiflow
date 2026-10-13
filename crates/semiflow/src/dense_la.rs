//! Small dense linear algebra shared by the kinetics (ADR-0208, ADR-0212): the
//! scaling-and-squaring Taylor-18 exponential (moved verbatim from `reaction.rs`,
//! bit-identical) and an in-place `LU` with partial pivoting for the per-node
//! `K×K` Rosenbrock systems.

use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
};

/// `e^{hC}` for a small dense row-major `K×K` matrix.
pub(crate) fn dense_exp<F: SemiflowFloat>(k: usize, c: &[F], h: F) -> Vec<F> {
    let norm = (0..k)
        .map(|col| (0..k).fold(F::zero(), |acc, row| acc + (c[row * k + col] * h).abs()))
        .fold(F::zero(), |a, b| if b > a { b } else { a });
    let mut squarings = 0_u32;
    let mut scale = F::one();
    while norm * scale > from_f64::<F>(0.5) && squarings < 1100 {
        scale *= from_f64::<F>(0.5);
        squarings += 1;
    }
    let a: Vec<F> = c.iter().map(|&v| v * h * scale).collect();
    let mut result = identity(k);
    let mut term = identity(k);
    for d in 1..=18_u32 {
        term = matmul(k, &term, &a);
        let inv = F::one() / from_f64::<F>(f64::from(d));
        for v in &mut term {
            *v *= inv;
        }
        for (r, &t) in result.iter_mut().zip(&term) {
            *r += t;
        }
    }
    for _ in 0..squarings {
        result = matmul(k, &result, &result);
    }
    result
}

pub(crate) fn identity<F: SemiflowFloat>(k: usize) -> Vec<F> {
    let mut m = vec![F::zero(); k * k];
    for i in 0..k {
        m[i * k + i] = F::one();
    }
    m
}

pub(crate) fn matmul<F: SemiflowFloat>(k: usize, a: &[F], b: &[F]) -> Vec<F> {
    let mut out = vec![F::zero(); k * k];
    for r in 0..k {
        for m in 0..k {
            let arm = a[r * k + m];
            for c in 0..k {
                out[r * k + c] += arm * b[m * k + c];
            }
        }
    }
    out
}

/// `PA = LU` of the row-major `k×k` matrix `a`, in place (`L` unit lower below the
/// diagonal, `U` on and above it); `piv[r]` is the original row now at `r`.
///
/// # Errors
/// `DomainViolation` on a zero or non-finite pivot (singular matrix).
#[allow(clippy::many_single_char_names)] // k, a, p, l, u: LU notation
pub(crate) fn lu_in_place<F: SemiflowFloat>(
    k: usize,
    a: &mut [F],
    piv: &mut [usize],
) -> Result<(), SemiflowError> {
    for (r, p) in piv.iter_mut().enumerate().take(k) {
        *p = r;
    }
    for col in 0..k {
        let mut best = col;
        for r in col + 1..k {
            if a[r * k + col].abs() > a[best * k + col].abs() {
                best = r;
            }
        }
        let p = a[best * k + col];
        if !p.is_finite() || p == F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "dense LU: singular or non-finite pivot",
                value: p.to_f64().unwrap_or(f64::NAN),
            });
        }
        if best != col {
            for c in 0..k {
                a.swap(best * k + c, col * k + c);
            }
            piv.swap(best, col);
        }
        for r in col + 1..k {
            let l = a[r * k + col] / p;
            a[r * k + col] = l;
            for c in col + 1..k {
                let u = a[col * k + c];
                a[r * k + c] -= l * u;
            }
        }
    }
    Ok(())
}

/// `b ← A⁻¹b` from [`lu_in_place`]'s factor; `work` holds `k` values.
pub(crate) fn lu_solve<F: SemiflowFloat>(
    k: usize,
    lu: &[F],
    piv: &[usize],
    b: &mut [F],
    work: &mut [F],
) {
    for (w, &p) in work.iter_mut().zip(piv).take(k) {
        *w = b[p];
    }
    for r in 0..k {
        for c in 0..r {
            let l = lu[r * k + c];
            let wc = work[c];
            work[r] -= l * wc;
        }
    }
    for r in (0..k).rev() {
        for c in r + 1..k {
            let u = lu[r * k + c];
            let wc = work[c];
            work[r] -= u * wc;
        }
        work[r] /= lu[r * k + r];
    }
    b[..k].copy_from_slice(&work[..k]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_exp_of_diagonal() {
        let e = dense_exp(2, &[-1.0, 0.0, 0.0, 2.0], 0.5);
        assert!((e[0] - (-0.5_f64).exp()).abs() < 1e-15 && (e[3] - 1.0_f64.exp()).abs() < 1e-14);
        assert!(e[1] == 0.0 && e[2] == 0.0);
    }

    #[test]
    fn lu_solves_and_rejects_singular() {
        let a = [0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 1.0];
        let (mut lu, mut piv, mut work) = (a, [0; 3], [0.0; 3]);
        lu_in_place(3, &mut lu, &mut piv).unwrap();
        let x = [1.0, -2.0, 0.5];
        let mut b: Vec<f64> = (0..3)
            .map(|r| (0..3).map(|c| a[r * 3 + c] * x[c]).sum())
            .collect();
        lu_solve(3, &lu, &piv, &mut b, &mut work);
        for (got, want) in b.iter().zip(x) {
            assert!((got - want).abs() < 1e-14, "{got} vs {want}");
        }
        let mut sing = [1.0, 2.0, 2.0, 4.0];
        assert!(lu_in_place(2, &mut sing, &mut [0; 2]).is_err());
    }
}
