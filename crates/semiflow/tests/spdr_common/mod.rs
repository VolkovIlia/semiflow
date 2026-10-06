//! Shared helpers for the ADR-0202 `G_SPDR_*` gate tests (test-local dense oracles).
#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

use semiflow::{
    assemble_conservative_csr_1d, boundary::BoundaryPolicy, grid::Grid1D, SemiflowError,
    SymmetricOperator,
};

/// 1-D conservative Neumann carrier on `[0,1]` with `k(x) = 1 + x²` (harmonic faces).
pub fn op_1d(n: usize) -> SymmetricOperator<f64> {
    let grid = Grid1D::new(0.0, 1.0, n).unwrap();
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 / (n - 1) as f64;
            1.0 + x * x
        })
        .collect();
    assemble_conservative_csr_1d(grid, &k, None, BoundaryPolicy::Neumann).unwrap()
}

/// Dense row-major copy of a CSR operator.
pub fn dense_of(op: &SymmetricOperator<f64>) -> Vec<f64> {
    let n = op.n();
    let (rp, ci, va) = op.csr();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for k in rp[i]..rp[i + 1] {
            a[i * n + ci[k] as usize] += va[k];
        }
    }
    a
}

/// `S = λ·diag(m) + A` (dense).
pub fn dense_shifted(op: &SymmetricOperator<f64>, lam: f64, mass: &[f64]) -> Vec<f64> {
    let n = op.n();
    let mut s = dense_of(op);
    for i in 0..n {
        s[i * n + i] += lam * mass[i];
    }
    s
}

/// Dense Gaussian elimination with partial pivoting.
pub fn solve_dense(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for c in 0..n {
        let piv = (c..n)
            .max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs()))
            .unwrap();
        for k in 0..n {
            m.swap(c * n + k, piv * n + k);
        }
        x.swap(c, piv);
        for r in c + 1..n {
            let f = m[r * n + c] / m[c * n + c];
            for k in c..n {
                m[r * n + k] -= f * m[c * n + k];
            }
            x[r] -= f * x[c];
        }
    }
    for r in (0..n).rev() {
        let mut s = x[r];
        for k in r + 1..n {
            s -= m[r * n + k] * x[k];
        }
        x[r] = s / m[r * n + r];
    }
    x
}

pub fn matvec(a: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
        .collect()
}

pub fn sup(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

pub fn rel_sup_err(got: &[f64], want: &[f64]) -> f64 {
    let d: Vec<f64> = got.iter().zip(want).map(|(a, b)| a - b).collect();
    sup(&d) / sup(want)
}

pub fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `‖b − S x‖₂ / ‖b‖₂` against a dense `S`.
pub fn dense_rel_residual(s: &[f64], x: &[f64], b: &[f64]) -> f64 {
    let sx = matvec(s, x);
    let r: Vec<f64> = b.iter().zip(&sx).map(|(bi, si)| bi - si).collect();
    norm2(&r) / norm2(b)
}

/// Half-cell lumped mass `[½, 1, …, 1, ½]`.
pub fn half_cell_mass(n: usize) -> Vec<f64> {
    let mut m = vec![1.0; n];
    m[0] = 0.5;
    m[n - 1] = 0.5;
    m
}

/// Deterministic pseudo-random vector in `[-1, 1]`.
pub fn lcg(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

pub fn is_domain(e: &SemiflowError) -> bool {
    matches!(e, SemiflowError::DomainViolation { .. })
}
