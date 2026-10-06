//! Shared dense oracles for the ADR-0202 φ gates (test-only; not a test binary).
//!
//! Independent of the library: scaling-and-squaring + Taylor on the dense
//! `(n+p)` augmented matrix of Al-Mohy–Higham 2011 Thm 2.1, plus the DST-I
//! eigen-exact scalar `φ_k` used by `G_PHI_AUG_DENSE`.
#![allow(dead_code, clippy::cast_precision_loss, clippy::many_single_char_names)]

/// Row-major `a·b`, both `d×d`.
fn matmul(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; d * d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i * d + k];
            for j in 0..d {
                c[i * d + j] += aik * b[k * d + j];
            }
        }
    }
    c
}

/// Dense `exp(m)` (row-major, `d×d`) by scaling-and-squaring with a 26-term Taylor kernel.
pub fn dense_expm(m: &[f64], d: usize) -> Vec<f64> {
    let norm = (0..d)
        .map(|i| (0..d).map(|j| m[i * d + j].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let mut s = 0_i32;
    while norm / 2.0_f64.powi(s) > 0.5 {
        s += 1;
    }
    let a: Vec<f64> = m.iter().map(|x| x / 2.0_f64.powi(s)).collect();
    let mut term = vec![0.0_f64; d * d];
    for i in 0..d {
        term[i * d + i] = 1.0;
    }
    let mut sum = term.clone();
    for k in 1..=26_u32 {
        term = matmul(&term, &a, d)
            .iter()
            .map(|x| x / f64::from(k))
            .collect();
        for (si, ti) in sum.iter_mut().zip(&term) {
            *si += ti;
        }
    }
    for _ in 0..s {
        sum = matmul(&sum, &sum, d);
    }
    sum
}

/// `Σ_{k=1}^{p} φ_k(τG) wt[k-1]  +  e^{τG} w0` via the dense augmented exponential.
///
/// `g` is the row-major `n×n` generator. `wt[k-1]` is the already-weighted
/// coefficient of `φ_k` (the caller applies any `τ^k`). `p = wt.len()`.
pub fn dense_phi_comb(g: &[f64], n: usize, tau: f64, w0: &[f64], wt: &[Vec<f64>]) -> Vec<f64> {
    let p = wt.len();
    let d = n + p;
    let mut b = vec![0.0_f64; d * d];
    for i in 0..n {
        for j in 0..n {
            b[i * d + j] = tau * g[i * n + j];
        }
    }
    for col in 0..p {
        // Column `col` of the coupling block carries the coefficient of φ_{p-col}.
        for i in 0..n {
            b[i * d + n + col] = wt[p - col - 1][i];
        }
    }
    for j in 0..p.saturating_sub(1) {
        b[(n + j) * d + n + j + 1] = 1.0;
    }
    let e = dense_expm(&b, d);
    let mut y0 = vec![0.0_f64; d];
    y0[..n].copy_from_slice(w0);
    if p > 0 {
        y0[n + p - 1] = 1.0;
    }
    (0..n)
        .map(|i| (0..d).map(|j| e[i * d + j] * y0[j]).sum())
        .collect()
}

/// `φ_k(τG)·v` alone (no `τ^k`), `k ≤ 3`.
pub fn dense_phi_k(g: &[f64], n: usize, tau: f64, k: usize, v: &[f64]) -> Vec<f64> {
    let zero = vec![0.0_f64; n];
    if k == 0 {
        return dense_phi_comb(g, n, tau, v, &[]);
    }
    let wt: Vec<Vec<f64>> = (1..=k)
        .map(|j| if j == k { v.to_vec() } else { zero.clone() })
        .collect();
    dense_phi_comb(g, n, tau, &zero, &wt)
}

/// Deterministic pseudo-random vector in `[-1, 1)` (LCG, seeded).
pub fn lcg_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// `max_i |a_i − b_i|`.
pub fn sup_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// `max_i |a_i|`.
pub fn sup(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Scalar `φ_k(z)` by its convergent Taylor series (no cancellation for z ≤ 0 of moderate size).
pub fn scalar_phi_k(k: usize, z: f64) -> f64 {
    let mut fact = 1.0_f64;
    for i in 1..=k {
        fact *= i as f64;
    }
    let mut term = 1.0 / fact;
    let mut sum = term;
    for n in 1_usize..120 {
        term *= z / (n + k) as f64;
        sum += term;
        if term.abs() < 1e-18 * (sum.abs() + 1e-300) {
            break;
        }
    }
    sum
}

/// Eigen-exact `φ_k(τA)·v`, `A = offdiag·tridiag(1,−2,1)` (DST-I spectrum).
pub fn eigen_exact_phi_k(k: usize, tau: f64, offdiag: f64, v: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let n = v.len();
    let norm = (2.0 / (n + 1) as f64).sqrt();
    let mut out = vec![0.0_f64; n];
    for j in 1..=n {
        let ang = j as f64 * PI / (n + 1) as f64;
        let tau_lam = tau * offdiag * (-2.0 + 2.0 * ang.cos());
        let inner: f64 = (1..=n)
            .map(|i| norm * (i as f64 * ang).sin() * v[i - 1])
            .sum();
        let phi_val = scalar_phi_k(k, tau_lam);
        for i in 1..=n {
            out[i - 1] += phi_val * inner * norm * (i as f64 * ang).sin();
        }
    }
    out
}
