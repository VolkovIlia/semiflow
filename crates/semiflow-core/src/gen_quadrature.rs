//! Generalized Gauss quadrature helpers (Golub-Welsch tridiagonal eigensolver).
//!
//! Provides `pub(crate)` quadrature rules used by [`crate::subordinated`]:
//! - [`sym_tridiag_eig`] — QL with Wilkinson shifts for n×n symmetric tridiagonal (n≤32).
//! - [`gen_laguerre_quadrature`] — generalized GL for weight `s^β e^{-s}` on (0,∞).
//! - [`gauss_legendre_interval`] — Gauss-Legendre on [a,b].
//! - [`ig_density_std`] — standardized Inverse-Gaussian density helper.

// Quadrature node indices and counts (usize) cast to f64 for Jacobi matrix entries; ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;

use alloc::vec::Vec;

// ─── Symmetric tridiagonal eigensolver (Golub-Welsch) ────────────────────────

/// QL with implicit Wilkinson shifts for a symmetric n×n tridiagonal (n ≤ 32).
///
/// Inputs: `diag[0..n]` = diagonal, `offdiag[0..n-1]` = sub-diagonal.
/// Outputs: `(eigenvalues ascending, first-row eigenvector components z)`.
/// Initial `z[0]=1`, rest 0 (first row of identity). Returns `Err(())` on
/// non-convergence (>30 iterations on any eigenpair). TQLI, Numerical Recipes §11.3.
// Symmetric tridiagonal QL eigensolver (Golub-Welsch / Numerical Recipes §11.3 TQLI).
// Single-char names (d, z, g, r, s, p, …) are standard; match the algorithm literature.
#[allow(clippy::many_single_char_names)]
pub(crate) fn sym_tridiag_eig(
    diag: [f64; 32],
    offdiag: [f64; 32],
    n: usize,
) -> Result<([f64; 32], [f64; 32]), ()> {
    let mut d = diag;
    let mut z = [0.0f64; 32];
    z[0] = 1.0;
    let mut en = [0.0f64; 33]; // en[i] = sub-diagonal above row i (NR: en[i]=offdiag[i-1])
    en[1..n].copy_from_slice(&offdiag[..(n - 1)]);
    for l in 0..n {
        ql_deflate_l(l, n, &mut d, &mut z, &mut en)?;
    }
    insertion_sort_eig(&mut d, &mut z, n);
    Ok((d, z))
}

/// Run the QL deflation loop for eigenvalue `l` (inner QL iteration until convergence).
#[allow(clippy::many_single_char_names)]
fn ql_deflate_l(
    l: usize,
    n: usize,
    d: &mut [f64; 32],
    z: &mut [f64; 32],
    en: &mut [f64; 33],
) -> Result<(), ()> {
    let mut niter = 0usize;
    'm_loop: loop {
        let m = ql_find_m(d, en, l, n);
        if m == l {
            return Ok(());
        }
        niter += 1;
        if niter > 30 {
            return Err(());
        }
        let mut g = (d[l + 1] - d[l]) / (2.0 * en[l + 1]);
        let r = (g * g + 1.0).sqrt();
        g = d[m] - d[l] + en[l + 1] / (g + if g >= 0.0 { r } else { -r });
        let (mut s, mut c, mut p) = (1.0f64, 1.0f64, 0.0f64);
        let mut i = m;
        loop {
            i -= 1;
            let (new_s, new_c, new_r, next_g, next_p, new_d_ip1) =
                ql_rotation_step(s, c, p, g, en, d, i);
            en[i + 2] = new_r;
            if new_r.abs() == 0.0 {
                d[i + 1] -= p;
                en[m + 1] = 0.0;
                continue 'm_loop;
            }
            (s, c, p, g, d[i + 1]) = (new_s, new_c, next_p, next_g, new_d_ip1);
            let zz = z[i + 1];
            z[i + 1] = s * z[i] + c * zz;
            z[i] = c * z[i] - s * zz;
            if i == l {
                break;
            }
        }
        d[l] -= p;
        en[l + 1] = g;
        en[m + 1] = 0.0;
    }
}

/// Find sub-diagonal index `m >= l` where `en[m+1]` is negligible vs `d` (convergence check).
#[allow(clippy::float_cmp)]
#[inline]
fn ql_find_m(d: &[f64; 32], en: &[f64; 33], l: usize, n: usize) -> usize {
    let mut m = l;
    while m < n - 1 {
        let dd = d[m].abs() + d[m + 1].abs();
        if en[m + 1].abs() + dd == dd {
            break;
        }
        m += 1;
    }
    m
}

/// One QL Givens-rotation step; returns `(s, c, r, next_g, next_p, new_d_ip1)`.
///
/// `r.abs() == 0.0` signals the zero-pivot branch (caller continues outer loop).
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
#[inline]
fn ql_rotation_step(
    s: f64,
    c: f64,
    p: f64,
    g: f64,
    en: &[f64; 33],
    d: &[f64; 32],
    i: usize,
) -> (f64, f64, f64, f64, f64, f64) {
    let f = s * en[i + 1];
    let b = c * en[i + 1];
    let r = (f * f + g * g).sqrt();
    if r.abs() == 0.0 {
        return (s, c, r, g, p, d[i + 1]);
    }
    let new_s = f / r;
    let new_c = g / r;
    let gg = d[i + 1] - p;
    let rr = (d[i] - gg) * new_s + 2.0 * new_c * b;
    let next_p = new_s * rr;
    (new_s, new_c, r, new_c * rr - b, next_p, gg + next_p)
}

/// Insertion sort for eigenvalues `d` and eigenvector components `z` (n ≤ 32).
#[inline]
fn insertion_sort_eig(d: &mut [f64; 32], z: &mut [f64; 32], n: usize) {
    for i in 1..n {
        let (di, zi) = (d[i], z[i]);
        let mut j = i;
        while j > 0 && d[j - 1] > di {
            d[j] = d[j - 1];
            z[j] = z[j - 1];
            j -= 1;
        }
        d[j] = di;
        z[j] = zi;
    }
}

// ─── Generalized Gauss-Laguerre quadrature ───────────────────────────────────

/// Generalized Gauss-Laguerre for weight `s^β e^{-s}` on (0,∞), β > -1, n ≤ 32.
///
/// Jacobi tridiagonal: `diag[k] = 2k + β + 1`, `offdiag[k] = sqrt(k(k+β))` for k≥1.
/// Returns `(nodes, weights)` where `weights_k = Γ(β+1) · v_{0,k}²`.
/// Falls back to the GL32 hardcoded table (β=0) on eigensolver failure.
pub(crate) fn gen_laguerre_quadrature(n: usize, beta: f64) -> (Vec<f64>, Vec<f64>) {
    debug_assert!((1..=32).contains(&n));
    let mut d = [0.0f64; 32];
    let mut e = [0.0f64; 32];
    for (k, d_k) in d[..n].iter_mut().enumerate() {
        *d_k = 2.0 * k as f64 + beta + 1.0;
    }
    for k in 1..n {
        e[k - 1] = ((k as f64) * (k as f64 + beta)).max(0.0).sqrt();
    }
    let mu0 = libm::tgamma(beta + 1.0);
    if let Ok((nodes, z)) = sym_tridiag_eig(d, e, n) {
        let nodes: Vec<f64> = nodes[..n].to_vec();
        let weights: Vec<f64> = z[..n].iter().map(|&vi| mu0 * vi * vi).collect();
        (nodes, weights)
    } else {
        let nodes: Vec<f64> = crate::resolvent_quad::GL32_NODES[..n].to_vec();
        let weights: Vec<f64> = crate::resolvent_quad::GL32_WEIGHTS[..n].to_vec();
        (nodes, weights)
    }
}

// ─── Gauss-Legendre quadrature on [a,b] ──────────────────────────────────────

/// Gauss-Legendre rule on [a,b], n ≤ 32.
///
/// Jacobi matrix for Legendre weight on [-1,1]: `diag[k]=0`,
/// `offdiag[k] = k / sqrt(4k²-1)`. Maps nodes and weights to [a,b].
// a, b, d, e, xi, z are standard GL quadrature names; single-char names match the literature.
#[allow(clippy::many_single_char_names)]
pub(crate) fn gauss_legendre_interval(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    debug_assert!((1..=32).contains(&n) && a < b);
    let d = [0.0f64; 32];
    let mut e = [0.0f64; 32];
    for k in 1..n {
        let kf = k as f64;
        e[k - 1] = kf / (4.0 * kf * kf - 1.0).sqrt();
    }
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    if let Ok((xi, z)) = sym_tridiag_eig(d, e, n) {
        let nodes: Vec<f64> = xi[..n].iter().map(|&x| mid + half * x).collect();
        let weights: Vec<f64> = z[..n].iter().map(|&vi| 2.0 * half * vi * vi).collect();
        (nodes, weights)
    } else {
        let h = (b - a) / n as f64;
        let nodes = (0..n).map(|k| a + (k as f64 + 0.5) * h).collect();
        let weights = vec![(b - a) / n as f64; n];
        (nodes, weights)
    }
}

// ─── Inverse-Gaussian density helper ─────────────────────────────────────────

/// Standardized Inverse-Gaussian density `f(v; kappa)` for `v = s/mean`.
///
/// `f(v; κ) = sqrt(κ/2π) · v^{-3/2} · exp(−κ(v−1)²/(2v))`, v > 0.
/// Used by [`crate::subordinated::InverseGaussianSubordinator`] quadrature.
#[inline]
pub(crate) fn ig_density_std(v: f64, kappa: f64) -> f64 {
    if v <= 0.0 || !v.is_finite() {
        return 0.0;
    }
    let exponent = -kappa * (v - 1.0) * (v - 1.0) / (2.0 * v);
    if exponent < -700.0 {
        return 0.0;
    }
    (kappa / (2.0 * core::f64::consts::PI)).sqrt() * v.powf(-1.5) * libm::exp(exponent)
}
