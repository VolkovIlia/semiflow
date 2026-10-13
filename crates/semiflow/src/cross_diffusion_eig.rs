//! Small dense real eigen-decomposition for [`super::CrossDiffusion`] (ADR-0213):
//! Hessenberg reduction by stabilised elimination, Francis double-shift QR for the
//! eigenvalues (the classical `elmhes` / `hqr` pair, Wilkinson & Reinsch), and the
//! eigenvectors as null spaces of `D − λI` by Gaussian elimination with complete
//! pivoting. The caller verifies every decomposition (conditioning and
//! reconstruction), so this file only has to be right or report failure.

// The classical single-letter notation (`p, q, r, s, w, x, y, z`) is kept so the
// port reads against Wilkinson & Reinsch; the exact `a + s == s` tests are the
// algorithm's negligibility criterion.
#![allow(clippy::many_single_char_names, clippy::float_cmp)]

use alloc::{vec, vec::Vec};

use libm::{copysign, fabs, sqrt};

/// 1-based view of a row-major `n×n` matrix (the classical algorithms are 1-based).
struct M<'a> {
    a: &'a mut [f64],
    n: usize,
}

impl M<'_> {
    fn g(&self, i: usize, j: usize) -> f64 {
        self.a[(i - 1) * self.n + (j - 1)]
    }
    fn s(&mut self, i: usize, j: usize, v: f64) {
        self.a[(i - 1) * self.n + (j - 1)] = v;
    }
}

/// Reduce `a` to upper Hessenberg form in place (elimination with pivoting).
fn elmhes(m: &mut M<'_>) {
    let n = m.n;
    for p in 2..n {
        let (mut x, mut i) = (0.0_f64, p);
        for j in p..=n {
            if fabs(m.g(j, p - 1)) > fabs(x) {
                x = m.g(j, p - 1);
                i = j;
            }
        }
        if i != p {
            for j in p - 1..=n {
                let (u, v) = (m.g(i, j), m.g(p, j));
                m.s(i, j, v);
                m.s(p, j, u);
            }
            for j in 1..=n {
                let (u, v) = (m.g(j, i), m.g(j, p));
                m.s(j, i, v);
                m.s(j, p, u);
            }
        }
        if x != 0.0 {
            for i in p + 1..=n {
                let y = m.g(i, p - 1) / x;
                if y != 0.0 {
                    m.s(i, p - 1, y);
                    for j in p..=n {
                        m.s(i, j, m.g(i, j) - y * m.g(p, j));
                    }
                    for j in 1..=n {
                        m.s(j, p, m.g(j, p) + y * m.g(j, i));
                    }
                }
            }
        }
    }
}

/// Eigenvalues `(re, im)` of the row-major `n×n` matrix `d` (`None`: QR did not
/// converge in 60 iterations per eigenvalue).
pub(super) fn eigenvalues(d: &[f64], n: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut a = d.to_vec();
    let mut m = M { a: &mut a, n };
    elmhes(&mut m);
    for i in 3..=n {
        for j in 1..i - 1 {
            m.s(i, j, 0.0);
        }
    }
    hqr(&mut m)
}

/// Francis double-shift QR on an upper Hessenberg matrix (1-based port of `hqr`).
fn hqr(m: &mut M<'_>) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = m.n;
    let (mut wr, mut wi) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let mut anorm = 0.0;
    for i in 1..=n {
        for j in i.saturating_sub(1).max(1)..=n {
            anorm += fabs(m.g(i, j));
        }
    }
    let (mut nn, mut t) = (n, 0.0_f64);
    while nn >= 1 {
        let mut its = 0;
        loop {
            let l = small_subdiagonal(m, nn, anorm);
            let x = m.g(nn, nn);
            if l == nn {
                wr[nn] = x + t;
                wi[nn] = 0.0;
                nn -= 1;
            } else if l == nn - 1 {
                let roots = two_roots(m, nn, t);
                (wr[nn - 1], wr[nn], wi[nn - 1], wi[nn]) = roots;
                nn -= 2;
            } else {
                if its == 60 {
                    return None;
                }
                let shift = if its == 10 || its == 20 {
                    exceptional_shift(m, nn, &mut t)
                } else {
                    (x, m.g(nn - 1, nn - 1), m.g(nn, nn - 1) * m.g(nn - 1, nn))
                };
                its += 1;
                let (mm, pqr) = sweep_start(m, (l, nn), shift);
                sweep_apply(m, (l, nn), mm, pqr);
            }
            if l + 1 >= nn {
                break;
            }
        }
    }
    Some((wr[1..].to_vec(), wi[1..].to_vec()))
}

/// The largest `l ≤ nn` with a negligible subdiagonal `a[l][l−1]` (set to 0), or 1.
fn small_subdiagonal(m: &mut M<'_>, nn: usize, anorm: f64) -> usize {
    let mut l = nn;
    while l >= 2 {
        let mut s = fabs(m.g(l - 1, l - 1)) + fabs(m.g(l, l));
        if s == 0.0 {
            s = anorm;
        }
        if fabs(m.g(l, l - 1)) + s == s {
            m.s(l, l - 1, 0.0);
            break;
        }
        l -= 1;
    }
    l
}

/// The eigenvalues of the trailing `2×2` block: `(wr[nn−1], wr[nn], wi[nn−1], wi[nn])`.
fn two_roots(m: &M<'_>, nn: usize, t: f64) -> (f64, f64, f64, f64) {
    let (x, y) = (m.g(nn, nn), m.g(nn - 1, nn - 1));
    let w = m.g(nn, nn - 1) * m.g(nn - 1, nn);
    let p = 0.5 * (y - x);
    let q = p * p + w;
    let z = sqrt(fabs(q));
    let x = x + t;
    if q >= 0.0 {
        let z = p + copysign(z, p);
        let second = if z == 0.0 { x + z } else { x - w / z };
        (x + z, second, 0.0, 0.0)
    } else {
        (x + p, x + p, -z, z)
    }
}

/// The ad-hoc shift of iterations 10 and 20 (`t` accumulates the applied shift).
fn exceptional_shift(m: &mut M<'_>, nn: usize, t: &mut f64) -> (f64, f64, f64) {
    let x = m.g(nn, nn);
    *t += x;
    for i in 1..=nn {
        m.s(i, i, m.g(i, i) - x);
    }
    let s = fabs(m.g(nn, nn - 1)) + fabs(m.g(nn - 1, nn - 2));
    (0.75 * s, 0.75 * s, -0.4375 * s * s)
}

/// The start row `mm` of the double-shift sweep and its first reflector `(p, q, r)`.
fn sweep_start(
    m: &M<'_>,
    (l, nn): (usize, usize),
    (x, y, w): (f64, f64, f64),
) -> (usize, [f64; 3]) {
    let mut mm = nn - 2;
    loop {
        let z = m.g(mm, mm);
        let (rr, s) = (x - z, y - z);
        let p = (rr * s - w) / m.g(mm + 1, mm) + m.g(mm, mm + 1);
        let q = m.g(mm + 1, mm + 1) - z - rr - s;
        let r = m.g(mm + 2, mm + 1);
        let s = fabs(p) + fabs(q) + fabs(r);
        let pqr = [p / s, q / s, r / s];
        if mm == l {
            return (mm, pqr);
        }
        let u = fabs(m.g(mm, mm - 1)) * (fabs(pqr[1]) + fabs(pqr[2]));
        let v = fabs(pqr[0]) * (fabs(m.g(mm - 1, mm - 1)) + fabs(z) + fabs(m.g(mm + 1, mm + 1)));
        if u + v == v {
            return (mm, pqr);
        }
        mm -= 1;
    }
}

/// Chase the bulge from row `mm` down to `nn` with `3×3` Householder reflectors.
fn sweep_apply(m: &mut M<'_>, (l, nn): (usize, usize), mm: usize, [p0, q0, r0]: [f64; 3]) {
    for i in mm + 2..=nn {
        m.s(i, i - 2, 0.0);
        if i != mm + 2 {
            m.s(i, i - 3, 0.0);
        }
    }
    let (mut p, mut q, mut r, mut x) = (p0, q0, r0, 0.0);
    for k in mm..nn {
        if k != mm {
            p = m.g(k, k - 1);
            q = m.g(k + 1, k - 1);
            r = if k == nn - 1 { 0.0 } else { m.g(k + 2, k - 1) };
            x = fabs(p) + fabs(q) + fabs(r);
            if x != 0.0 {
                (p, q, r) = (p / x, q / x, r / x);
            }
        }
        let s = copysign(sqrt(p * p + q * q + r * r), p);
        if s == 0.0 {
            continue;
        }
        if k == mm {
            if l != mm {
                m.s(k, k - 1, -m.g(k, k - 1));
            }
        } else {
            m.s(k, k - 1, -s * x);
        }
        p += s;
        let v = [p / s, q / s, r / s];
        (q, r) = (q / p, r / p);
        reflect(m, (k, l, nn), v, (q, r));
    }
}

/// Apply one reflector `I − v·[1, q, r]ᵀ` to rows `k..k+2` (columns `k..nn`) and
/// to columns `k..k+2` (rows `l..min(nn, k+3)`).
fn reflect(
    m: &mut M<'_>,
    (k, l, nn): (usize, usize, usize),
    [x, y, z]: [f64; 3],
    (q, r): (f64, f64),
) {
    let three = k != nn - 1;
    for j in k..=nn {
        let mut p = m.g(k, j) + q * m.g(k + 1, j);
        if three {
            p += r * m.g(k + 2, j);
            m.s(k + 2, j, m.g(k + 2, j) - p * z);
        }
        m.s(k + 1, j, m.g(k + 1, j) - p * y);
        m.s(k, j, m.g(k, j) - p * x);
    }
    for i in l..=nn.min(k + 3) {
        let mut p = x * m.g(i, k) + y * m.g(i, k + 1);
        if three {
            p += z * m.g(i, k + 2);
            m.s(i, k + 2, m.g(i, k + 2) - p * r);
        }
        m.s(i, k + 1, m.g(i, k + 1) - p * q);
        m.s(i, k, m.g(i, k) - p);
    }
}

/// Basis of the null space of the row-major `n×n` matrix `a` (pivots `≤ tol` count
/// as zero), by Gaussian elimination with complete pivoting; unit 2-norm columns.
pub(super) fn null_space(a: &[f64], n: usize, tol: f64) -> Vec<Vec<f64>> {
    let mut u = a.to_vec();
    let mut cols: Vec<usize> = (0..n).collect();
    let mut rank = 0;
    while rank < n {
        let (mut bi, mut bj, mut best) = (rank, rank, 0.0_f64);
        for i in rank..n {
            for j in rank..n {
                if fabs(u[i * n + j]) > best {
                    (bi, bj, best) = (i, j, fabs(u[i * n + j]));
                }
            }
        }
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // a NaN pivot ends the elimination
        if !(best > tol) {
            break;
        }
        for c in 0..n {
            u.swap(rank * n + c, bi * n + c);
        }
        for r in 0..n {
            u.swap(r * n + rank, r * n + bj);
        }
        cols.swap(rank, bj);
        for i in rank + 1..n {
            let f = u[i * n + rank] / u[rank * n + rank];
            for c in rank..n {
                u[i * n + c] -= f * u[rank * n + c];
            }
        }
        rank += 1;
    }
    (rank..n)
        .map(|free| {
            let mut x = vec![0.0; n];
            x[free] = 1.0;
            for r in (0..rank).rev() {
                let s: f64 = (r + 1..n).map(|c| u[r * n + c] * x[c]).sum();
                x[r] = -s / u[r * n + r];
            }
            let norm = sqrt(x.iter().map(|v| v * v).sum::<f64>());
            let mut out = vec![0.0; n];
            for (j, &c) in cols.iter().enumerate() {
                out[c] = x[j] / norm;
            }
            out
        })
        .collect()
}
