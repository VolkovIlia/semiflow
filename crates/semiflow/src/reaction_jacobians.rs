//! Analytic Jacobians of the catalogue kinetics (ADR-0212), layout
//! `jac[(k·K + l)·N + i] = ∂f_k/∂u_l` at node `i` (species-major `u`).

#![allow(clippy::many_single_char_names)] // the models' own parameter names

use crate::float::{from_f64, SemiflowFloat};

/// Nagumo `f = u(1 − u)(u − a)`: `f′ = −3u² + 2(1 + a)u − a`.
pub(crate) fn nagumo<F: SemiflowFloat>(a: F, n: usize, u: &[F], jac: &mut [F]) {
    let (two, three) = (from_f64::<F>(2.0), from_f64::<F>(3.0));
    for (j, &v) in jac[..n].iter_mut().zip(&u[..n]) {
        *j = two * (F::one() + a) * v - three * v * v - a;
    }
}

/// Fill the four `2×2` blocks from `entry(i) = [∂₀f₀, ∂₁f₀, ∂₀f₁, ∂₁f₁]`.
fn two_species<F: SemiflowFloat>(n: usize, jac: &mut [F], entry: impl Fn(usize) -> [F; 4]) {
    for i in 0..n {
        for (b, v) in entry(i).into_iter().enumerate() {
            jac[b * n + i] = v;
        }
    }
}

/// Gray–Scott `(−uv² + f(1 − u), uv² − (f + k)v)`.
pub(crate) fn gray_scott<F: SemiflowFloat>(feed: F, kill: F, n: usize, u: &[F], jac: &mut [F]) {
    let two = from_f64::<F>(2.0);
    let (uu, vv) = u.split_at(n);
    two_species(n, jac, |i| {
        let (a, b) = (uu[i], vv[i]);
        [
            -(b * b) - feed,
            -(two * a * b),
            b * b,
            two * a * b - (feed + kill),
        ]
    });
}

/// FitzHugh–Nagumo `(v − v³/3 − w + I, ε(v + a − b·w))`.
pub(crate) fn fitzhugh_nagumo<F: SemiflowFloat>(eps: F, b: F, n: usize, u: &[F], jac: &mut [F]) {
    let v = &u[..n];
    two_species(n, jac, |i| {
        [F::one() - v[i] * v[i], -F::one(), eps, -(eps * b)]
    });
}

/// Brusselator `(a − (b + 1)u + u²v, bu − u²v)`.
pub(crate) fn brusselator<F: SemiflowFloat>(b: F, n: usize, u: &[F], jac: &mut [F]) {
    let two = from_f64::<F>(2.0);
    let (uu, vv) = u.split_at(n);
    two_species(n, jac, |i| {
        let (p, q) = (uu[i], vv[i]);
        let uv2 = two * p * q;
        [uv2 - (b + F::one()), p * p, b - uv2, -(p * p)]
    });
}
