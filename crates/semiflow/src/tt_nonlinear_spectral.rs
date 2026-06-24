//! S³ nonlinear POC — Cole-Hopf Burgers (Seam A) + Strang-split RD (Seam B).
//!
//! See: `contracts/s3-nonlinear-poc.contract.md` (NORMATIVE),
//!      `.dev-docs/specs/s3-nonlinear.md`, `docs/adr/0168-nonlinear-curse-escape.md`.
//!
//! ## Mathematics (ADR-0168)
//!
//! **Seam A** — Cole-Hopf Burgers (EXACT in time):
//! ```text
//! u_t = nu u_xx - u u_x
//! phi = exp(-Psi / 2nu),  Psi = antiderivative of (u0 - mean(u0))
//! phi_t = nu phi_xx  (LINEAR heat, exact via ADR-0164 spectral factor)
//! u = -2nu phi_x / phi
//! ```
//!
//! **Seam B** — Strang-split reaction-diffusion (order-2):
//! ```text
//! u_t = nu Lap u + f(u)
//! Phi(tau) = react(tau/2) . heat(tau) . react(tau/2)
//! ```
//!
//! ## Solver-free (Theorem-6 R2)
//!
//! NO `lu_solve_inplace`, NO `dense_expm` in the evolvers.

// Grid/FFT indices (usize) cast to F; all << 2^52.
// pub(crate) fns used only by the slow-tests gate (separate compilation unit).
#![allow(clippy::cast_precision_loss, dead_code)]
#![cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]

extern crate alloc;
#[cfg(feature = "s3-poc")]
use alloc::vec;
use alloc::vec::Vec;

use crate::{
    float::{from_f64, SemiflowFloat},
    tt_drift_spectral::apply_drift_spectral_axis,
    tt_spectral::{dft_1d_real_to_cplx, idft_1d_cplx},
};

// ═══════════════════════════════════════════════════════════════════════════
// §1 — Reaction enum and exact pointwise flow (Seam B)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "s3-poc")]
/// Closed-form-flow polynomial reactions ONLY (the construction-time wall).
///
/// Generic / transcendental `f` is UNREPRESENTABLE by this enum — the wall.
/// Each variant carries the EXACT pointwise flow of `du/ds = f(u)`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Reaction<F: SemiflowFloat> {
    /// `f(u) = r u (1-u)` (Fisher-KPP logistic). Exact flow:
    /// `u0 * exp(rs) / (1 - u0 + u0 * exp(rs))`. Domain (0,1).
    Logistic {
        /// Growth rate `r` (Fisher-KPP).
        r: F,
    },
    /// `f(u) = c u`. Exact flow: `u0 * exp(cs)`.
    /// `c=0` is the identity (reduction assert 5).
    Linear {
        /// Linear rate `c`.
        c: F,
    },
    /// `f(u) = a u^2 + b u + c`. Exact flow via discriminant.
    /// Spot-checked only (out-of-scope per contract §4).
    Quadratic {
        /// Quadratic coefficient.
        a: F,
        /// Linear coefficient.
        b: F,
        /// Constant coefficient.
        c: F,
    },
}

/// Apply the EXACT pointwise reaction flow `Phi_f(u, s)` elementwise.
///
/// NO solve, NO semigroup — closed form per variant.
#[cfg(feature = "s3-poc")]
pub(crate) fn react_flow<F: SemiflowFloat>(u: &mut [F], reaction: &Reaction<F>, s: F) {
    match reaction {
        Reaction::Logistic { r } => react_logistic(u, *r, s),
        Reaction::Linear { c } => react_linear(u, *c, s),
        Reaction::Quadratic { a, b, c } => react_quadratic_dispatch(u, *a, *b, *c, s),
    }
}

#[cfg(feature = "s3-poc")]
fn react_logistic<F: SemiflowFloat>(u: &mut [F], r: F, s: F) {
    let e = (r * s).exp();
    for ui in u.iter_mut() {
        debug_assert!(
            *ui > F::zero() && *ui < F::one(),
            "Logistic IC out of (0,1)"
        );
        *ui = *ui * e / (F::one() + *ui * (e - F::one()));
    }
}

#[cfg(feature = "s3-poc")]
fn react_linear<F: SemiflowFloat>(u: &mut [F], c: F, s: F) {
    let e = (c * s).exp();
    for ui in u.iter_mut() {
        *ui *= e;
    }
}

/// Dispatch for Quadratic: split by discriminant to keep each branch short.
#[cfg(feature = "s3-poc")]
#[allow(clippy::many_single_char_names)]
fn react_quadratic_dispatch<F: SemiflowFloat>(u: &mut [F], a: F, b: F, c: F, s: F) {
    let eps = from_f64::<F>(1e-12);
    let four = from_f64::<F>(4.0);
    if a.abs() < eps {
        react_quad_degenerate(u, b, c, s, eps);
        return;
    }
    let disc = b * b - four * a * c;
    if disc.abs() < eps {
        react_quad_repeated(u, a, b, s, eps);
    } else if disc > F::zero() {
        react_quad_real_roots(u, a, b, s, disc, eps);
    } else {
        react_quad_complex_roots(u, a, b, s, disc);
    }
}

#[cfg(feature = "s3-poc")]
#[allow(clippy::many_single_char_names)]
fn react_quad_degenerate<F: SemiflowFloat>(u: &mut [F], b: F, c: F, s: F, eps: F) {
    for ui in u.iter_mut() {
        if b.abs() < eps {
            *ui += c * s;
        } else {
            let e = (b * s).exp();
            *ui = *ui * e + c * (e - F::one()) / b;
        }
    }
}

#[cfg(feature = "s3-poc")]
#[allow(clippy::many_single_char_names)]
fn react_quad_repeated<F: SemiflowFloat>(u: &mut [F], a: F, b: F, s: F, eps: F) {
    let two = from_f64::<F>(2.0);
    let r0 = -b / (two * a);
    for ui in u.iter_mut() {
        let w0 = *ui - r0;
        if w0.abs() >= eps {
            *ui = F::one() / (F::one() / w0 + a * s) + r0;
        }
    }
}

#[cfg(feature = "s3-poc")]
fn react_quad_real_roots<F: SemiflowFloat>(u: &mut [F], a: F, b: F, s: F, disc: F, eps: F) {
    let two = from_f64::<F>(2.0);
    let sq = disc.sqrt();
    let rp = (-b + sq) / (two * a);
    let rm = (-b - sq) / (two * a);
    let eas = (a * (rp - rm) * s).exp();
    for ui in u.iter_mut() {
        let ratio_t = (((*ui) - rp) / ((*ui) - rm)) * eas;
        let denom = ratio_t - F::one();
        *ui = if denom.abs() < eps {
            rp
        } else {
            (ratio_t * rm - rp) / denom
        };
    }
}

#[cfg(feature = "s3-poc")]
#[allow(clippy::many_single_char_names)]
fn react_quad_complex_roots<F: SemiflowFloat>(u: &mut [F], a: F, b: F, s: F, disc: F) {
    let two = from_f64::<F>(2.0);
    let p = -b / (two * a);
    let q = (-disc).sqrt() / (two * a.abs());
    let aq = a * q;
    for ui in u.iter_mut() {
        let w0 = *ui - p;
        let theta = (w0 / q).atan() + aq * s;
        *ui = q * theta.tan() + p;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §2 — Seam B: Strang-split reaction-diffusion evolver
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "s3-poc")]
/// Seam-B configuration (avoids 8-arg function limit).
pub(crate) struct StrangConfig<'a, F: SemiflowFloat> {
    /// Grid size (same on every axis).
    pub(crate) n: usize,
    /// Number of spatial dimensions.
    pub(crate) d: usize,
    /// Grid spacing.
    pub(crate) dx: F,
    /// Diffusion coefficient `ν > 0`.
    pub(crate) nu: F,
    /// Reaction term (closed-form-flow enum).
    pub(crate) reaction: &'a Reaction<F>,
}

/// Evolve `u0` (flat `n^d` real) by `u_t = nu Lap u + f(u)` via Strang sandwich:
///
/// `react(tau/2) . heat(tau) . react(tau/2)`
///
/// Heat = ADR-0164 spectral factor per axis (`b=0`), solver-free.
/// React = exact closed-form pointwise flow.
/// Order-2 in `tau`. NO `lu_solve_inplace`, NO `dense_expm`.
#[cfg(feature = "s3-poc")]
pub(crate) fn strang_rd_evolve<F: SemiflowFloat>(
    u0: &[F],
    cfg: &StrangConfig<F>,
    tau: F,
    nsteps: usize,
) -> Vec<F> {
    let half = tau / from_f64(2.0);
    let mut u = u0.to_vec();
    for _ in 0..nsteps {
        react_flow(&mut u, cfg.reaction, half);
        apply_heat_all_axes(&mut u, cfg.n, cfg.d, cfg.dx, cfg.nu, tau);
        react_flow(&mut u, cfg.reaction, half);
    }
    u
}

/// Apply `exp(tau nu Lap)` per axis via spectral factor (b=0).
#[cfg(feature = "s3-poc")]
fn apply_heat_all_axes<F: SemiflowFloat>(u: &mut [F], n: usize, d: usize, dx: F, nu: F, tau: F) {
    let nd = n_pow(n, d);
    for axis in 0..d {
        let stride = n_pow(n, axis);
        let outer = nd / (stride * n);
        let mut line = vec![F::zero(); n];
        for i_out in 0..outer {
            for i_in in 0..stride {
                gather_line(u, &mut line, n, stride, i_out, i_in);
                let _ = apply_drift_spectral_axis(&mut line, n, dx, nu, F::zero(), tau);
                scatter_line(u, &line, n, stride, i_out, i_in);
            }
        }
    }
}

/// Gather a 1-D line along an axis from the flat `n^d` buffer.
#[cfg(feature = "s3-poc")]
fn gather_line<F: SemiflowFloat>(
    u: &[F],
    line: &mut [F],
    n: usize,
    stride: usize,
    i_out: usize,
    i_in: usize,
) {
    let outer_stride = stride * n;
    for (k, lk) in line.iter_mut().enumerate() {
        *lk = u[i_out * outer_stride + k * stride + i_in];
    }
}

/// Scatter a 1-D line back into the flat `n^d` buffer.
#[cfg(feature = "s3-poc")]
fn scatter_line<F: SemiflowFloat>(
    u: &mut [F],
    line: &[F],
    n: usize,
    stride: usize,
    i_out: usize,
    i_in: usize,
) {
    let outer_stride = stride * n;
    for (k, &lk) in line.iter().enumerate() {
        u[i_out * outer_stride + k * stride + i_in] = lk;
    }
}

/// Compute `n^exp` as usize (avoids repeated `n.pow(exp as u32)` casts).
#[cfg(feature = "s3-poc")]
fn n_pow(n: usize, exp: usize) -> usize {
    n.pow(u32::try_from(exp).expect("dimension fits u32"))
}

// ═══════════════════════════════════════════════════════════════════════════
// §3 — Seam A helpers: spectral derivative + antiderivative
// ═══════════════════════════════════════════════════════════════════════════

/// Angular wavenumber for DFT index `m` in a grid of size `n` with spacing `dx`.
///
/// Uses the DFT wrapping convention: `2pi * fftfreq(m, n) / dx`.
/// Positive frequencies for `m <= n/2`; negative for `m > n/2`.
fn angular_wavenumber<F: SemiflowFloat>(m: usize, n: usize, dx: F) -> F {
    let two_pi = from_f64::<F>(core::f64::consts::TAU);
    let n_f = from_f64::<F>(n as f64);
    // DFT wrapping: positive for m <= n/2, negative for m > n/2.
    // m < n <= 2^53 so casting to f64 is exact; subtraction in f64 is exact.
    let m_f = if m <= n / 2 {
        m as f64
    } else {
        m as f64 - n as f64
    };
    two_pi * from_f64::<F>(m_f) / (n_f * dx)
}

/// Spectral first derivative `d/dx` of a 1-D periodic real line.
///
/// `ifft(i k fft(line)).real` with correct negative-frequency wrapping.
/// Reuses 1-D DFT helpers from `tt_spectral`. NO solve.
pub(crate) fn spectral_deriv_1d<F: SemiflowFloat>(line: &[F], n: usize, dx: F) -> Vec<F> {
    let mut cplx = dft_1d_real_to_cplx(line);
    for m in 0..n {
        let k = angular_wavenumber(m, n, dx);
        let (re, im) = (cplx[2 * m], cplx[2 * m + 1]);
        cplx[2 * m] = -im * k;
        cplx[2 * m + 1] = re * k;
    }
    let rec = idft_1d_cplx(&cplx);
    (0..n).map(|i| rec[2 * i]).collect()
}

/// Spectral periodic antiderivative `Psi` with `Psi' = u` (zero mean required).
///
/// `Psi_hat[m] = u_hat[m] / (i k_m)` for `k_m != 0`; `Psi_hat[0] = 0`.
/// Requires `sum(u) = 0` (caller enforces via mean subtraction). NO solve.
pub(crate) fn spectral_antideriv_1d<F: SemiflowFloat>(u: &[F], n: usize, dx: F) -> Vec<F> {
    let mut cplx = dft_1d_real_to_cplx(u);
    cplx[0] = F::zero();
    cplx[1] = F::zero();
    for m in 1..n {
        let k = angular_wavenumber(m, n, dx);
        let (re, im) = (cplx[2 * m], cplx[2 * m + 1]);
        // Divide by i k: (re + i im) / (i k) = im/k - i re/k
        cplx[2 * m] = im / k;
        cplx[2 * m + 1] = -re / k;
    }
    let rec = idft_1d_cplx(&cplx);
    (0..n).map(|i| rec[2 * i]).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// §4 — Seam A: Cole-Hopf Burgers evolver (EXACT in time)
// ═══════════════════════════════════════════════════════════════════════════

/// Evolve viscous Burgers `u_t = nu u_xx - u u_x` (1-D periodic) EXACTLY in time:
///
/// 1. `u0 <- u0 - mean(u0)` (enforce zero mean)
/// 2. `Psi = spectral_antideriv_1d(u0)` (forward Cole-Hopf)
/// 3. `phi = exp(-Psi / (2 nu))` (pointwise)
/// 4. `phi <- apply_drift_spectral_axis(phi, nu, 0, T)` (linear heat, exact-in-time)
/// 5. `u = -2 nu * spectral_deriv_1d(phi) / phi` (back Cole-Hopf)
///
/// EXACT in time: for ANY substep partition the result is identical (semigroup).
/// NO `lu_solve_inplace`, NO `dense_expm`.
pub(crate) fn burgers_cole_hopf_evolve<F: SemiflowFloat>(
    u0: &[F],
    n: usize,
    dx: F,
    nu: F,
    t_final: F,
) -> Vec<F> {
    let mean = mean_val(u0);
    let u_zm: Vec<F> = u0.iter().map(|&x| x - mean).collect();
    let psi = spectral_antideriv_1d(&u_zm, n, dx);
    let two_nu = from_f64::<F>(2.0) * nu;
    let mut phi: Vec<F> = psi.iter().map(|&p| (-p / two_nu).exp()).collect();
    let _ = apply_drift_spectral_axis(&mut phi, n, dx, nu, F::zero(), t_final);
    let phi_x = spectral_deriv_1d(&phi, n, dx);
    phi_x
        .iter()
        .zip(phi.iter())
        .map(|(&px, &p)| -two_nu * px / p)
        .collect()
}

fn mean_val<F: SemiflowFloat>(u: &[F]) -> F {
    let sum = u.iter().copied().fold(F::zero(), |acc, x| acc + x);
    sum / from_f64(u.len() as f64)
}

// ═══════════════════════════════════════════════════════════════════════════
// §5 — Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "s3-poc"))]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    include!("tt_nonlinear_spectral_tests_mod.rs");
}
