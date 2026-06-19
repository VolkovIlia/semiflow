//! Generic-over-F septic-Hermite sampler (§46.5.bis, ADR-0133 Amendment 1).
//!
//! Mirrors `sample_septic_1d` (f64 path) EXACTLY: same 8 weight polynomials
//! (§40.3) and 3 central-FD stencils (§40.2), with `f64` literals promoted
//! via `F::from(·).unwrap_or_else(F::zero)` and `bc_value → bc_value_generic`.
//! No SIMD (§46.5 carve-out). The f64 path is untouched.
//!
//! Extracted into this child module to keep `grid_chebyshev_septic.rs` under
//! the 500-line file budget (suckless/constitution Override #1).

use num_traits::Float;

use crate::{
    float::SemiflowFloat,
    grid::{bc_value_generic, BoundaryPolicy, Grid1D},
};

// ---------------------------------------------------------------------------
// Literal-conversion helper (avoids repeating unwrap_or_else at each call-site).
// ---------------------------------------------------------------------------

// Deliberate inline(always): hot inner loop for float literal conversion in spectral kernels.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fc<F: SemiflowFloat>(v: f64) -> F {
    F::from(v).unwrap_or_else(F::zero)
}

// ---------------------------------------------------------------------------
// Generic weight polynomials — Horner form, §40.3 NORMATIVE.
// Each mirrors the corresponding f64 helper in the parent module exactly.
// ---------------------------------------------------------------------------

#[inline]
fn h_a0_g<F: SemiflowFloat>(s: F) -> F {
    let s2 = s * s;
    let s4 = s2 * s2;
    s4 * (((fc::<F>(20.0) * s - fc::<F>(70.0)) * s + fc::<F>(84.0)) * s - fc::<F>(35.0)) + F::one()
}

#[inline]
fn h_a1_g<F: SemiflowFloat>(s: F) -> F {
    let s2 = s * s;
    let s4 = s2 * s2;
    s + s4 * (((fc::<F>(10.0) * s - fc::<F>(36.0)) * s + fc::<F>(45.0)) * s - fc::<F>(20.0))
}

#[inline]
fn h_a2_g<F: SemiflowFloat>(s: F) -> F {
    let s2 = s * s;
    let s4 = s2 * s2;
    fc::<F>(0.5)
        * (s2 + s4 * (((fc::<F>(4.0) * s - fc::<F>(15.0)) * s + fc::<F>(20.0)) * s - fc::<F>(10.0)))
}

#[inline]
fn h_a3_g<F: SemiflowFloat>(s: F) -> F {
    let s2 = s * s;
    let s3 = s2 * s;
    fc::<F>(1.0 / 6.0)
        * s3
        * ((((s - fc::<F>(4.0)) * s + fc::<F>(6.0)) * s - fc::<F>(4.0)) * s + F::one())
}

#[inline]
fn h_b0_g<F: SemiflowFloat>(s: F) -> F {
    let s2 = s * s;
    let s4 = s2 * s2;
    s4 * (((-fc::<F>(20.0) * s + fc::<F>(70.0)) * s - fc::<F>(84.0)) * s + fc::<F>(35.0))
}

#[inline]
fn h_b1_g<F: SemiflowFloat>(s: F) -> F {
    let s2 = s * s;
    let s4 = s2 * s2;
    s4 * (((fc::<F>(10.0) * s - fc::<F>(34.0)) * s + fc::<F>(39.0)) * s - fc::<F>(15.0))
}

#[inline]
fn h_b2_g<F: SemiflowFloat>(s: F) -> F {
    let s2 = s * s;
    let s4 = s2 * s2;
    fc::<F>(0.5)
        * s4
        * (((-fc::<F>(4.0) * s + fc::<F>(13.0)) * s - fc::<F>(14.0)) * s + fc::<F>(5.0))
}

#[inline]
fn h_b3_g<F: SemiflowFloat>(s: F) -> F {
    let s2 = s * s;
    let s4 = s2 * s2;
    fc::<F>(1.0 / 6.0) * s4 * (((s - fc::<F>(3.0)) * s + fc::<F>(3.0)) * s - F::one())
}

// ---------------------------------------------------------------------------
// Generic FD stencils — same Fornberg coefficients as the f64 helpers.
// ---------------------------------------------------------------------------

/// Generic 8-pt scaled first derivative `dx·f'` (Fornberg 1988, k=1, N=9).
#[allow(clippy::similar_names)]
#[inline]
fn fd_prime_g<F: SemiflowFloat>(
    values: &[F],
    bnd: BoundaryPolicy<F>,
    n: usize,
    idx: i64,
    dx: F,
) -> F {
    let fm4 = bc_value_generic(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value_generic(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value_generic(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value_generic(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value_generic(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value_generic(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value_generic(bnd, values, n, idx + 4, dx);
    (fc::<F>(3.0) * fm4 - fc::<F>(32.0) * fm3 + fc::<F>(168.0) * fm2 - fc::<F>(672.0) * fm1
        + fc::<F>(672.0) * fp1
        - fc::<F>(168.0) * fp2
        + fc::<F>(32.0) * fp3
        - fc::<F>(3.0) * fp4)
        / fc::<F>(840.0)
}

/// Generic 7-pt scaled second derivative `dx²·f''` (Fornberg 1988, k=2, N=7).
#[allow(clippy::similar_names)]
#[inline]
fn fd_double_g<F: SemiflowFloat>(
    values: &[F],
    bnd: BoundaryPolicy<F>,
    n: usize,
    idx: i64,
    dx: F,
) -> F {
    let fm3 = bc_value_generic(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value_generic(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value_generic(bnd, values, n, idx - 1, dx);
    let f0 = bc_value_generic(bnd, values, n, idx, dx);
    let fp1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value_generic(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value_generic(bnd, values, n, idx + 3, dx);
    (fc::<F>(2.0) * fm3 - fc::<F>(27.0) * fm2 + fc::<F>(270.0) * fm1 - fc::<F>(490.0) * f0
        + fc::<F>(270.0) * fp1
        - fc::<F>(27.0) * fp2
        + fc::<F>(2.0) * fp3)
        / fc::<F>(180.0)
}

/// Generic 6-pt scaled third derivative `dx³·f'''` (Fornberg 1988, k=3, N=7).
#[allow(clippy::similar_names)]
#[inline]
fn fd_triple_g<F: SemiflowFloat>(
    values: &[F],
    bnd: BoundaryPolicy<F>,
    n: usize,
    idx: i64,
    dx: F,
) -> F {
    let fm3 = bc_value_generic(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value_generic(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value_generic(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value_generic(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value_generic(bnd, values, n, idx + 3, dx);
    (fm3 - fc::<F>(8.0) * fm2 + fc::<F>(13.0) * fm1 - fc::<F>(13.0) * fp1 + fc::<F>(8.0) * fp2
        - fp3)
        / fc::<F>(8.0)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Generic septic-Hermite sampler for `F: SemiflowFloat` (incl. `Dual<f64>`).
pub(crate) fn sample_septic_1d_generic<F: SemiflowFloat>(values: &[F], grid: &Grid1D<F>, x: F) -> F {
    let dx = grid.dx();
    let t_frac = (x - grid.xmin) / dx;
    let t_floor = Float::floor(t_frac);
    #[allow(clippy::cast_possible_truncation)]
    let idx = t_floor.to_i64().unwrap_or(0);
    let s = t_frac - t_floor;

    let bnd = grid.boundary;
    let n = grid.n;

    let v0 = bc_value_generic(bnd, values, n, idx, dx);
    let v1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let v0p = fd_prime_g(values, bnd, n, idx, dx);
    let v1p = fd_prime_g(values, bnd, n, idx + 1, dx);
    let v0pp = fd_double_g(values, bnd, n, idx, dx);
    let v1pp = fd_double_g(values, bnd, n, idx + 1, dx);
    let v0ppp = fd_triple_g(values, bnd, n, idx, dx);
    let v1ppp = fd_triple_g(values, bnd, n, idx + 1, dx);

    h_a0_g(s) * v0
        + h_a1_g(s) * v0p
        + h_a2_g(s) * v0pp
        + h_a3_g(s) * v0ppp
        + h_b0_g(s) * v1
        + h_b1_g(s) * v1p
        + h_b2_g(s) * v1pp
        + h_b3_g(s) * v1ppp
}
