//! Generic-over-F octonic-Hermite sampler (§46.5.bis, ADR-0139).
//!
//! Mirrors `sample_octonic_1d` (f64 path) EXACTLY: same 10 weight polynomials
//! (§41.bis, ADR-0117) and 4 central-FD stencils, with `f64` literals promoted
//! via `F::from(·).unwrap_or_else(F::zero)` and `bc_value → bc_value_generic`.
//! No SIMD (§46.5 carve-out). The f64 path is untouched (byte-identical, G1).
//!
//! Extracted into this child module to keep `grid_chebyshev_octonic.rs` under
//! the 500-line file budget (suckless/constitution Override #1).

use num_traits::Float;

use crate::{
    float::SemiflowFloat,
    grid::{bc_value_generic, BoundaryPolicy, Grid1D},
};

// ---------------------------------------------------------------------------
// Literal-conversion helper (mirrors septic_generic.rs fc::<F>).
// ---------------------------------------------------------------------------

// Deliberate inline(always): hot inner loop for float literal conversion in spectral kernels.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fc<F: SemiflowFloat>(v: f64) -> F {
    F::from(v).unwrap_or_else(F::zero)
}

// ---------------------------------------------------------------------------
// Generic weight polynomials — Horner form, §41.bis NORMATIVE.
// Each mirrors the corresponding f64 helper in grid_chebyshev_octonic.rs.
// ---------------------------------------------------------------------------

#[inline]
fn h_a0_g<F: SemiflowFloat>(s: F) -> F {
    // -70s^9 + 315s^8 - 540s^7 + 420s^6 - 126s^5 + 1
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((-fc::<F>(70.0) * s + fc::<F>(315.0)) * s - fc::<F>(540.0)) * s + fc::<F>(420.0)) * s
        - fc::<F>(126.0))
        + F::one()
}

#[inline]
fn h_a1_g<F: SemiflowFloat>(s: F) -> F {
    // -35s^9 + 160s^8 - 280s^7 + 224s^6 - 70s^5 + s
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s + s5
        * ((((-fc::<F>(35.0) * s + fc::<F>(160.0)) * s - fc::<F>(280.0)) * s + fc::<F>(224.0)) * s
            - fc::<F>(70.0))
}

#[inline]
fn h_a2_g<F: SemiflowFloat>(s: F) -> F {
    // -15s^9/2 + 35s^8 - 63s^7 + 105s^6/2 - 35s^5/2 + s^2/2
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    fc::<F>(0.5) * s2
        + s5 * ((((-fc::<F>(7.5) * s + fc::<F>(35.0)) * s - fc::<F>(63.0)) * s + fc::<F>(52.5)) * s
            - fc::<F>(17.5))
}

#[inline]
fn h_a3_g<F: SemiflowFloat>(s: F) -> F {
    // -5s^9/6 + 4s^8 - 15s^7/2 + 20s^6/3 - 5s^5/2 + s^3/6
    let s2 = s * s;
    let s3 = s2 * s;
    let s5 = s2 * s3;
    fc::<F>(1.0 / 6.0) * s3
        + s5 * ((((-fc::<F>(5.0 / 6.0) * s + fc::<F>(4.0)) * s - fc::<F>(7.5)) * s
            + fc::<F>(20.0 / 3.0))
            * s
            - fc::<F>(2.5))
}

#[inline]
fn h_a4_g<F: SemiflowFloat>(s: F) -> F {
    // -s^9/24 + 5s^8/24 - 5s^7/12 + 5s^6/12 - 5s^5/24 + s^4/24
    let s2 = s * s;
    let s4 = s2 * s2;
    let s5 = s4 * s;
    fc::<F>(1.0 / 24.0) * s4
        + s5 * ((((-fc::<F>(1.0 / 24.0) * s + fc::<F>(5.0 / 24.0)) * s - fc::<F>(5.0 / 12.0)) * s
            + fc::<F>(5.0 / 12.0))
            * s
            - fc::<F>(5.0 / 24.0))
}

#[inline]
fn h_b0_g<F: SemiflowFloat>(s: F) -> F {
    // 70s^9 - 315s^8 + 540s^7 - 420s^6 + 126s^5
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((fc::<F>(70.0) * s - fc::<F>(315.0)) * s + fc::<F>(540.0)) * s - fc::<F>(420.0)) * s
        + fc::<F>(126.0))
}

#[inline]
fn h_b1_g<F: SemiflowFloat>(s: F) -> F {
    // -35s^9 + 155s^8 - 260s^7 + 196s^6 - 56s^5
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((-fc::<F>(35.0) * s + fc::<F>(155.0)) * s - fc::<F>(260.0)) * s + fc::<F>(196.0)) * s
        - fc::<F>(56.0))
}

#[inline]
fn h_b2_g<F: SemiflowFloat>(s: F) -> F {
    // 15s^9/2 - 65s^8/2 + 53s^7 - 77s^6/2 + 21s^5/2
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((fc::<F>(7.5) * s - fc::<F>(32.5)) * s + fc::<F>(53.0)) * s - fc::<F>(38.5)) * s
        + fc::<F>(10.5))
}

#[inline]
fn h_b3_g<F: SemiflowFloat>(s: F) -> F {
    // -5s^9/6 + 7s^8/2 - 11s^7/2 + 23s^6/6 - s^5
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((-fc::<F>(5.0 / 6.0) * s + fc::<F>(3.5)) * s - fc::<F>(5.5)) * s + fc::<F>(23.0 / 6.0))
        * s
        - F::one())
}

#[inline]
fn h_b4_g<F: SemiflowFloat>(s: F) -> F {
    // s^9/24 - s^8/6 + s^7/4 - s^6/6 + s^5/24
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((fc::<F>(1.0 / 24.0) * s - fc::<F>(1.0 / 6.0)) * s + fc::<F>(0.25)) * s
        - fc::<F>(1.0 / 6.0))
        * s
        + fc::<F>(1.0 / 24.0))
}

// ---------------------------------------------------------------------------
// Generic FD stencils — same Fornberg coefficients as the f64 helpers.
// ---------------------------------------------------------------------------

/// Generic 10-pt scaled first derivative `dx*f'` (Fornberg 1988, k=1).
#[allow(clippy::similar_names)]
#[inline]
fn fd_prime_g<F: SemiflowFloat>(
    values: &[F],
    bnd: BoundaryPolicy<F>,
    n: usize,
    idx: i64,
    dx: F,
) -> F {
    let fm5 = bc_value_generic(bnd, values, n, idx - 5, dx);
    let fm4 = bc_value_generic(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value_generic(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value_generic(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value_generic(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value_generic(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value_generic(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value_generic(bnd, values, n, idx + 4, dx);
    let fp5 = bc_value_generic(bnd, values, n, idx + 5, dx);
    // Fornberg 10-pt k=1: (-2,25,-150,600,-2100,2100,-600,150,-25,2)/2520
    (-fc::<F>(2.0) * fm5 + fc::<F>(25.0) * fm4 - fc::<F>(150.0) * fm3 + fc::<F>(600.0) * fm2
        - fc::<F>(2100.0) * fm1
        + fc::<F>(2100.0) * fp1
        - fc::<F>(600.0) * fp2
        + fc::<F>(150.0) * fp3
        - fc::<F>(25.0) * fp4
        + fc::<F>(2.0) * fp5)
        / fc::<F>(2520.0)
}

/// Generic 9-pt scaled second derivative `dx^2*f''` (Fornberg 1988, k=2).
#[allow(clippy::similar_names)]
#[inline]
fn fd_double_g<F: SemiflowFloat>(
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
    let f0 = bc_value_generic(bnd, values, n, idx, dx);
    let fp1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value_generic(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value_generic(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value_generic(bnd, values, n, idx + 4, dx);
    // Fornberg 9-pt k=2: (-9,128,-1008,8064,-14350,8064,-1008,128,-9)/5040
    (-fc::<F>(9.0) * fm4 + fc::<F>(128.0) * fm3 - fc::<F>(1008.0) * fm2 + fc::<F>(8064.0) * fm1
        - fc::<F>(14350.0) * f0
        + fc::<F>(8064.0) * fp1
        - fc::<F>(1008.0) * fp2
        + fc::<F>(128.0) * fp3
        - fc::<F>(9.0) * fp4)
        / fc::<F>(5040.0)
}

/// Generic 10-pt scaled third derivative `dx^3*f'''` (Fornberg 1988, k=3).
#[allow(clippy::similar_names)]
#[inline]
fn fd_triple_g<F: SemiflowFloat>(
    values: &[F],
    bnd: BoundaryPolicy<F>,
    n: usize,
    idx: i64,
    dx: F,
) -> F {
    let fm5 = bc_value_generic(bnd, values, n, idx - 5, dx);
    let fm4 = bc_value_generic(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value_generic(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value_generic(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value_generic(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value_generic(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value_generic(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value_generic(bnd, values, n, idx + 4, dx);
    let fp5 = bc_value_generic(bnd, values, n, idx + 5, dx);
    // Fornberg 10-pt k=3 anti-sym (205,-2522,14607,-52428,70098)/30240
    (fc::<F>(205.0) * fm5 - fc::<F>(2522.0) * fm4 + fc::<F>(14607.0) * fm3 - fc::<F>(52428.0) * fm2
        + fc::<F>(70098.0) * fm1
        - fc::<F>(70098.0) * fp1
        + fc::<F>(52428.0) * fp2
        - fc::<F>(14607.0) * fp3
        + fc::<F>(2522.0) * fp4
        - fc::<F>(205.0) * fp5)
        / fc::<F>(30240.0)
}

/// Generic 9-pt scaled fourth derivative `dx^4*f''''` (Fornberg 1988, k=4).
#[allow(clippy::similar_names)]
#[inline]
fn fd_quad_g<F: SemiflowFloat>(
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
    let f0 = bc_value_generic(bnd, values, n, idx, dx);
    let fp1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value_generic(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value_generic(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value_generic(bnd, values, n, idx + 4, dx);
    // Fornberg 9-pt k=4: (7,-96,676,-1952,2730,-1952,676,-96,7)/240
    (fc::<F>(7.0) * fm4 - fc::<F>(96.0) * fm3 + fc::<F>(676.0) * fm2 - fc::<F>(1952.0) * fm1
        + fc::<F>(2730.0) * f0
        - fc::<F>(1952.0) * fp1
        + fc::<F>(676.0) * fp2
        - fc::<F>(96.0) * fp3
        + fc::<F>(7.0) * fp4)
        / fc::<F>(240.0)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Generic octonic-Hermite sampler for `F: SemiflowFloat` (incl. `Dual<f64>`).
pub(crate) fn sample_octonic_1d_generic<F: SemiflowFloat>(
    values: &[F],
    grid: &Grid1D<F>,
    x: F,
) -> F {
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
    let v0pppp = fd_quad_g(values, bnd, n, idx, dx);
    let v1pppp = fd_quad_g(values, bnd, n, idx + 1, dx);

    h_a0_g(s) * v0
        + h_a1_g(s) * v0p
        + h_a2_g(s) * v0pp
        + h_a3_g(s) * v0ppp
        + h_a4_g(s) * v0pppp
        + h_b0_g(s) * v1
        + h_b1_g(s) * v1p
        + h_b2_g(s) * v1pp
        + h_b3_g(s) * v1ppp
        + h_b4_g(s) * v1pppp
}
