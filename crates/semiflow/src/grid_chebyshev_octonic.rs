//! Octonic-Hermite (degree-9) interpolation for v7.0.0 KEYSTONE spatial floor.
//!
//! Provides `sample_octonic_1d` used by `InterpKind::OctonicHermite` dispatch
//! in `grid::Grid1D::interp`.
//!
//! Per math.md §41.bis / ADR-0117, the octonic-Hermite interpolant matches nodal
//! values f, `dx·f'`, `dx²·f''`, `dx³·f'''`, `dx⁴·f''''` at both cell endpoints,
//! yielding a degree-9 polynomial with leading residue O(dx¹⁰) on smooth f.
//!
//! Scaled-data convention (NORMATIVE, unit-interval coordinate s ∈ [0,1]):
//!
//! ```text
//!   F0      = f(x_i),               F1      = f(x_{i+1})
//!   F0p     = dx   * f'(x_i),       F1p     = dx   * f'(x_{i+1})
//!   F0pp    = dx²  * f''(x_i),      F1pp    = dx²  * f''(x_{i+1})
//!   F0ppp   = dx³  * f'''(x_i),     F1ppp   = dx³  * f'''(x_{i+1})
//!   F0pppp  = dx⁴  * f''''(x_i),    F1pppp  = dx⁴  * f''''(x_{i+1})
//! ```
//!
//! Weight basis (sympy-derived via 10×10 endpoint solve, NORMATIVE):
//!
//! ```text
//!   a0(s) = −70s⁹+315s⁸−540s⁷+420s⁶−126s⁵+1   (ADR-0117 closed form)
//!   a1..a4, b0..b4: solved from 10×10 Hermite system (see below).
//! ```
//!
//! Predicted virtual-node floor at N=512: φ ≈ 9.1·10⁻¹⁶ (1638× below `SepticHermite`).
//! Sympy verification: `scripts/verify_octonic_hermite_weights.py` (5/5 PASS).
//!
//! Caller invariant: `f ∈ C⁴(ℝ)` (FD-computed ghost data for f', f'', f''', f'''').

// Grid node index i (usize) cast to f64 for coordinate x = i * dx; indices ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

use num_traits::float::FloatCore;

use crate::grid::{bc_value, BoundaryPolicy, Grid1D};
#[cfg(feature = "simd")]
use crate::simd::{F64x4, SimdF64x4};

// ---------------------------------------------------------------------------
// Horner-form weight polynomials for the octonic-Hermite kernel.
// Derived by sympy 10×10 endpoint solve (verify_octonic_hermite_weights.py). NORMATIVE.
// Convention: s ∈ [0, 1] where x = x_i + s * dx.
//
// Expanded (all coefficients sympy-verified):
//   a0(s) = -70s⁹ + 315s⁸ - 540s⁷ + 420s⁶ - 126s⁵ + 1
//   a1(s) = -35s⁹ + 160s⁸ - 280s⁷ + 224s⁶ - 70s⁵ + s
//   a2(s) = -15s⁹/2 + 35s⁸ - 63s⁷ + 105s⁶/2 - 35s⁵/2 + s²/2
//   a3(s) = -5s⁹/6 + 4s⁸ - 15s⁷/2 + 20s⁶/3 - 5s⁵/2 + s³/6
//   a4(s) = -s⁹/24 + 5s⁸/24 - 5s⁷/12 + 5s⁶/12 - 5s⁵/24 + s⁴/24
//   b0(s) = 70s⁹ - 315s⁸ + 540s⁷ - 420s⁶ + 126s⁵
//   b1(s) = -35s⁹ + 155s⁸ - 260s⁷ + 196s⁶ - 56s⁵
//   b2(s) = 15s⁹/2 - 65s⁸/2 + 53s⁷ - 77s⁶/2 + 21s⁵/2
//   b3(s) = -5s⁹/6 + 7s⁸/2 - 11s⁷/2 + 23s⁶/6 - s⁵
//   b4(s) = s⁹/24 - s⁸/6 + s⁷/4 - s⁶/6 + s⁵/24
// ---------------------------------------------------------------------------

#[inline]
fn h_a0(s: f64) -> f64 {
    // -70s⁹ + 315s⁸ - 540s⁷ + 420s⁶ - 126s⁵ + 1
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((-70.0 * s + 315.0) * s - 540.0) * s + 420.0) * s - 126.0) + 1.0
}

#[inline]
fn h_a1(s: f64) -> f64 {
    // -35s⁹ + 160s⁸ - 280s⁷ + 224s⁶ - 70s⁵ + s
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s + s5 * ((((-35.0 * s + 160.0) * s - 280.0) * s + 224.0) * s - 70.0)
}

#[inline]
fn h_a2(s: f64) -> f64 {
    // -15s⁹/2 + 35s⁸ - 63s⁷ + 105s⁶/2 - 35s⁵/2 + s²/2
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    0.5 * s2 + s5 * ((((-7.5 * s + 35.0) * s - 63.0) * s + 52.5) * s - 17.5)
}

#[inline]
fn h_a3(s: f64) -> f64 {
    // -5s⁹/6 + 4s⁸ - 15s⁷/2 + 20s⁶/3 - 5s⁵/2 + s³/6
    let s2 = s * s;
    let s3 = s2 * s;
    let s5 = s2 * s3;
    (1.0 / 6.0) * s3 + s5 * ((((-5.0 / 6.0 * s + 4.0) * s - 7.5) * s + 20.0 / 3.0) * s - 2.5)
}

#[inline]
fn h_a4(s: f64) -> f64 {
    // -s⁹/24 + 5s⁸/24 - 5s⁷/12 + 5s⁶/12 - 5s⁵/24 + s⁴/24
    let s2 = s * s;
    let s4 = s2 * s2;
    let s5 = s4 * s;
    (1.0 / 24.0) * s4
        + s5 * ((((-1.0 / 24.0 * s + 5.0 / 24.0) * s - 5.0 / 12.0) * s + 5.0 / 12.0) * s
            - 5.0 / 24.0)
}

#[inline]
fn h_b0(s: f64) -> f64 {
    // 70s⁹ - 315s⁸ + 540s⁷ - 420s⁶ + 126s⁵
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((70.0 * s - 315.0) * s + 540.0) * s - 420.0) * s + 126.0)
}

#[inline]
fn h_b1(s: f64) -> f64 {
    // -35s⁹ + 155s⁸ - 260s⁷ + 196s⁶ - 56s⁵
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((-35.0 * s + 155.0) * s - 260.0) * s + 196.0) * s - 56.0)
}

#[inline]
fn h_b2(s: f64) -> f64 {
    // 15s⁹/2 - 65s⁸/2 + 53s⁷ - 77s⁶/2 + 21s⁵/2
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((7.5 * s - 32.5) * s + 53.0) * s - 38.5) * s + 10.5)
}

#[inline]
fn h_b3(s: f64) -> f64 {
    // -5s⁹/6 + 7s⁸/2 - 11s⁷/2 + 23s⁶/6 - s⁵
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((-5.0 / 6.0 * s + 3.5) * s - 5.5) * s + 23.0 / 6.0) * s - 1.0)
}

#[inline]
fn h_b4(s: f64) -> f64 {
    // s⁹/24 - s⁸/6 + s⁷/4 - s⁶/6 + s⁵/24
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    s5 * ((((1.0 / 24.0 * s - 1.0 / 6.0) * s + 0.25) * s - 1.0 / 6.0) * s + 1.0 / 24.0)
}

// ---------------------------------------------------------------------------
// Central FD helpers — compute scaled derivatives from the values array.
// All weights are Fornberg (1988) exact rationals, baked as f64 constants.
// ---------------------------------------------------------------------------

/// Scaled first derivative `dx * f'` at grid index `idx` — scalar path.
///
/// Uses 10-point central {±1,±2,±3,±4,±5}: `Σ wⱼ·f[j] / (2520·dx)`.
/// Weights: (−2, 25, −150, 600, −2100, 2100, −600, 150, −25, 2) / 2520.
/// Leading error: O(dx¹⁰) on `f'`, preserving O(dx¹⁰) interpolant floor.
// used under #[cfg(not(feature = "simd"))] and test force-scalar path
#[allow(clippy::similar_names)]
#[allow(dead_code)]
#[inline]
fn fd_scaled_prime_scalar(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm5 = bc_value(bnd, values, n, idx - 5, dx);
    let fm4 = bc_value(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value(bnd, values, n, idx + 4, dx);
    let fp5 = bc_value(bnd, values, n, idx + 5, dx);
    // Fornberg 10-pt k=1: (-2,25,-150,600,-2100,0,2100,-600,150,-25,2)/2520 skipping f0
    (-2.0 * fm5 + 25.0 * fm4 - 150.0 * fm3 + 600.0 * fm2 - 2100.0 * fm1 + 2100.0 * fp1
        - 600.0 * fp2
        + 150.0 * fp3
        - 25.0 * fp4
        + 2.0 * fp5)
        / 2520.0
}

/// SIMD 10-pt `fd_scaled_prime`: 5+5 split into two F64x4 (padded).
///
/// Block A: (−2, 25, −150, 600, −2100, 0) × (fm5, fm4, fm3, fm2, fm1, 0)
/// Block B: (2100, −600, 150, −25, 2, 0) × (fp1, fp2, fp3, fp4, fp5, 0)
/// Uses 4+4 split since F64x4 holds 4 lanes; rem 1 added as scalar.
#[cfg(feature = "simd")]
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_prime_simd(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm5 = bc_value(bnd, values, n, idx - 5, dx);
    let fm4 = bc_value(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value(bnd, values, n, idx + 4, dx);
    let fp5 = bc_value(bnd, values, n, idx + 5, dx);

    let wa = [-2.0_f64, 25.0, -150.0, 600.0];
    let wb = [-2100.0_f64, 2100.0, -600.0, 150.0];
    let va = [fm5, fm4, fm3, fm2];
    let vb = [fm1, fp1, fp2, fp3];

    let sum_ab = F64x4::load_unaligned(&va)
        .mul(F64x4::load_unaligned(&wa))
        .horizontal_sum()
        + F64x4::load_unaligned(&vb)
            .mul(F64x4::load_unaligned(&wb))
            .horizontal_sum();
    // Remainder: -25*fp4 + 2*fp5
    (sum_ab - 25.0 * fp4 + 2.0 * fp5) / 2520.0
}

/// Scaled first derivative `dx * f'` at grid index `idx`.
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_prime(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    #[cfg(feature = "simd")]
    {
        if cfg!(test) && crate::simd::FORCE_SCALAR.with(core::cell::Cell::get) {
            return fd_scaled_prime_scalar(values, bnd, n, idx, dx);
        }
        fd_scaled_prime_simd(values, bnd, n, idx, dx)
    }
    #[cfg(not(feature = "simd"))]
    fd_scaled_prime_scalar(values, bnd, n, idx, dx)
}

/// Scaled second derivative `dx² * f''` at grid index `idx`.
///
/// Uses 9-point central {0,±1,±2,±3,±4}: weights (−9,128,−1008,8064,−14350,...)/5040.
/// Leading error: O(dx⁸) on `f''` → O(dx¹⁰) on `dx²·f''` (preserves floor).
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_double_prime(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm4 = bc_value(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let f0 = bc_value(bnd, values, n, idx, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value(bnd, values, n, idx + 4, dx);
    // Fornberg 9-pt k=2: (-9,128,-1008,8064,-14350,8064,-1008,128,-9)/5040
    (-9.0 * fm4 + 128.0 * fm3 - 1008.0 * fm2 + 8064.0 * fm1 - 14350.0 * f0 + 8064.0 * fp1
        - 1008.0 * fp2
        + 128.0 * fp3
        - 9.0 * fp4)
        / 5040.0
}

/// Scaled third derivative `dx³ * f'''` at grid index `idx`.
///
/// Uses 10-point anti-symmetric {±1,±2,±3,±4,±5}: weights /30240.
/// Leading error: O(dx⁸) on `f'''` → O(dx¹¹) on scaled `dx³·f'''`.
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_triple_prime(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm5 = bc_value(bnd, values, n, idx - 5, dx);
    let fm4 = bc_value(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value(bnd, values, n, idx + 4, dx);
    let fp5 = bc_value(bnd, values, n, idx + 5, dx);
    // Fornberg 10-pt k=3 anti-sym (205,-2522,14607,-52428,70098)/30240
    (205.0 * fm5 - 2522.0 * fm4 + 14607.0 * fm3 - 52428.0 * fm2 + 70098.0 * fm1 - 70098.0 * fp1
        + 52428.0 * fp2
        - 14607.0 * fp3
        + 2522.0 * fp4
        - 205.0 * fp5)
        / 30240.0
}

/// Scaled fourth derivative `dx⁴ * f''''` at grid index `idx`.
///
/// Uses 9-point central {0,±1,±2,±3,±4}: weights (7,−96,676,−1952,2730,...)/240.
/// Leading error: O(dx⁶) on `f''''` → O(dx¹⁰) on scaled `dx⁴·f''''`.
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_quad_prime(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm4 = bc_value(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let f0 = bc_value(bnd, values, n, idx, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value(bnd, values, n, idx + 4, dx);
    // Fornberg 9-pt k=4: (7,-96,676,-1952,2730,-1952,676,-96,7)/240
    (7.0 * fm4 - 96.0 * fm3 + 676.0 * fm2 - 1952.0 * fm1 + 2730.0 * f0 - 1952.0 * fp1 + 676.0 * fp2
        - 96.0 * fp3
        + 7.0 * fp4)
        / 240.0
}

// ---------------------------------------------------------------------------
// Generic octonic-Hermite sampler — implementation lives in a child module
// to keep this file within the 500-line budget (§46.5.bis, ADR-0139).
// ---------------------------------------------------------------------------

/// Generic octonic-Hermite sampler for `F: SemiflowFloat` (incl. `Dual<f64>`).
///
/// Mirrors `sample_octonic_1d` EXACTLY — same 10 ADR-0117 weight polynomials,
/// same 4 central-FD stencils — with `f64` literals promoted via `F::from(·)`
/// and `bc_value → bc_value_generic`. No SIMD (§46.5 carve-out); leaves the
/// existing f64 path byte-identical (additive-only change).
///
/// Called by `Grid1D::interp_generic` for the `OctonicHermite` arm (§46.5.bis).
pub(crate) use octonic_generic::sample_octonic_1d_generic;

pub(crate) mod octonic_generic;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Sample an octonic-Hermite interpolant at off-grid `x`.
///
/// Ghost `dx·f'`, `dx²·f''`, `dx³·f'''`, `dx⁴·f''''` data are computed via
/// central Fornberg FD on `values` using `BoundaryPolicy` for out-of-range nodes.
///
/// # Contract
/// - `values.len() == grid.n` (`Grid1D` invariant, not re-checked here).
/// - `x` may be arbitrary real; BC extension handles out-of-domain.
/// - Achieves O(dx¹⁰) on smooth f ∈ C⁴(ℝ); floor ≈ 9.1e-16 at N=512 (ADR-0117).
pub(crate) fn sample_octonic_1d(values: &[f64], grid: &Grid1D, x: f64) -> f64 {
    let dx = grid.dx();
    let t_frac = (x - grid.xmin) / dx;
    let t_floor = FloatCore::floor(t_frac);
    #[allow(clippy::cast_possible_truncation)]
    let idx = t_floor as i64;
    let s = t_frac - t_floor;

    let bnd = grid.boundary;
    let n = grid.n;

    let v0 = bc_value(bnd, values, n, idx, dx);
    let v1 = bc_value(bnd, values, n, idx + 1, dx);

    let v0p = fd_scaled_prime(values, bnd, n, idx, dx);
    let v1p = fd_scaled_prime(values, bnd, n, idx + 1, dx);
    let v0pp = fd_scaled_double_prime(values, bnd, n, idx, dx);
    let v1pp = fd_scaled_double_prime(values, bnd, n, idx + 1, dx);
    let v0ppp = fd_scaled_triple_prime(values, bnd, n, idx, dx);
    let v1ppp = fd_scaled_triple_prime(values, bnd, n, idx + 1, dx);
    let v0pppp = fd_scaled_quad_prime(values, bnd, n, idx, dx);
    let v1pppp = fd_scaled_quad_prime(values, bnd, n, idx + 1, dx);

    h_a0(s) * v0
        + h_a1(s) * v0p
        + h_a2(s) * v0pp
        + h_a3(s) * v0ppp
        + h_a4(s) * v0pppp
        + h_b0(s) * v1
        + h_b1(s) * v1p
        + h_b2(s) * v1pp
        + h_b3(s) * v1ppp
        + h_b4(s) * v1pppp
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_and_values(n: usize, f: impl Fn(f64) -> f64) -> (Grid1D, Vec<f64>) {
        let grid = Grid1D::new(0.0, 1.0, n)
            .expect("valid grid")
            .with_boundary(BoundaryPolicy::LinearExtrapolate);
        let dx = grid.dx();
        let values: Vec<f64> = (0..n).map(|i| f(i as f64 * dx)).collect();
        (grid, values)
    }

    #[test]
    fn octonic_node_exact() {
        let f = |x: f64| (2.0 * x + 1.0).powi(3) * (-x).exp();
        let n = 32;
        let (grid, values) = make_grid_and_values(n, f);
        let dx = grid.dx();
        for i in 0..n {
            let x = i as f64 * dx;
            let got = sample_octonic_1d(&values, &grid, x);
            assert!(
                (got - values[i]).abs() < 1e-13,
                "node {i}: got {got}, expected {}",
                values[i]
            );
        }
    }

    #[test]
    fn octonic_linear_exact() {
        let f = |x: f64| 3.0 * x - 1.5;
        let n = 16;
        let (grid, values) = make_grid_and_values(n, f);
        let dx = grid.dx();
        for i in 0..(n - 1) {
            let x = (i as f64 + 0.5) * dx;
            let got = sample_octonic_1d(&values, &grid, x);
            let exact = f(x);
            assert!(
                (got - exact).abs() < 1e-14,
                "midpoint {i}: got {got}, exact {exact}"
            );
        }
    }

    #[test]
    fn octonic_cubic_exact() {
        let f = |x: f64| x * x * x - 0.5 * x * x + 0.25 * x;
        let n = 64;
        let (grid, values) = make_grid_and_values(n, f);
        let dx = grid.dx();
        // Start at cell 5 so all 10-pt prime stencils (±5) stay fully in-range.
        for i in 5..(n - 6) {
            let x = (i as f64 + 0.333) * dx;
            let got = sample_octonic_1d(&values, &grid, x);
            let exact = f(x);
            assert!(
                (got - exact).abs() < 1e-11,
                "cell {i} x={x:.6}: got {got:.15e}, exact {exact:.15e}, err {:.3e}",
                (got - exact).abs()
            );
        }
    }

    #[test]
    fn weight_partition_of_unity() {
        assert!((h_a0(0.0) - 1.0).abs() < 1e-15);
        assert!(h_a0(1.0).abs() < 1e-15);
        assert!(h_b0(0.0).abs() < 1e-15);
        assert!((h_b0(1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn weight_derivative_endpoints() {
        // a1..a4 own s=0 → must vanish at s=1.
        assert!(h_a1(1.0).abs() < 1e-14, "h_a1(1)={}", h_a1(1.0));
        assert!(h_a2(1.0).abs() < 1e-14, "h_a2(1)={}", h_a2(1.0));
        assert!(h_a3(1.0).abs() < 1e-14, "h_a3(1)={}", h_a3(1.0));
        assert!(h_a4(1.0).abs() < 1e-14, "h_a4(1)={}", h_a4(1.0));
        // b1..b4 own s=1 → must vanish at s=0.
        assert!(h_b1(0.0).abs() < 1e-14, "h_b1(0)={}", h_b1(0.0));
        assert!(h_b2(0.0).abs() < 1e-14, "h_b2(0)={}", h_b2(0.0));
        assert!(h_b3(0.0).abs() < 1e-14, "h_b3(0)={}", h_b3(0.0));
        assert!(h_b4(0.0).abs() < 1e-14, "h_b4(0)={}", h_b4(0.0));
    }
}
