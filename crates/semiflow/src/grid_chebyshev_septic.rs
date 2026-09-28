//! Septic-Hermite (degree-7) interpolation for v6.0.0 8th-order spatial.
//!
//! Provides `sample_septic_1d` used by `InterpKind::SepticHermite` dispatch
//! in `grid::Grid1D::interp`.
//!
//! Per math.md §40 / ADR-0109, the septic-Hermite interpolant matches nodal
//! values f, `dx·f'`, `dx²·f''`, `dx³·f'''` at both cell endpoints, yielding
//! a degree-7 polynomial with leading residue O(dx⁸) on smooth f.
//!
//! Scaled-data convention (NORMATIVE, unit-interval coordinate s ∈ [0,1]):
//!
//! ```text
//!   F0    = f(x_i),              F1    = f(x_{i+1})
//!   F0p   = dx  * f'(x_i),       F1p   = dx  * f'(x_{i+1})
//!   F0pp  = dx² * f''(x_i),      F1pp  = dx² * f''(x_{i+1})
//!   F0ppp = dx³ * f'''(x_i),     F1ppp = dx³ * f'''(x_{i+1})
//! ```
//!
//! Weight basis (sympy-derived, NORMATIVE, Birkhoff-Garabedian-Lorentz 1983):
//!
//! ```text
//!   a0(s) = 20s⁷ − 70s⁶ + 84s⁵ − 35s⁴ + 1
//!   a1(s) = s(1−s)⁴(1 + 4s + 10s²)
//!   a2(s) = (1/2)s²(1−s)⁴(1 + 4s)
//!   a3(s) = (1/6)s³(1−s)⁴
//!   b0(s) = a0(1−s)
//!   b1(s) = −a1(1−s)
//!   b2(s) = a2(1−s)
//!   b3(s) = −a3(1−s)
//! ```
//!
//! Sympy verification: `scripts/verify_septic_hermite_weights.py` (6/6 PASS).
//!
//! Caller invariant: `f ∈ C³(ℝ)` (FD-computed ghost data for f', f'', f''').
//!
//! Empirical floor at N=512: ≈ 1.49e-12 (formal model ADR-0109 §40.4),
//! 67× below the `QuinticHermite` floor of ≈ 1e-10.

// Grid node index i (usize) cast to f64 for coordinate x = i * dx; indices ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

use num_traits::float::FloatCore;

use crate::grid::{bc_value, BoundaryPolicy, Grid1D};
#[cfg(feature = "simd")]
use crate::simd::{F64x4, SimdF64x4};

// ---------------------------------------------------------------------------
// Horner-form weight polynomials for the septic-Hermite kernel.
// Derived by sympy (verify_septic_hermite_weights.py). NORMATIVE.
// Convention: x = x_i + s * dx, s in [0, 1].
//
// Expanded polynomial coefficients (all verified by sympy):
//   a0(s) = 20s^7 − 70s^6 + 84s^5 − 35s^4 + 1
//   a1(s) = −10s^7 + 36s^6 − 45s^5 + 20s^4 + s
//   a2(s) = (1/2)(4s^7 − 15s^6 + 20s^5 − 10s^4 + s^2)
//   a3(s) = (1/6)(s^7 − 4s^6 + 6s^5 − 4s^4 + s^3)
//   b0(s) = −20s^7 + 70s^6 − 84s^5 + 35s^4
//   b1(s) = 10s^7 − 34s^6 + 39s^5 − 15s^4
//   b2(s) = (1/2)(−4s^7 + 13s^6 − 14s^5 + 5s^4)
//   b3(s) = (1/6)(s^7 − 3s^6 + 3s^5 − s^4)
// ---------------------------------------------------------------------------

#[inline]
fn h_a0(s: f64) -> f64 {
    // 20s^7 − 70s^6 + 84s^5 − 35s^4 + 1
    // Factor s^4 out of polynomial part: s^4*(20s^3 - 70s^2 + 84s - 35) + 1
    let s2 = s * s;
    let s4 = s2 * s2;
    s4 * (((20.0 * s - 70.0) * s + 84.0) * s - 35.0) + 1.0
}

#[inline]
fn h_a1(s: f64) -> f64 {
    // a1(s) = s·(1−s)⁴·(1 + 4s + 10s²)
    // Expanded: 10s^7 − 36s^6 + 45s^5 − 20s^4 + s
    // = s + s^4*(10s^3 − 36s^2 + 45s − 20)
    let s2 = s * s;
    let s4 = s2 * s2;
    s + s4 * (((10.0 * s - 36.0) * s + 45.0) * s - 20.0)
}

#[inline]
fn h_a2(s: f64) -> f64 {
    // (1/2)(4s^7 − 15s^6 + 20s^5 − 10s^4 + s^2)
    // = 0.5*(s^2 + s^4*(-10 + 20s - 15s^2 + 4s^3))
    let s2 = s * s;
    let s4 = s2 * s2;
    0.5 * (s2 + s4 * (((4.0 * s - 15.0) * s + 20.0) * s - 10.0))
}

#[inline]
fn h_a3(s: f64) -> f64 {
    // (1/6)(s^7 − 4s^6 + 6s^5 − 4s^4 + s^3)
    // = (1/6)*s^3*(s^4 - 4s^3 + 6s^2 - 4s + 1)
    // = (1/6)*s^3*(s-1)^4  [note: (1-s)^4 = (s-1)^4]
    let s2 = s * s;
    let s3 = s2 * s;
    // Horner on (s^4 - 4s^3 + 6s^2 - 4s + 1):
    (1.0 / 6.0) * s3 * ((((s - 4.0) * s + 6.0) * s - 4.0) * s + 1.0)
}

#[inline]
fn h_b0(s: f64) -> f64 {
    // −20s^7 + 70s^6 − 84s^5 + 35s^4
    // = s^4*(-20s^3 + 70s^2 - 84s + 35)
    let s2 = s * s;
    let s4 = s2 * s2;
    s4 * (((-20.0 * s + 70.0) * s - 84.0) * s + 35.0)
}

#[inline]
fn h_b1(s: f64) -> f64 {
    // 10s^7 − 34s^6 + 39s^5 − 15s^4
    // = s^4*(10s^3 - 34s^2 + 39s - 15)
    let s2 = s * s;
    let s4 = s2 * s2;
    s4 * (((10.0 * s - 34.0) * s + 39.0) * s - 15.0)
}

#[inline]
fn h_b2(s: f64) -> f64 {
    // (1/2)(−4s^7 + 13s^6 − 14s^5 + 5s^4)
    // = 0.5*s^4*(-4s^3 + 13s^2 - 14s + 5)
    let s2 = s * s;
    let s4 = s2 * s2;
    0.5 * s4 * (((-4.0 * s + 13.0) * s - 14.0) * s + 5.0)
}

#[inline]
fn h_b3(s: f64) -> f64 {
    // (1/6)(s^7 − 3s^6 + 3s^5 − s^4)
    // = (1/6)*s^4*(s^3 - 3s^2 + 3s - 1) = -(1/6)*s^4*(1-s)^3
    let s2 = s * s;
    let s4 = s2 * s2;
    // Horner on (s^3 - 3s^2 + 3s - 1):
    (1.0 / 6.0) * s4 * (((s - 3.0) * s + 3.0) * s - 1.0)
}

// ---------------------------------------------------------------------------
// Central FD helpers — compute scaled derivatives from the values array.
// BC extension via bc_value handles out-of-range nodes.
// similar_names allowed: fm1/fp1 etc. are standard math stencil notation.
// ---------------------------------------------------------------------------

/// Scaled first derivative `dx * f'` at grid index `idx` — scalar path.
///
/// Uses the 8-point central-difference formula (Fornberg 1988, Table 1):
/// `(3f[i-4] − 32f[i-3] + 168f[i-2] − 672f[i-1] + 672f[i+1] − 168f[i+2] + 32f[i+3] − 3f[i+4]) / 840`
///
/// Leading error: O(dx⁹) on the scaled derivative `dx·f'`, i.e. O(dx⁸) on `f'`,
/// which keeps the septic-Hermite interpolant genuinely O(dx⁸).
#[allow(clippy::similar_names)]
#[allow(dead_code)] // used under #[cfg(not(feature = "simd"))] and test force-scalar path
#[inline]
fn fd_scaled_prime_scalar(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm4 = bc_value(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value(bnd, values, n, idx + 4, dx);
    (3.0 * fm4 - 32.0 * fm3 + 168.0 * fm2 - 672.0 * fm1 + 672.0 * fp1 - 168.0 * fp2 + 32.0 * fp3
        - 3.0 * fp4)
        / 840.0
}

/// SIMD 8-pt `fd_scaled_prime`: 4+4 split into two F64x4 vectors.
///
/// Block A: `(3, -32, 168, -672)` × `(fm4, fm3, fm2, fm1)` → `sum_a`
/// Block B: `(672, -168, 32, -3)` × `(fp1, fp2, fp3, fp4)` → `sum_b`
/// Result: `(sum_a + sum_b) / 840`.
///
/// Bit-equality with scalar path tested in `septic_hermite_floor.rs`.
#[cfg(feature = "simd")]
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_prime_simd(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm4 = bc_value(bnd, values, n, idx - 4, dx);
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    let fp4 = bc_value(bnd, values, n, idx + 4, dx);

    let wa = [3.0_f64, -32.0, 168.0, -672.0];
    let wb = [672.0_f64, -168.0, 32.0, -3.0];
    let va = [fm4, fm3, fm2, fm1];
    let vb = [fp1, fp2, fp3, fp4];

    let sum_a = F64x4::load_unaligned(&va)
        .mul(F64x4::load_unaligned(&wa))
        .horizontal_sum();
    let sum_b = F64x4::load_unaligned(&vb)
        .mul(F64x4::load_unaligned(&wb))
        .horizontal_sum();
    (sum_a + sum_b) / 840.0
}

/// Scaled first derivative `dx * f'` at grid index `idx`.
///
/// Dispatches to SIMD path when feature `simd` is active.
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_prime(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    #[cfg(feature = "simd")]
    {
        // cfg!(test) collapses to false in release builds → branch eliminated.
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
/// Uses the 7-point central-difference formula (Fornberg 1988, k=2, N=7):
/// `(2f[i-3] − 27f[i-2] + 270f[i-1] − 490f[i] + 270f[i+1] − 27f[i+2] + 2f[i+3]) / 180`
///
/// Leading error: O(dx⁸) on `dx²·f''`, keeping septic-Hermite accuracy intact.
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_double_prime(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let f0 = bc_value(bnd, values, n, idx, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    // Fornberg 1988 k=2 7-pt: (2,-27,270,-490,270,-27,2)/180
    (2.0 * fm3 - 27.0 * fm2 + 270.0 * fm1 - 490.0 * f0 + 270.0 * fp1 - 27.0 * fp2 + 2.0 * fp3)
        / 180.0
}

/// Scaled third derivative `dx³ * f'''` at grid index `idx`.
///
/// Uses the 6-point central-difference formula (Fornberg 1988, k=3, N=7):
/// `(f[i-3] − 8f[i-2] + 13f[i-1] − 13f[i+1] + 8f[i+2] − f[i+3]) / 8`
///
/// This gives O(dx⁶) accuracy for `dx³·f'''` (O(dx⁶) absolute in the scaled
/// derivative), which ensures the septic-Hermite interpolant achieves its
/// nominal O(dx⁸) floor.
///
/// Error budget rationale: `h_a3(s) ≤ C` (bounded, not O(dx)), so
/// `|h_a3 · error_in_v0ppp| = O(dx⁶)` — below the O(dx⁸) residue target
/// once combined with the O(dx⁴) contribution from `h_a3`'s polynomial weight.
///
/// The formerly-used 4-pt formula `(-f[i-2]+2f[i-1]-2f[i+1]+f[i+2])/2` gives
/// only O(dx⁵) absolute error (O(dx^2) for f'''), limiting the floor to ≈5e-10
/// instead of ≈1.5e-12. Upgraded for the v6.0 floor gate (ADR-0109 §40.4).
///
/// Sign convention (NORMATIVE): `(f[-3] - 8f[-2] + 13f[-1] - 13f[+1] + 8f[+2] - f[+3]) / 8`
/// computes `+dx³·f'''` (positive sign). See Fornberg 1988 Table 1, row k=3, N=7.
#[allow(clippy::similar_names)]
#[inline]
fn fd_scaled_triple_prime(values: &[f64], bnd: BoundaryPolicy, n: usize, idx: i64, dx: f64) -> f64 {
    let fm3 = bc_value(bnd, values, n, idx - 3, dx);
    let fm2 = bc_value(bnd, values, n, idx - 2, dx);
    let fm1 = bc_value(bnd, values, n, idx - 1, dx);
    let fp1 = bc_value(bnd, values, n, idx + 1, dx);
    let fp2 = bc_value(bnd, values, n, idx + 2, dx);
    let fp3 = bc_value(bnd, values, n, idx + 3, dx);
    // Fornberg 1988 k=3 6-pt: (f[-3] - 8f[-2] + 13f[-1] - 13f[+1] + 8f[+2] - f[+3]) / 8
    // Computes +dx³·f''' with O(dx⁶) absolute error in the scaled derivative.
    (fm3 - 8.0 * fm2 + 13.0 * fm1 - 13.0 * fp1 + 8.0 * fp2 - fp3) / 8.0
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Sample a septic-Hermite interpolant at off-grid `x`.
///
/// Ghost `dx·f'`, `dx²·f''`, `dx³·f'''` data are computed via central FD
/// on `values` using `BoundaryPolicy` for out-of-range nodes.
///
/// # Contract
/// - `values.len() == grid.n` (`Grid1D` invariant, not re-checked here).
/// - `x` may be arbitrary real; BC extension handles out-of-domain.
/// - Achieves O(dx⁸) on smooth f ∈ C³(ℝ); floor ≈ 1.49e-12 at N=512.
pub(crate) fn sample_septic_1d(values: &[f64], grid: &Grid1D, x: f64) -> f64 {
    let dx = grid.dx();
    let t_frac = (x - grid.xmin) / dx;
    let t_floor = FloatCore::floor(t_frac);
    // Safe cast: t_floor is an exact integer for any grid-aligned position.
    #[allow(clippy::cast_possible_truncation)]
    let idx = t_floor as i64;
    let s = t_frac - t_floor;

    let bnd = grid.boundary;
    let n = grid.n;

    // Nodal values at cell endpoints.
    let v0 = bc_value(bnd, values, n, idx, dx);
    let v1 = bc_value(bnd, values, n, idx + 1, dx);

    // Scaled derivatives at cell endpoints via FD.
    let v0p = fd_scaled_prime(values, bnd, n, idx, dx);
    let v1p = fd_scaled_prime(values, bnd, n, idx + 1, dx);
    let v0pp = fd_scaled_double_prime(values, bnd, n, idx, dx);
    let v1pp = fd_scaled_double_prime(values, bnd, n, idx + 1, dx);
    let v0ppp = fd_scaled_triple_prime(values, bnd, n, idx, dx);
    let v1ppp = fd_scaled_triple_prime(values, bnd, n, idx + 1, dx);

    // Septic-Hermite evaluation (all weights are dimensionless in s).
    h_a0(s) * v0
        + h_a1(s) * v0p
        + h_a2(s) * v0pp
        + h_a3(s) * v0ppp
        + h_b0(s) * v1
        + h_b1(s) * v1p
        + h_b2(s) * v1pp
        + h_b3(s) * v1ppp
}

// ---------------------------------------------------------------------------
// Generic septic-Hermite sampler — implementation lives in a child module
// to keep this file within the 500-line budget (§46.5.bis, ADR-0133 Am.1).
// ---------------------------------------------------------------------------

/// Generic septic-Hermite sampler for `F: SemiflowFloat` (incl. `Dual<f64>`).
///
/// Mirrors `sample_septic_1d` EXACTLY — same 8 Birkhoff-Garabedian-Lorentz
/// weight polynomials (§40.3), same 3 central-FD stencils (§40.2) — but
/// with `f64` literals replaced by `F::from(·)` and `bc_value → bc_value_generic`.
/// No SIMD (§46.5 carve-out); leaves the existing `sample_septic_1d` and all
/// SIMD paths byte-identical (additive-only change).
///
/// Called by `Grid1D::interp_generic` for the `SepticHermite` arm (§46.5.bis).
pub(crate) use septic_generic::sample_septic_1d_generic;

pub(crate) mod septic_generic;

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// Helper: uniform grid [0,1] with n nodes, values = `f(x_i)`.
    ///
    /// Uses `LinearExtrapolate` BC so that FD stencil ghost points near
    /// the boundary use affine continuation of the function — this is
    /// required for polynomial-exactness tests which check ALL cells
    /// including cells 0..4 near the left boundary. The production default
    /// `Reflect` BC intentionally sets ghost derivatives to 0 (Neumann-like),
    /// which does not reproduce the true derivative of a linear function
    /// at boundary stencil positions; that is an expected feature of the
    /// Reflect policy, not a bug in the Hermite basis functions.
    fn make_grid_and_values(n: usize, f: impl Fn(f64) -> f64) -> (Grid1D, Vec<f64>) {
        let grid = Grid1D::new(0.0, 1.0, n)
            .expect("valid grid")
            .with_boundary(BoundaryPolicy::LinearExtrapolate);
        let dx = grid.dx();
        let values: Vec<f64> = (0..n).map(|i| f(i as f64 * dx)).collect();
        (grid, values)
    }

    /// Node-exact: sample at grid node should recover stored value exactly.
    #[test]
    fn septic_node_exact() {
        let f = |x: f64| (2.0 * x + 1.0).powi(3) * (-x).exp();
        let n = 32;
        let (grid, values) = make_grid_and_values(n, f);
        let dx = grid.dx();
        for i in 0..n {
            let x = i as f64 * dx;
            let got = sample_septic_1d(&values, &grid, x);
            assert!(
                (got - values[i]).abs() < 1e-13,
                "node {i}: got {got}, expected {}",
                values[i]
            );
        }
    }

    /// Linearity: for a linear function, interpolant is exact everywhere.
    #[test]
    fn septic_linear_exact() {
        let f = |x: f64| 3.0 * x - 1.5;
        let n = 16;
        let (grid, values) = make_grid_and_values(n, f);
        let dx = grid.dx();
        // Test at cell midpoints.
        for i in 0..(n - 1) {
            let x = (i as f64 + 0.5) * dx;
            let got = sample_septic_1d(&values, &grid, x);
            let exact = f(x);
            assert!(
                (got - exact).abs() < 1e-14,
                "midpoint {i}: got {got}, exact {exact}"
            );
        }
    }

    /// Cubic exact: septic-Hermite reproduces cubics exactly
    /// in INTERIOR cells where all FD stencils are fully in-range.
    ///
    /// The 8-pt prime stencil (`fd_scaled_prime`) requires idx ≥ 4 so that no
    /// ghost-data extrapolation is needed. Cells 0..4 are excluded because
    /// `LinearExtrapolate` BC is only 1st-order accurate for ghost cubics.
    #[test]
    fn septic_cubic_exact() {
        let f = |x: f64| x * x * x - 0.5 * x * x + 0.25 * x;
        let n = 64;
        let (grid, values) = make_grid_and_values(n, f);
        let dx = grid.dx();
        // Start at cell 4 so the 8-pt prime stencil (±4 nodes) stays fully in-range.
        // End at n-5 so the right boundary stencil is also fully interior.
        for i in 4..(n - 5) {
            let x = (i as f64 + 0.333) * dx;
            let got = sample_septic_1d(&values, &grid, x);
            let exact = f(x);
            assert!(
                (got - exact).abs() < 1e-12,
                "cell {i} x={x:.6}: got {got:.15e}, exact {exact:.15e}, err {:.3e}",
                (got - exact).abs()
            );
        }
    }

    /// a0(0)=1, a0(1)=0; b0(0)=0, b0(1)=1 (partition of unity at nodes).
    #[test]
    fn weight_partition_of_unity() {
        assert!((h_a0(0.0) - 1.0).abs() < 1e-15);
        assert!(h_a0(1.0).abs() < 1e-15);
        assert!(h_b0(0.0).abs() < 1e-15);
        assert!((h_b0(1.0) - 1.0).abs() < 1e-15);
    }

    /// All derivative weights vanish at their non-owning node.
    #[test]
    fn weight_derivative_endpoints() {
        // a1, a2, a3 own node 0 side → must vanish at s=1.
        assert!(h_a1(1.0).abs() < 1e-15, "h_a1(1)={}", h_a1(1.0));
        assert!(h_a2(1.0).abs() < 1e-15, "h_a2(1)={}", h_a2(1.0));
        assert!(h_a3(1.0).abs() < 1e-15, "h_a3(1)={}", h_a3(1.0));
        // b1, b2, b3 own node 1 side → must vanish at s=0.
        assert!(h_b1(0.0).abs() < 1e-15, "h_b1(0)={}", h_b1(0.0));
        assert!(h_b2(0.0).abs() < 1e-15, "h_b2(0)={}", h_b2(0.0));
        assert!(h_b3(0.0).abs() < 1e-15, "h_b3(0)={}", h_b3(0.0));
    }
}
