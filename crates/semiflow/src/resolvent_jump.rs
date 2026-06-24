//! F2 — Resolvent time-jump amortization (math.md §47, ADR-0134).
//!
//! [`ResolventJumpChernoff`] computes `e^{tA}g` for a LARGE step `t` via the
//! Trefethen–Weideman–Schmelzer (2006) parabolic-contour inverse Laplace quadrature:
//!
//! ```text
//! e^{tA}g ≈ (1/2πi) Σ_k e^{λ_k t} (λ_k I − A)⁻¹ g · λ'(θ_k) · (2π/M)
//! ```
//!
//! The `M/t` contour scaling decouples node count from `t` (geometric convergence,
//! Theorem 47.1). Each resolvent `(λ_k I − A)⁻¹ g` is evaluated by a LEFT-half-
//! plane-capable complex Thomas solve (O(N) for the 3-pt Laplacian). The shipped
//! Gauss-Laguerre `eval`/`eval_complex` (§22.9) is NOT used — it diverges for
//! `Re λ ≤ 0`, which is where the optimal contour places most nodes.
//!
//! ## NARROW scope (§47.4 — NORMATIVE)
//!
//! Self-adjoint / sectorial generators only (every shipped diffusion kernel).
//! Non-self-adjoint / advection-dominated generators are OUT of scope for v8.0.0.
//!
//! ## References
//!
//! - Trefethen, Weideman, Schmelzer, *BIT Numer. Math.* 46:3 (2006), pp. 653–670.
//! - math.md §47 (NORMATIVE), ADR-0134 (ACCEPTED-NARROW, 2026-06-07).
//! - Gate `G_RESOLVENT_JUMP_ORDER` (`RELEASE_BLOCKING`) + oracle `T_RESOLVENT_JUMP`.

// Contour node count M and grid sizes (usize) cast to f64 for step/scale; ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use num_complex::Complex;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
};

// ---------------------------------------------------------------------------
// TWS-2006 parabolic-contour constants (§47.2, NORMATIVE — do NOT alter).
// ---------------------------------------------------------------------------

/// TWS-2006 parabolic-contour coefficients `(a₀, a₁, a₂)`.
/// `λ(θ) = (M/t)(a₀ − a₁ θ² + i a₂ θ)`.
const TWS: [f64; 3] = [0.1309, 0.1194, 0.2500];

// ---------------------------------------------------------------------------
// Minimum node count: below this the geometric regime is not yet reached.
// ---------------------------------------------------------------------------

/// Minimum allowed `m_nodes` (construction gate).
const M_MIN: usize = 6;

// ---------------------------------------------------------------------------
// Public type
// ---------------------------------------------------------------------------

/// Resolvent time-jump Chernoff function (F2 NARROW-GO, ADR-0134, §47).
///
/// Computes `e^{tA}g` for a large time step `t` via a constant-`M` contour
/// quadrature over `(λ_k I − A)⁻¹ g`, evaluated by a complex Thomas solve.
///
/// **Construction**: `ResolventJumpChernoff::new(inner, m_nodes)`. The `inner`
/// must be a 1D diffusion-family Chernoff function (self-adjoint, sectorial);
/// `inner.grid` is used to reconstruct the discrete Laplacian for each LHP node.
///
/// **Usage**: call [`jump`](Self::jump) for a single large-step approximation.
/// This is NOT a [`ChernoffFunction`] implementation: it represents the resolved
/// semigroup directly, not a single-step approximant.
pub struct ResolventJumpChernoff<C, F = f64>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    /// Inner Chernoff function carrying the operator geometry.
    pub inner: C,
    /// Number of contour nodes `M`. Invariant: `m_nodes >= M_MIN`.
    pub m_nodes: usize,
    /// Grid geometry (extracted from `inner.grid` at construction).
    pub grid: Grid1D<F>,
}

impl<C, F> ResolventJumpChernoff<C, F>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    /// Construct from an inner Chernoff function and a node count `M ≥ 6`.
    ///
    /// The grid is extracted from `inner` at construction; `inner.grid` must
    /// be accessible (all shipped 1D diffusion kernels expose `pub grid`).
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `m_nodes < 6`.
    pub fn new(inner: C, m_nodes: usize, grid: Grid1D<F>) -> Result<Self, SemiflowError> {
        if m_nodes < M_MIN {
            return Err(SemiflowError::DomainViolation {
                what: "ResolventJumpChernoff: m_nodes must be >= 6 (geometric regime)",
                value: m_nodes as f64,
            });
        }
        Ok(Self {
            inner,
            m_nodes,
            grid,
        })
    }

    /// Approximate `e^{tA}g` via the TWS parabolic-contour inverse Laplace sum.
    ///
    /// Implements math.md §47.3 (NORMATIVE).  Each contour node is solved by
    /// `resolve_lhp` — a complex Thomas O(N) solve that is
    /// valid for any `λ ∉ σ(A)`, including `Re λ < 0`.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `t ≤ 0` or `t` non-finite.
    // t, g, k, m, s, p: standard resolvent quadrature names from math §47.3.
    #[allow(clippy::many_single_char_names)]
    pub fn jump(&self, t: F, g: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError>
    where
        C: ChernoffFunction<F, S = GridFn1D<F>>,
    {
        if !t.is_finite() || t <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "ResolventJumpChernoff::jump: t must be finite and positive",
                value: t.to_f64().unwrap_or(f64::NAN),
            });
        }
        let n = self.grid.n;
        // Accumulator in f64 complex (contour arithmetic uses f64 throughout).
        let mut acc: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
        let m = self.m_nodes;
        let t_f64 = t.to_f64().unwrap_or(f64::NAN);
        let scale = m as f64 / t_f64; // M/t
        let step = 2.0 * core::f64::consts::PI / m as f64; // 2π/M
        for k in 0..m {
            let (lam, dlam) = contour_node(scale, k, step);
            let r = self.resolve_lhp(lam, g)?;
            // weight = e^{λ t} · λ'(θ) · (2π/M) / (2π i) = e^{λt} λ' / (i M)
            let weight =
                (lam * t_f64).exp() * dlam * step / Complex::new(0.0, 2.0 * core::f64::consts::PI);
            for i in 0..n {
                acc[i] += weight * r[i];
            }
        }
        // Take real part into GridFn1D<F>.
        let values: Vec<F> = acc.iter().map(|z| from_f64::<F>(z.re)).collect();
        Ok(GridFn1D {
            values,
            grid: self.grid,
        })
    }

    /// Complex Thomas solve of `(λI − A)g = r` for the discrete 3-pt Laplacian.
    ///
    /// Valid for any `λ ∉ σ(A)` — including `Re λ < 0`.  The system is
    /// complex tridiagonal; the Thomas algorithm is unconditionally stable
    /// off-spectrum for sectorial `A`.
    ///
    /// Returns the per-node complex values as `Vec<Complex<f64>>` (contour
    /// arithmetic stays in f64; real part is taken in `jump`).
    fn resolve_lhp(
        &self,
        lam: Complex<f64>,
        g: &GridFn1D<F>,
    ) -> Result<Vec<Complex<f64>>, SemiflowError> {
        let n = self.grid.n;
        let dx = self.grid.dx().to_f64().unwrap_or(f64::NAN);
        let inv_dx2 = 1.0 / (dx * dx);
        // Assemble tridiagonal (λI − A) for the 3-pt Neumann Laplacian.
        // Neumann BC: boundary rows use one-sided stencil (mirror of laplacian_1d in kit).
        let sub = Complex::new(-inv_dx2, 0.0);
        let sup = Complex::new(-inv_dx2, 0.0);
        // Build RHS and run Thomas forward sweep.
        lhp_thomas(lam, inv_dx2, sub, sup, n, g)
    }
}

// ---------------------------------------------------------------------------
// Contour helpers
// ---------------------------------------------------------------------------

/// TWS parabolic node and derivative at index `k`.
///
/// Returns `(λ_k, λ'_k)` with `θ_k = −π + (k + ½)(2π/M)`.
/// `pub(crate)` so that `resolvent_jump_nd` can reuse the same coefficients.
#[inline]
pub(crate) fn contour_node(scale: f64, k: usize, step: f64) -> (Complex<f64>, Complex<f64>) {
    let theta = -core::f64::consts::PI + (k as f64 + 0.5) * step;
    let lam = Complex::new(
        scale * (TWS[0] - TWS[1] * theta * theta),
        scale * TWS[2] * theta,
    );
    let dlam = Complex::new(scale * (-2.0 * TWS[1] * theta), scale * TWS[2]);
    (lam, dlam)
}

// ---------------------------------------------------------------------------
// Scalar complex Thomas solve for (λI − A)r = g
// ---------------------------------------------------------------------------

/// Forward-backward Thomas solve for the complex tridiagonal `(λI − A)r = g`.
///
/// `A` is the 3-pt Neumann Laplacian: interior diag `−2/dx²`, off-diag `1/dx²`;
/// boundary diag `−1/dx²` (one-sided). Returns complex nodal solution.
///
/// # Errors
/// [`SemiflowError::DomainViolation`] if a pivot is (near) zero (λ on spectrum).
fn lhp_thomas<F: SemiflowFloat>(
    lam: Complex<f64>,
    inv_dx2: f64,
    sub: Complex<f64>,
    sup: Complex<f64>,
    n: usize,
    g: &GridFn1D<F>,
) -> Result<Vec<Complex<f64>>, SemiflowError> {
    // Diagonal entries: d[0] = λ − (−1/dx²) = λ + 1/dx²  (Neumann boundary)
    //                   d[k] = λ − (−2/dx²) = λ + 2/dx²  (interior)
    //                   d[n-1] same as d[0] by symmetry.
    let d_bnd = lam + Complex::new(inv_dx2, 0.0);
    let d_int = lam + Complex::new(2.0 * inv_dx2, 0.0);
    // Thomas forward sweep.
    let mut c_prime: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    let mut d_prime: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    // Row 0 (boundary, no sub-diagonal).
    let piv0 = d_bnd;
    if piv0.norm() < 1e-300 {
        return Err(SemiflowError::DomainViolation {
            what: "resolve_lhp: pivot near zero (λ on spectrum)",
            value: piv0.norm(),
        });
    }
    c_prime[0] = sup / piv0;
    d_prime[0] = Complex::new(g.values[0].to_f64().unwrap_or(0.0), 0.0) / piv0;
    for k in 1..n {
        let dk = if k == n - 1 { d_bnd } else { d_int };
        let piv = dk - sub * c_prime[k - 1];
        if piv.norm() < 1e-300 {
            return Err(SemiflowError::DomainViolation {
                what: "resolve_lhp: pivot near zero (λ on spectrum)",
                value: piv.norm(),
            });
        }
        c_prime[k] = sup / piv;
        let gk = Complex::new(g.values[k].to_f64().unwrap_or(0.0), 0.0);
        d_prime[k] = (gk - sub * d_prime[k - 1]) / piv;
    }
    // Back substitution.
    let mut r: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    r[n - 1] = d_prime[n - 1];
    for k in (0..n - 1).rev() {
        r[k] = d_prime[k] - c_prime[k] * r[k + 1];
    }
    Ok(r)
}
