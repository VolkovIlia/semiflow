//! Higher-order divergence-form stencils for `∂_x(a(x) ∂_x ·)` (ADR-0118).
#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::doc_markdown)] // LoC, TRUTHFUL_ORDER are intentional technical terms
//!
//! Provides `apply_div_form_4th` (O(dx⁴)) and `apply_div_form_6th` (O(dx⁶))
//! as `pub(crate)` helpers for the ζ⁶/ζ⁸ TRUTHFUL_ORDER gate machinery.
//!
//! Extracted from `diffusion4_zeta4.rs` (633 LoC at extraction time) to stay
//! within the 700-LoC architectural cap. The existing 2nd-order `apply_div_form`
//! in `diffusion4_zeta4.rs` is UNCHANGED — callers that want the conservative
//! operator keep using it directly.
//!
//! ## Stencil definitions (sympy-verified, NORMATIVE — ADR-0118)
//!
//! Both stencils use staggered half-node evaluation: `a` is evaluated EXACTLY at
//! `x_{i±½} = x_i ± dx/2` (known coefficient, no FD error). The first derivative
//! at the half-node uses:
//!
//! 4th order: `(∂_x f)_{i+½} = (−f[i−1] + 27f[i] − 27f[i+1] + f[i+2]) / (24 dx)`
//! — offsets {−1, 0, 1, 2} relative to i+½.
//!
//! 6th order: `(∂_x f)_{i+½} = (−9f[i−2] + 125f[i−1] − 2250f[i] + 2250f[i+1]
//!             − 125f[i+2] + 9f[i+3]) / (1920 dx)`
//! — offsets {−5/2, −3/2, −1/2, +1/2, +3/2, +5/2} relative to i+½.
//!
//! The outer flux divergence applies the SAME 4- or 6-weight stencil to the half-node
//! flux values. Neumann BC: edge half-node fluxes are clamped per the existing
//! `apply_div_form` idiom (mirror the 3-point conservative pattern).
//!
//! Sympy oracles: `T_DIV_STENCIL_4TH` 4/4 PASS, `T_DIV_STENCIL_6TH` 4/4 PASS (ADR-0118).

use crate::{diffusion4::Diffusion4thChernoff, error::SemiflowError, grid_fn::GridFn1D};

// ---------------------------------------------------------------------------
// 4th-order stencil: apply_div_form_4th
// ---------------------------------------------------------------------------

/// Apply 4th-order divergence-form `A = ∂_x(a(x)·∂_x)` with Neumann BCs.
///
/// Half-node first derivative: `(∂_x f)_{i+½} = (−f_{i−1} + 27f_i − 27f_{i+1} + f_{i+2}) / (24 dx)`.
/// Outer divergence: same 4-weight stencil applied to flux values at half-node centres.
/// `a` evaluated EXACTLY at `x_{i+½} = x_i + 0.5·dx` — no FD error in coefficient.
///
/// Minimum grid size: n ≥ 5 (required for the 4-point inner stencil without full
/// ghost padding). Returns `DomainViolation` for n < 5.
#[allow(clippy::cast_precision_loss)] // n ≤ grid size; well within f64 mantissa
#[allow(dead_code)] // reserved for future 4th-order jet callers
pub(crate) fn apply_div_form_4th(
    dc: &Diffusion4thChernoff<f64>,
    f: &GridFn1D<f64>,
    out: &mut GridFn1D<f64>,
) -> Result<(), SemiflowError> {
    let n = f.values.len();
    if n < 5 {
        return Err(SemiflowError::DomainViolation {
            what: "4th-order divergence stencil requires >= 5 grid points",
            value: n as f64,
        });
    }
    let dx = dc.grid.dx();
    out.values.resize(n, 0.0);

    // Helper: clamp index to [0, n-1] for Neumann (zero-flux) BC.
    let clamp = |k: i64| -> usize {
        if k < 0 {
            0
        } else if k as usize >= n {
            n - 1
        } else {
            k as usize
        }
    };

    // Compute half-node fluxes F_{i+½} = a(x_{i+½}) · (∂_x f)_{i+½}.
    // 4-point stencil: (∂_x f)_{i+½} = (-f[i-1]+27f[i]-27f[i+1]+f[i+2]) / (24 dx).
    let get = |k: i64| f.values[clamp(k)];
    let x_at = |i: usize| dc.grid.x_at(i);
    let flux_cap = n + 2;
    let flux = build_half_node_flux(dc, n, dx, flux_cap, &get, &x_at);

    // Outer 4-pt divergence: apply to half-node flux values at each node.
    apply_outer_div_4pt(&flux, flux_cap, n, dx, &mut out.values);
    Ok(())
}

/// Apply the 4-point outer divergence operator to half-node flux values.
///
/// `flux[k]` is the flux at half-node `(k-1)+½`.
/// Uses clamping for boundary half-nodes (invariant L1/Neumann).
#[allow(clippy::cast_precision_loss)]
fn apply_outer_div_4pt(flux: &[f64], flux_cap: usize, n: usize, dx: f64, out: &mut [f64]) {
    // flux at half-nodes: node i uses: i-3/2 → k=i-1, i-1/2 → k=i, i+1/2 → k=i+1, i+3/2 → k=i+2
    let fk = |k: i64| {
        let idx = k + 1; // shift: flux stored at k+1 for half-node (k-1)+½
        if idx < 0 {
            flux[0]
        } else if idx as usize >= flux_cap {
            flux[flux_cap - 1]
        } else {
            flux[idx as usize]
        }
    };
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        let ii = i as i64;
        *slot = (-fk(ii - 1) + 27.0 * fk(ii) - 27.0 * fk(ii + 1) + fk(ii + 2)) / (24.0 * dx);
    }
}

// Build the extended half-node flux array for the 4th-order divergence stencil.
// flux[k] = F at half-node (k-1)+½ for k = 0..flux_cap-1.
// All 6 args are required: dc, n, dx, flux_cap, get, x_at.
#[allow(clippy::cast_precision_loss)]
fn build_half_node_flux(
    dc: &Diffusion4thChernoff<f64>,
    n: usize,
    dx: f64,
    flux_cap: usize,
    get: &impl Fn(i64) -> f64,
    x_at: &impl Fn(usize) -> f64,
) -> alloc::vec::Vec<f64> {
    let mut flux = alloc::vec![0.0_f64; flux_cap];
    for (k, slot) in flux.iter_mut().enumerate() {
        let j = k as i64 - 1;
        let x_half = if j >= 0 && (j as usize) < n {
            x_at(j as usize) + 0.5 * dx
        } else if j < 0 {
            x_at(0) + (j as f64 + 0.5) * dx
        } else {
            x_at(n - 1) + (j as f64 - (n as f64 - 1.0) + 0.5) * dx
        };
        let a_half = dc.eval_a(x_half);
        let deriv = (-get(j - 1) + 27.0 * get(j) - 27.0 * get(j + 1) + get(j + 2)) / (24.0 * dx);
        *slot = a_half * deriv;
    }
    flux
}

// ---------------------------------------------------------------------------
// 6th-order stencil: apply_div_form_6th  (THE GATE STENCIL per ADR-0118)
// ---------------------------------------------------------------------------

/// Apply 6th-order divergence-form `A = ∂_x(a(x)·∂_x)` with Neumann BCs.
///
/// Half-node first derivative (6-weight staggered, offsets {−5/2,−3/2,−1/2,+1/2,+3/2,+5/2}):
/// `(∂_x f)_{i+½} = (−9f[i−2] + 125f[i−1] − 2250f[i] + 2250f[i+1] − 125f[i+2] + 9f[i+3]) / (1920 dx)`.
/// Outer divergence: same 6-weight stencil applied to flux values.
/// `a` evaluated EXACTLY at `x_{i+½}` — no FD error in coefficient.
///
/// Minimum grid size: n ≥ 7 (6-point half-node stencil coverage). Returns
/// `DomainViolation` for n < 7.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn apply_div_form_6th(
    dc: &Diffusion4thChernoff<f64>,
    f: &GridFn1D<f64>,
    out: &mut GridFn1D<f64>,
) -> Result<(), SemiflowError> {
    let n = f.values.len();
    if n < 7 {
        return Err(SemiflowError::DomainViolation {
            what: "6th-order divergence stencil requires >= 7 grid points",
            value: n as f64,
        });
    }
    let dx = dc.grid.dx();
    out.values.resize(n, 0.0);
    let clamp = |k: i64| -> usize {
        if k < 0 {
            0
        } else if k as usize >= n {
            n - 1
        } else {
            k as usize
        }
    };
    let get = |k: i64| f.values[clamp(k)];
    let flux = build_half_node_flux_6th(dc, n, dx, get);
    let flux_cap = flux.len();
    let offset = 3_i64;
    for i in 0..n {
        let ii = i as i64;
        let fj = |j: i64| {
            let k = (j + offset) as usize;
            flux[k.min(flux_cap - 1)]
        };
        out.values[i] = (-9.0 * fj(ii - 3) + 125.0 * fj(ii - 2) - 2250.0 * fj(ii - 1)
            + 2250.0 * fj(ii)
            - 125.0 * fj(ii + 1)
            + 9.0 * fj(ii + 2))
            / (1920.0 * dx);
    }
    Ok(())
}

/// Build half-node flux vector for the 6th-order stencil.
///
/// Returns `flux[j + 3]` = F at half-node j+½ for j = −3..n+2.
fn build_half_node_flux_6th(
    dc: &Diffusion4thChernoff<f64>,
    n: usize,
    dx: f64,
    get: impl Fn(i64) -> f64,
) -> alloc::vec::Vec<f64> {
    let offset = 3_usize;
    let flux_cap = n + 2 * offset;
    let mut flux = alloc::vec![0.0_f64; flux_cap];
    for (k, slot) in flux.iter_mut().enumerate() {
        let j = k as i64 - offset as i64;
        let x_half = if j >= 0 && (j as usize) < n {
            dc.grid.x_at(j as usize) + 0.5 * dx
        } else if j < 0 {
            dc.grid.x_at(0) + (j as f64 + 0.5) * dx
        } else {
            dc.grid.x_at(n - 1) + (j as f64 - (n as f64 - 1.0) + 0.5) * dx
        };
        let deriv = (-9.0 * get(j - 2) + 125.0 * get(j - 1) - 2250.0 * get(j)
            + 2250.0 * get(j + 1)
            - 125.0 * get(j + 2)
            + 9.0 * get(j + 3))
            / (1920.0 * dx);
        *slot = dc.eval_a(x_half) * deriv;
    }
    flux
}

// ---------------------------------------------------------------------------
// K-jet iterators
// ---------------------------------------------------------------------------

/// Compute K-jet `[f, Af, ..., A^K f]` using the 4th-order operator (K iterations).
///
/// `out` must have length K+1. `out[0] = f` (identity).
#[allow(dead_code)] // reserved for future 4th-order jet callers
pub(crate) fn apply_jet_iter_4th(
    dc: &Diffusion4thChernoff<f64>,
    f: &GridFn1D<f64>,
    out: &mut [GridFn1D<f64>],
    k: usize,
) -> Result<(), SemiflowError> {
    out[0].values.clone_from(&f.values);
    apply_div_form_4th(dc, f, &mut out[1])?;
    for j in 1..k {
        let prev = out[j].clone();
        apply_div_form_4th(dc, &prev, &mut out[j + 1])?;
    }
    Ok(())
}

/// Compute K-jet `[f, Af, ..., A^K f]` using the 6th-order operator (K iterations).
///
/// `out` must have length K+1. `out[0] = f` (identity).
pub(crate) fn apply_jet_iter_6th(
    dc: &Diffusion4thChernoff<f64>,
    f: &GridFn1D<f64>,
    out: &mut [GridFn1D<f64>],
    k: usize,
) -> Result<(), SemiflowError> {
    out[0].values.clone_from(&f.values);
    apply_div_form_6th(dc, f, &mut out[1])?;
    for j in 1..k {
        let prev = out[j].clone();
        apply_div_form_6th(dc, &prev, &mut out[j + 1])?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (fast)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Diffusion4thChernoff, Grid1D, GridFn1D};

    fn make_dc(n: usize) -> Diffusion4thChernoff<f64> {
        let grid = Grid1D::new(-4.0, 4.0, n).expect("grid");
        Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid)
    }

    #[test]
    fn div_form_4th_produces_finite() {
        let dc = make_dc(32);
        let f = GridFn1D::from_fn(dc.grid, |x| (-x * x).exp());
        let mut out = f.zeroed_like();
        apply_div_form_4th(&dc, &f, &mut out).expect("4th-order div ok");
        assert!(out.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn div_form_6th_produces_finite() {
        let dc = make_dc(32);
        let f = GridFn1D::from_fn(dc.grid, |x| (-x * x).exp());
        let mut out = f.zeroed_like();
        apply_div_form_6th(&dc, &f, &mut out).expect("6th-order div ok");
        assert!(out.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn div_form_6th_rejects_small_grid() {
        let dc = make_dc(6);
        let f = GridFn1D::from_fn(dc.grid, |x| (-x * x).exp());
        let mut out = f.zeroed_like();
        assert!(apply_div_form_6th(&dc, &f, &mut out).is_err());
    }

    #[test]
    fn jet_iter_6th_produces_finite() {
        let dc = make_dc(32);
        let f = GridFn1D::from_fn(dc.grid, |x| (-x * x).exp());
        let mut out: [GridFn1D<f64>; 5] = core::array::from_fn(|_| f.zeroed_like());
        apply_jet_iter_6th(&dc, &f, &mut out, 4).expect("jet_iter_6th ok");
        for (j, s) in out.iter().enumerate() {
            assert!(s.values.iter().all(|v| v.is_finite()), "jet[{j}] has nan");
        }
    }
}
