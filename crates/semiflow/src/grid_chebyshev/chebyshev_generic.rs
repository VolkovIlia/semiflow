//! Generic-over-F Chebyshev spectral sampler (§46.5.bis, ADR-0139).
//!
//! Mirrors `sample_chebyshev_1d` (f64 path) EXACTLY: same Chebyshev-Lobatto
//! virtual-node placement, same barycentric Lagrange formula (Berrut-Trefethen
//! 2004 §5.1), with f64 table values promoted via `F::from(·)` and virtual-node
//! sampling via `sample_septic_1d_generic`.
//! No SIMD (§46.5 carve-out).  The f64 path is untouched (byte-identical, G1).
//!
//! Extracted into this child module to keep `grid_chebyshev.rs` under the
//! 500-line file budget (suckless/constitution Override #1).

use num_traits::Float;

use crate::{
    boundary::BoundaryPolicy,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_chebyshev_nodes::{chebyshev_nodes, chebyshev_weights, is_supported_m},
    grid_chebyshev_septic::sample_septic_1d_generic as sample_virtual_node_g,
};

// Guard factor: same as f64 path (8 x machine-epsilon x dx).
const EPSILON_FACTOR: f64 = 8.0;

// ---------------------------------------------------------------------------
// Literal-conversion helper.
// ---------------------------------------------------------------------------

// Deliberate inline(always): hot inner loop for float literal conversion in spectral kernels.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fc<F: SemiflowFloat>(v: f64) -> F {
    F::from(v).unwrap_or_else(F::zero)
}

// ---------------------------------------------------------------------------
// rem_euclid for generic F (no .rem_euclid on generic floats).
// ---------------------------------------------------------------------------

#[inline]
fn rem_euclid_g<F: SemiflowFloat>(a: F, b: F) -> F {
    let r = a - Float::floor(a / b) * b;
    if r < F::zero() {
        r + b.abs()
    } else {
        r
    }
}

// ---------------------------------------------------------------------------
// Out-of-domain helpers (generic mirrors of f64 grid_chebyshev.rs helpers)
// ---------------------------------------------------------------------------

fn reflect_into_domain_g<F: SemiflowFloat>(x: F, xmin: F, xmax: F) -> F {
    let width = xmax - xmin;
    let period = fc::<F>(2.0) * width;
    let t = rem_euclid_g(x - xmin, period);
    let folded = if t > width { period - t } else { t };
    folded + xmin
}

fn wrap_periodic_g<F: SemiflowFloat>(x: F, xmin: F, xmax: F) -> F {
    let width = xmax - xmin;
    let t = rem_euclid_g(x - xmin, width);
    let eps: F = fc::<F>(f64::EPSILON);
    xmin + t.min(width - eps * width.abs())
}

fn linear_extrapolate_g<F: SemiflowFloat>(values: &[F], grid: &Grid1D<F>, x: F, m: usize) -> F {
    let half = (grid.xmax - grid.xmin) * fc::<F>(0.5);
    let mid = (grid.xmax + grid.xmin) * fc::<F>(0.5);
    let nodes_ref = chebyshev_nodes(m).unwrap_or(&[]);
    let x0 = mid + half;
    let x1 = mid + half * fc::<F>(nodes_ref.get(1).copied().unwrap_or(0.0));
    let f0 = sample_virtual_node_g(values, grid, x0);
    let f1 = sample_virtual_node_g(values, grid, x1);
    let eps: F = fc::<F>(f64::EPSILON);
    let slope = if (x0 - x1).abs() > eps {
        (f0 - f1) / (x0 - x1)
    } else {
        F::zero()
    };
    if x > grid.xmax {
        return f0 + slope * (x - x0);
    }
    let xl = mid - half;
    let xl1 = mid + half * fc::<F>(nodes_ref.get(m.saturating_sub(1)).copied().unwrap_or(0.0));
    let fl = sample_virtual_node_g(values, grid, xl);
    let fl1 = sample_virtual_node_g(values, grid, xl1);
    let slope_l = if (xl - xl1).abs() > eps {
        (fl - fl1) / (xl - xl1)
    } else {
        F::zero()
    };
    fl + slope_l * (x - xl)
}

fn out_of_domain_sample_g<F: SemiflowFloat>(values: &[F], grid: &Grid1D<F>, x: F, m: usize) -> F {
    match grid.boundary {
        BoundaryPolicy::Reflect | BoundaryPolicy::Robin { .. } => {
            let reflected = reflect_into_domain_g(x, grid.xmin, grid.xmax);
            sample_chebyshev_spectral_1d_generic(values, grid, reflected, m)
                .unwrap_or_else(|_| F::zero())
        }
        BoundaryPolicy::Periodic => {
            let wrapped = wrap_periodic_g(x, grid.xmin, grid.xmax);
            sample_chebyshev_spectral_1d_generic(values, grid, wrapped, m)
                .unwrap_or_else(|_| F::zero())
        }
        BoundaryPolicy::ZeroExtend => F::zero(),
        BoundaryPolicy::LinearExtrapolate => linear_extrapolate_g(values, grid, x, m),
        BoundaryPolicy::Dirichlet { value } => value,
        BoundaryPolicy::Neumann => {
            let nearest = if x > grid.xmax { grid.xmax } else { grid.xmin };
            sample_virtual_node_g(values, grid, nearest)
        }
        // Odd-image: reflect into domain, negate. Mirrors Reflect path with sign flip.
        BoundaryPolicy::OddReflect => {
            let reflected = reflect_into_domain_g(x, grid.xmin, grid.xmax);
            let v = sample_chebyshev_spectral_1d_generic(values, grid, reflected, m)
                .unwrap_or_else(|_| F::zero());
            F::zero() - v
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Generic Chebyshev spectral sampler for `F: SemiflowFloat` (incl. `Dual<f64>`).
///
/// Same algorithm as `sample_chebyshev_1d` (f64 path): Chebyshev-Lobatto
/// virtual-node placement + barycentric Lagrange (Berrut-Trefethen 2004).
/// Virtual nodes are sampled via `sample_septic_1d_generic`.
///
/// # Errors
/// Returns `SemiflowError::DomainViolation` if `m` is not in the supported set.
pub(crate) fn sample_chebyshev_spectral_1d_generic<F: SemiflowFloat>(
    values: &[F],
    grid: &Grid1D<F>,
    x: F,
    m: usize,
) -> Result<F, SemiflowError> {
    if !is_supported_m(m) {
        return Err(SemiflowError::DomainViolation {
            what: "Chebyshev M must be in {8,16,32,64,128,256,512}",
            value: m as f64,
        });
    }

    if x < grid.xmin || x > grid.xmax {
        return Ok(out_of_domain_sample_g(values, grid, x, m));
    }

    let nodes_ref = chebyshev_nodes(m)?;
    let weights_ref = chebyshev_weights(m)?;

    let mid = (grid.xmax + grid.xmin) * fc::<F>(0.5);
    let half = (grid.xmax - grid.xmin) * fc::<F>(0.5);
    let dx_abs = grid.dx().abs();
    let guard = fc::<F>(EPSILON_FACTOR * f64::EPSILON) * dx_abs;

    let mut num = F::zero();
    let mut den = F::zero();

    for k in 0..=m {
        let x_k = mid + half * fc::<F>(nodes_ref[k]);
        let w_k = fc::<F>(weights_ref[k]);
        let diff = x - x_k;
        if diff.abs() < guard {
            return Ok(sample_virtual_node_g(values, grid, x_k));
        }
        let f_k = sample_virtual_node_g(values, grid, x_k);
        let term = w_k / diff;
        num += term * f_k;
        den += term;
    }

    Ok(num / den)
}
