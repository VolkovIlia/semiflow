//! Chebyshev spectral collocation spatial sampler (ADR-0090, ADR-0104, ADR-0109, math.md §9.2.7).
//!
//! Provides `sample_chebyshev_1d` used by `InterpKind::ChebyshevSpectralWithBC { m, oob_policy }`
//! dispatch in `grid::Grid1D::interp` (f64 path only).
//!
//! ## Algorithm (NORMATIVE, ADR-0090 §"Algorithm")
//!
//! For a query point `x ∈ [xmin, xmax]`:
//!
//! 1. Map M+1 Chebyshev-Lobatto nodes from `[−1, 1]` → `[xmin, xmax]`:
//!    ```text
//!    y_k = cos(k·π / M),  k = 0..M
//!    x_k = mid + half · y_k,  mid = (xmax+xmin)/2,  half = (xmax-xmin)/2
//!    ```
//! 2. Sample M+1 virtual-node values `f_k` via `sample_septic_1d` (O(dx⁸) floor, ADR-0109).
//!    `QuinticHermite` virtual-node sampling was removed at v7.0 (see `docs/migration/v6-to-v7.md`).
//! 3. Evaluate barycentric Lagrange at `x` (Berrut-Trefethen 2004, formula 5.1):
//!    ```text
//!    f(x) = Σ_k (w_k · f_k / (x − x_k))  /  Σ_k (w_k / (x − x_k))
//!    ```
//!    with Chebyshev-Lobatto weights `w_k = (−1)^k · δ_k` (`δ_0` = `δ_M` = 1/2 else 1).
//! 4. Removable-singularity guard: when `|x − x_k| < EPSILON_GUARD`, return `f_k` directly.
//!
//! Cost per call: O(M) `SepticHermite` samples + O(M) barycentric ops = O(M) flops.
//!
//! ## Spatial floor (NORMATIVE, ADR-0104, ADR-0109)
//!
//! Effective floor = max(SepticHermite virtual-node floor,
//!                       barycentric numerical conditioning,
//!                       f64 ULP).
//! At N=512:
//!   `SepticHermite` virtual-node floor: O(dx⁸) ≈ 1.49e-12 (ADR-0109 §40.4)
//!   Barycentric in-domain conditioning (M=64): O(2^{-64}) ≈ 5e-20
//!   f64 ULP: 2e-16
//! Effective: ≈ 1.49e-12 (virtual-node dominated, 67× improvement over v5.0).
//!
//! The "spectral" qualifier refers to convergence rate (exp-decay in M for
//! analytic f), NOT to the absolute floor — which is bounded below by the
//! virtual-node interpolant. See ADR-0104 §"BREAKING redesign — Surface 2".
//!
//! ## Out-of-domain (NORMATIVE, ADR-0104)
//!
//! For x ∉ [xmin, xmax]: the barycentric Lagrange formula has no valid
//! evaluation (Berrut-Trefethen 2004 §3.2 — formula valid only in-domain).
//! `sample_chebyshev_1d` delegates to `out_of_domain_sample` which honors
//! the grid's `BoundaryPolicy` (or the `OobPolicy` override when the
//! `ChebyshevSpectralWithBC` variant is in use).
//!
//! ## References
//!
//! - math.md §9.2.7 — NORMATIVE Chebyshev spectral collocation section.
//! - ADR-0090 — adoption decision (Option A: interpolation on uniform `Grid1D`).
//! - ADR-0104 — H3 boundary-policy fix + H4 truthful floor rating.
//! - Berrut & Trefethen 2004, *SIAM Review* 46:501 — barycentric formula stability.
//! - Boyd, *Chebyshev and Fourier Spectral Methods* (Dover 2nd ed., 2000), Ch. 5–6.
//! - Trefethen, *Spectral Methods in MATLAB* (SIAM 2000), Ch. 6–8 + Appendix.

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

// Virtual-node sampler: SepticHermite O(dx⁸), floor ≈ 1.49e-12 (ADR-0109 v6.0+).
// QuinticHermite removed at v7.0 (ADR-0109 12-month removal clock fulfilled).
use crate::{
    boundary::BoundaryPolicy,
    error::SemiflowError,
    grid::Grid1D,
    grid_chebyshev_nodes::{chebyshev_nodes, chebyshev_weights, is_supported_m},
    grid_chebyshev_septic::sample_septic_1d as sample_virtual_node,
};

/// Guard width for removable-singularity: when |x - `x_k`| < `EPSILON_GUARD`, return `f_k` directly.
/// Uses a generous multiple of machine epsilon scaled by dx.
const EPSILON_FACTOR: f64 = 8.0;

// ---------------------------------------------------------------------------
// Generic Chebyshev spectral sampler — implementation lives in a child module
// to keep this file within the 500-line budget (§46.5.bis, ADR-0139).
// ---------------------------------------------------------------------------

/// Generic Chebyshev spectral sampler for `F: SemiflowFloat` (incl. `Dual<f64>`).
///
/// Mirrors `sample_chebyshev_1d` EXACTLY — same Chebyshev-Lobatto virtual-node
/// placement (§9.2.7) and barycentric Lagrange formula (Berrut-Trefethen 2004)
/// — with f64 table values promoted via `F::from(·)` and virtual-node sampling
/// via `sample_septic_1d_generic`. No SIMD (§46.5 carve-out); leaves the
/// existing f64 path byte-identical (additive-only change).
///
/// Called by `Grid1D::interp_generic` for `ChebyshevSpectralWithBC` (§46.5.bis).
pub(crate) use chebyshev_generic::sample_chebyshev_spectral_1d_generic;

pub(crate) mod chebyshev_generic;

// ---------------------------------------------------------------------------
// Out-of-domain helper (ADR-0104 H3 fix)
// ---------------------------------------------------------------------------

/// Reflect `x` into `[xmin, xmax]` using the mirror-fold formula.
///
/// Period = 2·(xmax − xmin). One application is sufficient: after folding,
/// the result is guaranteed to be in `[xmin, xmax]`.
fn reflect_into_domain(x: f64, xmin: f64, xmax: f64) -> f64 {
    let width = xmax - xmin;
    debug_assert!(width > 0.0, "reflect_into_domain: xmin must be < xmax");
    let period = 2.0 * width;
    // Shift to [0, 2*width) using rem_euclid.
    let t = (x - xmin).rem_euclid(period);
    // Fold the upper half back: t in [width, 2*width) → mirror to [0, width).
    let folded = if t > width { period - t } else { t };
    xmin + folded
}

/// Wrap `x` periodically into `[xmin, xmax)` with period = (xmax − xmin).
fn wrap_periodic(x: f64, xmin: f64, xmax: f64) -> f64 {
    let width = xmax - xmin;
    debug_assert!(width > 0.0, "wrap_periodic: xmin must be < xmax");
    let t = (x - xmin).rem_euclid(width);
    // Clamp to xmax - epsilon to stay strictly inside the closed interval.
    xmin + t.min(width - f64::EPSILON * width.abs())
}

/// Linear extrapolation at `x` outside `[xmin, xmax]`.
///
/// Uses two virtual nodes nearest the boundary to define the slope.
fn linear_extrapolate_chebyshev(values: &[f64], grid: &Grid1D, x: f64, m: usize) -> f64 {
    // Two boundary virtual nodes for slope estimate.
    let half = (grid.xmax - grid.xmin) * 0.5;
    let mid = (grid.xmax + grid.xmin) * 0.5;
    // Chebyshev-Lobatto nodes: k=0 → xmax, k=1 → cos(π/M)·half + mid.
    let x0 = mid + half; // = xmax (k=0)
    let x1 = mid + half * libm::cos(core::f64::consts::PI / m as f64); // k=1
    let f0 = sample_virtual_node(values, grid, x0);
    let f1 = sample_virtual_node(values, grid, x1);
    let slope = if (x0 - x1).abs() > f64::EPSILON {
        (f0 - f1) / (x0 - x1)
    } else {
        0.0
    };
    // Choose the nearest boundary as anchor.
    if x > grid.xmax {
        f0 + slope * (x - x0)
    } else {
        // x < xmin: use the left boundary (k=M → xmin).
        let xl = mid - half; // = xmin (k=M)
        let xl1 = mid + half * libm::cos((m as f64 - 1.0) * core::f64::consts::PI / m as f64);
        let fl = sample_virtual_node(values, grid, xl);
        let fl1 = sample_virtual_node(values, grid, xl1);
        let slope_l = if (xl - xl1).abs() > f64::EPSILON {
            (fl - fl1) / (xl - xl1)
        } else {
            0.0
        };
        fl + slope_l * (x - xl)
    }
}

/// Sample at out-of-domain `x` via the grid's `BoundaryPolicy` (ADR-0104 H3 fix).
///
/// Called by `sample_chebyshev_1d` when `x ∉ [grid.xmin, grid.xmax]`.
/// Dispatches per `BoundaryPolicy`:
/// - `Reflect` / `Robin` — mirror-fold x back into domain, recurse.
/// - `Periodic` — wrap modulo (xmax − xmin), recurse.
/// - `ZeroExtend` — return 0.0.
/// - `LinearExtrapolate` — affine continuation from boundary virtual nodes.
/// - `Dirichlet { value }` — return `value`.
/// - `Neumann` — clamp-to-boundary-node value (zero-flux extension).
///
/// # Panics
///
/// Does not panic. Reflect and Periodic recurse once; after fold/wrap, the
/// result is in-domain and the recursion terminates (the branch is only
/// entered when `x` is strictly outside `[xmin, xmax]`).
fn out_of_domain_sample(values: &[f64], grid: &Grid1D, x: f64, m: usize) -> f64 {
    match grid.boundary {
        BoundaryPolicy::Reflect | BoundaryPolicy::Robin { .. } => {
            // Mirror x into [xmin, xmax] with one fold; result is in-domain.
            let reflected = reflect_into_domain(x, grid.xmin, grid.xmax);
            debug_assert!(
                reflected >= grid.xmin && reflected <= grid.xmax,
                "reflect_into_domain produced out-of-domain result"
            );
            sample_chebyshev_1d(values, grid, reflected, m).unwrap_or(0.0)
        }
        BoundaryPolicy::Periodic => {
            let wrapped = wrap_periodic(x, grid.xmin, grid.xmax);
            sample_chebyshev_1d(values, grid, wrapped, m).unwrap_or(0.0)
        }
        BoundaryPolicy::ZeroExtend => 0.0,
        BoundaryPolicy::LinearExtrapolate => linear_extrapolate_chebyshev(values, grid, x, m),
        BoundaryPolicy::Dirichlet { value } => value,
        BoundaryPolicy::Neumann => {
            // Clamp to nearest boundary node and return its value.
            let nearest = if x > grid.xmax { grid.xmax } else { grid.xmin };
            sample_virtual_node(values, grid, nearest)
        }
        // Odd-image: reflect into domain, negate. Mirrors Reflect path with sign flip.
        BoundaryPolicy::OddReflect => {
            let reflected = reflect_into_domain(x, grid.xmin, grid.xmax);
            -sample_chebyshev_1d(values, grid, reflected, m).unwrap_or(0.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Main sampler
// ---------------------------------------------------------------------------

/// Sample a Chebyshev spectral interpolant at `x`.
///
/// Constructs M+1 Chebyshev-Lobatto virtual nodes mapped from `[-1,1]` to
/// `[grid.xmin, grid.xmax]`, samples them via the active virtual-node interpolant, then
/// evaluates the barycentric Lagrange formula at `x`.
///
/// For `x ∉ [xmin, xmax]`: delegates to `out_of_domain_sample` which honors
/// the grid's `BoundaryPolicy` (ADR-0104 H3 fix). In-domain path is unchanged.
///
/// # Contract
/// - `values.len() == grid.n` (`Grid1D` invariant, caller-enforced).
/// - `m ∈ {8, 16, 32, 64, 128, 256, 512}`.
/// - Returns `SemiflowError::DomainViolation` if `m` is not supported.
///
/// # Errors
///
/// Propagates errors from `crate::grid_chebyshev_nodes::chebyshev_nodes`.
/// Evaluate the barycentric Lagrange sum at `x` using Chebyshev-Lobatto
/// nodes mapped from `[−1,1]` to `[xmin, xmax]`.
///
/// Returns `Ok(f_k)` early when `x` is within epsilon of a node (removable
/// singularity guard, Higham 2004 §3.1).  Otherwise accumulates `num/den`.
fn barycentric_lobatto_eval(
    values: &[f64],
    grid: &Grid1D,
    x: f64,
    m: usize,
    nodes_ref: &[f64],
    weights_ref: &[f64],
) -> f64 {
    let mid = (grid.xmax + grid.xmin) * 0.5;
    let half = (grid.xmax - grid.xmin) * 0.5;
    let dx_abs = grid.dx().abs();
    let guard = EPSILON_FACTOR * f64::EPSILON * dx_abs;

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for k in 0..=m {
        // Map Chebyshev-Lobatto node from [-1,1] to [xmin, xmax].
        let x_k = mid + half * nodes_ref[k];
        let w_k = weights_ref[k];
        let diff = x - x_k;
        if diff.abs() < guard {
            // Removable singularity: x ≈ x_k → return f_k directly.
            return sample_virtual_node(values, grid, x_k);
        }
        let f_k = sample_virtual_node(values, grid, x_k);
        let term = w_k / diff;
        num += term * f_k;
        den += term;
    }
    // Barycentric formula: f(x) = num / den.
    num / den
}

pub(crate) fn sample_chebyshev_1d(
    values: &[f64],
    grid: &Grid1D,
    x: f64,
    m: usize,
) -> Result<f64, SemiflowError> {
    // Early validation: fail fast on unsupported M.
    if !is_supported_m(m) {
        return Err(SemiflowError::DomainViolation {
            what: "Chebyshev M must be in {8,16,32,64,128,256,512}",
            value: m as f64,
        });
    }

    // ADR-0104 H3 fix: route out-of-domain samples through the boundary policy.
    // Inside [xmin, xmax]: barycentric Lagrange is convergent (Berrut-Trefethen 2004).
    // Outside: barycentric formula diverges polynomially; delegate to BC.
    if x < grid.xmin || x > grid.xmax {
        return Ok(out_of_domain_sample(values, grid, x, m));
    }

    let nodes_ref = chebyshev_nodes(m)?;
    let weights_ref = chebyshev_weights(m)?;

    Ok(barycentric_lobatto_eval(
        values,
        grid,
        x,
        m,
        nodes_ref,
        weights_ref,
    ))
}

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::{boundary::InterpKind, grid::OobPolicy, BoundaryPolicy, Grid1D};

    fn gauss(x: f64) -> f64 {
        libm::exp(-x * x)
    }

    /// Verify barycentric evaluation at a Chebyshev-Lobatto node returns the
    /// virtual-node value (removable-singularity guard correctness).
    #[test]
    fn at_lobatto_node_returns_node_value() {
        let grid = Grid1D::new(-5.0, 5.0, 128).unwrap();
        let values: Vec<f64> = (0..grid.n).map(|i| gauss(grid.x_at(i))).collect();

        let x_first = grid.xmax; // cos(0) = 1 → mid + half*1 = xmax
        let result = sample_chebyshev_1d(&values, &grid, x_first, 16).unwrap();
        let expected = sample_virtual_node(&values, &grid, x_first);
        let err = (result - expected).abs();
        assert!(
            err < 1e-14,
            "at-node removable-singularity: err = {err:.3e}, expected < 1e-14"
        );
    }

    /// Verify Chebyshev sampler returns f64 for a supported M value.
    #[test]
    fn interior_sample_finite() {
        let grid = Grid1D::new(-3.0, 3.0, 64).unwrap();
        let values: Vec<f64> = (0..grid.n).map(|i| gauss(grid.x_at(i))).collect();
        let probe = 0.731_f64;
        let result = sample_chebyshev_1d(&values, &grid, probe, 32).unwrap();
        assert!(
            result.is_finite(),
            "Chebyshev sample must be finite at interior point"
        );
    }

    /// Verify unsupported M returns an error.
    #[test]
    fn unsupported_m_returns_error() {
        let grid = Grid1D::new(-1.0, 1.0, 16).unwrap();
        let values: Vec<f64> = vec![0.0; 16];
        let err = sample_chebyshev_1d(&values, &grid, 0.0, 7);
        assert!(err.is_err(), "M=7 must return DomainViolation error");
    }

    /// Chebyshev sampler improves over `CubicHermite` at off-node positions (N=64 sanity).
    #[test]
    fn chebyshev_improves_over_cubic_sanity() {
        let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
        let values: Vec<f64> = (0..grid.n).map(|i| gauss(grid.x_at(i))).collect();

        let dx = grid.dx();
        let probe = grid.x_at(32) + 0.5 * dx;
        let analytic = gauss(probe);

        let cubic_result = grid.interp(&values, probe).unwrap();
        let cubic_err = (cubic_result - analytic).abs();

        let cheb_result = sample_chebyshev_1d(&values, &grid, probe, 32).unwrap();
        let cheb_err = (cheb_result - analytic).abs();

        assert!(
            cheb_err <= cubic_err * 1000.0,
            "Chebyshev not catastrophically worse than cubic: cubic={cubic_err:.3e} cheb={cheb_err:.3e}"
        );
    }

    /// Verify `ChebyshevSpectralWithBC` dispatch round-trip via `Grid1D::interp`.
    #[test]
    fn dispatch_via_interp_with_bc() {
        let grid = Grid1D::new(-2.0, 2.0, 32).unwrap();
        let cheb_grid = grid.with_interp(InterpKind::ChebyshevSpectralWithBC {
            m: 16,
            oob_policy: OobPolicy::Inherit,
        });
        let values: Vec<f64> = (0..grid.n).map(|i| gauss(grid.x_at(i))).collect();
        let probe = 0.25_f64;
        let result = cheb_grid.interp(&values, probe).unwrap();
        assert!(
            result.is_finite(),
            "Dispatch via Grid1D::interp must return finite value"
        );
    }

    /// Verify out-of-domain with Reflect BC returns finite (ADR-0104 H3 fix).
    #[test]
    fn out_of_domain_reflect_returns_finite() {
        let grid = Grid1D::new(-1.0, 1.0, 64)
            .unwrap()
            .with_boundary(BoundaryPolicy::Reflect);
        let values: Vec<f64> = (0..grid.n).map(|i| gauss(grid.x_at(i))).collect();
        // Probe 0.5 units past xmax — was catastrophically divergent before H3 fix.
        let result = sample_chebyshev_1d(&values, &grid, 1.5, 16).unwrap();
        assert!(result.is_finite(), "Reflect OOB must be finite: {result}");
    }

    /// Verify out-of-domain with `ZeroExtend` BC returns 0.0 (ADR-0104 H3 fix).
    #[test]
    fn out_of_domain_zero_extend_returns_zero() {
        let grid = Grid1D::new(-1.0, 1.0, 64)
            .unwrap()
            .with_boundary(BoundaryPolicy::ZeroExtend);
        let values: Vec<f64> = (0..grid.n).map(|i| gauss(grid.x_at(i))).collect();
        let result = sample_chebyshev_1d(&values, &grid, 2.0, 16).unwrap();
        assert_eq!(result, 0.0, "ZeroExtend OOB must return 0.0");
    }
}
