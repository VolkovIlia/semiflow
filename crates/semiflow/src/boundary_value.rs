//! Accessor-based boundary resolution — the half of [`crate::boundary`] that
//! turns a [`crate::boundary::BoundaryHit`] into a value.
//!
//! Split out of `boundary.rs` for the 500-line budget (constitution Override #1
//! was retired at v6.0.0, so the cap is hard). The seam is the natural one:
//! `boundary.rs` answers *which node* an out-of-range index maps to,
//! this module answers *what value* that mapping yields, over an arbitrary
//! accessor so the contiguous 1-D and strided N-D paths share one implementation.

use crate::{
    boundary::{bc_index, BoundaryHit, BoundaryPolicy},
    float::SemiflowFloat,
};

// ---------------------------------------------------------------------------
// bc_value_by (accessor-based — serves strided N-D lines, ADR-0191)
// ---------------------------------------------------------------------------

/// `bc_value_generic` over an arbitrary node accessor instead of a slice.
///
/// `get(i)` must return the axis value at in-range node `i ∈ [0, n)`. This is
/// the single source of truth for boundary-policy resolution: `bc_value_generic`
/// delegates here with `|i| values[i]`, and [`crate::grid_nd::GridFnND::sample`]
/// delegates here with a strided accessor `|i| values[base + i * stride]`, so
/// the 1-D and N-D paths cannot drift apart (ADR-0191).
pub(crate) fn bc_value_by<F: SemiflowFloat, G: Fn(usize) -> F>(
    boundary: BoundaryPolicy<F>,
    get: G,
    n: usize,
    idx: i64,
    dx: F,
) -> F {
    bc_value_from_hit(bc_index(boundary, n, idx), boundary, get, n, dx)
}

/// The value half of [`bc_value_by`], with the index resolution already done.
///
/// Split out so a caller resolving the same `(boundary, n, idx)` repeatedly can
/// call [`bc_index`] once and reuse the hit — `GridFnND::sample` re-visits each
/// axis's stencil `K^(D-1)` times, i.e. 1364 resolutions per sample at `D = 5`
/// against 20 distinct ones (ADR-0191 AM 4). The arithmetic is byte-for-byte
/// what was inline in `bc_value_by`, in the same order, so results are
/// bit-identical.
pub(crate) fn bc_value_from_hit<F: SemiflowFloat, G: Fn(usize) -> F>(
    hit: BoundaryHit<F>,
    boundary: BoundaryPolicy<F>,
    get: G,
    n: usize,
    dx: F,
) -> F {
    let half = crate::float::half::<F>();
    let three = F::from(3.0_f64).unwrap_or_else(F::zero);
    let four = F::from(4.0_f64).unwrap_or_else(F::zero);
    match hit {
        BoundaryHit::Inside(i) => get(i),
        BoundaryHit::Zero => F::zero(),
        BoundaryHit::Dirichlet(v) => v,
        BoundaryHit::OutsideLeft(d) => {
            let slope_combo = -three * get(0) + four * get(1) - get(2);
            let d_f = F::from(f64::from(d)).unwrap_or_else(F::zero);
            get(0) - d_f * half * slope_combo
        }
        BoundaryHit::OutsideRight(d) => {
            let slope_combo = three * get(n - 1) - four * get(n - 2) + get(n - 3);
            let d_f = F::from(f64::from(d)).unwrap_or_else(F::zero);
            get(n - 1) + d_f * half * slope_combo
        }
        BoundaryHit::RobinSkew { reflected, depth } => {
            let BoundaryPolicy::Robin { alpha, beta } = boundary else {
                // Unreachable: RobinSkew is only produced by Robin policy.
                return get(reflected); // even-reflect fallback, no panic
            };
            let two = F::from(2.0_f64).unwrap_or_else(F::zero);
            let d_f = F::from(f64::from(depth)).unwrap_or_else(F::zero);
            let exponent = -(two * (alpha / beta) * d_f * dx);
            exponent.exp() * get(reflected)
        }
        // Odd-image: negate the mirrored interior value (ADR-0176, math §21.9).
        BoundaryHit::OddReflected { reflected } => F::zero() - get(reflected),
    }
}
