//! [`interp_stencil`] — node offsets and weights for a 1-D interpolant (ADR-0190).
//!
//! Every interpolant `SemiFlow` supports is **linear in the node values**: sampling
//! at a cell-local fraction `s ∈ [0, 1)` is a fixed-offset weighted sum
//!
//! ```text
//!   f(x) ≈ Σ_k w_k(s) · f[idx + o_k]
//! ```
//!
//! where `idx = floor((x − xmin)/dx)`. The 1-D samplers in [`crate::grid`]
//! evaluate that sum directly; the d-dimensional sampler
//! [`crate::grid_nd::GridFnND::sample`] needs the `(offsets, weights)` pair
//! separately, because it forms the tensor product `Π_d w^{(d)}` over `K^D`
//! nodes. This module is the single source of truth for both: the generic 1-D
//! Catmull-Rom path (`crate::grid::catmull_rom_scalar_generic`) is expressed in
//! terms of these weights, so the 1-D and N-D paths cannot drift apart.
//!
//! ## Scope (honest limits, ADR-0190)
//!
//! `CubicHermite` (K=4) and `Linear` (K=2) are covered. `SepticHermite`,
//! `OctonicHermite` and `ChebyshevSpectralWithBC` are **not**: their nodal
//! weights are a composition of the Birkhoff–Garabedian–Lorentz polynomials
//! (§40.3) with the central-FD derivative stencils (§40.2), and extracting them
//! would mean rewriting samplers that carry a release-blocking bit-equality
//! contract (ADR-0018). They return [`SemiflowError::Unsupported`] from the
//! N-D path. This is not a limitation in practice for the defect ADR-0190
//! fixes: Catmull-Rom already removes the accumulated interpolation variance
//! entirely (residual ~1e-9, flat in the step count), whereas septic would cost
//! `8^D` nodes per sample against Catmull-Rom's `4^D`.

use crate::{error::SemiflowError, float::SemiflowFloat, grid::InterpKind};

/// Maximum stencil width returned by [`interp_stencil`].
///
/// `CubicHermite` uses all 4; `Linear` uses 2. Raising this is what a future
/// septic/octonic N-D extension would need (K=8 / K=10).
pub(crate) const K_MAX: usize = 4;

/// Whether `kind` has a `D > 1` tensor-product stencil.
///
/// Checked once when an N-D state is constructed ([`crate::grid_nd::GridFnND::new`])
/// so that per-sample calls cannot fail for an in-shape coordinate — the hot
/// kernels sample inside tight loops and must not carry a per-node `Result`.
pub(crate) fn supports_nd(kind: InterpKind) -> bool {
    matches!(kind, InterpKind::CubicHermite | InterpKind::Linear)
}

/// Node offsets (relative to `idx`) and weights for `kind` at cell fraction `s`.
///
/// Returns `(k, offsets, weights)` where only the first `k` entries are
/// meaningful. `Σ w_i = 1` exactly at `s = 0` and `s = 1` for every supported
/// kind, so a constant field is reproduced exactly (the `F(0) = I` precondition
/// the Chernoff product depends on).
///
/// # Errors
/// - [`SemiflowError::Unsupported`] for `SepticHermite`, `OctonicHermite` and
///   `ChebyshevSpectralWithBC` — see the module-level honest-limits note.
pub(crate) fn interp_stencil<F: SemiflowFloat>(
    kind: InterpKind,
    s: F,
) -> Result<(usize, [i64; K_MAX], [F; K_MAX]), SemiflowError> {
    match kind {
        InterpKind::CubicHermite => {
            let (offsets, weights) = catmull_rom_weights(s);
            Ok((4, offsets, weights))
        }
        InterpKind::Linear => {
            let zero = F::zero();
            Ok((
                2,
                [0, 1, 0, 0],
                [F::one() - s, s, zero, zero],
            ))
        }
        InterpKind::SepticHermite => Err(SemiflowError::Unsupported {
            feature: "SepticHermite in GridFnND::sample (use InterpKind::CubicHermite)",
        }),
        InterpKind::OctonicHermite => Err(SemiflowError::Unsupported {
            feature: "OctonicHermite in GridFnND::sample (use InterpKind::CubicHermite)",
        }),
        InterpKind::ChebyshevSpectralWithBC { .. } => Err(SemiflowError::Unsupported {
            feature: "ChebyshevSpectralWithBC in GridFnND::sample (global stencil)",
        }),
    }
}

/// Catmull-Rom (4-point cubic Hermite) nodal weights at fraction `s`.
///
/// Expanding the standard form
/// `½(2p₀ + (−p₋₁+p₁)s + (2p₋₁−5p₀+4p₁−p₂)s² + (−p₋₁+3p₀−3p₁+p₂)s³)`
/// by node gives the four polynomials below. Offsets are `[−1, 0, 1, 2]`.
fn catmull_rom_weights<F: SemiflowFloat>(s: F) -> ([i64; K_MAX], [F; K_MAX]) {
    let two = crate::float::two::<F>();
    let three = F::from(3.0_f64).unwrap_or_else(F::zero);
    let four = F::from(4.0_f64).unwrap_or_else(F::zero);
    let five = F::from(5.0_f64).unwrap_or_else(F::zero);
    let half = crate::float::half::<F>();
    let s2 = s * s;
    let s3 = s2 * s;
    // Named by node position rather than by offset (`w_m1`/`w_1` differ in one
    // character and read as typos of one another).
    let w_prev = half * (-s + two * s2 - s3);
    let w_cur = half * (two - five * s2 + three * s3);
    let w_next = half * (s + four * s2 - three * s3);
    let w_far = half * (-s2 + s3);
    ([-1, 0, 1, 2], [w_prev, w_cur, w_next, w_far])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catmull_rom_weights_are_a_partition_of_unity() {
        for k in 0..=10 {
            let s = f64::from(k) / 10.0;
            let (_, w) = catmull_rom_weights::<f64>(s);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-15, "s={s} sum={sum}");
        }
    }

    #[test]
    // Bit-exactness at `s = 0` is the claim, not an accidental float compare:
    // it is why sampling at a node reproduces that node's value to the last ULP.
    #[allow(clippy::float_cmp)]
    fn catmull_rom_weights_interpolate_nodes() {
        let (_, w0) = catmull_rom_weights::<f64>(0.0);
        assert_eq!(w0, [0.0, 1.0, 0.0, 0.0]);
        let (_, w1) = catmull_rom_weights::<f64>(1.0);
        assert!(w1[0].abs() < 1e-15 && (w1[2] - 1.0).abs() < 1e-15);
    }

    // The binding gate that ties these weights to the arithmetic the generic
    // 1-D sampler actually performs lives in `grid_tests.rs`
    // (`catmull_rom_matches_interp_stencil`), because `catmull_rom_scalar_generic`
    // is private to `grid` and `grid.rs` has no line budget to spare.

    #[test]
    fn linear_weights_are_a_partition_of_unity() {
        let (k, off, w) = interp_stencil::<f64>(InterpKind::Linear, 0.25).unwrap();
        assert_eq!(k, 2);
        assert_eq!(&off[..2], &[0, 1]);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn high_order_kinds_are_unsupported_in_nd() {
        assert!(interp_stencil::<f64>(InterpKind::SepticHermite, 0.5).is_err());
        assert!(interp_stencil::<f64>(InterpKind::OctonicHermite, 0.5).is_err());
    }
}
