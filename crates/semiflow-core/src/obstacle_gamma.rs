//! Inactive-set Γ = V″ primitive for [`ObstacleChernoff`] (math §44.5.bis).
//!
//! Separated from `obstacle.rs` to keep each file under 500 lines (constitution
//! v4.0.0 default cap; same split precedent as `reflection.rs` /
//! `reflection_regions.rs`).
//!
//! ## Honesty (NORMATIVE, math §44.5.bis)
//!
//! The value function V of the obstacle VI is C¹ across the free boundary `x*`
//! (smooth-fit, Peskir 2005), but Γ = V″ is **discontinuous** there:
//! - `Γ = 0` on the active (contact/stopping) set `{V = g}` (payoff is linear);
//! - `Γ > 0` strictly inside the continuation set `{V > g}`.
//!
//! Perpetual-American-put witness `(K, r, σ) = (1, 0.05, 0.20)`:
//! **Γ(S*⁺) ≈ 4.90**, Γ(S*⁻) = 0 — V is C¹ but NOT C² at `x*`.
//!
//! No classical **global** Γ exists. `apply_inactive_gamma_into` deliberately
//! exposes Γ ONLY where it mathematically exists (the open continuation set),
//! and REFUSES it (companion mask `defined[i] = false`) on the active set, at
//! the contact line, and within the one-node guard band. This is NOT
//! `ChernoffFunction::apply_gamma`; `ObstacleChernoff` does NOT implement a
//! global C² Greek surface.

extern crate alloc;
use alloc::vec;

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid_fn::GridFn1D,
    obstacle::{Obstacle, ObstacleChernoff},
};

// ---------------------------------------------------------------------------
// Private helper: inactive indicator with 3-point guard band
// ---------------------------------------------------------------------------

/// Compute the `defined` mask: `defined[i] = true` iff node `i` is in the
/// OPEN continuation set AND its 3-point centred stencil `{i−1, i, i+1}` lies
/// entirely within the continuation set (so the stencil never straddles `x*`).
///
/// Boundary nodes (`i = 0`, `i = n−1`) are always `false` (no centred stencil).
/// Any node adjacent to an active node is `false` (one-node guard band).
///
/// # Errors
/// `DomainViolation` if `v.grid.n != defined.len()`.
#[allow(clippy::cast_precision_loss)]
fn inactive_defined_mask<O, F>(
    v: &GridFn1D<F>,
    obstacle: &O,
    defined: &mut [bool],
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    O: Obstacle<F>,
{
    let n = v.grid.n;
    if defined.len() != n {
        return Err(SemiflowError::DomainViolation {
            what: "inactive_defined_mask: defined.len() must equal v.grid.n",
            value: defined.len() as f64,
        });
    }
    // Pass 1: per-node inactive indicator (strict: v > g).
    let mut inactive = vec![false; n];
    for (i, inp) in inactive.iter_mut().enumerate() {
        let x = v.grid.x_at(i);
        *inp = v.values[i] > obstacle.value_at(&[x]);
    }
    // Pass 2: defined iff interior AND all 3-point stencil nodes are inactive.
    for i in 0..n {
        defined[i] = i >= 1 && i + 1 < n && inactive[i - 1] && inactive[i] && inactive[i + 1];
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// apply_inactive_gamma_into — inherent method on ObstacleChernoff
// ---------------------------------------------------------------------------

impl<C, O, F> ObstacleChernoff<C, O, F>
where
    F: SemiflowFloat,
    O: Obstacle<F>,
{
    /// Inactive-set Γ = V″ of the projected value field, on the OPEN continuation
    /// set only (math §44.5.bis). Writes the central-difference second derivative
    /// `gamma[i] = (v[i+1] − 2·v[i] + v[i−1]) / dx²` at every node `i` that is
    /// strictly inside the continuation set AND whose 3-point stencil is itself
    /// entirely inside it; sets `defined[i] = true` there. On the active set, at
    /// the contact line `x*`, and within a one-node guard band of any active node
    /// (the stencil would straddle the kink), writes `gamma[i] = F::zero()` and
    /// `defined[i] = false` — Γ is REFUSED, not fabricated.
    ///
    /// Returns the number of nodes where Γ is defined (the inactive-set size).
    ///
    /// # Honesty (NORMATIVE, math §44.5.bis)
    ///
    /// Γ is C¹-not-C² at the free boundary: it JUMPS across `x*` (perpetual-put
    /// witness Γ(S*⁺) ≈ 4.90, Γ(S*⁻) = 0). There is NO classical global Γ; this
    /// primitive deliberately exposes Γ ONLY where it mathematically exists. Callers
    /// MUST consult `defined[i]` before reading `gamma[i]`; a `false` entry means
    /// "Γ undefined here", never "Γ = 0".
    ///
    /// This method is a SEPARATE inherent primitive, NOT part of any
    /// `ChernoffFunction` Greek or `AdjointApply` surface. It is D=1 only;
    /// multi-asset (D≥2) free-surface Γ is deferred (math §44.5.ter).
    ///
    /// # Errors
    ///
    /// `DomainViolation` if `v.grid.n != gamma.grid.n` or `!= defined.len()`,
    /// or if `v.grid.n < 3` (no interior 3-point stencil).
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_inactive_gamma_into(
        &self,
        v: &GridFn1D<F>,
        gamma: &mut GridFn1D<F>,
        defined: &mut [bool],
    ) -> Result<usize, SemiflowError>
    where
        C: crate::chernoff::ChernoffFunction<F, S = GridFn1D<F>>,
    {
        let n = v.grid.n;
        if gamma.grid.n != n || defined.len() != n {
            return Err(SemiflowError::DomainViolation {
                what: "apply_inactive_gamma_into: v/gamma/defined length mismatch",
                value: n as f64,
            });
        }
        if n < 3 {
            return Err(SemiflowError::DomainViolation {
                what: "apply_inactive_gamma_into: n < 3 (no interior stencil)",
                value: n as f64,
            });
        }
        // Compute guard-band inactive mask.
        inactive_defined_mask(v, self.obstacle(), defined)?;
        // dx_step = uniform grid spacing.
        let dx_step = (v.grid.xmax - v.grid.xmin) / from_f64::<F>((n - 1) as f64);
        let dx2 = dx_step * dx_step;
        let mut count = 0_usize;
        // Indexed loop required: stencil needs i-1 and i+1.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if defined[i] {
                // Central-difference: (v[i+1] - 2*v[i] + v[i-1]) / dx²
                let g = (v.values[i + 1] - v.values[i] - v.values[i] + v.values[i - 1]) / dx2;
                gamma.values[i] = g;
                count += 1;
            } else {
                gamma.values[i] = F::zero();
            }
        }
        Ok(count)
    }
}
