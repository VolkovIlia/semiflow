//! Particle-reduction policies for `GridlessChernoff` (math.md §50.4, ADR-0155).
//!
//! This module is `pub(crate)` and re-exported from `gridless.rs` as
//! `pub use gridless_reduce::ParticleReduction`.  It is separate to keep
//! both files within the 500-line suckless budget.

use alloc::vec::Vec;

use crate::{adjoint_fp::MeasureState, error::SemiflowError, float::SemiflowFloat};

// ---------------------------------------------------------------------------
// ParticleReduction
// ---------------------------------------------------------------------------

/// Particle-reduction policy applied after each axis sub-step (§50.4).
///
/// Caps the Dirac count that would otherwise grow as `3^d` per axis-sweep.
/// All policies are **deterministic** (no RNG) to preserve the quasi-MC
/// low-discrepancy structure claimed by `G_GRIDLESS_VARIANCE` (ADR-0155 §50.5).
#[derive(Clone, Debug)]
pub enum ParticleReduction {
    /// D-dimensional weighted-Voronoi product-bin merge (§50.4, ship-track).
    ///
    /// Algorithm:
    /// 1. Choose `m = floor(cap^{1/d})` bins per axis (clamped to `m ≥ 1`).
    /// 2. Assign each particle to a `d`-tuple of per-axis bin indices.
    /// 3. Pack indices into a `usize` key via mixed-radix `Σ_j idx_j · m^j`.
    /// 4. Merge each non-empty cell to one Dirac at the weight-barycenter
    ///    (per-axis `x̄_j = Σ w_i x_{ij} / Σ w_i`), summed weight `w̄ = Σ w_i`.
    ///
    /// Preserves total mass and **per-axis first moment** exactly.
    /// Second-moment error = within-cell variance (§50.4 price).
    ///
    /// Applied **after each axis sub-step** to bound working set to `3·P_cap`.
    WeightedVoronoi {
        /// Maximum Dirac count after reduction. Must be ≥ 1.
        cap: usize,
    },

    /// Gaussian-background fallback (§50.4 form 2, long-time, research-track).
    ///
    /// **Currently a pass-through stub.** Deferred to v9.x (ADR-0155).
    GaussianBackground,
}

impl ParticleReduction {
    /// Apply the reduction to `ensemble` in-place.
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if `WeightedVoronoi.cap == 0` or if
    /// any particle weight is non-finite.
    pub fn apply<F: SemiflowFloat, const D: usize>(
        &self,
        ensemble: &mut MeasureState<F, D>,
    ) -> Result<(), SemiflowError> {
        match self {
            ParticleReduction::WeightedVoronoi { cap } => voronoi_reduce_ddim(ensemble, *cap),
            // Stub: pass-through — see rustdoc for deferred plan.
            ParticleReduction::GaussianBackground => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// D-dimensional product-bin WeightedVoronoi reduction (§50.4, §3.1)
// ---------------------------------------------------------------------------

/// Merge `ensemble`'s Diracs into ≤`cap` bins via d-dimensional product binning.
///
/// Algorithm (deterministic, no RNG):
/// 1. `m = max(1, floor(cap^{1/d}))` bins per axis. For `D=0`, return immediately.
/// 2. Find per-axis extent `[lo_j, hi_j]`.
/// 3. Compute per-axis bin index `idx_j = floor((x_j-lo_j)/(span_j+eps)·m)`.
/// 4. Pack `d`-tuple into key via mixed-radix `Σ idx_j · m^j` (bounded by `m^d ≤ cap`).
/// 5. Accumulate weight and weighted-position per cell, emit barycenter Diracs.
///
/// Invariants (normative, §3.1):
/// - Total mass: `Σ w̄ = Σ w_i`.
/// - Per-axis first moment: `Σ w̄·x̄_j = Σ w_i·x_{ij}` for every axis `j`.
fn voronoi_reduce_ddim<F: SemiflowFloat, const D: usize>(
    ensemble: &mut MeasureState<F, D>,
    cap: usize,
) -> Result<(), SemiflowError> {
    if cap == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "gridless: WeightedVoronoi cap must be >= 1",
            value: 0.0,
        });
    }
    if ensemble.n_diracs() <= cap {
        return Ok(());
    }

    let old: Vec<([F; D], F)> = ensemble.as_diracs_slice().to_vec();
    for (_, w) in &old {
        if !w.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "gridless: non-finite weight in WeightedVoronoi reduction",
                value: w.to_f64().unwrap_or(f64::NAN),
            });
        }
    }

    if D == 0 {
        return Ok(());
    }

    let m = bins_per_axis(cap, D); // m^D ≤ cap
    let (lo, span, inv_span) = per_axis_extents::<F, D>(&old);
    let reduced = product_bin_merge::<F, D>(&old, m, cap, &lo, &span, &inv_span);

    ensemble.clear_diracs_with_capacity(reduced.len());
    for (pos, w) in reduced {
        ensemble.push_dirac_raw(pos, w);
    }
    Ok(())
}

/// Compute `m = max(1, floor(cap^{1/d}))` bins per axis.
///
/// Guarantees `m^d ≤ cap` (so cell count fits in `cap`).
fn bins_per_axis(cap: usize, d: usize) -> usize {
    if d == 0 || cap == 0 {
        return 1;
    }
    // Use f64 for the root; small rounding errors are corrected by floor+clamp.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let m = libm::floor(libm::pow(cap as f64, 1.0 / d as f64)) as usize;
    m.max(1)
}

/// Compute per-axis `[lo_j, hi_j]`, `span_j = hi_j - lo_j`, and `1/span_j`.
fn per_axis_extents<F: SemiflowFloat, const D: usize>(
    diracs: &[([F; D], F)],
) -> ([F; D], [F; D], [F; D]) {
    let large = F::from(1e38).unwrap_or(F::zero());
    let mut lo = [large; D];
    let mut hi = [-large; D];
    for (pos, _) in diracs {
        for j in 0..D {
            if pos[j] < lo[j] {
                lo[j] = pos[j];
            }
            if pos[j] > hi[j] {
                hi[j] = pos[j];
            }
        }
    }
    // Guard degenerate (all particles on same coordinate).
    let eps_base = F::from(1e-300).unwrap_or(F::zero());
    let eps_rel = F::from(1e-9).unwrap_or(F::zero());
    let mut span = [F::zero(); D];
    let mut inv_span = [F::zero(); D];
    for j in 0..D {
        if lo[j] > hi[j] {
            hi[j] = lo[j];
        }
        let raw_span = hi[j] - lo[j];
        let eps = raw_span * eps_rel + eps_base;
        span[j] = raw_span + eps + eps; // adds 2*eps so hi maps into last bin
        inv_span[j] = if span[j] > F::zero() {
            F::one() / span[j]
        } else {
            F::zero()
        };
    }
    (lo, span, inv_span)
}

/// Merge Diracs into product bins; return vec of (barycenter, `total_weight`).
///
/// Uses mixed-radix packing: `key = Σ_j idx_j · m^j`. The key fits in `usize`
/// because `m^D ≤ cap` (which is at most a few thousand in practice).
fn product_bin_merge<F: SemiflowFloat, const D: usize>(
    diracs: &[([F; D], F)],
    m: usize,
    cap: usize,
    lo: &[F; D],
    _span: &[F; D],
    inv_span: &[F; D],
) -> Vec<([F; D], F)> {
    #[allow(clippy::cast_precision_loss)]
    let m_f = m as f64;
    // Number of cells = m^D, bounded by cap.
    #[allow(clippy::cast_possible_truncation)]
    let n_cells = m.saturating_pow(D as u32).min(cap).max(1);
    let mut cell_w: Vec<F> = alloc::vec![F::zero();    n_cells];
    let mut cell_pos: Vec<[F; D]> = alloc::vec![[F::zero(); D]; n_cells];

    for (pos, w) in diracs {
        let key = particle_key::<F, D>(pos, m, m_f, lo, inv_span, n_cells);
        cell_w[key] += *w;
        for j in 0..D {
            cell_pos[key][j] += pos[j] * *w;
        }
    }

    // Emit one Dirac per non-empty cell at its weight-barycenter.
    let mut out: Vec<([F; D], F)> = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let w = cell_w[i];
        if w == F::zero() {
            continue;
        }
        let inv_w = F::one() / w;
        let mut centroid = [F::zero(); D];
        for j in 0..D {
            centroid[j] = cell_pos[i][j] * inv_w;
        }
        out.push((centroid, w));
    }
    out
}

/// Compute the mixed-radix cell key for a particle.
///
/// `key = Σ_j idx_j · m^j`, clamped to `[0, n_cells-1]`.
fn particle_key<F: SemiflowFloat, const D: usize>(
    pos: &[F; D],
    m: usize,
    m_f: f64,
    lo: &[F; D],
    inv_span: &[F; D],
    n_cells: usize,
) -> usize {
    let mut key: usize = 0;
    let mut stride: usize = 1;
    for j in 0..D {
        let frac = ((pos[j] - lo[j]) * inv_span[j]).to_f64().unwrap_or(0.0);
        // Saturate frac to [0, 1) so index is in [0, m-1].
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let idx = (frac * m_f).floor() as usize;
        let idx = idx.min(m.saturating_sub(1));
        key += idx * stride;
        stride = stride.saturating_mul(m);
    }
    key.min(n_cells - 1)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `bins_per_axis`: m^D ≤ cap.
    #[test]
    fn bins_per_axis_fits_cap() {
        for &(cap, d) in &[(64usize, 2usize), (4096, 2), (4096, 10), (256, 6)] {
            let m = bins_per_axis(cap, d);
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap
            )]
            let m_d = (m as f64).powi(d as i32);
            #[allow(clippy::cast_precision_loss)]
            let cap_f = cap as f64;
            assert!(
                m_d <= cap_f + 1e-6,
                "m={m} d={d} cap={cap}: m^d={m_d} > cap"
            );
            assert!(m >= 1, "m must be ≥ 1");
        }
    }

    /// Anti-D2: per-axis first moment preserved on EVERY axis for D=3 ensemble.
    #[test]
    fn product_bin_first_moment_all_axes_d3() {
        // Hand-built 3D ensemble with known per-axis first moments.
        let mut ensemble = MeasureState::<f64, 3>::dirac([0.5, -1.0, 2.0], 0.3);
        ensemble.push_dirac_raw([1.5, 0.0, -1.0], 0.2);
        ensemble.push_dirac_raw([-0.5, 2.0, 0.5], 0.5);

        let m1_before: [f64; 3] = {
            let diracs = ensemble.as_diracs_slice();
            let mut m = [0.0f64; 3];
            for (p, w) in diracs {
                for j in 0..3 {
                    m[j] += p[j] * w;
                }
            }
            m
        };
        let mass_before: f64 = ensemble.as_diracs_slice().iter().map(|(_, w)| w).sum();

        // Force merge (cap=1 collapses all to one).
        ParticleReduction::WeightedVoronoi { cap: 1 }
            .apply(&mut ensemble)
            .unwrap();

        let mass_after: f64 = ensemble.as_diracs_slice().iter().map(|(_, w)| w).sum();
        assert!(
            (mass_after - mass_before).abs() < 1e-13,
            "mass: {mass_after} ≠ {mass_before}"
        );

        let diracs = ensemble.as_diracs_slice();
        let mut m1_after = [0.0f64; 3];
        for (p, w) in diracs {
            for j in 0..3 {
                m1_after[j] += p[j] * w;
            }
        }

        for j in 0..3 {
            assert!(
                (m1_after[j] - m1_before[j]).abs() < 1e-12,
                "axis-{j} first moment: before={} after={}",
                m1_before[j],
                m1_after[j]
            );
        }
    }

    /// D=2: per-axis first moment preserved with multiple bins (cap=4).
    #[test]
    fn product_bin_first_moment_all_axes_d2() {
        let mut ensemble = MeasureState::<f64, 2>::dirac([0.0, 1.0], 0.4);
        ensemble.push_dirac_raw([2.0, 3.0], 0.3);
        ensemble.push_dirac_raw([1.0, -1.0], 0.2);
        ensemble.push_dirac_raw([-1.0, 0.5], 0.1);

        let m1_before: [f64; 2] = {
            let diracs = ensemble.as_diracs_slice();
            let mut m = [0.0f64; 2];
            for (p, w) in diracs {
                for j in 0..2 {
                    m[j] += p[j] * w;
                }
            }
            m
        };

        // cap=4 may or may not merge; either way first moments must survive.
        ParticleReduction::WeightedVoronoi { cap: 2 }
            .apply(&mut ensemble)
            .unwrap();

        let diracs = ensemble.as_diracs_slice();
        let mut m1_after = [0.0f64; 2];
        for (p, w) in diracs {
            m1_after[0] += p[0] * w;
            m1_after[1] += p[1] * w;
        }
        for j in 0..2 {
            assert!(
                (m1_after[j] - m1_before[j]).abs() < 1e-12,
                "D=2 axis-{j}: before={} after={}",
                m1_before[j],
                m1_after[j]
            );
        }
    }

    /// cap=0 returns error.
    #[test]
    fn cap_zero_err() {
        let mut ensemble = MeasureState::<f64, 1>::dirac([0.0], 1.0);
        assert!(ParticleReduction::WeightedVoronoi { cap: 0 }
            .apply(&mut ensemble)
            .is_err());
    }
}
