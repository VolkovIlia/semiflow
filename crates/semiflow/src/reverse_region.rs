//! [`RegionMap`] — DoF-aligned region partition for K>1 reverse-AD
//! (math §51.10, ADR-0177; issue #1).
//!
//! Defines a contiguous partition ρ: node index → region id so that
//! `a(x_i) = θ_{ρ(i)}` is a piecewise-constant diffusion coefficient
//! built as ONE `DiffusionChernoff` via `with_closure`.
//!
//! ## Scope (NARROW — §51.10)
//!
//! - Region boundaries are **DoF-aligned**: node `i` belongs to exactly one Ω_r.
//! - Const-per-region coefficients only; variable-a within a region is out of scope.
//! - K=1 is the special case (one region = all nodes), byte-identical to §51.9.
//!
//! Zero new runtime dependencies.

use alloc::{sync::Arc, vec::Vec};

use crate::error::SemiflowError;

// ---------------------------------------------------------------------------
// RegionMap
// ---------------------------------------------------------------------------

/// DoF-aligned region partition ρ: node index → region id (math §51.10).
///
/// Contiguous by default; [`Self::contiguous`] partitions `n_grid` nodes into
/// `k` equal-size (± 1) regions. Each node belongs to exactly one region.
#[derive(Clone, Debug)]
pub struct RegionMap {
    /// `region_of[i]` = region id r of node i.  len == `n_grid`.
    region_of: Vec<usize>,
    region_count: usize,
}

impl RegionMap {
    /// Contiguous DoF-aligned partition of `n_grid` nodes into `k` regions.
    ///
    /// Nodes `0..floor(n/k)` → region 0; `floor(n/k)..2·floor(n/k)` → region 1; …
    /// The last region absorbs the remainder (at most k-1 extra nodes).
    ///
    /// # Errors
    /// Returns `SemiflowError::UnsupportedOperation` if `k == 0` or `k > n_grid`.
    pub fn contiguous(n_grid: usize, k: usize) -> Result<Self, SemiflowError> {
        if k == 0 || k > n_grid {
            return Err(SemiflowError::UnsupportedOperation {
                what: "RegionMap::contiguous: k must be 1..=n_grid",
            });
        }
        let per = n_grid / k;
        let region_of: Vec<usize> = (0..n_grid).map(|i| (i / per).min(k - 1)).collect();
        Ok(Self {
            region_of,
            region_count: k,
        })
    }

    /// Build from an explicit node-to-region assignment vector.
    ///
    /// `region_of[i]` must be in `0..k`. All k regions must be non-empty.
    ///
    /// # Errors
    /// Returns `SemiflowError::UnsupportedOperation` if any id is out of range.
    pub fn from_vec(region_of: Vec<usize>, k: usize) -> Result<Self, SemiflowError> {
        if k == 0 {
            return Err(SemiflowError::UnsupportedOperation {
                what: "RegionMap::from_vec: k must be >= 1",
            });
        }
        if region_of.iter().any(|&r| r >= k) {
            return Err(SemiflowError::UnsupportedOperation {
                what: "RegionMap::from_vec: region id out of range",
            });
        }
        Ok(Self {
            region_of,
            region_count: k,
        })
    }

    /// Region id of node `i` (unchecked — caller must ensure `i < n_grid`).
    #[inline]
    #[must_use]
    pub fn region_of(&self, i: usize) -> usize {
        self.region_of[i]
    }

    /// Number of regions K.
    #[inline]
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.region_count
    }

    /// Number of grid nodes.
    #[inline]
    #[must_use]
    pub fn n_grid(&self) -> usize {
        self.region_of.len()
    }
}

// ---------------------------------------------------------------------------
// Arc-wrapped map for closure capture (cheap Clone, Send+Sync)
// ---------------------------------------------------------------------------

/// A reference-counted [`RegionMap`] for use in `DiffusionChernoff` closures.
///
/// `Arc<RegionMap>` is `Clone` in O(1) and is `Send + Sync` (all fields are
/// plain `Vec<usize>` / `usize`). Required by `with_closure` which needs
/// `Send + Sync + 'static` on the closure.
pub type SharedRegionMap = Arc<RegionMap>;
