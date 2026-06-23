//! Wave B2 cache types for `TruncatedExp4thDiffusionChernoff` (v0.13.0, ADR-0034 Amendment 1).
//!
//! Kept in a separate module so that `truncated_exp4.rs` stays within the
//! 700-line constitution cap (Override #1).
//!
//! ## What is here
//!
//! - [`HalfNodeCoeffCache`]: pre-evaluated `a(x)` at the four G⁴ half-node offsets.
//! - [`TruncatedExp4WithCache`]: wrapper that substitutes cache lookups for closure
//!   dispatch in the stencil hot loop, enabling Wave B3 SIMD lane saturation.

use alloc::vec::Vec;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
    truncated_exp4::{
        apply_power_series_f64_at_node, validate_a_norm_bound_f64, validate_cfl_4th,
        validate_tau_f64, TruncatedExp4thDiffusionChernoff, TRUNC_ORDER_USIZE,
    },
};

// Wave B3: SIMD dispatch imports (ADR-0019 Amendment 2).
// Re-exported from simd/mod.rs; arch-specific impls in simd/x86_64.rs / aarch64.rs.
#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
use crate::simd::apply_g4_stencil_avx2_4nodes;
#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
use crate::simd::apply_g4_stencil_neon_4nodes;

// ---------------------------------------------------------------------------
// HalfNodeCoeffCache
// ---------------------------------------------------------------------------

/// Pre-evaluated `a(x)` at the four half-node offsets required by the G⁴ stencil.
///
/// Layout (N = number of grid nodes):
/// - `ar3h[i]` = `a(x_i + 3·dx/2)` for i in 0..N
/// - `ar1h[i]` = `a(x_i + dx/2)`   for i in 0..N
/// - `al1h[i]` = `a(x_i - dx/2)`   for i in 0..N
/// - `al3h[i]` = `a(x_i - 3·dx/2)` for i in 0..N
///
/// Eliminates four `a(x)` function-pointer calls per node per stencil application.
/// `precompute_g4_grids_f64` calls the stencil K=4 times, so Wave B2 removes
/// `4 × K × N = 16N` function-pointer dispatches per `apply()`.
///
/// # Caller contract
///
/// The function pointer `a` passed to [`HalfNodeCoeffCache::new`] MUST be **pure
/// and deterministic** (same input always yields the same output).  If `a` has
/// side effects or non-deterministic behaviour the cached values will diverge
/// from the live-dispatch path and `TEXP4_CACHED_COEFF_BIT_EQUAL` will fail.
///
/// See ADR-0034 Amendment 1.
#[derive(Clone)]
pub struct HalfNodeCoeffCache {
    /// `a(x_i + 3·dx/2)` for i in 0..N
    pub ar3h: Vec<f64>,
    /// `a(x_i + dx/2)` for i in 0..N
    pub ar1h: Vec<f64>,
    /// `a(x_i - dx/2)` for i in 0..N
    pub al1h: Vec<f64>,
    /// `a(x_i - 3·dx/2)` for i in 0..N
    pub al3h: Vec<f64>,
}

impl HalfNodeCoeffCache {
    /// Populate cache by evaluating `a(x)` at all half-node offsets for grid `g`.
    ///
    /// Boundary half-nodes (`i=0` left side, `i=N-1` right side) are evaluated by
    /// calling `a()` at the extrapolated coordinate — matching the live-closure path
    /// in `apply_g4_at_node_f64` which calls `(mc.a)(x_i ± offset)` without clamping
    /// (ADR-0007 boundary policy applies at the `GridFn` level, not here).
    // ar3h/ar1h/al1h/al3h: a×right/left×3/1×half — mathematically unambiguous.
    #[allow(clippy::similar_names)]
    pub fn new(a: fn(f64) -> f64, grid: Grid1D<f64>) -> Self {
        let n = grid.n;
        let dx = grid.dx();
        let mut ar3h = Vec::with_capacity(n);
        let mut ar1h = Vec::with_capacity(n);
        let mut al1h = Vec::with_capacity(n);
        let mut al3h = Vec::with_capacity(n);
        for i in 0..n {
            let x = grid.x_at(i);
            ar3h.push(a(x + 1.5 * dx));
            ar1h.push(a(x + 0.5 * dx));
            al1h.push(a(x - 0.5 * dx));
            al3h.push(a(x - 1.5 * dx));
        }
        Self {
            ar3h,
            ar1h,
            al1h,
            al3h,
        }
    }
}

// ---------------------------------------------------------------------------
// TruncatedExp4WithCache
// ---------------------------------------------------------------------------

/// [`TruncatedExp4thDiffusionChernoff`] with pre-evaluated `a(x)` half-node cache.
///
/// Implements [`ChernoffFunction`] identically to the wrapped type but replaces
/// the 4-per-node function-pointer dispatches in `apply_g4_at_node_f64` with
/// direct `Vec<f64>` index reads.
///
/// # Construction
///
/// ```rust
/// use semiflow::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExp4WithCache};
/// let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
/// let me4c = TruncatedExp4WithCache::with_cached_coefficients(
///     |_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid,
/// );
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let u1 = me4c.apply_chernoff(0.001, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
///
/// # Caller invariants
///
/// Same as [`TruncatedExp4thDiffusionChernoff`].  Additionally, `a(x)` MUST be
/// **pure and deterministic** — see [`HalfNodeCoeffCache`] doc.
///
/// See ADR-0034 Amendment 1.
#[derive(Clone)]
pub struct TruncatedExp4WithCache {
    /// Wrapped Chernoff operator (closure path, unchanged, for validation reuse).
    pub(crate) inner: TruncatedExp4thDiffusionChernoff<f64>,
    /// Pre-evaluated half-node coefficient cache.
    pub(crate) cache: HalfNodeCoeffCache,
}

impl TruncatedExp4WithCache {
    /// Construct with pre-evaluated `a(x)` half-node coefficients.
    ///
    /// The hot loop indexes the cache instead of calling the closure, enabling
    /// SIMD lane saturation in Wave B3.
    ///
    /// Memory cost: 4N × 8 bytes ≈ 16 KB at N=512, negligible vs the 2.8 MB
    /// per-call baseline measured in iter-3 benchmarks.
    ///
    /// # Panics
    ///
    /// Does not panic. All grid operations are deterministic.
    #[must_use]
    pub fn with_cached_coefficients(
        a: fn(f64) -> f64,
        a_prime: fn(f64) -> f64,
        a_double_prime: fn(f64) -> f64,
        a_norm_bound: f64,
        grid: Grid1D<f64>,
    ) -> Self {
        let inner =
            TruncatedExp4thDiffusionChernoff::new(a, a_prime, a_double_prime, a_norm_bound, grid);
        let cache = HalfNodeCoeffCache::new(a, grid);
        Self { inner, cache }
    }
}

impl ChernoffFunction<f64> for TruncatedExp4WithCache {
    type S = GridFn1D<f64>;

    /// Consistency order **2**.
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)`.
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }

    /// Allocation-free output override using pre-evaluated coefficient cache.
    ///
    /// Uses the same `precompute_g4_grids_cached` stencil path (including SIMD
    /// dispatch) as `apply` to guarantee bit-identical results. The only savings vs
    /// the default bridge: output is written directly into `dst` instead of returning
    /// a new `GridFn1D` allocation.
    ///
    /// The four g-grid `GridFn1D` intermediates are still heap-allocated per call.
    /// Scratch is accepted but unused (`GridFn1D` state cannot be stored in `ScratchPool`).
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        validate_tau_f64(tau)?;
        validate_a_norm_bound_f64(self.inner.a_norm_bound)?;
        let dx = self.inner.grid.dx();
        validate_cfl_4th(tau, self.inner.a_norm_bound, dx)?;
        let n = src.values.len();
        let g_grids = precompute_g4_grids_cached(&self.inner, &self.cache, src)?;
        dst.values.resize(n, 0.0);
        // No drift conjugation — direct index (bit-identical to apply).
        for i in 0..n {
            dst.values[i] = apply_power_series_f64_at_node(tau, &g_grids, i);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cached stencil helpers
// ---------------------------------------------------------------------------

/// Precompute G⁴ power-series grids using cache lookups instead of closure dispatch.
fn precompute_g4_grids_cached(
    mc: &TruncatedExp4thDiffusionChernoff<f64>,
    cache: &HalfNodeCoeffCache,
    f: &GridFn1D<f64>,
) -> Result<[GridFn1D<f64>; TRUNC_ORDER_USIZE + 1], SemiflowError> {
    let g0 = f.clone();
    let g1 = apply_g4_stencil_cached(mc, cache, &g0)?;
    let g2 = apply_g4_stencil_cached(mc, cache, &g1)?;
    let g3 = apply_g4_stencil_cached(mc, cache, &g2)?;
    let g4 = apply_g4_stencil_cached(mc, cache, &g3)?;
    Ok([g0, g1, g2, g3, g4])
}

/// Apply the G⁴ stencil once; reads coefficients from cache instead of closure.
///
/// Dispatches to the 4-node SIMD path (Wave B3) for interior chunks when
/// `feature = "simd"` is active and the arch supports AVX2/NEON.
/// Boundary nodes (first/last 2) always use the scalar path (requires `sample()`
/// for out-of-bounds extrapolation).
fn apply_g4_stencil_cached(
    mc: &TruncatedExp4thDiffusionChernoff<f64>,
    cache: &HalfNodeCoeffCache,
    prev: &GridFn1D<f64>,
) -> Result<GridFn1D<f64>, SemiflowError> {
    let n = prev.values.len();
    let dx = mc.grid.dx();
    let dx_sq = dx * dx;
    let mut out = prev.zeroed_like();

    #[cfg(feature = "simd")]
    {
        if !(cfg!(test) && crate::simd::FORCE_SCALAR.with(core::cell::Cell::get)) {
            return apply_g4_stencil_cached_simd(mc, cache, prev, &mut out, n, dx, dx_sq);
        }
    }

    // Scalar path — also reached under FORCE_SCALAR in tests.
    apply_g4_stencil_cached_scalar(mc, cache, prev, &mut out, n, dx, dx_sq)?;
    Ok(out)
}

/// Scalar implementation of the G⁴ stencil (all nodes, no SIMD).
#[allow(clippy::too_many_arguments)]
fn apply_g4_stencil_cached_scalar(
    mc: &TruncatedExp4thDiffusionChernoff<f64>,
    cache: &HalfNodeCoeffCache,
    prev: &GridFn1D<f64>,
    out: &mut GridFn1D<f64>,
    n: usize,
    dx: f64,
    dx_sq: f64,
) -> Result<(), SemiflowError> {
    for i in 0..n {
        let x_i = mc.grid.x_at(i);
        out.values[i] = apply_g4_at_node_cached(cache, prev, i, n, x_i, dx, dx_sq)?;
    }
    Ok(())
}

/// SIMD dispatch path: boundary nodes scalar, interior chunks of 4 via arch SIMD.
///
/// Boundary invariant: the first/last 2 nodes may require `prev.sample()` for
/// out-of-bounds extrapolation — those are always processed by the scalar path.
/// Interior nodes [2 .. n-2] are safe for the 5-point stencil without extrapolation.
#[cfg(feature = "simd")]
#[allow(clippy::too_many_arguments)]
fn apply_g4_stencil_cached_simd(
    mc: &TruncatedExp4thDiffusionChernoff<f64>,
    cache: &HalfNodeCoeffCache,
    prev: &GridFn1D<f64>,
    out: &mut GridFn1D<f64>,
    n: usize,
    dx: f64,
    dx_sq: f64,
) -> Result<GridFn1D<f64>, SemiflowError> {
    let dx_sq_inv = 1.0 / dx_sq;
    let scalar_left = 2_usize.min(n);
    let simd_end = n.saturating_sub(2); // exclusive: last 2 nodes stay scalar

    // Scalar left boundary (nodes 0, 1).
    for i in 0..scalar_left {
        let x_i = mc.grid.x_at(i);
        out.values[i] = apply_g4_at_node_cached(cache, prev, i, n, x_i, dx, dx_sq)?;
    }

    // SIMD interior: chunks of 4 where all 5-point stencil reads are in-bounds.
    // Condition: base >= 2 (scalar_left) and base+5 <= n (i.e. base <= n-5).
    // simd_end = n-2, so base+4 <= n-2 → base <= n-6 → base+5 <= n-1 < n. Safe.
    let mut i = scalar_left;
    while i + 4 <= simd_end {
        apply_g4_4nodes_simd(i, prev, cache, dx_sq_inv, out);
        i += 4;
    }

    // Scalar tail + right boundary (remainder and last 2 nodes).
    while i < n {
        let x_i = mc.grid.x_at(i);
        out.values[i] = apply_g4_at_node_cached(cache, prev, i, n, x_i, dx, dx_sq)?;
        i += 1;
    }
    Ok(out.clone())
}

/// Dispatch 4-node SIMD kernel: selects AVX2, NEON, or scalar for other arches.
///
/// Interior invariant: `base >= 2` and `base + 5 <= n` — caller ensures this.
#[cfg(feature = "simd")]
#[allow(clippy::similar_names)]
fn apply_g4_4nodes_simd(
    base: usize,
    prev: &GridFn1D<f64>,
    cache: &HalfNodeCoeffCache,
    dx_sq_inv: f64,
    out: &mut GridFn1D<f64>,
) {
    // AVX2 path.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    apply_g4_stencil_avx2_4nodes(
        base,
        &prev.values,
        &cache.ar3h,
        &cache.ar1h,
        &cache.al1h,
        &cache.al3h,
        dx_sq_inv,
        &mut out.values,
    );

    // NEON path.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    apply_g4_stencil_neon_4nodes(
        base,
        &prev.values,
        &cache.ar3h,
        &cache.ar1h,
        &cache.al1h,
        &cache.al3h,
        dx_sq_inv,
        &mut out.values,
    );

    // Scalar fallback for arches without AVX2/NEON (simd feature but no intrinsics).
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    apply_g4_4nodes_scalar_fallback(base, prev, cache, dx_sq_inv, out);
}

/// Pure-scalar 4-node G⁴ stencil — fallback when AVX2/NEON are not enabled.
///
/// Interior invariant: `base >= 2` and `base + 5 <= n` — caller ensures this.
#[cfg(all(
    feature = "simd",
    not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))
))]
#[allow(clippy::similar_names)]
fn apply_g4_4nodes_scalar_fallback(
    base: usize,
    prev: &GridFn1D<f64>,
    cache: &HalfNodeCoeffCache,
    dx_sq_inv: f64,
    out: &mut GridFn1D<f64>,
) {
    let dx_sq = 1.0 / dx_sq_inv;
    for j in base..base + 4 {
        // Interior invariant guarantees prev[j-2..j+3] are all in-bounds.
        let rp2 = prev.values[j + 2];
        let rp1 = prev.values[j + 1];
        let ctr = prev.values[j];
        let lm1 = prev.values[j - 1];
        let lm2 = prev.values[j - 2];
        let ar3h = cache.ar3h[j];
        let ar1h = cache.ar1h[j];
        let al1h = cache.al1h[j];
        let al3h = cache.al3h[j];
        let flux_right = 5.0 * ar1h * (rp1 - ctr) / 4.0;
        let flux_right_far = -ar3h * (rp2 - rp1) / 12.0;
        let flux_left = -5.0 * al1h * (ctr - lm1) / 4.0;
        let flux_left_far = al3h * (lm1 - lm2) / 12.0;
        out.values[j] = (flux_right_far + flux_right + flux_left + flux_left_far) / dx_sq;
    }
}

#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn apply_g4_at_node_cached(
    cache: &HalfNodeCoeffCache,
    prev: &GridFn1D<f64>,
    i: usize,
    n: usize,
    x_i: f64,
    dx: f64,
    dx_sq: f64,
) -> Result<f64, SemiflowError> {
    let rp2 = if i + 2 < n {
        prev.values[i + 2]
    } else {
        prev.sample(x_i + 2.0 * dx)?
    };
    let rp1 = if i + 1 < n {
        prev.values[i + 1]
    } else {
        prev.sample(x_i + dx)?
    };
    let ctr = prev.values[i];
    let lm1 = if i >= 1 {
        prev.values[i - 1]
    } else {
        prev.sample(x_i - dx)?
    };
    let lm2 = if i >= 2 {
        prev.values[i - 2]
    } else {
        prev.sample(x_i - 2.0 * dx)?
    };

    // Cache lookup — eliminates function-pointer dispatch.
    let ar3h = cache.ar3h[i];
    let ar1h = cache.ar1h[i];
    let al1h = cache.al1h[i];
    let al3h = cache.al3h[i];

    let flux_right = 5.0 * ar1h * (rp1 - ctr) / 4.0;
    let flux_right_far = -ar3h * (rp2 - rp1) / 12.0;
    let flux_left = -5.0 * al1h * (ctr - lm1) / 4.0;
    let flux_left_far = al3h * (lm1 - lm2) / 12.0;
    Ok((flux_right_far + flux_right + flux_left + flux_left_far) / dx_sq)
}
