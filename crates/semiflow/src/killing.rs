//! Operator-level Dirichlet via Feynman–Kac killing (Butko 2018).
//!
//! `F(τ)f(x) = 𝟙_R(x) · C.apply(τ, f)(x)`. Order-1 globally (math §21).
//!
//! ## Dimension scope (v2.6)
//!
//! v2.6 ships `D = 1` fully tested (`BoxRegion<F, 1>`, `BallRegion<F, 1>`,
//! `KillingChernoff` paired with `GridFn1D<F>`).
//! `D = 2` and `D = 3` compile (const generics) but `mask_in_place` requires
//! an `impl GridFn1D`-like state with accessible node coordinates; those
//! implementations are deferred to v2.7 alongside the `GridFn2D`/`GridFn3D`
//! coordinate-access helper trait.
//!
//! ## Design choice: concrete state type (v2.6)
//!
//! The `KillingRegion::mask_in_place` method takes a `GridFn1D<F>` reference
//! directly (not the abstract `State<F>` trait) because `State<F>` does not
//! expose per-cell coordinate access. Adding per-cell coordinate access to
//! `State<F>` would be a breaking change; instead, the trait is specialized for
//! v2.6 to `GridFn1D<F>`. A future `CoordinateState<F, D>` supertrait (v2.7+)
//! will generalize this.
//!
//! ## Mathematical contract
//!
//! `KillingChernoff<C, R>` implements `ChernoffFunction<F>`:
//!   1. `inner.apply_into(τ, src, dst, scratch)` — unrestricted Chernoff step.
//!   2. `region.mask_in_place(dst)` — zero out cells outside `R` (post-multiply).
//!
//! Order-1 globally per Butko 2018 §3.2 (commutator `[L, 𝟙_R]` ≡ O(τ)). See
//! math.md §21 for the full derivation and §21.5 for G23 + T18N acceptance gates.

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// KillingRegion<F> — trait
// ---------------------------------------------------------------------------

/// Indicator function `𝟙_R(x)` for a bounded region `R ⊊ ℝᴰ`.
///
/// Consumed by `KillingChernoff<C, R>` to post-multiply each Chernoff step
/// by the indicator (Feynman–Kac killing, math §21).
///
/// ## v2.6 concrete impls
///
/// - `BoxRegion<F, D>` — axis-aligned box `{x : lo[k] ≤ x[k] < hi[k]}`.
/// - `BallRegion<F, D>` — Euclidean ball `{x : Σ(x[k]-c[k])² ≤ r²}`.
///
/// ## Implementing for custom regions
///
/// Override `is_inside` and `mask_in_place`. The default `mask_in_place`
/// iterates `is_inside` per cell; override for SIMD batch paths.
pub trait KillingRegion<F: SemiflowFloat = f64> {
    /// Test whether a point (as a slice of `D` coordinates) is inside `R`.
    fn is_inside(&self, point: &[F]) -> bool;

    /// Zero cells of `dst` where `!is_inside(coord(i))`. Returns `Ok(())`.
    ///
    /// Default impl iterates `is_inside` per node (O(N · D)). Concrete impls
    /// (`BoxRegion`, `BallRegion`) override with scalar tight loops sufficient for
    /// the eigenmode oracle (no SIMD intrinsics in v2.6).
    ///
    /// # Errors
    /// Returns [`SemiflowError`] if the grid state is invalid.
    fn mask_in_place(&self, dst: &mut GridFn1D<F>) -> Result<(), SemiflowError>;

    /// Spatial dimension `D` of the region (informational).
    fn dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// BoxRegion<F, D> — axis-aligned box
// ---------------------------------------------------------------------------

/// Axis-aligned box `R = {x ∈ ℝᴰ : lo[k] ≤ x[k] < hi[k] for all k}`.
///
/// The boundary `∂R` (where equality holds) is **excluded** from `R`:
/// cells on the boundary are zeroed by `mask_in_place` (math §21.4, open-R
/// convention).
///
/// ## Errors on construction
///
/// `BoxRegion::new` returns `Err(DomainViolation)` if `lo[k] >= hi[k]` for
/// any `k`, so that the box is always non-degenerate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxRegion<F: SemiflowFloat = f64, const D: usize = 1> {
    /// Lower bounds (inclusive). Invariant: `lo[k] < hi[k]` for all k.
    pub lo: [F; D],
    /// Upper bounds (exclusive). Invariant: `lo[k] < hi[k]` for all k.
    pub hi: [F; D],
}

impl<F: SemiflowFloat, const D: usize> BoxRegion<F, D> {
    /// Construct a `BoxRegion` with validated bounds.
    ///
    /// # Errors
    ///
    /// Returns `SemiflowError::DomainViolation` if `lo[k] >= hi[k]` for any `k`.
    pub fn new(lo: [F; D], hi: [F; D]) -> Result<Self, SemiflowError> {
        for k in 0..D {
            if lo[k] >= hi[k] {
                return Err(SemiflowError::DomainViolation {
                    what: "BoxRegion: lo[k] must be < hi[k] for all k",
                    value: lo[k].to_f64().unwrap_or(f64::NAN),
                });
            }
        }
        Ok(Self { lo, hi })
    }
}

impl<F: SemiflowFloat, const D: usize> KillingRegion<F> for BoxRegion<F, D> {
    /// `lo[k] <= point[k] < hi[k]` for all `k` (open-R convention).
    fn is_inside(&self, point: &[F]) -> bool {
        debug_assert_eq!(point.len(), D, "point dimension mismatch");
        (0..D).all(|k| point[k] >= self.lo[k] && point[k] < self.hi[k])
    }

    fn mask_in_place(&self, dst: &mut GridFn1D<F>) -> Result<(), SemiflowError> {
        let n = dst.grid.n;
        for i in 0..n {
            let x = dst.grid.x_at(i);
            if !self.is_inside(&[x]) {
                dst.values[i] = F::zero();
            }
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        D
    }
}

// ---------------------------------------------------------------------------
// BallRegion<F, D> — Euclidean ball
// ---------------------------------------------------------------------------

/// Euclidean ball `R = {x ∈ ℝᴰ : Σ(x[k]-c[k])² ≤ r²}`.
///
/// The boundary (where the sum equals `r²`) is included in `R` for `is_inside`
/// (closed-ball convention matching math §21.4). `mask_in_place` zeros cells
/// strictly outside the closed ball.
///
/// ## Errors on construction
///
/// `BallRegion::new` returns `Err(DomainViolation)` if `radius <= 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BallRegion<F: SemiflowFloat = f64, const D: usize = 1> {
    /// Center of the ball.
    pub center: [F; D],
    /// Radius. Invariant: `radius > 0`.
    pub radius: F,
}

impl<F: SemiflowFloat, const D: usize> BallRegion<F, D> {
    /// Construct a `BallRegion` with validated radius.
    ///
    /// # Errors
    ///
    /// Returns `SemiflowError::DomainViolation` if `radius <= 0`.
    pub fn new(center: [F; D], radius: F) -> Result<Self, SemiflowError> {
        if radius <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "BallRegion: radius must be > 0",
                value: radius.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self { center, radius })
    }
}

impl<F: SemiflowFloat, const D: usize> KillingRegion<F> for BallRegion<F, D> {
    /// `Σ(x[k]-c[k])² ≤ r²` (closed-ball convention).
    fn is_inside(&self, point: &[F]) -> bool {
        debug_assert_eq!(point.len(), D, "point dimension mismatch");
        let r_sq = self.radius * self.radius;
        let dist_sq: F = (0..D)
            .map(|k| {
                let d = point[k] - self.center[k];
                d * d
            })
            .fold(F::zero(), |acc, v| acc + v);
        dist_sq <= r_sq
    }

    fn mask_in_place(&self, dst: &mut GridFn1D<F>) -> Result<(), SemiflowError> {
        let n = dst.grid.n;
        for i in 0..n {
            let x = dst.grid.x_at(i);
            if !self.is_inside(&[x]) {
                dst.values[i] = F::zero();
            }
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        D
    }
}

// ---------------------------------------------------------------------------
// KillingChernoff<C, R, F> — wrapper
// ---------------------------------------------------------------------------

/// Chernoff function for the Feynman–Kac killed semigroup on region `R`.
///
/// `F(τ)f = 𝟙_R · C(τ)f` (post-multiply). Order-1 globally per Butko 2018
/// §3.2. Inner `C` may have higher order; killing dominates. Growth bound
/// inherited from inner (killing is sub-Markov: `‖𝟙_R · f‖_∞ ≤ ‖f‖_∞`).
///
/// ## v2.6 constraint
///
/// `C::S` must be `GridFn1D<F>` (the only state type with accessible node
/// coordinates in v2.6). Higher-dimensional states are deferred to v2.7.
#[derive(Debug, Clone)]
pub struct KillingChernoff<C, R, F: SemiflowFloat = f64> {
    inner: C,
    region: R,
    _f: PhantomData<F>,
}

impl<C, R, F> KillingChernoff<C, R, F>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: KillingRegion<F>,
{
    /// Wrap `inner` with killing region `region`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the region is invalid (e.g., empty `BoxRegion`).
    /// In practice the region is validated at construction time by `BoxRegion::new`
    /// or `BallRegion::new`; this method always returns `Ok` for pre-validated regions.
    pub fn new(inner: C, region: R) -> Result<Self, SemiflowError> {
        Ok(Self {
            inner,
            region,
            _f: PhantomData,
        })
    }
}

impl<C, R, F> ChernoffFunction<F> for KillingChernoff<C, R, F>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    R: KillingRegion<F>,
{
    type S = GridFn1D<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.inner.apply_into(tau, src, dst, scratch)?;
        self.region.mask_in_place(dst)
    }

    /// Order-1 globally (Butko 2018 §3.2 — commutator `[L, 𝟙_R]` is O(τ)).
    fn order(&self) -> u32 {
        1
    }

    /// Killing is sub-Markov; growth bounded by inner.
    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::{BallRegion, BoxRegion, KillingRegion};
    use crate::{
        chernoff::ChernoffFunction, diffusion::DiffusionChernoff, error::SemiflowError,
        grid::Grid1D, grid_fn::GridFn1D, killing::KillingChernoff,
    };

    // --- BoxRegion construction ---

    #[test]
    fn box_region_valid_bounds() {
        let r = BoxRegion::<f64, 1>::new([0.0], [1.0]);
        assert!(r.is_ok());
    }

    #[test]
    fn box_region_inverted_bounds_err() {
        let err = BoxRegion::<f64, 1>::new([1.0], [0.5]).unwrap_err();
        assert!(
            matches!(err, SemiflowError::DomainViolation { .. }),
            "expected DomainViolation, got {err:?}"
        );
    }

    #[test]
    fn box_region_is_inside_corner_cases() {
        let r = BoxRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        // Open-R convention: lo is included, hi is excluded.
        assert!(r.is_inside(&[0.0])); // lo endpoint: included
        assert!(!r.is_inside(&[1.0])); // hi endpoint: excluded
        assert!(r.is_inside(&[0.5])); // interior
        assert!(!r.is_inside(&[-0.1])); // left outside
        assert!(!r.is_inside(&[1.5])); // right outside
    }

    // --- BallRegion construction ---

    #[test]
    fn ball_region_zero_radius_err() {
        let err = BallRegion::<f64, 1>::new([0.0], 0.0).unwrap_err();
        assert!(matches!(err, SemiflowError::DomainViolation { .. }));
    }

    #[test]
    fn ball_region_negative_radius_err() {
        let err = BallRegion::<f64, 1>::new([0.0], -1.0).unwrap_err();
        assert!(matches!(err, SemiflowError::DomainViolation { .. }));
    }

    #[test]
    fn ball_region_is_inside_unit_ball() {
        let ball = BallRegion::<f64, 1>::new([0.0], 1.0).unwrap();
        assert!(ball.is_inside(&[0.0]));
        assert!(ball.is_inside(&[1.0])); // exactly on boundary (closed)
        assert!(!ball.is_inside(&[1.1])); // outside
    }

    // --- KillingChernoff::apply_into zeroes cells outside BoxRegion ---

    #[test]
    fn killing_chernoff_zeros_outside_region() {
        use crate::chernoff::ApplyChernoffExt;
        // Grid: [0, 2], n=9 → nodes at 0, 0.25, 0.5, ..., 2.0
        let grid = Grid1D::new(0.0_f64, 2.0, 9).unwrap();
        // Constant-a diffusion (a=0.5, a'=0, a''=0)
        let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        // Kill region: [0.5, 1.5) — only interior nodes survive
        let region = BoxRegion::<f64, 1>::new([0.5], [1.5]).unwrap();
        let killing = KillingChernoff::new(diff, region).unwrap();
        // Initial datum: f(x) = 1 everywhere
        let u0 = GridFn1D::from_fn(grid, |_| 1.0_f64);
        let u1 = killing.apply_chernoff(0.001, &u0).unwrap();
        // Nodes outside [0.5, 1.5): x in {0.0, 0.25, 1.5, 1.75, 2.0} → must be 0
        let dx = grid.dx();
        for i in 0..grid.n {
            let x = grid.xmin + (i as f64) * dx;
            if !(0.5..1.5).contains(&x) {
                assert_eq!(
                    u1.values[i], 0.0,
                    "node {i} (x={x}) should be zero outside region"
                );
            }
        }
        // Nodes inside [0.5, 1.5): should be non-zero (diffusion doesn't kill)
        assert!(
            u1.values[2] != 0.0 || u1.values[3] != 0.0 || u1.values[4] != 0.0,
            "interior nodes should be non-zero"
        );
        assert_eq!(killing.order(), 1);
    }
}
