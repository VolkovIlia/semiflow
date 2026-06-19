//! Additive `ReflectingRegion<F>` impls for `BoxRegion<F, D>` and `BallRegion<F, D>`.
//!
//! Split from `reflection.rs` to stay within the suckless ≤500 `LoC` budget (ADR-0072).
//! The structs themselves live in `killing.rs` (v2.6); only the trait impls are here.
//!
//! ## `BoxRegion`
//! Per-axis flip: for p[k] outside [lo[k], hi[k]], reflect across the nearest edge.
//! Implements both `KillingRegion<F>` (v2.6) and `ReflectingRegion<F>` (v2.8).
//!
//! ## `BallRegion`
//! Spherical inversion for D ≥ 2; returns `DomainViolation` for D = 1.
//! For 1D reflecting BCs use `HalfSpaceRegion<F, 1>` instead.

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    killing::{BallRegion, BoxRegion},
    reflection::ReflectingRegion,
};

// ---------------------------------------------------------------------------
// Additive impl ReflectingRegion<F> for BoxRegion<F, D>
// ---------------------------------------------------------------------------

/// Per-axis flip: for p[k] outside [lo[k], hi[k]], reflect across nearest edge.
///
/// Reuses v2.6 `BoxRegion<F, D>` struct unchanged. The same struct now
/// implements both `KillingRegion<F>` (v2.6) and `ReflectingRegion<F>` (v2.8).
impl<F: SemiflowFloat, const D: usize> ReflectingRegion<F> for BoxRegion<F, D> {
    fn dim(&self) -> usize {
        D
    }

    fn is_inside(&self, point: &[F]) -> bool {
        (0..D).all(|k| point[k] >= self.lo[k] && point[k] < self.hi[k])
    }

    /// Build ghost state for a box via per-axis flip (D=1 path; v2.9 for multi-D).
    fn reflect_in_place(
        &self,
        dst: &mut GridFn1D<F>,
        src: &GridFn1D<F>,
    ) -> Result<(), SemiflowError> {
        let n = dst.grid.n;
        for i in 0..n {
            let x = dst.grid.x_at(i);
            let coord = [x];
            if !<Self as ReflectingRegion<F>>::is_inside(self, &coord) {
                // Per-axis flip for D = 1 (multi-D deferred to v2.9).
                let reflected_x = if x < self.lo[0] {
                    self.lo[0] + (self.lo[0] - x)
                } else {
                    self.hi[0] - (x - self.hi[0])
                };
                dst.values[i] = src.sample_generic(reflected_x)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Additive impl ReflectingRegion<F> for BallRegion<F, D>
// ---------------------------------------------------------------------------

/// Spherical inversion for D ≥ 2; `DomainViolation` for D = 1.
///
/// Reuses v2.6 `BallRegion<F, D>` struct unchanged (additive impl only).
/// For 1D reflecting BCs, use `HalfSpaceRegion<F, 1>` instead.
impl<F: SemiflowFloat, const D: usize> ReflectingRegion<F> for BallRegion<F, D> {
    fn dim(&self) -> usize {
        D
    }

    fn is_inside(&self, point: &[F]) -> bool {
        let r_sq = self.radius * self.radius;
        let dist_sq: F = (0..D)
            .map(|k| {
                let d = point[k] - self.center[k];
                d * d
            })
            .fold(F::zero(), |a, b| a + b);
        dist_sq <= r_sq
    }

    /// Returns `DomainViolation` for D=1 (use `HalfSpaceRegion<F, 1>`).
    /// Multi-D spherical inversion on `GridFn2D`/`GridFn3D` is deferred to v2.9.
    fn reflect_in_place(
        &self,
        _dst: &mut GridFn1D<F>,
        _src: &GridFn1D<F>,
    ) -> Result<(), SemiflowError> {
        if D == 1 {
            return Err(SemiflowError::DomainViolation {
                what: "BallRegion<F, 1>: use HalfSpaceRegion<F, 1> for 1D reflecting BCs",
                value: 1.0,
            });
        }
        #[allow(clippy::cast_precision_loss)]
        Err(SemiflowError::DomainViolation {
            what: "BallRegion D>1: GridFn2D/3D path deferred to v2.9",
            value: D as f64,
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::{
        error::SemiflowError,
        grid::Grid1D,
        grid_fn::GridFn1D,
        killing::{BallRegion, BoxRegion},
        reflection::ReflectingRegion,
    };

    // --- BoxRegion ReflectingRegion ---

    #[test]
    fn box_region_reflecting_is_inside() {
        let b = BoxRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
        assert!(<BoxRegion<f64, 1> as ReflectingRegion<f64>>::is_inside(
            &b,
            &[0.5]
        ));
        assert!(!<BoxRegion<f64, 1> as ReflectingRegion<f64>>::is_inside(
            &b,
            &[1.5]
        ));
    }

    // --- BallRegion D=1 returns DomainViolation ---

    #[test]
    fn ball_region_d1_reflect_in_place_err() {
        let ball = BallRegion::<f64, 1>::new([0.0], 1.0).unwrap();
        let grid = Grid1D::new(0.0_f64, 2.0, 4).unwrap();
        let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
        let src = GridFn1D::from_fn(grid, |x| x);
        let result =
            <BallRegion<f64, 1> as ReflectingRegion<f64>>::reflect_in_place(&ball, &mut dst, &src);
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }
}
