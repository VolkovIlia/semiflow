//! D≥2 projective-splitting obstacle evolver (math §44.5.ter).
//!
//! Provides [`ObstacleChernoffND`] — a thin newtype that wraps any
//! `ChernoffFunction<F, S = GridFnND<F, D>>` inner with post-projection
//! `Π_g = max(·, g)` over the flat row-major ND state.
//!
//! Separated from `obstacle.rs` for two reasons:
//! 1. **Coherence**: Rust stable cannot distinguish `C: ChernoffFunction<F, S = GridFn1D<F>>`
//!    from `C: ChernoffFunction<F, S = GridFnND<F, D>>` in two impls on the same struct.
//!    The newtype makes both impls non-overlapping by construction.
//! 2. **Line budget**: Keeps `obstacle.rs` under the 500-line constitution cap (v4.0.0).
//!
//! ## Scope (NORMATIVE, math §44.5.ter)
//!
//! v8.2.0 ships D≥2 **forward evolution only**. The active-set adjoint
//! (`apply_active_set_adjoint_into`) and inactive-set Γ (`apply_inactive_gamma_into`)
//! remain D=1; multi-asset free-surface Γ and ND adjoint are deferred.

use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid_nd::GridFnND,
    obstacle::Obstacle,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// ObstacleChernoffND<C, O, F, D>
// ---------------------------------------------------------------------------

/// D≥2 projective-splitting Chernoff function `Π_g ∘ S(Δτ)` on
/// [`GridFnND<F, D>`] state (math §44.5.ter).
///
/// `F(τ)f = max(C(τ)f, g)` (elementwise post-projection). Order-1 globally
/// (declared; projection cap dominates, §44.4). Stability rests on
/// `Π_g`-nonexpansiveness (Theorem 44.1). The projection `Π_g(W) = max(W, g)`
/// and active-set mask `𝟙[W > g]` are **elementwise over flat row-major
/// storage** and carry no dimension assumption (ADR-0150 sub-check 5).
///
/// ## Coherence note
///
/// `ObstacleChernoff` (D=1, `GridFn1D`) and this type (D≥2, `GridFnND`) are
/// **distinct types** so Rust's coherence checker accepts both `ChernoffFunction`
/// impls without conflict.
///
/// ## Scope (NORMATIVE, math §44.5.ter)
///
/// v8.2.0 ships D≥2 **forward evolution only**. Active-set adjoint and
/// inactive-set Γ remain D=1 (see `obstacle.rs` / `obstacle_gamma.rs`).
#[derive(Debug, Clone)]
pub struct ObstacleChernoffND<C, O, F: SemiflowFloat = f64, const D: usize = 2> {
    inner: C,
    obstacle: O,
    _f: PhantomData<F>,
}

impl<C, O, F, const D: usize> ObstacleChernoffND<C, O, F, D>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFnND<F, D>>,
    O: Obstacle<F>,
{
    /// Wrap `inner` with post-projection onto `{V ≥ obstacle}` for D-dimensional
    /// state (`GridFnND<F, D>`).
    ///
    /// # Errors
    /// Always `Ok` for pre-validated obstacles.
    pub fn new(inner: C, obstacle: O) -> Result<Self, SemiflowError> {
        Ok(Self {
            inner,
            obstacle,
            _f: PhantomData,
        })
    }

    /// Borrow the inner (linear) D-dimensional Chernoff propagator.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Borrow the obstacle `g`.
    pub fn obstacle(&self) -> &O {
        &self.obstacle
    }
}

impl<C, O, F, const D: usize> ChernoffFunction<F> for ObstacleChernoffND<C, O, F, D>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFnND<F, D>>,
    O: Obstacle<F>,
{
    type S = GridFnND<F, D>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.inner.apply_into(tau, src, dst, scratch)?;
        self.obstacle.project_in_place_nd(dst)
    }

    /// Order-1 globally (projection cap dominates, §44.4). Same honest
    /// declaration as the D=1 `ObstacleChernoff`.
    fn order(&self) -> u32 {
        1
    }

    /// Returns the inner's homogeneous growth (math §44.6 affine-Π_g note).
    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::ObstacleChernoffND;
    use crate::{
        chernoff::{ChernoffFunction, Growth},
        error::SemiflowError,
        float::SemiflowFloat,
        grid::Grid1D,
        grid_nd::{GridFnND, GridND},
        obstacle::{ConstantObstacle, Obstacle},
        scratch::ScratchPool,
        state::State,
    };

    // Minimal ND inner for testing: identity propagator on GridFnND.
    #[derive(Clone)]
    struct IdentityND<F: SemiflowFloat, const D: usize>(core::marker::PhantomData<F>);

    impl<F: SemiflowFloat, const D: usize> IdentityND<F, D> {
        fn new() -> Self {
            Self(core::marker::PhantomData)
        }
    }

    impl<F: SemiflowFloat, const D: usize> ChernoffFunction<F> for IdentityND<F, D> {
        type S = GridFnND<F, D>;
        fn apply_into(
            &self,
            _tau: F,
            src: &Self::S,
            dst: &mut Self::S,
            _s: &mut ScratchPool<F>,
        ) -> Result<(), SemiflowError> {
            dst.copy_from(src);
            Ok(())
        }
        fn order(&self) -> u32 {
            1
        }
        fn growth(&self) -> Growth<F> {
            Growth::contraction()
        }
    }

    #[test]
    fn nd_projection_lifts_below_floor() {
        let ax = Grid1D::new(0.0_f64, 1.0, 4).unwrap();
        let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
        // Initial values all below the obstacle floor 0.5.
        let v0 = GridFnND::from_fn(grid.clone(), |_: &[f64; 2]| 0.0_f64);
        let obs = ConstantObstacle::new(0.5_f64).unwrap();
        let kernel: ObstacleChernoffND<IdentityND<f64, 2>, _, f64, 2> =
            ObstacleChernoffND::new(IdentityND::new(), obs).unwrap();
        let mut dst = GridFnND::from_fn(grid, |_: &[f64; 2]| 0.0_f64);
        let mut scratch = ScratchPool::new();
        kernel
            .apply_into(0.01, &v0, &mut dst, &mut scratch)
            .unwrap();
        for &v in &dst.values {
            assert!(v >= 0.5 - 1e-12, "value {v} below obstacle 0.5");
        }
        assert_eq!(kernel.order(), 1);
    }

    #[test]
    fn nd_project_in_place_nd_dim2() {
        let ax = Grid1D::new(0.0_f64, 1.0, 4).unwrap();
        let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
        let mut u = GridFnND::from_fn(grid, |x: &[f64; 2]| x[0] * x[1]);
        let obs = ConstantObstacle::new(0.5_f64).unwrap();
        obs.project_in_place_nd(&mut u).unwrap();
        for &v in &u.values {
            assert!(v >= 0.5 - 1e-12, "projected value {v} below floor 0.5");
        }
    }

    #[test]
    fn nd_active_set_nd_into_dim2() {
        let ax = Grid1D::new(0.0_f64, 1.0, 4).unwrap();
        let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
        // Values: x0*x1; some will be > 0.3, some will not.
        let w = GridFnND::from_fn(grid, |x: &[f64; 2]| x[0] * x[1]);
        let obs = ConstantObstacle::new(0.3_f64).unwrap();
        let mut active = vec![false; w.values.len()];
        obs.active_set_nd_into(&w, &mut active).unwrap();
        for (flat, &a) in active.iter().enumerate() {
            assert_eq!(a, w.values[flat] > 0.3_f64);
        }
    }

    #[test]
    fn nd_active_set_nd_length_mismatch_err() {
        let ax = Grid1D::new(0.0_f64, 1.0, 4).unwrap();
        let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
        let w = GridFnND::from_fn(grid, |_: &[f64; 2]| 1.0_f64);
        let obs = ConstantObstacle::new(0.0_f64).unwrap();
        let mut active = vec![false; 3]; // wrong length
        let err = obs.active_set_nd_into(&w, &mut active).unwrap_err();
        assert!(matches!(err, SemiflowError::DomainViolation { .. }));
    }
}
