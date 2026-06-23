//! Property tests for `AdaptivePI<C>` contractivity (v0.6.0, ADR-0014).
//!
//! Gate: `‖final_state‖_∞ ≤ ‖u0‖_∞ · 1.001` (Lady Windermere's fan:
//! adaptive sub-stepping preserves contractivity of the underlying
//! semigroup within a tiny FP tolerance).
//!
//! Setup: random Gaussian IC on N=200 grid, `a = 0.5` (constant-diffusion),
//! random `tol_rel ∈ [1e-8, 1e-3]`, random `t ∈ [0.1, 2.0]`.
//! Uses `DiffusionChernoff` (order 2) as inner function.
//!
//! 200 cases, Proptest 1.4.0.

use proptest::prelude::*;
use semiflow::{state::State, AdaptivePI, BoundaryPolicy, DiffusionChernoff, Grid1D, GridFn1D};

// Thread-local amplitude so fn-pointer is compatible with proptest lambdas.
// We re-build the grid/IC inside each test case from the proptest parameters.

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// `AdaptivePI<DiffusionChernoff>` is contractive: ‖out‖ ≤ ‖u0‖ · 1.001.
    ///
    /// Contractivity is inherited from the underlying diffusion semigroup;
    /// adaptive sub-stepping preserves it via the semigroup property.
    #[test]
    fn adaptive_contractivity(
        mu    in -5.0f64..=5.0f64,
        sigma in  0.5f64..=3.0f64,
        t     in  0.1f64..=2.0f64,
        tol_rel in 1.0e-8f64..=1.0e-3f64,
    ) {
        let grid = Grid1D::new(-10.0, 10.0, 200)
            .expect("grid")
            .with_boundary(BoundaryPolicy::Reflect);

        let u0 = GridFn1D::from_fn(grid, |x| {
            libm::exp(-(x - mu) * (x - mu) / (2.0 * sigma * sigma))
        });
        let norm_u0 = u0.norm_sup();

        let func = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let mut pi = AdaptivePI::new(func).with_tolerance(0.0, tol_rel);

        let outcome = pi.evolve_adaptive(t, &u0)
            .expect("DiffusionChernoff adaptive should not error at tol_rel >= 1e-8");

        let out_norm = outcome.final_state.norm_sup();
        let bound = norm_u0 * 1.001;

        prop_assert!(
            out_norm <= bound,
            "contractivity violated: ‖out‖={out_norm:.6e} > 1.001·‖u0‖={bound:.6e} \
             (t={t:.3}, tol_rel={tol_rel:.3e}, mu={mu:.3}, sigma={sigma:.3})"
        );

        prop_assert!(
            outcome.steps_accepted >= 1,
            "AdaptivePI took zero steps for t={t:.3} > 0"
        );

        let total = outcome.steps_accepted + outcome.steps_rejected;
        prop_assert!(
            total <= pi.max_substeps,
            "steps_accepted+steps_rejected={total} > max_substeps={} \
             — should only err via AdaptiveStepRejected",
            pi.max_substeps
        );
    }
}
