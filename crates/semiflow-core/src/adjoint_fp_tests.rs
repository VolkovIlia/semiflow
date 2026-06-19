// Tests for adjoint_fp — split from adjoint_fp.rs (batch H8).
// Included via `include!` so test items see all private symbols from the parent module.

mod tests {
    use super::*;
    use crate::{DiffusionChernoff, Grid1D, ScratchPool};

    fn brownian_adj(n: usize) -> AdjointFokkerPlanckChernoff<DiffusionChernoff<f64>, f64, 1> {
        let grid = Grid1D::new(-4.0_f64, 4.0, n).unwrap();
        let fwd = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        AdjointFokkerPlanckChernoff::new(fwd, 0.5_f64, 0.0, 0.0)
    }

    #[test]
    fn order_inherits_from_forward() {
        assert_eq!(brownian_adj(32).order(), 2);
    }

    #[test]
    fn dirac_expands_to_four_children() {
        let adj = brownian_adj(32);
        let rho0 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
        let mut rho1 = rho0.clone();
        let mut pool = ScratchPool::<f64>::new();
        adj.apply_into(0.1_f64, &rho0, &mut rho1, &mut pool)
            .unwrap();
        assert_eq!(rho1.n_diracs(), 4);
        assert!(rho1.total_variation().is_finite());
    }

    #[test]
    fn mass_conservation_c_zero() {
        let adj = brownian_adj(32);
        let rho0 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
        let mut rho1 = rho0.clone();
        let mut pool = ScratchPool::<f64>::new();
        adj.apply_into(0.01_f64, &rho0, &mut rho1, &mut pool)
            .unwrap();
        let mass: f64 = rho1.diracs.iter().map(|(_, w)| w).sum();
        assert!((mass - 1.0).abs() < 1e-14, "mass drift: {mass}");
    }

    #[test]
    fn negative_tau_returns_err() {
        let adj = brownian_adj(32);
        let rho0 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
        let mut rho1 = rho0.clone();
        let mut pool = ScratchPool::<f64>::new();
        assert!(adj
            .apply_into(-0.01_f64, &rho0, &mut rho1, &mut pool)
            .is_err());
    }

    #[test]
    fn total_variation_abs_weight() {
        let rho = MeasureState::<f64, 1>::dirac([1.0], -2.0);
        assert!((rho.total_variation() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn pair_cosine_at_origin() {
        let rho = MeasureState::<f64, 1>::dirac([0.0], 1.0);
        assert!((rho.pair(|pos| (1.5 * pos[0]).cos()) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn gaussian_state_variance_grows() {
        let rho0 = MeasureState::<f64, 1>::gaussian([0.0], 1.0, 1.0).unwrap();
        let mut rho1 = MeasureState::<f64, 1>::dirac([0.0], 0.0);
        lemma_a1_push(0.1_f64, 0.5, 0.0, 0.0, &rho0, &mut rho1).unwrap();
        let expected_var = 1.0 + 2.0 * 0.5 * 0.1;
        assert!((rho1.gaussians[0].variance - expected_var).abs() < 1e-14);
    }

    #[test]
    fn second_moment_dirac_formula() {
        // δ_x with weight w: second moment = x² · w
        let rho = MeasureState::<f64, 1>::dirac([3.0], 2.0);
        assert!((rho.second_moment() - 18.0).abs() < 1e-14);
    }
}
