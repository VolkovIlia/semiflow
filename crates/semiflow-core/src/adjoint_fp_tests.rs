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

    // ── §38.12 variance diagnostic ───────────────────────────────────────────

    /// Explicit D=2 mixture: two Diracs + one Gaussian.
    ///
    /// Mixture:
    ///   δ_{[1.0, 0.0]} weight 1.0
    ///   δ_{[3.0, 0.0]} weight 1.0
    ///   Gaussian mean=[2.0, 0.0], var=1.0, weight=2.0
    ///
    /// mass = 1 + 1 + 2 = 4
    /// mu_x = (1·1 + 1·3 + 2·2) / 4 = 8/4 = 2.0
    /// mu_y = 0.0
    /// E[x²] = (1·1 + 1·9 + 2·(4 + 1·2)) / 4 = (1+9+12)/4 = 22/4 = 5.5
    ///   (D=2 Gaussian contribution: w·(|mean|²+D·var) = 2·(4+2) = 12)
    /// Var = E[x²] - (mu_x²+mu_y²) = 5.5 - 4.0 = 1.5
    ///
    /// Per-axis:
    ///   E[x_0²] = (1·1 + 1·9 + 2·(4+1)) / 4 = (1+9+10)/4 = 5.0
    ///   Var_0 = 5.0 - 4.0 = 1.0
    ///   E[x_1²] = (0+0 + 2·(0+1)) / 4 = 2/4 = 0.5
    ///   Var_1 = 0.5 - 0.0 = 0.5
    ///   Var_0 + Var_1 = 1.5 ✓
    #[test]
    fn variance_dirac_gaussian_d2() {
        let mut rho = MeasureState::<f64, 2>::dirac([1.0, 0.0], 1.0);
        rho.push_dirac_raw([3.0, 0.0], 1.0);
        let gauss = MeasureState::<f64, 2>::gaussian([2.0, 0.0], 1.0, 2.0).unwrap();
        // Merge by axpy (α=1, so weights transfer unchanged)
        use crate::state::State;
        rho.axpy_into(1.0, &gauss);

        let tol = 1e-12_f64;

        // first_moment
        let mu = rho.first_moment();
        assert!((mu[0] - 2.0).abs() < tol, "mu[0]={}", mu[0]);
        assert!(mu[1].abs() < tol, "mu[1]={}", mu[1]);

        // variance (scalar)
        let var_s = rho.variance();
        assert!((var_s - 1.5).abs() < tol, "variance={var_s}");

        // variance_per_axis
        let var_ax = rho.variance_per_axis();
        assert!((var_ax[0] - 1.0).abs() < tol, "var_ax[0]={}", var_ax[0]);
        assert!((var_ax[1] - 0.5).abs() < tol, "var_ax[1]={}", var_ax[1]);

        // consistency: Var = Var_0 + Var_1
        assert!((var_s - (var_ax[0] + var_ax[1])).abs() < tol, "Var≠Σ_d Var_d");
    }

    #[test]
    fn variance_zero_mass_returns_zero() {
        // Empty measure: first_moment/variance/variance_per_axis must not NaN.
        let rho = MeasureState::<f64, 2>::from_particles(&[]);
        let mu = rho.first_moment();
        assert_eq!(mu, [0.0, 0.0]);
        assert_eq!(rho.variance(), 0.0);
        assert_eq!(rho.variance_per_axis(), [0.0, 0.0]);
    }
}
