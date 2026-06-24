    use core::f64::consts::TAU;

    use super::*;

    fn grid_xs(n: usize) -> Vec<f64> {
        let dx = TAU / n as f64;
        (0..n).map(|i| i as f64 * dx).collect()
    }

    fn max_abs(v: &[f64]) -> f64 {
        v.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
    }

    // ── Antiderivative round-trip: Psi' = u ──────────────────────────────────
    #[test]
    fn antideriv_roundtrip() {
        let n = 16usize;
        let dx = TAU / n as f64;
        let xs = grid_xs(n);
        let u: Vec<f64> = xs
            .iter()
            .map(|&x| x.sin() + 0.3 * (2.0 * x).cos())
            .collect();
        let psi = spectral_antideriv_1d(&u, n, dx);
        let u_rec = spectral_deriv_1d(&psi, n, dx);
        let err = max_abs(
            &u.iter()
                .zip(u_rec.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        );
        assert!(err < 1e-11, "antideriv round-trip error {err:.3e}");
    }

    // ── Logistic flow: matches explicit formula ───────────────────────────────
    #[test]
    fn logistic_flow_correctness() {
        let n = 8usize;
        let r = 3.0f64;
        let s = 0.1f64;
        let u0: Vec<f64> = (0..n).map(|i| 0.3 + 0.05 * i as f64).collect();
        let mut u = u0.clone();
        react_flow(&mut u, &Reaction::Logistic { r }, s);
        let e = (r * s).exp();
        for (got, &u0i) in u.iter().zip(u0.iter()) {
            let expected = u0i * e / (1.0 - u0i + u0i * e);
            assert!(
                (got - expected).abs() < 1e-14,
                "logistic err {:.3e}",
                got - expected
            );
        }
    }

    // ── Linear flow with c=0 is identity ────────────────────────────────────
    #[test]
    fn linear_flow_zero_is_identity() {
        let mut u: Vec<f64> = (0..10).map(|i| f64::from(i) * 0.1 + 0.05).collect();
        let u_orig = u.clone();
        react_flow(&mut u, &Reaction::Linear { c: 0.0 }, 1.234);
        let err = max_abs(
            &u.iter()
                .zip(u_orig.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        );
        assert!(err < 1e-15, "Linear c=0 not identity: {err:.3e}");
    }

    // ── Reduction: Strang(Linear{c:0}) == pure heat (0 ULP) ─────────────────
    #[test]
    fn strang_linear_zero_equals_heat() {
        let n = 8usize;
        let d = 2usize;
        let dx = TAU / n as f64;
        let nu = 0.15f64;
        let tau = 0.05f64;
        let nsteps = 4usize;
        let xs = grid_xs(n);
        let base: Vec<f64> = xs.iter().map(|&x| 0.3 + 0.2 * x.cos()).collect();
        let nd = n.pow(d as u32);
        let u0: Vec<f64> = (0..nd)
            .map(|flat| base[flat % n] * base[flat / n])
            .collect();
        let reaction = Reaction::Linear { c: 0.0 };
        let cfg = StrangConfig {
            n,
            d,
            dx,
            nu,
            reaction: &reaction,
        };
        let u_strang = strang_rd_evolve(&u0, &cfg, tau, nsteps);
        let mut u_heat = u0.clone();
        for _ in 0..nsteps {
            apply_heat_all_axes(&mut u_heat, n, d, dx, nu, tau);
        }
        for (i, (&gs, &gh)) in u_strang.iter().zip(u_heat.iter()).enumerate() {
            assert_eq!(gs.to_bits(), gh.to_bits(), "Strang(c=0) vs heat at i={i}");
        }
    }

    // ── Cole-Hopf: heat semigroup EXACT (1-shot == 2-step on phi) ────────────
    #[test]
    fn cole_hopf_semigroup_fast() {
        let n = 32usize;
        let dx = TAU / n as f64;
        let nu = 0.10f64;
        let t = 0.10f64;
        let xs = grid_xs(n);
        let u0: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        // Forward Cole-Hopf: get phi0.
        let mean = u0.iter().sum::<f64>() / n as f64;
        let u_zm: Vec<f64> = u0.iter().map(|&x| x - mean).collect();
        let psi = spectral_antideriv_1d(&u_zm, n, dx);
        let two_nu = 2.0 * nu;
        let phi0: Vec<f64> = psi.iter().map(|&p| (-p / two_nu).exp()).collect();
        // 1-shot: apply heat once for T.
        let mut phi_1shot = phi0.clone();
        let _ = apply_drift_spectral_axis(&mut phi_1shot, n, dx, nu, 0.0f64, t);
        // 2-step: apply heat twice for T/2.
        let mut phi_2step = phi0.clone();
        let _ = apply_drift_spectral_axis(&mut phi_2step, n, dx, nu, 0.0f64, t / 2.0);
        let _ = apply_drift_spectral_axis(&mut phi_2step, n, dx, nu, 0.0f64, t / 2.0);
        let err = max_abs(
            &phi_1shot
                .iter()
                .zip(phi_2step.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        );
        assert!(
            err < 1e-9,
            "Cole-Hopf heat semigroup (1-shot vs 2-step on phi): {err:.3e}"
        );
    }
