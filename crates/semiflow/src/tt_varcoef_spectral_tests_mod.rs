    use core::f64::consts::TAU;

    use super::*;
    use crate::tt_drift_spectral::apply_drift_spectral_axis;

    // ── §1.6b: constant a, b=0 → residual_tridiag all ≈ 0 (≤1e-13) ────
    // R = L_j - a0·Lap_fd; for const a=a0 and b=0, R=0 exactly.
    #[test]
    fn residual_zero_for_constant_a_zero_drift() {
        let n = 8usize;
        let dx = TAU / n as f64;
        let a0 = 0.7f64;
        let a_coef: Vec<f64> = vec![a0; n];
        let b_zero: Vec<f64> = vec![0.0; n];
        let v_zero: Vec<f64> = vec![0.0; n];
        let (rl, rm, ru) = residual_tridiag(&a_coef, &b_zero, &v_zero, dx, a0);
        let max_r = rl
            .iter()
            .chain(rm.iter())
            .chain(ru.iter())
            .map(|x: &f64| x.abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_r < 1e-13,
            "residual_tridiag nonzero for const a, b=0: max={max_r:.3e} (expected <1e-13)"
        );
    }

    // ── P₂ identity when R=0 ────────────────────────────────────────────
    #[test]
    fn p2_identity_when_r_zero() {
        let n = 6usize;
        let sub = vec![0.0f64; n];
        let main_d = vec![0.0f64; n];
        let sup = vec![0.0f64; n];
        let mut u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3 + 0.1).sin()).collect();
        let u_orig = u.clone();
        p2_apply_tridiag(&mut u, &sub, &main_d, &sup, 0.05);
        let max_err = u
            .iter()
            .zip(u_orig.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-15,
            "P₂(s)·u ≠ u when R=0: max_err={max_err:.3e}"
        );
    }

    // ── §1.6a: const-a, b=0 step equals ADR-0164 spectral (≤1e-12) ─────
    // When a is constant and b=0: R=0 → P₂=I → step = k(τ) = spectral(a0, 0, τ).
    #[test]
    fn varcoef_step_const_a_zero_drift_equals_spectral() {
        let n = 9usize;
        let dx = TAU / n as f64;
        let a0 = 0.6f64;
        let a_coef: Vec<f64> = vec![a0; n];
        let b_zero: Vec<f64> = vec![0.0; n];
        let v_zero: Vec<f64> = vec![0.0; n];
        let tau = 0.02f64;
        let mut line1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.47 + 0.2).cos()).collect();
        let mut line2 = line1.clone();

        // varcoef path (P₂=I for const a, b=0, so pure k(τ))
        varcoef_axis_step(&mut line1, n, dx, &a_coef, &b_zero, &v_zero, tau);

        // ADR-0164 spectral path with b=0
        apply_drift_spectral_axis(&mut line2, n, dx, a0, 0.0, tau);

        let max_err = line1
            .iter()
            .zip(line2.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-12,
            "varcoef(const a, b=0) ≠ spectral(a0,0): max_err={max_err:.3e} (expected <1e-12)"
        );
    }

    // ── varcoef_evolve produces finite output ───────────────────────────
    #[test]
    fn evolve_produces_finite_output() {
        let n = 5usize;
        let d = 3usize;
        let dx = TAU / n as f64;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();
        let coef = AxisCoef {
            a_axis: (0..d)
                .map(|j| {
                    xs.iter()
                        .map(|&x| 0.5 + 0.2 * (x + 0.4 * j as f64).cos())
                        .collect()
                })
                .collect(),
            b_axis: (0..d)
                .map(|j| {
                    xs.iter()
                        .map(|&x| 0.3 * (x + 0.2 * j as f64).sin())
                        .collect()
                })
                .collect(),
            v_axis: (0..d).map(|_| vec![0.0f64; n]).collect(),
        };
        let nd = n.pow(d as u32);
        let u0: Vec<f64> = (0..nd).map(|i| (i as f64 * 0.31).sin()).collect();
        let (u_out, max_imag) = varcoef_evolve(&u0, n, d, dx, &coef, 0.01, 4);
        assert!(
            u_out.iter().all(|x| x.is_finite()),
            "evolve produced non-finite output"
        );
        assert!(
            max_imag < 1e-9,
            "max imag residue too large: {max_imag:.3e}"
        );
    }
