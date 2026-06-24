    use core::f64::consts::TAU;

    use super::*;
    use crate::tt_drift_spectral::apply_drift_spectral_axis;

    fn grid_xs(n: usize) -> Vec<f64> {
        let dx = TAU / n as f64;
        (0..n).map(|i| i as f64 * dx).collect()
    }

    // ── core_tridiag: Diffusion sums to zero per row (periodic FD) ──────
    #[test]
    fn core_diffusion_row_sum_zero() {
        let n = 8;
        let dx = TAU / n as f64;
        let (sub, main, sup) = core_tridiag::<f64>(CoefRole::Diffusion, dx, n);
        for i in 0..n {
            let s = sub[i] + main[i] + sup[i];
            assert!(s.abs() < 1e-14, "Diffusion row sum ≠ 0 at i={i}: {s:.3e}");
        }
    }

    // ── P₂ identity when coefs empty ────────────────────────────────────
    #[test]
    fn p2_identity_no_coefs() {
        let n = 5;
        let d = 2;
        let nd = n * n;
        let dx = TAU / n as f64;
        let mut u: Vec<f64> = (0..nd).map(|i| (i as f64 * 0.3 + 0.1).sin()).collect();
        let u_orig = u.clone();
        let mut scratch = vec![0.0f64; nd];
        p2_apply(&mut u, &mut scratch, n, d, dx, &[], 0.05);
        let max_err = u
            .iter()
            .zip(u_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-15,
            "P₂ with no coefs ≠ identity: {max_err:.3e}"
        );
    }

    // ── const-coef reduction: k_spectral(a0, b=0) step equals ADR-0164 ─
    #[test]
    fn k_spectral_const_a_equals_spectral_axis() {
        let n: usize = 7;
        let d: usize = 2;
        let nd = n.pow(d as u32);
        let dx = TAU / n as f64;
        let a0 = 0.5f64;
        let tau = 0.02f64;
        let u0: Vec<f64> = (0..nd).map(|i| (i as f64 * 0.37 + 0.1).cos()).collect();

        // Our k_spectral (applies on all d axes).
        let mut u1 = u0.clone();
        k_spectral(&mut u1, n, d, dx, a0, tau);

        // Reference: apply ADR-0164 axis-by-axis manually.
        let stride = n; // stride of axis 0 for d=2
        let mut u2 = u0.clone();
        // axis 0
        for i_inner in 0..stride {
            let mut line: Vec<f64> = (0..n).map(|i0| u2[i0 * stride + i_inner]).collect();
            apply_drift_spectral_axis(&mut line, n, dx, a0, 0.0, tau);
            for i0 in 0..n {
                u2[i0 * stride + i_inner] = line[i0];
            }
        }
        // axis 1
        for i_outer in 0..n {
            let mut line: Vec<f64> = (0..n).map(|i1| u2[i_outer * n + i1]).collect();
            apply_drift_spectral_axis(&mut line, n, dx, a0, 0.0, tau);
            for i1 in 0..n {
                u2[i_outer * n + i1] = line[i1];
            }
        }

        let max_err = u1
            .iter()
            .zip(u2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-12,
            "k_spectral ≠ axis-by-axis spectral: {max_err:.3e}"
        );
    }

    // ── apply_residual: zero for empty terms ────────────────────────────
    #[test]
    fn residual_zero_for_empty_terms() {
        let n = 5;
        let d = 2;
        let nd = n * n;
        let dx = TAU / n as f64;
        let coef = CpCoef::<f64> {
            c0: 0.5,
            terms: vec![],
            role: CoefRole::Diffusion,
        };
        let u: Vec<f64> = (0..nd).map(|i| (i as f64).sin()).collect();
        let mut out = vec![1.0f64; nd]; // non-zero to check zeroing
        apply_residual(&u, &mut out, n, d, dx, &coef);
        let max_abs = out.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(
            max_abs < 1e-15,
            "residual with no terms not zero: {max_abs:.3e}"
        );
    }

    // ── nonsep_evolve produces finite output ────────────────────────────
    #[test]
    fn evolve_produces_finite_output() {
        let n = 5;
        let d = 2;
        let nd = n * n;
        let dx = TAU / n as f64;
        let xs = grid_xs(n);
        let a0 = 0.5f64;

        // rank-1 non-separable coefficient: 0.25 * cos(x) * sin(y) [diffusion role]
        let term = CpTerm::<f64> {
            factor: vec![
                xs.iter().map(|&x| 0.25 * x.cos()).collect(),
                xs.iter().map(|&x| x.sin()).collect(),
            ],
        };
        let coefs = vec![CpCoef {
            c0: a0,
            terms: vec![term],
            role: CoefRole::Diffusion,
        }];
        let u0: Vec<f64> = (0..nd).map(|i| (i as f64 * 0.31).sin()).collect();
        let (u_out, max_imag) = nonsep_evolve(&u0, n, d, dx, a0, &coefs, 0.01, 4);
        assert!(
            u_out.iter().all(|x| x.is_finite()),
            "evolve produced non-finite"
        );
        assert!(
            max_imag < 1e-9,
            "max imag residue too large: {max_imag:.3e}"
        );
    }
