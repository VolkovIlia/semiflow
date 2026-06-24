    use super::*;

    /// §1.4(b): D = diag(a), b = 0 → symbol equals separable diffusion.
    ///
    /// With no off-diagonal coupling and no drift the cross sum vanishes and the
    /// imaginary part is zero. The real part is the separable diffusion symbol.
    #[test]
    fn diagonal_d_zero_b_is_separable() {
        let n = 7usize;
        let d = 3usize;
        let dx = 1.0_f64 / n as f64;
        let a_diag = [0.5f64, 0.7, 0.4];
        let b_zero = [0.0f64, 0.0, 0.0];
        let tau = 0.02f64;

        // Dense path with D = diag(a), b = 0.
        let d_mat = rank1_dense_matrix(&a_diag, &[0.0f64, 0.0, 0.0], 0.0);
        let es_dense = dense_expsym_nd(n, d, dx, &d_mat, &b_zero, tau);

        // Reference: compute separable symbol manually.
        let two_pi = core::f64::consts::TAU;
        let sym_d2: Vec<f64> = (0..n)
            .map(|m| (2.0 * (two_pi * m as f64 / n as f64).cos() - 2.0) / (dx * dx))
            .collect();
        let nd = n.pow(d as u32);
        for flat in 0..nd {
            let mut f = flat;
            let mut sym_re = 0.0f64;
            for j in (0..d).rev() {
                let mj = f % n;
                sym_re += a_diag[j] * sym_d2[mj];
                f /= n;
            }
            let expected_re = (tau * sym_re).exp();
            let re = es_dense[2 * flat];
            let im = es_dense[2 * flat + 1];
            assert!(im.abs() < 1e-14, "im nonzero at flat={flat}: {im:.3e}");
            assert!(
                (re - expected_re).abs() < 1e-12,
                "re mismatch at flat={flat}: got {re:.10}, expected {expected_re:.10}"
            );
        }
    }

    /// §1.4(a): tridiagonal D → dense expsym must equal adjacent-only symbol bit-for-bit.
    ///
    /// When D\[j,k\] = 0 for |j-k| > 1 the pair sum reduces to adjacent pairs only.
    #[test]
    fn tridiagonal_d_equals_adjacent_symbol() {
        let n = 5usize;
        let d = 3usize;
        let dx = 1.0_f64 / n as f64;
        let tau = 0.02f64;
        let rho = 0.15f64;
        let a_val = 0.5f64;
        let b_vals = [0.6f64, 0.7, 0.8];

        let d_diag = [a_val, a_val, a_val];
        let mut d_tri = rank1_dense_matrix(&d_diag, &[0.0f64, 0.0, 0.0], 0.0);
        d_tri[1] = rho;
        d_tri[d] = rho;
        d_tri[d + 2] = rho;
        d_tri[2 * d + 1] = rho;

        let es_dense = dense_expsym_nd(n, d, dx, &d_tri, &b_vals, tau);
        check_adjacent_only_symbol(&es_dense, n, d, dx, tau, rho, a_val, &b_vals);
    }

    /// Compute adjacent-only symbol (re, im) for one Fourier mode tuple.
    #[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
    fn adjacent_symbol_one(
        modes: &[usize; 3],
        sd2: &[f64],
        sd1r: &[f64],
        tau: f64,
        rho: f64,
        a: f64,
        b: &[f64],
    ) -> (f64, f64) {
        let d = modes.len();
        let mut sym_re: f64 = (0..d).map(|j| a * sd2[modes[j]]).sum();
        sym_re -= 2.0 * rho * sd1r[modes[0]] * sd1r[modes[1]];
        sym_re -= 2.0 * rho * sd1r[modes[1]] * sd1r[modes[2]];
        let sym_im: f64 = b.iter().zip(modes.iter()).map(|(&bj, &m)| bj * sd1r[m]).sum();
        let exp_re = (tau * sym_re).exp();
        let phase = tau * sym_im;
        (exp_re * phase.cos(), exp_re * phase.sin())
    }

    /// Compare expsym against the adjacent-only reference formula (d=3 case).
    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
    fn check_adjacent_only_symbol(
        es_dense: &[f64],
        n: usize,
        d: usize,
        dx: f64,
        tau: f64,
        rho: f64,
        a: f64,
        b: &[f64],
    ) {
        let two_pi = core::f64::consts::TAU;
        let nf = n as f64;
        let sd2: Vec<f64> =
            (0..n).map(|m| (2.0 * (two_pi * m as f64 / nf).cos() - 2.0) / (dx * dx)).collect();
        let sd1r: Vec<f64> = (0..n).map(|m| (two_pi * m as f64 / nf).sin() / dx).collect();
        let nd = n.pow(d as u32);
        for flat in 0..nd {
            let mut f = flat;
            let mut modes = [0usize; 3];
            for j in (0..d).rev() {
                modes[j] = f % n;
                f /= n;
            }
            let (ere, eim) = adjacent_symbol_one(&modes, &sd2, &sd1r, tau, rho, a, b);
            assert_eq!(
                es_dense[2 * flat].to_bits(),
                ere.to_bits(),
                "tridiag re mismatch flat={flat}: got {:.12}, exp {ere:.12}",
                es_dense[2 * flat]
            );
            assert_eq!(
                es_dense[2 * flat + 1].to_bits(),
                eim.to_bits(),
                "tridiag im mismatch flat={flat}: got {:.12}, exp {eim:.12}",
                es_dense[2 * flat + 1]
            );
        }
    }

    /// Smoke test: rank1_dense_matrix has non-zero off-diagonals.
    #[test]
    fn rank1_dense_nonzero_offdiag() {
        let d = 4usize;
        let a: Vec<f64> = vec![0.5; d];
        let g: Vec<f64> = (0..d).map(|k| (k as f64 * 0.3 + 0.5).cos() * 0.6).collect();
        let mat = rank1_dense_matrix(&a, &g, 0.25);
        let n_offdiag = (0..d)
            .flat_map(|i| (0..d).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j && mat[i * d + j].abs() > 1e-14)
            .count();
        assert_eq!(n_offdiag, d * (d - 1), "all off-diag should be non-zero");
    }

    /// Round-trip: fft_nd_real → ifft_nd recovers original.
    #[test]
    fn fft_nd_roundtrip() {
        let n = 5usize;
        let d = 3usize;
        let nd = n.pow(d as u32);
        let u0: Vec<f64> = (0..nd).map(|i| ((i as f64) * 0.37 + 0.1).sin()).collect();
        let cplx = fft_nd_real(&u0, n, d);
        let (recovered, max_imag) = ifft_nd(&cplx, n, d);
        let max_err = u0
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-12, "round-trip err={max_err:.3e}");
        assert!(max_imag < 1e-12, "round-trip max_imag={max_imag:.3e}");
    }
