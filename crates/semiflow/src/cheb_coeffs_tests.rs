// Unit tests for `cheb_coeffs` — included as `mod tests`.

use alloc::vec::Vec;

use super::{exp_chebyshev_coefficients, scaled_bessel_into, MIN_CHEB_DEGREE};

/// `Σ' aₖ Tₖ(cos θ) = Σ aₖ cos kθ` against `e^{−z(1+cos θ)}`.
fn max_series_error(z: f64, tol: f64) -> (f64, usize) {
    let mut a = Vec::new();
    let m = exp_chebyshev_coefficients(z, tol, &mut a);
    let mut worst = 0.0_f64;
    for j in 0..=64 {
        let theta = core::f64::consts::PI * f64::from(j) / 64.0;
        let x = libm::cos(theta);
        #[allow(clippy::cast_precision_loss)]
        let sum: f64 = a
            .iter()
            .enumerate()
            .map(|(k, &ak)| ak * libm::cos(k as f64 * theta))
            .sum();
        let exact = libm::exp(-z * (1.0 + x));
        worst = worst.max((sum - exact).abs());
    }
    (worst, m)
}

/// The truncated series reproduces `e^{−z(1+x)}` to `tol` for `z` from `1e-6` to `1e7`.
#[test]
fn chebyshev_series_matches_exponential() {
    for &tol in &[1e-6, 1e-10, 1e-12] {
        for &z in &[1e-6, 0.3, 1.0, 1.0 + 1e-9, 2.5, 10.0, 200.0, 1e3, 1e5, 1e7] {
            let (err, m) = max_series_error(z, tol);
            // Truncation ≤ tol/4; the rest is rounding of ~m terms of size ≤ 2/√(2πz).
            #[allow(clippy::cast_precision_loss)]
            let budget = tol / 4.0 + 64.0 * f64::EPSILON * (1.0 + (m as f64).sqrt());
            assert!(
                err <= budget,
                "z={z} tol={tol}: max error {err:e} > {budget:e} (m={m})"
            );
            assert!(m >= MIN_CHEB_DEGREE);
        }
    }
}

/// Degree grows like `√(2z·ln(1/tol))`, not like `z` (the substepping it replaces).
#[test]
fn chebyshev_degree_is_sqrt_z() {
    for &z in &[1e2, 1e4, 1e6, 1e8] {
        let mut a = Vec::new();
        let m = exp_chebyshev_coefficients(z, 1e-12, &mut a);
        let ln = -libm::log(1e-12 / 8.0);
        #[allow(clippy::cast_precision_loss)]
        let bound = libm::sqrt(2.0 * z * ln) + 2.0 * ln + 8.0;
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        assert!(mf <= bound, "z={z}: degree {m} > {bound}");
    }
}

/// Series and Miller branches agree across the switch at `z = 1`, and the
/// scaled values satisfy `c₀ + 2Σcₖ = 1`.
#[test]
fn scaled_bessel_continuity_and_normalisation() {
    let (mut lo, mut hi) = ([0.0; 40], [0.0; 40]);
    scaled_bessel_into(1.0, &mut lo);
    scaled_bessel_into(1.0 + 1e-12, &mut hi);
    for k in 0..12 {
        let rel = (lo[k] - hi[k]).abs() / lo[k];
        assert!(
            rel < 1e-10,
            "k={k}: series {} vs Miller {} (rel {rel:e})",
            lo[k],
            hi[k]
        );
    }
    for &z in &[0.5, 3.0, 40.0, 900.0] {
        let mut c = [0.0; 400];
        scaled_bessel_into(z, &mut c);
        let total = c[0] + 2.0 * c[1..].iter().sum::<f64>();
        assert!((total - 1.0).abs() < 1e-13, "z={z}: c0 + 2Σc = {total}");
    }
}
