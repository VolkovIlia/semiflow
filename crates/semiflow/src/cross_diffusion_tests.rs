//! Unit tests of the cross-diffusion decomposition (ADR-0213). The gates live in
//! `tests/g_cross_diffusion*.rs`.

use alloc::vec::Vec;

use super::*;

fn reconstruct(cd: &CrossDiffusion<f64>) -> Vec<f64> {
    let k = cd.k;
    (0..k * k)
        .map(|e| {
            let (i, j) = (e / k, e % k);
            (0..k)
                .map(|l| cd.p[i * k + l] * cd.lambda[l] * cd.p_inv[l * k + j])
                .sum()
        })
        .collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| libm::fabs(x - y))
        .fold(0.0, f64::max)
}

fn scale_of(d: &[f64]) -> f64 {
    d.iter().fold(0.0, |m, x| m.max(libm::fabs(*x)))
}

#[test]
fn decomposes_non_symmetric_and_repeated_spectra() {
    let cases: [&[f64]; 5] = [
        &[2.0, 1.0, 0.5, 3.0],                          // distinct, non-symmetric
        &[1.0, 0.0, 0.0, 1.0],                          // repeated, diagonalisable
        &[4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0], // symmetric 3×3
        &[1.0, 0.2, 0.0, 0.0, 2.0, 0.3, 0.1, 0.0, 3.0], // non-symmetric 3×3
        &[0.0, 0.0, 0.0, 0.0],                          // zero (no diffusion)
    ];
    for d in cases {
        let cd = CrossDiffusion::<f64>::new(d).unwrap();
        assert!(max_diff(&reconstruct(&cd), d) <= 1e-12, "{d:?}");
        assert!(cd.eigenvalues().iter().all(|&l| l >= 0.0));
    }
    let tri = CrossDiffusion::<f64>::new(&[4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0]).unwrap();
    let want = [4.0 - libm::sqrt(2.0), 4.0, 4.0 + libm::sqrt(2.0)];
    assert!(
        max_diff(tri.eigenvalues(), &want) <= 1e-13,
        "{:?}",
        tri.eigenvalues()
    );
}

/// Well-conditioned matrices the review found rejected: non-normal (`cond₁(P) ≈
/// 2e3`), nearly repeated symmetric (cond 1), well-separated but `‖D‖ ≫ λ`.
#[test]
fn accepts_well_conditioned_non_normal_and_close_spectra() {
    let cases: [&[f64]; 5] = [
        &[1.0, 0.0, 2000.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        &[1.0, 0.0, 100.0, 0.0, 1.05, 0.0, 0.0, 0.0, 1.1],
        &[1.0, 1e-8, 1e-8, 1.0],
        &[0.1, 1e6, 0.0, 0.5],
        &[2.0, 1e-7, 0.0, 0.0, 2.0 + 1e-7, 1e-7, 0.0, 0.0, 2.0 + 2e-7],
    ];
    for d in cases {
        let cd = CrossDiffusion::<f64>::new(d).unwrap_or_else(|e| panic!("{d:?}: {e:?}"));
        assert!(
            max_diff(&reconstruct(&cd), d) <= 1e-12 * scale_of(d),
            "{d:?}"
        );
    }
}

/// The iteration runs on a power-of-two normalised copy: `c·D` decomposes for
/// any representable scale, with eigenvalues `c·λ`.
#[test]
fn decomposition_is_scale_invariant() {
    let base = [1.0, 0.5, 0.2, 0.6];
    let lam = CrossDiffusion::<f64>::new(&base).unwrap().lambda;
    for c in [1e-300, 1e-160, 0.5, 1e155, 1e300] {
        let d: Vec<f64> = base.iter().map(|x| c * x).collect();
        let cd = CrossDiffusion::<f64>::new(&d).unwrap_or_else(|e| panic!("{c}: {e:?}"));
        for (got, want) in cd.eigenvalues().iter().zip(&lam) {
            assert!(
                libm::fabs(got - c * want) <= 1e-14 * c,
                "{c}: {got} vs {}",
                c * want
            );
        }
    }
}

/// `f32` stores the `f64`-verified decomposition rounded.
#[test]
fn f32_decomposition_is_the_rounded_f64_one() {
    for d in [[0.1, 0.0, 0.0, 0.2], [1.0, 0.5, 0.2, 0.6]] {
        let (c32, c64) = (
            CrossDiffusion::<f32>::new(&d).unwrap(),
            CrossDiffusion::<f64>::new(&d).unwrap(),
        );
        #[allow(clippy::cast_possible_truncation)]
        let rounded = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        assert_eq!(c32.modes(), rounded(c64.modes()).as_slice());
        assert_eq!(c32.inverse_modes(), rounded(c64.inverse_modes()).as_slice());
    }
}

fn reason(r: Result<CrossDiffusion<f64>, SemiflowError>) -> &'static str {
    match r {
        Err(SemiflowError::DomainViolation { what, .. }) => what,
        other => panic!("expected a DomainViolation, got {other:?}"),
    }
}

#[test]
fn rejects_with_the_reason() {
    let bad: [(&[f64], &str); 8] = [
        (&[1.0, -2.0, 2.0, 1.0], "complex eigenvalues"),
        (&[1.0, 0.0, 0.0, -1.0], "negative eigenvalue"),
        (&[0.0, 1.0, 1.0, 0.0], "negative eigenvalue"),
        (&[1.0, 0.0, 0.0, -1e-8], "negative eigenvalue"),
        (&[1.0, -1e-8, 1e-8, 1.0], "complex eigenvalues"),
        (&[1.0, 1.0, 0.0, 1.0], "does not reproduce D"),
        (&[1.0, 1.0, 0.0, 1.0 + 1e-9], "ill-conditioned"),
        (
            &[1.0, 1.0, 0.0, 0.0, 1.0 + 1e-5, 1.0, 0.0, 0.0, 1.0 + 2e-5],
            "ill-conditioned",
        ),
    ];
    for (d, why) in bad {
        let what = reason(CrossDiffusion::<f64>::new(d));
        assert!(what.contains(why), "{d:?}: expected {why:?}, got {what:?}");
    }
    assert!(reason(CrossDiffusion::<f64>::new(&[1.0, 2.0, 3.0])).contains("K×K"));
    assert!(reason(CrossDiffusion::<f64>::new(&[f64::NAN])).contains("non-finite"));
}

#[test]
fn modes_round_trip_and_user_decomposition() {
    let cd = CrossDiffusion::<f64>::new(&[2.0, 1.0, 0.5, 3.0]).unwrap();
    let n = 3;
    let u = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
    let (mut m, mut back) = ([0.0; 6], [0.0; 6]);
    cd.to_modes(&u, &mut m, n);
    cd.to_species(&m, &mut back, n);
    assert!(max_diff(&back, &u) <= 1e-14);
    let user = CrossDiffusion::<f64>::from_modes(&[1.0, 1.0, 1.0, -1.0], &[3.0, 1.0]).unwrap();
    assert!(max_diff(&reconstruct(&user), &[2.0, 1.0, 1.0, 2.0]) <= 1e-15);
    let cases: [(&[f64], &[f64], &str); 3] = [
        (
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0, 2.0],
            "singular eigenvector matrix",
        ),
        (
            &[1.0, 1.0, 1.0, 1.0 + 1e-10],
            &[1.0, 2.0],
            "ill-conditioned",
        ),
        (&[1.0, 0.0, 0.0, 1.0], &[1.0, -2.0], "negative eigenvalue"),
    ];
    for (p, l, why) in cases {
        let what = reason(CrossDiffusion::<f64>::from_modes(p, l));
        assert!(what.contains(why), "{why}: {what}");
    }
}
