//! Unit tests of the cross-diffusion decomposition (ADR-0213). The gates live in
//! `tests/g_cross_diffusion.rs`.

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
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
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
    let want = [4.0 - 2f64.sqrt(), 4.0, 4.0 + 2f64.sqrt()];
    assert!(
        max_diff(tri.eigenvalues(), &want) <= 1e-13,
        "{:?}",
        tri.eigenvalues()
    );
}

#[test]
fn rejects_complex_negative_and_defective() {
    let bad: [(&[f64], &str); 5] = [
        (&[1.0, -2.0, 2.0, 1.0], "complex"),
        (&[1.0, 0.0, 0.0, -1.0], "negative"),
        (&[0.0, 1.0, 1.0, 0.0], "negative"),
        (&[1.0, 1.0, 0.0, 1.0], "defective"),
        (&[1.0, 1e-9, 0.0, 1.0 + 1e-12], "ill-conditioned"),
    ];
    for (d, why) in bad {
        let r = CrossDiffusion::<f64>::new(d);
        assert!(
            matches!(r, Err(SemiflowError::DomainViolation { .. })),
            "{why}: {r:?}"
        );
    }
    assert!(CrossDiffusion::<f64>::new(&[1.0, 2.0, 3.0]).is_err());
    assert!(CrossDiffusion::<f64>::new(&[f64::NAN]).is_err());
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
    assert!(CrossDiffusion::<f64>::from_modes(&[1.0, 1.0, 1.0, 1.0], &[1.0, 2.0]).is_err());
    assert!(CrossDiffusion::<f64>::from_modes(&[1.0, 0.0, 0.0, 1.0], &[1.0, -2.0]).is_err());
}
