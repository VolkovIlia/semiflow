use super::*;
use crate::{chernoff::ChernoffFunction, scratch::ScratchPool};

fn heat_d1(cap: usize) -> GridlessChernoff<f64, 1> {
    GridlessChernoff::isotropic(0.5, 0.0, 0.0, ParticleReduction::WeightedVoronoi { cap })
}

// ── D=1 push-forward ────────────────────────────────────────────────────────

#[test]
fn push_forward_single_dirac_count_and_mass() {
    let mut rho1 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    let rho0 = rho1.clone();
    heat_d1(16)
        .apply_into(0.1, &rho0, &mut rho1, &mut ScratchPool::new())
        .unwrap();
    assert_eq!(rho1.as_diracs_slice().len(), 3);
    let w: f64 = rho1.as_diracs_slice().iter().map(|(_, w)| w).sum();
    assert!((w - 1.0).abs() < 1e-13, "mass={w}");
}

#[test]
fn push_forward_reaction_mass() {
    let ev = GridlessChernoff::<f64, 1>::isotropic(
        0.5,
        0.0,
        1.0,
        ParticleReduction::WeightedVoronoi { cap: 16 },
    );
    let tau = 0.05_f64;
    let rho0 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    let mut rho1 = rho0.clone();
    ev.apply_into(tau, &rho0, &mut rho1, &mut ScratchPool::new())
        .unwrap();
    let w: f64 = rho1.as_diracs_slice().iter().map(|(_, w)| w).sum();
    assert!((w - (1.0 + tau)).abs() < 1e-12, "mass={w}");
}

#[test]
fn push_forward_branch_weights_exact() {
    let tau = 0.1_f64;
    let rho0 = MeasureState::<f64, 1>::dirac([1.0], 2.0);
    let mut rho1 = rho0.clone();
    heat_d1(16)
        .apply_into(tau, &rho0, &mut rho1, &mut ScratchPool::new())
        .unwrap();
    let h = 2.0 * (0.5 * tau).sqrt();
    let d = rho1.as_diracs_slice();
    let fw = |t: f64| {
        d.iter()
            .find(|(p, _)| (p[0] - t).abs() < 1e-11)
            .map_or(f64::NAN, |(_, w)| *w)
    };
    assert!((fw(1.0 + h) - 0.5).abs() < 1e-13, "+h");
    assert!((fw(1.0 - h) - 0.5).abs() < 1e-13, "-h");
    assert!((fw(1.0) - 1.0).abs() < 1e-13, "k=0");
}

#[test]
fn push_forward_positions_exact() {
    let (a, b, tau, x0) = (0.25_f64, 0.5_f64, 0.04_f64, 1.0_f64);
    let ev = GridlessChernoff::<f64, 1>::isotropic(
        a,
        b,
        0.0,
        ParticleReduction::WeightedVoronoi { cap: 16 },
    );
    let rho0 = MeasureState::<f64, 1>::dirac([x0], 1.0);
    let mut rho1 = rho0.clone();
    ev.apply_into(tau, &rho0, &mut rho1, &mut ScratchPool::new())
        .unwrap();
    let h = 2.0 * (a * tau).sqrt();
    let k = 2.0 * b * tau;
    let ps: alloc::vec::Vec<f64> = rho1.as_diracs_slice().iter().map(|(p, _)| p[0]).collect();
    assert!(
        ps.iter().any(|&p| (p - (x0 + h)).abs() < 1e-12),
        "+h missing"
    );
    assert!(
        ps.iter().any(|&p| (p - (x0 - h)).abs() < 1e-12),
        "-h missing"
    );
    assert!(
        ps.iter().any(|&p| (p - (x0 + k)).abs() < 1e-12),
        "+k missing"
    );
}

// ── Anti-D1: per-axis spread on D=2 and D=3 ────────────────────────────────

#[test]
fn per_axis_second_moment_nonzero_d2() {
    let ev = GridlessChernoff::<f64, 2>::isotropic(
        0.5,
        0.0,
        0.0,
        ParticleReduction::WeightedVoronoi { cap: 256 },
    );
    let rho0 = MeasureState::<f64, 2>::dirac([0.0, 0.0], 1.0);
    let mut rho1 = rho0.clone();
    ev.apply_into(0.1, &rho0, &mut rho1, &mut ScratchPool::new())
        .unwrap();
    let d = rho1.as_diracs_slice();
    for j in 0..2 {
        let m2: f64 = d.iter().map(|(p, w)| p[j] * p[j] * w).sum();
        assert!(m2 > 1e-12, "D=2 axis-{j} m2={m2}");
    }
}

#[test]
fn per_axis_second_moment_nonzero_d3() {
    let ev = GridlessChernoff::<f64, 3>::isotropic(
        0.5,
        0.0,
        0.0,
        ParticleReduction::WeightedVoronoi { cap: 512 },
    );
    let rho0 = MeasureState::<f64, 3>::dirac([0.0; 3], 1.0);
    let mut rho1 = rho0.clone();
    ev.apply_into(0.1, &rho0, &mut rho1, &mut ScratchPool::new())
        .unwrap();
    let d = rho1.as_diracs_slice();
    for j in 0..3 {
        let m2: f64 = d.iter().map(|(p, w)| p[j] * p[j] * w).sum();
        assert!(m2 > 1e-12, "D=3 axis-{j} m2={m2}");
    }
}

// ── Voronoi reduction ───────────────────────────────────────────────────────

#[test]
fn voronoi_mass_preserved() {
    let mut e = MeasureState::<f64, 1>::dirac([0.0], 0.6);
    e.push_dirac_raw([1.0], 0.4);
    let mb: f64 = e.as_diracs_slice().iter().map(|(_, w)| w).sum();
    ParticleReduction::WeightedVoronoi { cap: 1 }
        .apply(&mut e)
        .unwrap();
    let ma: f64 = e.as_diracs_slice().iter().map(|(_, w)| w).sum();
    assert!((ma - mb).abs() < 1e-13, "mass drift {mb}→{ma}");
}

#[test]
fn voronoi_first_moment_preserved() {
    let mut e = MeasureState::<f64, 1>::dirac([0.0], 0.6);
    e.push_dirac_raw([1.0], 0.4);
    let mb: f64 = e.as_diracs_slice().iter().map(|(p, w)| p[0] * w).sum();
    ParticleReduction::WeightedVoronoi { cap: 1 }
        .apply(&mut e)
        .unwrap();
    let ma: f64 = e.as_diracs_slice().iter().map(|(p, w)| p[0] * w).sum();
    assert!((ma - mb).abs() < 1e-13, "m1 drift {mb}→{ma}");
}

#[test]
fn voronoi_noop_within_cap() {
    let mut e = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    e.push_dirac_raw([1.0], 0.5);
    let n = e.n_diracs();
    ParticleReduction::WeightedVoronoi { cap: 10 }
        .apply(&mut e)
        .unwrap();
    assert_eq!(e.n_diracs(), n);
}

#[test]
fn voronoi_cap_zero_err() {
    let mut e = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    assert!(ParticleReduction::WeightedVoronoi { cap: 0 }
        .apply(&mut e)
        .is_err());
}

// ── Multi-step ──────────────────────────────────────────────────────────────

#[test]
fn multistep_mass_conserved() {
    let ev = heat_d1(32);
    let mut rho = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    let mut pool = ScratchPool::new();
    for _ in 0..8 {
        let s = rho.clone();
        ev.apply_into(0.05, &s, &mut rho, &mut pool).unwrap();
    }
    let w: f64 = rho.as_diracs_slice().iter().map(|(_, w)| w).sum();
    assert!((w - 1.0).abs() < 1e-10, "mass={w}");
}

#[test]
fn multistep_second_moment_grows_heat() {
    let (a, tau, n) = (0.5_f64, 0.01_f64, 20_u32);
    let ev = GridlessChernoff::<f64, 1>::isotropic(
        a,
        0.0,
        0.0,
        ParticleReduction::WeightedVoronoi { cap: 64 },
    );
    let mut rho = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    let mut pool = ScratchPool::new();
    for _ in 0..n {
        let s = rho.clone();
        ev.apply_into(tau, &s, &mut rho, &mut pool).unwrap();
    }
    let m2: f64 = rho
        .as_diracs_slice()
        .iter()
        .map(|(p, w)| p[0] * p[0] * w)
        .sum();
    let ex = f64::from(n) * 2.0 * a * tau;
    assert!((m2 - ex).abs() < 0.30 * ex + 1e-8, "m2={m2} ex={ex}");
}

// ── Error paths ─────────────────────────────────────────────────────────────

#[test]
fn negative_tau_err() {
    let rho0 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    let mut rho1 = rho0.clone();
    assert!(heat_d1(16)
        .apply_into(-0.01, &rho0, &mut rho1, &mut ScratchPool::new())
        .is_err());
}

#[test]
fn negative_a_err() {
    let ev = GridlessChernoff::<f64, 1>::isotropic(
        -0.1,
        0.0,
        0.0,
        ParticleReduction::WeightedVoronoi { cap: 4 },
    );
    let rho0 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    let mut rho1 = rho0.clone();
    assert!(ev
        .apply_into(0.1, &rho0, &mut rho1, &mut ScratchPool::new())
        .is_err());
}

// ── D=2 smoke ───────────────────────────────────────────────────────────────

#[test]
fn push_forward_2d_mass_and_spread() {
    let ev = GridlessChernoff::<f64, 2>::isotropic(
        0.5,
        0.0,
        0.0,
        ParticleReduction::WeightedVoronoi { cap: 256 },
    );
    let rho0 = MeasureState::<f64, 2>::dirac([0.0, 0.0], 1.0);
    let mut rho1 = rho0.clone();
    ev.apply_into(0.1, &rho0, &mut rho1, &mut ScratchPool::new())
        .unwrap();
    let w: f64 = rho1.as_diracs_slice().iter().map(|(_, w)| w).sum();
    assert!((w - 1.0).abs() < 1e-12, "mass={w}");
    let d = rho1.as_diracs_slice();
    for j in 0..2 {
        let m2: f64 = d.iter().map(|(p, w)| p[j] * p[j] * w).sum();
        assert!(m2 > 1e-12, "D=2 axis-{j} not spread");
    }
}
