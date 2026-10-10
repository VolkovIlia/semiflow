//! Unit tests of the reaction–diffusion plumbing (ADR-0208).

use alloc::{boxed::Box, vec, vec::Vec};

use super::*;
use crate::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid2d::Grid2D,
    grid3d::Grid3D,
    reaction::{FnReaction, GrayScott, Nagumo},
};

/// Coordinates follow the storage order `from_fn` fills (x fastest).
#[test]
fn coordinates_match_storage_order() {
    let g = Grid1D::new(-1.0, 2.0, 4).unwrap();
    let h = Grid1D::new(0.0, 1.0, 5).unwrap();
    let f2 = GridFn2D::from_fn(Grid2D::new(g, h), |x, y| 10.0 * x + y);
    let c2 = f2.node_coordinates();
    let n2 = f2.values.len();
    for i in 0..n2 {
        assert_eq!(
            f2.values[i].to_bits(),
            (10.0 * c2[i] + c2[n2 + i]).to_bits()
        );
    }
    let grid3 = Grid3D::new(g, h, Grid1D::new(5.0, 6.0, 4).unwrap()).unwrap();
    let f3 = GridFn3D::from_fn(grid3, |x, y, z| 100.0 * x + 10.0 * y + z);
    let c3 = f3.node_coordinates();
    let n3 = f3.values.len();
    assert_eq!(c3.len(), 3 * n3);
    for i in 0..n3 {
        let want = 100.0 * c3[i] + 10.0 * c3[n3 + i] + c3[2 * n3 + i];
        assert_eq!(f3.values[i].to_bits(), want.to_bits());
    }
    let f1 = GridFn1D::from_fn(g, |x| x);
    assert_eq!(f1.node_coordinates(), f1.values);
}

#[test]
fn species_state_ops() {
    let g = Grid1D::new(0.0, 1.0, 4).unwrap();
    let mut a = Species::new(vec![
        GridFn1D::from_fn(g, |x| x),
        GridFn1D::from_fn(g, |x| -2.0 * x),
    ]);
    let b = a.clone();
    assert_eq!(State::<f64>::len(&a), 8);
    assert!((a.norm_sup() - 2.0).abs() < 1e-15);
    a.axpy_into(1.0, &b);
    assert!((a.norm_sup() - 4.0).abs() < 1e-15);
    a.scale_into(0.5);
    assert!(same(&a, &b));
    a.zero_into();
    assert_eq!(a.norm_sup().to_bits(), 0.0_f64.to_bits());
    a.copy_from(&b);
    assert!(same(&a, &b));
}

fn same(a: &Species<GridFn1D<f64>>, b: &Species<GridFn1D<f64>>) -> bool {
    a.fields
        .iter()
        .zip(&b.fields)
        .all(|(x, y)| x.values == y.values)
}

fn heat(grid: Grid1D<f64>) -> SpeciesEngine<'static, f64, GridFn1D<f64>> {
    Box::new(DiffusionChernoff::new_const_a(0.5, 0.5, grid))
}

#[test]
fn rejects_bad_configurations() {
    let g = Grid1D::new(0.0, 1.0, 9).unwrap();
    // Two engines for one species.
    assert!(ReactionDiffusion::new(vec![heat(g), heat(g)], Box::new(Nagumo { a: 0.3 })).is_err());
    assert!(
        ReactionDiffusion::<f64, GridFn1D<f64>>::new(Vec::new(), Box::new(Nagumo { a: 0.3 }))
            .is_err()
    );
    let rd = ReactionDiffusion::new(
        vec![heat(g), heat(g)],
        Box::new(GrayScott {
            feed: 0.04,
            kill: 0.06,
        }),
    )
    .unwrap();
    let mut scratch = ScratchPool::new();
    // Wrong species count and mismatched node counts.
    let one = Species::new(vec![GridFn1D::from_fn(g, |_| 1.0)]);
    let mut out = one.clone();
    assert!(rd
        .evolve_into(0.0, 1.0, 2, &one, &mut out, &mut scratch)
        .is_err());
    let g2 = Grid1D::new(0.0, 1.0, 5).unwrap();
    let mixed = Species::new(vec![
        GridFn1D::from_fn(g, |_| 1.0),
        GridFn1D::from_fn(g2, |_| 0.0),
    ]);
    let mut out = mixed.clone();
    assert!(rd
        .evolve_into(0.0, 1.0, 2, &mixed, &mut out, &mut scratch)
        .is_err());
    let ok = Species::new(vec![
        GridFn1D::from_fn(g, |_| 1.0),
        GridFn1D::from_fn(g, |_| 0.0),
    ]);
    let mut out = ok.clone();
    assert!(rd
        .evolve_into(0.0, f64::NAN, 2, &ok, &mut out, &mut scratch)
        .is_err());
    assert!(rd
        .evolve_into(0.0, 1.0, 0, &ok, &mut out, &mut scratch)
        .is_err());
    assert!(rd
        .evolve_into(0.0, 1.0, 2, &ok, &mut out, &mut scratch)
        .is_ok());
}

/// A blow-up (`u′ = u²` from `u = 10` over `t = 1`) is an error, not `inf`.
#[test]
fn non_finite_reaction_is_an_error() {
    let g = Grid1D::new(0.0, 1.0, 9).unwrap();
    let blow = FnReaction::new(1, |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        du[0] = u[0] * u[0] * u[0];
    });
    let rd = ReactionDiffusion::new(vec![heat(g)], Box::new(blow)).unwrap();
    let u0 = Species::new(vec![GridFn1D::from_fn(g, |_| 1e3)]);
    let mut out = u0.clone();
    let r = rd.evolve_into(0.0, 1.0, 2, &u0, &mut out, &mut ScratchPool::new());
    assert!(
        matches!(r, Err(SemiflowError::DomainViolation { .. })),
        "{r:?}"
    );
}

/// `step_into` equals a one-step `evolve_into` (same splitting, nothing merged).
#[test]
fn single_step_matches_evolve() {
    let g = Grid1D::new(-3.0, 3.0, 31).unwrap();
    let rd = ReactionDiffusion::new(vec![heat(g)], Box::new(Nagumo { a: 0.3 }))
        .unwrap()
        .with_reaction_substeps(2);
    assert_eq!(rd.order(), 2);
    let u0 = Species::new(vec![GridFn1D::from_fn(g, |x| 1.0 / (1.0 + x.exp()))]);
    let mut scratch = ScratchPool::new();
    let mut a = u0.clone();
    rd.step_into(0.25, 0.1, &mut a, &mut scratch).unwrap();
    let mut b = u0.clone();
    rd.evolve_into(0.25, 0.1, 1, &u0, &mut b, &mut scratch)
        .unwrap();
    for (x, y) in a.fields[0].values.iter().zip(&b.fields[0].values) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}
