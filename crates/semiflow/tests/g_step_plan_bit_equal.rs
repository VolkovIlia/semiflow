//! `G_STEP_PLAN_BIT_EQUAL` (`RELEASE_BLOCKING`, ADR-0215): τ-keyed step plans are
//! bit-identical to the unplanned step.
//!
//! The reference wraps each engine in `NoPlan` (it forwards `apply_into` only, so
//! every consumer takes today's path). Planned and unplanned results agree to
//! 0 ULP for `DiffusionChernoff` (constant `a`, fn-pointer `a ≡ ½`, variable
//! closure `a`) and `ShiftChernoff1D` (drift, reaction) × every interpolant
//! (septic, octonic, Catmull–Rom, Chebyshev `M = 16`, linear) × every
//! `BoundaryPolicy` × `n ∈ {8, 37, 128}` × `τ ∈ {1e−4, 5}` (the large step shifts
//! samples past the ghost-table margin) through `Evolver::evolve_into` and
//! `ChernoffSemigroup::evolve`; `Strang2D` (closure × fn-pointer) and `Strang3D`
//! (const, fn-pointer, closure) on `24²`/`8³` (serial sweeps) × septic, Catmull–Rom
//! × every policy × `τ ∈ {1e−4, 5}`, and on `64²`/`32³` (the threaded sweeps with the
//! `parallel` feature) × septic, Catmull–Rom, linear × every policy ×
//! `τ ∈ {1e−3, 5}` plus every interpolant on a periodic grid. Fallbacks: a plan
//! presented with another `τ`, another engine of the same kind, or a sampling grid
//! differing in one field is not used. Errors: the planned consumers report the
//! error `apply_into` reports first (`G_STEP_PLAN_USED` shows the plans run).
//! Richardson levels, `ReactionDiffusion` and the threaded 1-D fill are in
//! `g_step_plan_consumers.rs`.

#![allow(clippy::cast_precision_loss)]

mod step_plan_common;

use semiflow::{
    BoundaryPolicy, ChernoffFunction, ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D, Grid3D,
    GridFn1D, GridFn2D, GridFn3D, InterpKind, OobPolicy, ScratchPool, SemiflowError,
    ShiftChernoff1D, Strang2D, Strang3D,
};
use step_plan_common::{bump, engines, evolve_1d, interps, same, shift, zero, NoPlan, POLICIES};

#[test]
fn g_step_plan_bit_equal_1d() {
    let mut checked = 0;
    for n in [8, 37, 128] {
        for policy in POLICIES {
            for interp in interps() {
                let grid = Grid1D::new(-1.0, 1.0, n)
                    .unwrap()
                    .with_boundary(policy)
                    .with_interp(interp);
                let u0 = GridFn1D::from_fn(grid, bump);
                for tau in [1e-4, 5.0] {
                    for ((name, e), (_, r)) in engines(grid).into_iter().zip(engines(grid)) {
                        let planned = evolve_1d(e, tau, &u0);
                        let reference = evolve_1d(NoPlan(r), tau, &u0);
                        assert!(
                            same(&planned, &reference),
                            "{name} n={n} {policy:?} {interp:?} tau={tau}: {planned:?} vs {reference:?}"
                        );
                        checked += 1;
                    }
                    let planned = evolve_1d(shift(grid), tau, &u0);
                    let reference = evolve_1d(NoPlan(shift(grid)), tau, &u0);
                    assert!(
                        same(&planned, &reference),
                        "shift n={n} {policy:?} {interp:?} tau={tau}"
                    );
                    checked += 1;
                }
            }
        }
    }
    eprintln!("G_STEP_PLAN_BIT_EQUAL 1-D: {checked} configurations, 0 ULP");
}

/// 2-D (`n×n`) and 3-D (`m³`) Strang sweeps, planned vs `NoPlan`, 2 steps each.
fn strang_pair(policy: BoundaryPolicy<f64>, interp: InterpKind, (n, m): (usize, usize), tau: f64) {
    let ax = |k: usize| {
        Grid1D::new(-1.0, 1.0, k)
            .unwrap()
            .with_boundary(policy)
            .with_interp(interp)
    };
    let g2 = Grid2D::new(ax(n), ax(n));
    let f2 = GridFn2D::from_fn(g2, |x, y| bump(x) * bump(0.7 * y));
    let [(name, ex), (_, ey)] = [engines(g2.x)[2].clone(), engines(g2.y)[1].clone()];
    let run2 = |s: &dyn Fn() -> Result<GridFn2D<f64>, SemiflowError>| s().map(|f| f.values);
    let planned = run2(&|| {
        ChernoffSemigroup::new(Strang2D::new(ex.clone(), ey.clone()), 2)?.evolve(2.0 * tau, &f2)
    });
    let reference = run2(&|| {
        ChernoffSemigroup::new(Strang2D::new(NoPlan(ex.clone()), NoPlan(ey.clone())), 2)?
            .evolve(2.0 * tau, &f2)
    });
    assert!(
        same(&planned, &reference),
        "2-D {name} {policy:?} {interp:?} tau={tau}"
    );
    let g3 = Grid3D::new(ax(m), ax(m), ax(m)).unwrap();
    let f3 = GridFn3D::from_fn(g3, |x, y, z| bump(x) * bump(y) * bump(0.5 * z));
    let [(_, e1), (_, e2), (_, e3)] = engines(g3.x);
    let run3 = |s: &dyn Fn() -> Result<GridFn3D<f64>, SemiflowError>| s().map(|f| f.values);
    let planned = run3(&|| {
        ChernoffSemigroup::new(Strang3D::new(e1.clone(), e2.clone(), e3.clone()), 2)?
            .evolve(2.0 * tau, &f3)
    });
    let reference = run3(&|| {
        let s = Strang3D::new(NoPlan(e1.clone()), NoPlan(e2.clone()), NoPlan(e3.clone()));
        ChernoffSemigroup::new(s, 2)?.evolve(2.0 * tau, &f3)
    });
    assert!(
        same(&planned, &reference),
        "3-D {policy:?} {interp:?} tau={tau}"
    );
}

#[test]
fn g_step_plan_bit_equal_strang() {
    for policy in POLICIES {
        for interp in [InterpKind::SepticHermite, InterpKind::CubicHermite] {
            for tau in [1e-4, 5.0] {
                strang_pair(policy, interp, (24, 8), tau);
            }
        }
    }
    // Grids large enough for the threaded sweeps (with the `parallel` feature).
    for policy in POLICIES {
        for interp in [
            InterpKind::SepticHermite,
            InterpKind::CubicHermite,
            InterpKind::Linear,
        ] {
            for tau in [1e-3, 5.0] {
                strang_pair(policy, interp, (64, 32), tau);
            }
        }
    }
    for interp in interps() {
        strang_pair(BoundaryPolicy::Periodic, interp, (64, 32), 1e-3);
    }
}

/// `apply_planned_into` with a plan of `c` (or the caller's `data`) vs `apply_into`.
fn planned_vs_plain(
    plan_of: &DiffusionChernoff<f64>,
    (tau_plan, u_plan): (f64, &GridFn1D<f64>),
    c: &DiffusionChernoff<f64>,
    (tau, u): (f64, &GridFn1D<f64>),
) -> (
    Result<Vec<f64>, SemiflowError>,
    Result<Vec<f64>, SemiflowError>,
) {
    let (d, _) = plan_of.plan_len(tau_plan, u_plan).expect("plan");
    let mut data = vec![0.0; d];
    plan_of.fill_plan(tau_plan, u_plan, &mut data).unwrap();
    let (mut pool, mut work) = (ScratchPool::new(), vec![0.0; 4 * 200]);
    let mut dst = u.clone();
    let planned = c
        .apply_planned_into(tau, &data, &mut work, u, &mut dst, &mut pool)
        .map(|()| dst.values.clone());
    let plain = c
        .apply_into(tau, u, &mut dst, &mut pool)
        .map(|()| dst.values.clone());
    (planned, plain)
}

/// A plan presented with another τ, sampling grid (each field alone) or engine
/// of the same kind is not used; in the τ and engine cases the stale plan's own
/// result differs, so the fallback is observable.
#[test]
fn g_step_plan_fallbacks() {
    let grid = Grid1D::new(-1.0, 1.0, 37).unwrap();
    let u0 = GridFn1D::from_fn(grid, bump);
    let [(_, e), (_, fp), (_, cl)] = engines(grid);
    let e2 = DiffusionChernoff::new_const_a(0.5, 0.5, grid);
    for (who, owner, c, tau) in [
        ("tau", &e, &e, 0.02),
        ("const engine", &e, &e2, 0.01),
        ("fn-ptr -> closure", &fp, &cl, 0.01),
        ("closure -> fn-ptr", &cl, &fp, 0.01),
    ] {
        let (planned, plain) = planned_vs_plain(owner, (0.01, &u0), c, (tau, &u0));
        assert!(same(&planned, &plain), "{who}");
        let (stale, _) = planned_vs_plain(owner, (0.01, &u0), owner, (0.01, &u0));
        assert!(
            !same(&stale, &plain),
            "{who}: a used stale plan would not be seen"
        );
    }
    let dir = |v: f64| grid.with_boundary(BoundaryPolicy::Dirichlet { value: v });
    for other in [
        Grid1D::new(-1.0, 1.2, 37).unwrap(),
        Grid1D::new(-0.9, 1.0, 37).unwrap(),
        Grid1D::new(-1.0, 1.0, 38).unwrap(),
        grid.with_interp(InterpKind::CubicHermite),
        dir(0.7),
    ] {
        let u = GridFn1D::from_fn(other, bump);
        let plan_grid = if other.boundary == dir(0.7).boundary {
            dir(0.3)
        } else {
            grid
        };
        let up = GridFn1D::from_fn(plan_grid, bump);
        let (planned, plain) = planned_vs_plain(&fp, (0.01, &up), &fp, (0.01, &u));
        assert!(same(&planned, &plain), "{other:?}");
    }
}

/// Errors: the planned consumers report what `apply_into` reports, including
/// which error comes first in node order (an unsupported interpolant sampled at
/// node 0 before an invalid `a` at a later node).
#[test]
fn g_step_plan_errors() {
    let grid = Grid1D::new(-1.0, 1.0, 37).unwrap();
    let u0 = GridFn1D::from_fn(grid, bump);
    let bad = DiffusionChernoff::new(|x: f64| -x, zero, zero, 1.0, grid);
    let (planned, plain) = (
        evolve_1d(bad.clone(), 0.01, &u0),
        evolve_1d(NoPlan(bad), 0.01, &u0),
    );
    assert_eq!(format!("{planned:?}"), format!("{plain:?}"));
    assert!(matches!(plain, Err(SemiflowError::DomainViolation { .. })));
    let cheb = InterpKind::ChebyshevSpectralWithBC {
        m: 10,
        oob_policy: OobPolicy::Inherit,
    };
    for interp in [InterpKind::Linear, cheb] {
        let g = Grid1D::new(-1.0, 1.0, 8).unwrap().with_interp(interp);
        let u = GridFn1D::from_fn(g, bump);
        let d = DiffusionChernoff::new(|x: f64| -x, zero, zero, 1.0, g);
        let s = ShiftChernoff1D::new(|x: f64| -x, zero, zero, 0.0, g);
        let pairs = [
            (
                evolve_1d(d.clone(), 0.01, &u),
                evolve_1d(NoPlan(d), 0.01, &u),
            ),
            (
                evolve_1d(s, 0.01, &u),
                evolve_1d(NoPlan(shift_neg(g)), 0.01, &u),
            ),
        ];
        for (planned, plain) in pairs {
            assert_eq!(format!("{planned:?}"), format!("{plain:?}"), "{interp:?}");
        }
    }
    let e = DiffusionChernoff::new_const_a(0.35, 0.35, grid);
    let mut data = vec![0.0; e.plan_len(0.01, &u0).unwrap().0];
    assert!(e.fill_plan(-1.0, &u0, &mut data).is_err());
}

/// `ShiftChernoff1D` with `a(x) = −x` (invalid from the grid's middle on).
fn shift_neg(g: Grid1D<f64>) -> ShiftChernoff1D<f64> {
    ShiftChernoff1D::new(|x: f64| -x, zero, zero, 0.0, g)
}
