//! `G_STEP_PLAN_BIT_EQUAL` (`RELEASE_BLOCKING`, ADR-0215), consumers: the plans
//! built by Richardson levels (`Evolver::evolve_extrapolated_into`, base
//! `n ∈ {1, 2}`, levels 1–4) and by `ReactionDiffusion` (one plan per species, with
//! and without cross-diffusion, plain and extrapolated) agree to 0 ULP with the
//! `NoPlan` reference for every `DiffusionChernoff` storage × reflect, periodic,
//! Neumann × septic, Catmull–Rom; with the `parallel` feature the threaded 1-D plan
//! fill and planned apply (4 threads forced, `n = 8192`) do too, including the
//! error variant.

mod step_plan_common;

use semiflow::{
    extrapolate_into, BoundaryPolicy, ChernoffFunction, CrossDiffusion, DiffusionChernoff, Evolver,
    FnReaction, Grid1D, GridFn1D, InterpKind, ReactionDiffusion, ScratchPool, SemiflowError,
    Species, SpeciesEngine,
};
use step_plan_common::{bump, engines, same, NoPlan};
#[cfg(feature = "parallel")]
use step_plan_common::{evolve_1d, shift, zero};

/// Richardson levels over `Evolver` (base `n ∈ {1, 2}`, levels 1–4).
fn richardson<C: ChernoffFunction<f64, S = GridFn1D<f64>>>(
    e: C,
    n: usize,
    levels: usize,
    u0: &GridFn1D<f64>,
) -> Result<Vec<f64>, SemiflowError> {
    let (mut out, mut pool) = (u0.clone(), ScratchPool::new());
    Evolver::new(e, n)?.evolve_extrapolated_into(0.5, levels, u0, &mut out, &mut pool)?;
    Ok(out.values)
}

/// A two-species Gray–Scott-like system on `engines` (cross-diffusion when `cd`).
fn rd_run(
    engines: Vec<SpeciesEngine<'static, f64, GridFn1D<f64>>>,
    cd: Option<CrossDiffusion<f64>>,
    u0: &[GridFn1D<f64>],
    (n, levels): (usize, usize),
) -> Result<Vec<f64>, SemiflowError> {
    let kin = FnReaction::new(2, |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        let r = u[0] * u[1] * u[1];
        du[0] = -r + 0.04 * (1.0 - u[0]);
        du[1] = r - 0.1 * u[1];
    });
    let mut rd = ReactionDiffusion::new(engines, Box::new(kin))?;
    if let Some(cd) = cd {
        rd = rd.with_cross_diffusion(cd)?;
    }
    let src = Species::new(u0.to_vec());
    let mut dst = src.clone();
    let mut pool = ScratchPool::new();
    if levels <= 1 {
        rd.evolve_into(0.0, 0.3, n, &src, &mut dst, &mut pool)?;
    } else {
        extrapolate_into(&rd, 0.0, 0.3, n, levels, &src, &mut dst, &mut pool)?;
    }
    Ok(dst.fields.into_iter().flat_map(|f| f.values).collect())
}

/// Mode engines for `D` (or the plain closure/fn-pointer pair), planned or not.
fn rd_engines(
    g: Grid1D<f64>,
    cd: Option<&CrossDiffusion<f64>>,
    plain: bool,
) -> Vec<SpeciesEngine<'static, f64, GridFn1D<f64>>> {
    let boxed = |e: DiffusionChernoff<f64>| -> SpeciesEngine<'static, f64, GridFn1D<f64>> {
        if plain {
            Box::new(NoPlan(e))
        } else {
            Box::new(e)
        }
    };
    if let Some(cd) = cd {
        cd.eigenvalues()
            .iter()
            .map(|&l| boxed(DiffusionChernoff::new_const_a(l, l, g)))
            .collect()
    } else {
        let [_, (_, fp), (_, cl)] = engines(g);
        vec![boxed(cl), boxed(fp)]
    }
}

/// Richardson levels and `ReactionDiffusion` (with and without cross-diffusion),
/// planned vs `NoPlan`, 0 ULP.
#[test]
fn g_step_plan_bit_equal_consumers() {
    for policy in [
        BoundaryPolicy::Reflect,
        BoundaryPolicy::Periodic,
        BoundaryPolicy::Neumann,
    ] {
        for interp in [InterpKind::SepticHermite, InterpKind::CubicHermite] {
            let g = Grid1D::new(-1.0, 1.0, 37)
                .unwrap()
                .with_boundary(policy)
                .with_interp(interp);
            let u0 = GridFn1D::from_fn(g, bump);
            for ((name, e), (_, r)) in engines(g).into_iter().zip(engines(g)) {
                for (n, levels) in [(1, 2), (2, 1), (2, 3), (1, 4)] {
                    let (p, q) = (
                        richardson(e.clone(), n, levels, &u0),
                        richardson(NoPlan(r.clone()), n, levels, &u0),
                    );
                    assert!(
                        same(&p, &q),
                        "richardson {name} {policy:?} {interp:?} n={n} L={levels}"
                    );
                }
            }
            let fields = [
                GridFn1D::from_fn(g, |x| 1.0 - 0.5 * bump(x)),
                GridFn1D::from_fn(g, |x| 0.25 * bump(x)),
            ];
            let cd = CrossDiffusion::<f64>::new(&[0.3, 0.1, 0.05, 0.2]).unwrap();
            for cross in [None, Some(&cd)] {
                for job in [(2, 1), (5, 1), (2, 2), (3, 3)] {
                    let p = rd_run(rd_engines(g, cross, false), cross.cloned(), &fields, job);
                    let q = rd_run(rd_engines(g, cross, true), cross.cloned(), &fields, job);
                    assert!(
                        same(&p, &q),
                        "reaction-diffusion cross={} {policy:?} {interp:?} {job:?}",
                        cross.is_some()
                    );
                }
            }
        }
    }
}

/// The threaded 1-D plan fill and planned apply (`parallel` feature, 4 threads
/// forced, `n = 8192` above the per-thread minimum) vs `NoPlan`, 0 ULP.
#[cfg(feature = "parallel")]
#[test]
fn g_step_plan_bit_equal_threaded_1d() {
    semiflow::parallel1d::FORCE_THREADS_1D.with(|c| c.set(Some(4)));
    for policy in [
        BoundaryPolicy::Reflect,
        BoundaryPolicy::Periodic,
        BoundaryPolicy::Dirichlet { value: 0.3 },
    ] {
        for interp in [InterpKind::SepticHermite, InterpKind::CubicHermite] {
            let g = Grid1D::new(-4.0, 4.0, 8192)
                .unwrap()
                .with_boundary(policy)
                .with_interp(interp);
            let u0 = GridFn1D::from_fn(g, bump);
            for tau in [1e-3, 5.0] {
                for ((name, e), (_, r)) in engines(g).into_iter().zip(engines(g)) {
                    let (p, q) = (evolve_1d(e, tau, &u0), evolve_1d(NoPlan(r), tau, &u0));
                    assert!(
                        same(&p, &q),
                        "threaded {name} {policy:?} {interp:?} tau={tau}"
                    );
                }
                let (p, q) = (
                    evolve_1d(shift(g), tau, &u0),
                    evolve_1d(NoPlan(shift(g)), tau, &u0),
                );
                assert!(
                    same(&p, &q),
                    "threaded shift {policy:?} {interp:?} tau={tau}"
                );
            }
        }
    }
    let g = Grid1D::new(-1.0, 1.0, 8192).unwrap();
    let u0 = GridFn1D::from_fn(g, bump);
    let bad = DiffusionChernoff::new(|x: f64| -x, zero, zero, 1.0, g);
    let (p, q) = (
        evolve_1d(bad.clone(), 0.01, &u0),
        evolve_1d(NoPlan(bad), 0.01, &u0),
    );
    assert!(same(&p, &q) && p.is_err(), "threaded error variant");
    semiflow::parallel1d::FORCE_THREADS_1D.with(|c| c.set(None));
}
