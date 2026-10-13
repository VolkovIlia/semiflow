//! `G_STEP_PLAN_USED` (`RELEASE_BLOCKING`, ADR-0215): the planned branch actually
//! runs. `G_STEP_PLAN_BIT_EQUAL` compares against `apply_into`, which is also every
//! plan's fallback, so it cannot see a plan that is silently never used.
//!
//! * Producers: perturbing the last stored weight of a filled plan changes the
//!   planned result (`DiffusionChernoff` const / fn-pointer / closure,
//!   `ShiftChernoff1D`), while the intact plan reproduces `apply_into` bitwise.
//! * Consumers: the coefficient function `a` is called once per plan — the call
//!   count of `Evolver`, `ChernoffSemigroup` and a `ReactionDiffusion` evolution
//!   does not grow with the step count, and a `Strang2D`/`Strang3D` sweep's does not
//!   grow with the number of pencils (serial `16²`/`8³`, and `64²`/`32³`, threaded
//!   with the `parallel` feature).

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use semiflow::{
    ApplyChernoffExt, ChernoffFunction, ChernoffSemigroup, DiffusionChernoff, Evolver, FnReaction,
    Grid1D, Grid2D, Grid3D, GridFn1D, GridFn2D, GridFn3D, ReactionDiffusion, ScratchPool,
    ShiftChernoff1D, Species, SpeciesEngine, Strang2D, Strang3D,
};

static A_CALLS: AtomicUsize = AtomicUsize::new(0);

fn a_counted(x: f64) -> f64 {
    A_CALLS.fetch_add(1, Relaxed);
    0.5 + 0.1 * x * x
}
fn a_prime(x: f64) -> f64 {
    0.2 * x
}
fn a_second(_: f64) -> f64 {
    0.2
}
fn zero(_: f64) -> f64 {
    0.0
}
fn half(_: f64) -> f64 {
    0.5
}

fn bump(x: f64) -> f64 {
    (-2.0 * x * x).exp() + 0.1 * x
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Intact plan == `apply_into`; a perturbed last weight != `apply_into`.
fn plan_is_used<C: ChernoffFunction<f64, S = GridFn1D<f64>>>(name: &str, c: &C) {
    let g = Grid1D::new(-1.0, 1.0, 37).unwrap();
    let u = GridFn1D::from_fn(g, bump);
    let (d, w) = c.plan_len(0.01, &u).expect("plan");
    let mut data = vec![0.0; d];
    c.fill_plan(0.01, &u, &mut data).unwrap();
    let (mut pool, mut work, mut dst) = (ScratchPool::new(), vec![0.0; w], u.clone());
    c.apply_into(0.01, &u, &mut dst, &mut pool).unwrap();
    let plain = bits(&dst.values);
    c.apply_planned_into(0.01, &data, &mut work, &u, &mut dst, &mut pool)
        .unwrap();
    assert_eq!(bits(&dst.values), plain, "{name}: intact plan");
    *data.last_mut().unwrap() += 1.0;
    c.apply_planned_into(0.01, &data, &mut work, &u, &mut dst, &mut pool)
        .unwrap();
    assert_ne!(bits(&dst.values), plain, "{name}: the plan is never used");
}

fn engine(g: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new(a_counted, a_prime, a_second, 0.7, g)
}

/// `a` calls made by `run`.
fn calls(run: impl FnOnce()) -> usize {
    let before = A_CALLS.load(Relaxed);
    run();
    A_CALLS.load(Relaxed) - before
}

fn evolver_calls(n: usize) -> usize {
    let g = Grid1D::new(-1.0, 1.0, 37).unwrap();
    let u = GridFn1D::from_fn(g, bump);
    let ev = Evolver::new(engine(g), n).unwrap();
    let semi = ChernoffSemigroup::new(engine(g), n).unwrap();
    calls(|| {
        let (mut dst, mut pool) = (u.clone(), ScratchPool::new());
        ev.evolve_into(0.01, &u, &mut dst, &mut pool).unwrap();
        let _ = semi.evolve(0.01, &u).unwrap();
    })
}

fn rd_calls(n: usize) -> usize {
    let g = Grid1D::new(-1.0, 1.0, 37).unwrap();
    let engines: Vec<SpeciesEngine<'static, f64, GridFn1D<f64>>> = vec![Box::new(engine(g))];
    let kin = FnReaction::new(1, |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        du[0] = -u[0];
    });
    let rd = ReactionDiffusion::new(engines, Box::new(kin)).unwrap();
    let src = Species::new(vec![GridFn1D::from_fn(g, bump)]);
    calls(|| {
        let mut dst = src.clone();
        rd.evolve_into(0.0, 0.01, n, &src, &mut dst, &mut ScratchPool::new())
            .unwrap();
    })
}

/// One Strang step with the counted engine on x and `ny` rows / `m` z-pencils.
fn strang_calls((nx, ny): (usize, usize), m: usize) -> (usize, usize) {
    let gx = Grid1D::new(-1.0, 1.0, nx).unwrap();
    let gy = Grid1D::new(-1.0, 1.0, ny).unwrap();
    let plain = |g| DiffusionChernoff::new(half, zero, zero, 0.5, g);
    let s2 = Strang2D::new(engine(gx), plain(gy));
    let f2 = GridFn2D::from_fn(Grid2D::new(gx, gy), |x, y| bump(x) * bump(y));
    let c2 = calls(|| {
        let _ = s2.apply_chernoff(0.01, &f2).unwrap();
    });
    let gm = Grid1D::new(-1.0, 1.0, m).unwrap();
    let s3 = Strang3D::new(plain(gm), plain(gm), engine(gm));
    let f3 = GridFn3D::from_fn(Grid3D::new(gm, gm, gm).unwrap(), |x, y, z| {
        bump(x) * bump(y) * bump(z)
    });
    let c3 = calls(|| {
        let _ = s3.apply_chernoff(0.01, &f3).unwrap();
    });
    (c2, c3)
}

#[test]
fn g_step_plan_used() {
    let g = Grid1D::new(-1.0, 1.0, 37).unwrap();
    plan_is_used("const", &DiffusionChernoff::new_const_a(0.35, 0.35, g));
    // Variable `a` with a′ ≠ 0: the last stored weight (a ζ sample) matters.
    plan_is_used("fn-ptr", &engine(g));
    let closure = DiffusionChernoff::with_closure(|x: f64| 0.5 + 0.1 * x, |_| 0.1, |_| 0.0, 0.7, g);
    plan_is_used("closure", &closure);
    plan_is_used(
        "shift",
        &ShiftChernoff1D::new(half, |x| 0.3 * x, |x| -0.2 * x * x, 0.2, g),
    );
    // All counts in one test: the counter is shared.
    let (e2, e50) = (evolver_calls(2), evolver_calls(50));
    let (r2, r50) = (rd_calls(2), rd_calls(50));
    let (s_small, s_big) = (strang_calls((16, 16), 8), strang_calls((16, 48), 12));
    let (t_small, t_big) = (strang_calls((64, 64), 32), strang_calls((64, 128), 40));
    eprintln!("G_STEP_PLAN_USED a-calls: evolver {e2}/{e50}, rd {r2}/{r50}, strang serial {s_small:?}/{s_big:?}, large {t_small:?}/{t_big:?}");
    assert_eq!(e2, e50, "Evolver/ChernoffSemigroup: one plan per call");
    assert_eq!(r2, r50, "ReactionDiffusion: one plan per evolution");
    // Strang2D: x-sweeps call a once per plan (two x half-sweeps), whatever ny.
    assert_eq!(s_small.0, s_big.0, "Strang2D serial");
    assert_eq!(
        t_small.0, t_big.0,
        "Strang2D 64² (threaded with `parallel`)"
    );
    // Strang3D: the z engine's calls grow with nz (its own nodes), not with nx·ny.
    assert_eq!(
        s_small.1 * 12,
        s_big.1 * 8,
        "Strang3D serial: one plan per z sweep"
    );
    assert_eq!(
        t_small.1 * 40,
        t_big.1 * 32,
        "Strang3D 32³ (threaded with `parallel`)"
    );
}
