//! `G_STEP_PLAN_BIT_EQUAL` (`RELEASE_BLOCKING`, ADR-0215): τ-keyed step plans are
//! bit-identical to the unplanned step.
//!
//! The reference wraps each engine in `NoPlan` (it forwards `apply_into` only, so
//! every consumer takes today's path). Planned and unplanned results must agree to
//! 0 ULP for `DiffusionChernoff` with constant `a`, fn-pointer `a ≡ ½` and a
//! variable closure `a`, and `ShiftChernoff1D` (drift and reaction), × every interpolant (septic, octonic, Catmull–Rom,
//! Chebyshev `M = 16`, linear) × every `BoundaryPolicy` × `n ∈ {8, 37, 128}` ×
//! `τ ∈ {1e−4, 5}` (the large step shifts samples past the ghost-table margin), through
//! `Evolver::evolve_into`, `ChernoffSemigroup::evolve`, `Strang2D` and `Strang3D`
//! (serial and, with the `parallel` feature, the threaded sweeps), plus the
//! fallbacks — a plan presented with another grid, `τ` or engine is not used —
//! and error equivalence (invalid `a`, invalid `τ`).

#![allow(clippy::cast_precision_loss)]

use semiflow::{
    BoundaryPolicy, ChernoffFunction, ChernoffSemigroup, DiffusionChernoff, Evolver, Grid1D,
    Grid2D, Grid3D, GridFn1D, GridFn2D, GridFn3D, Growth, InterpKind, OobPolicy, ScratchPool,
    SemiflowError, ShiftChernoff1D, Strang2D, Strang3D,
};

/// Forwards `apply_into` only: the unplanned reference path.
#[derive(Clone)]
struct NoPlan<C>(C);

impl<C: ChernoffFunction<f64>> ChernoffFunction<f64> for NoPlan<C> {
    type S = C::S;
    fn apply_into(
        &self,
        tau: f64,
        src: &C::S,
        dst: &mut C::S,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        self.0.apply_into(tau, src, dst, scratch)
    }
    fn order(&self) -> u32 {
        self.0.order()
    }
    fn growth(&self) -> Growth<f64> {
        self.0.growth()
    }
}

fn a_half(_: f64) -> f64 {
    0.5
}
fn zero(_: f64) -> f64 {
    0.0
}

/// The three storage variants of `DiffusionChernoff` on `grid`.
fn engines(grid: Grid1D<f64>) -> [(&'static str, DiffusionChernoff<f64>); 3] {
    [
        ("const", DiffusionChernoff::new_const_a(0.35, 0.35, grid)),
        (
            "fn-ptr",
            DiffusionChernoff::new(a_half, zero, zero, 0.5, grid),
        ),
        (
            "closure",
            DiffusionChernoff::with_closure(
                |x: f64| 0.5 + 0.2 * (0.5 * x).sin(),
                |x: f64| 0.1 * (0.5 * x).cos(),
                |x: f64| -0.05 * (0.5 * x).sin(),
                0.7,
                grid,
            ),
        ),
    ]
}

fn drift(x: f64) -> f64 {
    0.3 * x
}
fn sink(x: f64) -> f64 {
    -0.2 - 0.1 * x * x
}

/// `ShiftChernoff1D` with `a ≡ ½`, drift `0.3x`, reaction `−0.2 − 0.1x²`.
fn shift(grid: Grid1D<f64>) -> ShiftChernoff1D<f64> {
    ShiftChernoff1D::new(a_half, drift, sink, 0.3, grid)
}

const POLICIES: [BoundaryPolicy<f64>; 8] = [
    BoundaryPolicy::Reflect,
    BoundaryPolicy::ZeroExtend,
    BoundaryPolicy::Periodic,
    BoundaryPolicy::LinearExtrapolate,
    BoundaryPolicy::Dirichlet { value: 0.3 },
    BoundaryPolicy::Neumann,
    BoundaryPolicy::Robin {
        alpha: 1.0,
        beta: 2.0,
    },
    BoundaryPolicy::OddReflect,
];

fn interps() -> [InterpKind; 5] {
    [
        InterpKind::SepticHermite,
        InterpKind::OctonicHermite,
        InterpKind::CubicHermite,
        InterpKind::ChebyshevSpectralWithBC {
            m: 16,
            oob_policy: OobPolicy::Inherit,
        },
        InterpKind::Linear,
    ]
}

/// Bitwise equality of two results (values, or the same error variant).
fn same(a: &Result<Vec<f64>, SemiflowError>, b: &Result<Vec<f64>, SemiflowError>) -> bool {
    match (a, b) {
        (Ok(x), Ok(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (Err(x), Err(y)) => core::mem::discriminant(x) == core::mem::discriminant(y),
        _ => false,
    }
}

fn bump(x: f64) -> f64 {
    (-2.0 * x * x).exp() + 0.1 * x
}

/// `Evolver` (3 steps, caller pool used twice) and `ChernoffSemigroup` (2 steps).
fn evolve_1d<C: ChernoffFunction<f64, S = GridFn1D<f64>>>(
    e: C,
    tau: f64,
    u0: &GridFn1D<f64>,
) -> Result<Vec<f64>, SemiflowError> {
    let ev = Evolver::new(e, 3)?;
    let (mut dst, mut pool) = (u0.clone(), ScratchPool::new());
    ev.evolve_into(3.0 * tau, u0, &mut dst, &mut pool)?;
    ev.evolve_into(3.0 * tau, &dst.clone(), &mut dst, &mut pool)?;
    let semi = ChernoffSemigroup::new(ev.func, 2)?;
    let out = semi.evolve(2.0 * tau, &dst)?;
    Ok(out.values)
}

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
    for interp in interps() {
        strang_pair(BoundaryPolicy::Periodic, interp, (64, 32), 1e-3);
    }
}

#[test]
fn g_step_plan_fallbacks_and_errors() {
    let grid = Grid1D::new(-1.0, 1.0, 37).unwrap();
    let other = grid.with_boundary(BoundaryPolicy::Periodic);
    let [(_, e), (_, f), _] = engines(grid);
    let u0 = GridFn1D::from_fn(grid, bump);
    let (data_len, work_len) = e.plan_len(0.01, &u0).expect("plan");
    let mut data = vec![0.0; data_len];
    e.fill_plan(0.01, &u0, &mut data).unwrap();
    let mut work = vec![0.0; work_len];
    let mut pool = ScratchPool::new();
    let apply =
        |c: &DiffusionChernoff<f64>, tau: f64, u: &GridFn1D<f64>, pool: &mut ScratchPool<f64>| {
            let mut dst = u.clone();
            c.apply_into(tau, u, &mut dst, pool).map(|()| dst.values)
        };
    let planned = |c: &DiffusionChernoff<f64>,
                   tau: f64,
                   u: &GridFn1D<f64>,
                   work: &mut [f64],
                   pool: &mut ScratchPool<f64>| {
        let mut dst = u.clone();
        c.apply_planned_into(tau, &data, work, u, &mut dst, pool)
            .map(|()| dst.values)
    };
    let u_other = GridFn1D::from_fn(other, bump);
    for (who, c, tau, u) in [
        ("same", &e, 0.01, &u0),
        ("tau", &e, 0.02, &u0),
        ("grid", &e, 0.01, &u_other),
        ("engine", &f, 0.01, &u0),
    ] {
        let got = planned(c, tau, u, &mut work, &mut pool);
        assert!(same(&got, &apply(c, tau, u, &mut pool)), "{who}");
    }
    // Errors: the plan reports what apply_into reports.
    let bad = DiffusionChernoff::new(|x: f64| x, zero, zero, 1.0, grid);
    assert!(bad
        .fill_plan(
            0.01,
            &u0,
            &mut vec![0.0; bad.plan_len(0.01, &u0).unwrap().0]
        )
        .is_err());
    assert!(bad
        .apply_into(0.01, &u0, &mut u0.clone(), &mut pool)
        .is_err());
    assert!(e.fill_plan(-1.0, &u0, &mut data).is_err());
    let semi = ChernoffSemigroup::new(bad, 2).unwrap();
    assert!(matches!(
        semi.evolve(0.02, &u0),
        Err(SemiflowError::DomainViolation { .. })
    ));
}
