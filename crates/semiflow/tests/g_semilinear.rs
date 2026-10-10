//! Semilinear reaction–diffusion gates (`RELEASE_BLOCKING`, ADR-0208).
//!
//! * `G_SEMILIN_LINEAR_SYSTEM_EXACT`: equal diffusivities and a constant
//!   coupling matrix make diffusion and reaction commute, so the split result is
//!   `e^{tC}` (pointwise) after the pure diffusion run, to rounding; the RK4 path
//!   (no exact flow) agrees to `O(τ⁴)`.
//! * `G_SEMILIN_SOURCE_MMS`: `u_t = a u_xx + f(t, x)` with the manufactured
//!   solution `e^{−t}e^{−x²}`: order 2 (Strang), order 3 with two Richardson
//!   levels.
//! * `G_SEMILIN_FISHER_WAVE`: Fisher–KPP against the Ablowitz–Zeppetella exact
//!   travelling wave (exact logistic flow).
//! * `G_SEMILIN_NAGUMO_WAVE`: Nagumo against its exact front (RK4 reaction).
//! * `G_SEMILIN_GRAY_SCOTT_2D`: Gray–Scott on a periodic 2-D grid through
//!   `Strang2D` engines: self-convergence of order 2.

#![allow(clippy::cast_precision_loss)]

use semiflow::{
    BoundaryPolicy, DiffusionChernoff, Evolver, FisherKpp, FnReaction, GrayScott, Grid1D, Grid2D,
    GridFn1D, GridFn2D, Kinetics, LinearReaction, Nagumo, ReactionDiffusion, ScratchPool, Species,
    SpeciesEngine, Strang2D,
};

fn sup_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) })
}

fn heat(a: f64, grid: Grid1D<f64>) -> SpeciesEngine<'static, f64, GridFn1D<f64>> {
    Box::new(DiffusionChernoff::new_const_a(a, a, grid))
}

fn run_1d(
    engines: Vec<SpeciesEngine<'static, f64, GridFn1D<f64>>>,
    reaction: Box<dyn Kinetics<f64>>,
    u0: &Species<GridFn1D<f64>>,
    t: f64,
    n: usize,
    levels: usize,
) -> Species<GridFn1D<f64>> {
    let rd = ReactionDiffusion::new(engines, reaction).expect("rd");
    let mut out = u0.clone();
    let mut scratch = ScratchPool::new();
    if levels == 1 {
        rd.evolve_into(0.0, t, n, u0, &mut out, &mut scratch)
            .expect("evolve");
    } else {
        semiflow::extrapolate_into(&rd, 0.0, t, n, levels, u0, &mut out, &mut scratch)
            .expect("extrapolate");
    }
    out
}

/// "Diffuse each species, then apply `e^{tC}` pointwise" (flat, species-major).
fn diffuse_then_react(
    u0: &Species<GridFn1D<f64>>,
    c: &LinearReaction<f64>,
    t: f64,
    n: usize,
) -> Vec<f64> {
    let grid = u0.fields[0].grid;
    let mut flat = Vec::new();
    for f in &u0.fields {
        let mut out = f.clone();
        Evolver::new(DiffusionChernoff::new_const_a(0.3, 0.3, grid), n)
            .expect("evolver")
            .evolve_into(t, f, &mut out, &mut ScratchPool::new())
            .expect("diffuse");
        flat.extend(out.values);
    }
    c.exact_flow(0.0, t, grid.n, &[], &mut flat)
        .expect("exact flow")
        .expect("ok");
    flat
}

#[test]
fn g_semilin_linear_system_exact() {
    let grid = Grid1D::new(-10.0, 10.0, 401).expect("grid");
    let c = vec![-0.5, 0.3, 0.2, -0.1];
    let u0 = Species::new(vec![
        GridFn1D::from_fn(grid, |x| (-x * x).exp()),
        GridFn1D::from_fn(grid, |x| 0.5 * (-(x - 1.0) * (x - 1.0)).exp()),
    ]);
    let (t, n) = (0.5, 10);
    let exact_c = LinearReaction::new(c.clone()).expect("C");
    let flat = diffuse_then_react(&u0, &exact_c, t, n);
    let split = run_1d(
        vec![heat(0.3, grid), heat(0.3, grid)],
        Box::new(exact_c.clone()),
        &u0,
        t,
        n,
        1,
    );
    let scale = flat.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let got: Vec<f64> = split.fields.iter().flat_map(|f| f.values.clone()).collect();
    let err = sup_diff(&got, &flat) / scale;
    // RK4 path: the same linear reaction without an exact flow.
    let rk = FnReaction::new(2, move |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        du[0] = c[0] * u[0] + c[1] * u[1];
        du[1] = c[2] * u[0] + c[3] * u[1];
    });
    let split_rk = run_1d(
        vec![heat(0.3, grid), heat(0.3, grid)],
        Box::new(rk),
        &u0,
        t,
        n,
        1,
    );
    let got_rk: Vec<f64> = split_rk
        .fields
        .iter()
        .flat_map(|f| f.values.clone())
        .collect();
    let err_rk = sup_diff(&got_rk, &flat) / scale;
    eprintln!("G_SEMILIN_LINEAR_SYSTEM_EXACT exact-flow rel={err:.3e} rk4 rel={err_rk:.3e}");
    assert!(err <= 1e-13, "commuting split not exact: {err:e}");
    assert!(err_rk <= 1e-8, "RK4 reaction path: {err_rk:e}");
}

/// MMS: `u = e^{−t}e^{−x²}` for `u_t = a u_xx + f`, `a = 0.5`.
fn mms_error(n: usize, levels: usize) -> f64 {
    const A: f64 = 0.5;
    let grid = Grid1D::new(-10.0, 10.0, 801).expect("grid");
    let source = FnReaction::new(1, |t: f64, x: &[f64], _u: &[f64], du: &mut [f64]| {
        let x = x[0];
        du[0] = -(4.0 * A * x * x - 2.0 * A + 1.0) * (-t).exp() * (-x * x).exp();
    });
    let u0 = Species::new(vec![GridFn1D::from_fn(grid, |x| (-x * x).exp())]);
    let out = run_1d(vec![heat(A, grid)], Box::new(source), &u0, 1.0, n, levels);
    let exact: Vec<f64> = (0..grid.n)
        .map(|i| (-1.0_f64).exp() * (-grid.x_at(i).powi(2)).exp())
        .collect();
    sup_diff(&out.fields[0].values, &exact)
}

#[test]
fn g_semilin_source_mms() {
    let mut fails = Vec::new();
    for (levels, min_slope) in [(1, 1.9), (2, 2.8)] {
        let (e1, e2, e3) = (
            mms_error(10, levels),
            mms_error(20, levels),
            mms_error(40, levels),
        );
        let (s1, s2) = ((e1 / e2).log2(), (e2 / e3).log2());
        eprintln!("G_SEMILIN_SOURCE_MMS L={levels}: err {e1:.3e} {e2:.3e} {e3:.3e} slopes {s1:.3} {s2:.3}");
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must fail
        if !(s1 >= min_slope && s2 >= min_slope) {
            fails.push(format!("L={levels}: slopes {s1:.3}, {s2:.3} < {min_slope}"));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}

/// Fisher–KPP `u_t = u_xx + u(1 − u)`: `u = (1 + e^{(x − ct)/√6})^{−2}`, `c = 5/√6`.
fn fisher_error(n: usize) -> f64 {
    let grid = Grid1D::new(-60.0, 60.0, 1201)
        .expect("grid")
        .with_boundary(BoundaryPolicy::Reflect);
    let c = 5.0 / 6f64.sqrt();
    let wave = move |t: f64, x: f64| (1.0 + ((x - c * t) / 6f64.sqrt()).exp()).powi(-2);
    let u0 = Species::new(vec![GridFn1D::from_fn(grid, |x| wave(0.0, x))]);
    let reaction = FisherKpp {
        rate: 1.0,
        capacity: 1.0,
    };
    let out = run_1d(vec![heat(1.0, grid)], Box::new(reaction), &u0, 2.0, n, 1);
    let exact: Vec<f64> = (0..grid.n).map(|i| wave(2.0, grid.x_at(i))).collect();
    sup_diff(&out.fields[0].values, &exact)
}

/// Nagumo `u_t = u_xx + u(1 − u)(u − a)`: `u = 1/(1 + e^{(x − ct)/√2})`, `c = √2(½ − a)`.
fn nagumo_error(n: usize) -> f64 {
    let a = 0.25;
    let grid = Grid1D::new(-40.0, 40.0, 801)
        .expect("grid")
        .with_boundary(BoundaryPolicy::Reflect);
    let c = 2f64.sqrt() * (0.5 - a);
    let front = move |t: f64, x: f64| 1.0 / (1.0 + ((x - c * t) / 2f64.sqrt()).exp());
    let u0 = Species::new(vec![GridFn1D::from_fn(grid, |x| front(0.0, x))]);
    let out = run_1d(
        vec![heat(1.0, grid)],
        Box::new(Nagumo { a }),
        &u0,
        4.0,
        n,
        1,
    );
    let exact: Vec<f64> = (0..grid.n).map(|i| front(4.0, grid.x_at(i))).collect();
    sup_diff(&out.fields[0].values, &exact)
}

#[test]
fn g_semilin_fronts() {
    let mut fails = Vec::new();
    for (name, f) in [
        ("FISHER", fisher_error as fn(usize) -> f64),
        ("NAGUMO", nagumo_error),
    ] {
        let (e1, e2, e3) = (f(10), f(20), f(40));
        let (s1, s2) = ((e1 / e2).log2(), (e2 / e3).log2());
        eprintln!("G_SEMILIN_{name}_WAVE err {e1:.3e} {e2:.3e} {e3:.3e} slopes {s1:.3} {s2:.3}");
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must fail
        if !(s1 >= 1.8 && s2 >= 1.8 && e3 <= 1e-3) {
            fails.push(format!("{name}: errors {e1:e} {e2:e} {e3:e}"));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}

/// Gray–Scott on the periodic unit square, `D_u = 2e−3`, `D_v = 1e−3`.
fn gray_scott(n: usize) -> Vec<f64> {
    let g1 = Grid1D::new(0.0, 1.0, 65)
        .expect("grid")
        .with_boundary(BoundaryPolicy::Periodic);
    let grid = Grid2D::new(g1, g1);
    let bump =
        |x: f64, y: f64| (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) / (2.0 * 0.15 * 0.15)).exp();
    let u0 = Species::new(vec![
        GridFn2D::from_fn(grid, |x, y| 1.0 - 0.5 * bump(x, y)),
        GridFn2D::from_fn(grid, |x, y| 0.25 * bump(x, y)),
    ]);
    let axis = |d: f64| DiffusionChernoff::new_const_a(d, d, g1);
    let engines: Vec<SpeciesEngine<'static, f64, GridFn2D<f64>>> = vec![
        Box::new(Strang2D::new(axis(2e-3), axis(2e-3))),
        Box::new(Strang2D::new(axis(1e-3), axis(1e-3))),
    ];
    let reaction = GrayScott {
        feed: 0.04,
        kill: 0.06,
    };
    let rd = ReactionDiffusion::new(engines, Box::new(reaction)).expect("rd");
    let mut out = u0.clone();
    rd.evolve_into(0.0, 10.0, n, &u0, &mut out, &mut ScratchPool::new())
        .expect("evolve");
    out.fields.iter().flat_map(|f| f.values.clone()).collect()
}

#[test]
fn g_semilin_gray_scott_2d() {
    let (a, b, c) = (gray_scott(10), gray_scott(20), gray_scott(40));
    let (d1, d2) = (sup_diff(&a, &b), sup_diff(&b, &c));
    let slope = (d1 / d2).log2();
    let min = c.iter().fold(f64::INFINITY, |m, &v| m.min(v));
    eprintln!("G_SEMILIN_GRAY_SCOTT_2D |U10−U20|={d1:.3e} |U20−U40|={d2:.3e} slope {slope:.3} min {min:.3e}");
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must fail
    let ok = slope >= 1.8 && min >= -1e-10 && d1 > 1e-8;
    assert!(ok, "slope {slope}, min {min}, d1 {d1}");
}
