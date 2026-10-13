//! Cross-diffusion gates (`RELEASE_BLOCKING`, ADR-0213, math §70).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use semiflow::{
    BoundaryPolicy, ChernoffFunction, CrossDiffusion, DiffusionChernoff, FnReaction, Graph,
    GraphKrylovChernoff, GraphSignal, Grid1D, Grid2D, GridFn1D, GridFn2D, Growth, KrylovPath,
    Laplacian, ReactionDiffusion, ScratchPool, SemiflowError, Species, SpeciesEngine, State,
    Strang2D,
};

/// `e^{−λτL}` from an engine for `e^{−τL}` (exact for a semigroup engine).
struct Scaled<C> {
    inner: C,
    lambda: f64,
}

impl<C: ChernoffFunction<f64>> ChernoffFunction<f64> for Scaled<C> {
    type S = C::S;
    fn apply_into(
        &self,
        tau: f64,
        src: &C::S,
        dst: &mut C::S,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        self.inner.apply_into(self.lambda * tau, src, dst, scratch)
    }
    fn order(&self) -> u32 {
        self.inner.order()
    }
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

/// The identity engine (a mode with `λ = 0`).
struct Identity<S>(std::marker::PhantomData<fn() -> S>);

impl<S: State<f64>> ChernoffFunction<f64> for Identity<S> {
    type S = S;
    fn apply_into(
        &self,
        _tau: f64,
        src: &S,
        dst: &mut S,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        dst.copy_from(src);
        Ok(())
    }
    fn order(&self) -> u32 {
        u32::MAX
    }
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

/// `e^{A}` of a small dense row-major matrix (Taylor, scaling and squaring).
fn expm(a: &[f64], k: usize) -> Vec<f64> {
    let norm = a.iter().fold(0.0_f64, |m, x| m.max(x.abs())) * k as f64;
    let s = norm.log2().ceil().max(0.0) as i32 + 1;
    let scaled: Vec<f64> = a.iter().map(|x| x / 2f64.powi(s)).collect();
    let mul = |x: &[f64], y: &[f64]| -> Vec<f64> {
        (0..k * k)
            .map(|e| (0..k).map(|l| x[e / k * k + l] * y[l * k + e % k]).sum())
            .collect()
    };
    let mut out: Vec<f64> = (0..k * k)
        .map(|e| f64::from(u8::from(e / k == e % k)))
        .collect();
    let mut term = out.clone();
    for j in 1..=30 {
        term = mul(&term, &scaled)
            .iter()
            .map(|x| x / f64::from(j))
            .collect();
        out.iter_mut().zip(&term).for_each(|(o, t)| *o += t);
    }
    for _ in 0..s {
        out = mul(&out, &out);
    }
    out
}

fn matvec(m: &[f64], c: &[f64]) -> Vec<f64> {
    let k = c.len();
    (0..k)
        .map(|i| (0..k).map(|l| m[i * k + l] * c[l]).sum())
        .collect()
}

/// The gate matrices: non-symmetric 2×2, singular (a non-diffusing species driven
/// by the other's Laplacian), non-normal upper-triangular 3×3.
const MATRICES: [&[f64]; 3] = [
    &[1.0, 0.5, 0.2, 0.6],
    &[1.0, 0.5, 0.0, 0.0],
    &[0.8, 0.3, 0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.3],
];

/// `C` of the linear reaction `u′ = Cu` (does not commute with any gate `D`).
fn reaction_matrix(k: usize) -> Vec<f64> {
    (0..k * k)
        .map(|e| {
            let (i, j) = (e / k, e % k);
            if i == j {
                -0.3 * (i as f64 + 1.0)
            } else {
                0.4 * (i as f64 - j as f64)
            }
        })
        .collect()
}

/// Cycle graph on `N` nodes, Fourier mode `sin(2πfj/N)`, eigenvalue `μ`.
struct Ring {
    graph: Arc<Graph<f64>>,
    lap: Arc<Laplacian<f64>>,
    mode: Vec<f64>,
    mu: f64,
}

fn ring(n: usize, freq: f64) -> Ring {
    let graph: Arc<Graph<f64>> = Arc::new(Graph::cycle(n));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&graph));
    let w = 2.0 * std::f64::consts::PI * freq / n as f64;
    Ring {
        mode: (0..n).map(|j| (w * j as f64).sin()).collect(),
        mu: 2.0 - 2.0 * w.cos(),
        graph,
        lap,
    }
}

/// `max_k ‖u_k − ref_k·mode‖∞ / max_k |ref_k|`.
fn rel_err(u: &Species<impl AsRef<[f64]>>, want: &[f64], mode: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let mut worst = 0.0_f64;
    for (f, &w) in u.fields.iter().zip(want) {
        for (v, m) in f.as_ref().iter().zip(mode) {
            worst = worst.max((v - w * m).abs());
        }
    }
    worst / scale
}

struct Vals<'a>(&'a [f64]);
impl AsRef<[f64]> for Vals<'_> {
    fn as_ref(&self) -> &[f64] {
        self.0
    }
}

/// Evolve `c·mode` on the ring under `D` (and `u′ = Cu` when `react`).
fn ring_run(r: &Ring, d: &[f64], c: &[f64], (t, n, react): (f64, usize, bool)) -> Vec<Vec<f64>> {
    let k = c.len();
    let cd = CrossDiffusion::<f64>::new(d).unwrap();
    let engines: Vec<SpeciesEngine<'static, f64, GraphSignal<f64>>> = cd
        .eigenvalues()
        .iter()
        .map(|&lambda| -> SpeciesEngine<'static, f64, GraphSignal<f64>> {
            let gk = GraphKrylovChernoff::new(Arc::clone(&r.lap), KrylovPath::Chebyshev, 1e-15);
            Box::new(Scaled {
                inner: gk.unwrap(),
                lambda,
            })
        })
        .collect();
    let cm = reaction_matrix(k);
    let gain = if react { 1.0 } else { 0.0 };
    let kin = FnReaction::new(k, move |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        for (i, slot) in du.iter_mut().enumerate() {
            *slot = gain
                * (0..u.len())
                    .map(|l| cm[i * u.len() + l] * u[l])
                    .sum::<f64>();
        }
    });
    let rd = ReactionDiffusion::new(engines, Box::new(kin))
        .unwrap()
        .with_cross_diffusion(cd)
        .unwrap();
    let fields = c
        .iter()
        .map(|&ck| GraphSignal::from_fn(Arc::clone(&r.graph), |j| ck * r.mode[j as usize]))
        .collect();
    let src = Species::new(fields);
    let mut dst = src.clone();
    rd.evolve_into(0.0, t, n, &src, &mut dst, &mut ScratchPool::new())
        .unwrap();
    dst.fields.iter().map(|f| f.values().to_vec()).collect()
}

/// The exact `e^{T(−μD + C)}c` (`C = 0` without reaction).
fn ring_exact(mu: f64, d: &[f64], c: &[f64], t: f64, react: bool) -> Vec<f64> {
    let k = c.len();
    let cm = reaction_matrix(k);
    let a: Vec<f64> = (0..k * k)
        .map(|e| t * (-mu * d[e] + if react { cm[e] } else { 0.0 }))
        .collect();
    matvec(&expm(&a, k), c)
}

fn species_err(u: &[Vec<f64>], want: &[f64], mode: &[f64]) -> f64 {
    let fields: Vec<Vals<'_>> = u.iter().map(|v| Vals(v)).collect();
    rel_err(&Species::new(fields), want, mode)
}

/// `G_CROSSDIFF_GRAPH_EXACT`: exact diffusion on the ring — the mixing is exact,
/// so the result matches `e^{−TμD}c` to rounding, for every gate `D`.
#[test]
fn g_crossdiff_graph_exact() {
    let r = ring(48, 3.0);
    for d in MATRICES {
        let k = (d.len() as f64).sqrt() as usize;
        let c: Vec<f64> = (0..k).map(|i| 1.0 - 0.7 * i as f64).collect();
        let u = ring_run(&r, d, &c, (3.0, 4, false));
        let err = species_err(&u, &ring_exact(r.mu, d, &c, 3.0, false), &r.mode);
        eprintln!("graph exact K={k}: {err:.3e}");
        assert!(err <= 1e-12, "D = {d:?}: {err:e}");
    }
}

/// `G_CROSSDIFF_STRANG_ORDER`: with a non-commuting linear reaction the Strang
/// splitting converges with order 2 to `e^{T(−μD + C)}c` (the reaction acts on the
/// physical species; acting on the modes would converge to a different limit).
#[test]
fn g_crossdiff_strang_order() {
    let r = ring(48, 3.0);
    for d in MATRICES {
        let k = (d.len() as f64).sqrt() as usize;
        let c: Vec<f64> = (0..k).map(|i| 1.0 - 0.7 * i as f64).collect();
        let want = ring_exact(r.mu, d, &c, 3.0, true);
        let errs: Vec<f64> = [40, 80, 160]
            .iter()
            .map(|&n| species_err(&ring_run(&r, d, &c, (3.0, n, true)), &want, &r.mode))
            .collect();
        let slopes = [(errs[0] / errs[1]).log2(), (errs[1] / errs[2]).log2()];
        eprintln!("strang K={k}: errs {errs:?} slopes {slopes:?}");
        for s in slopes {
            assert!((1.9..=2.1).contains(&s), "D = {d:?}: slopes {slopes:?}");
        }
    }
}

/// A periodic axis of period 1 with `n` nodes: the engines' ghost nodes wrap the
/// node index modulo `n` (node `n` is node `0`), so the axis is `[0, 1 − 1/n]`.
fn periodic(n: usize) -> Grid1D<f64> {
    Grid1D::new(0.0, 1.0 - 1.0 / n as f64, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic)
}

fn grid_engine_1d(lambda: f64, g: Grid1D<f64>) -> SpeciesEngine<'static, f64, GridFn1D<f64>> {
    if lambda == 0.0 {
        Box::new(Identity(std::marker::PhantomData))
    } else {
        Box::new(DiffusionChernoff::new_const_a(lambda, lambda, g))
    }
}

fn grid_engine_2d(lambda: f64, g: Grid2D<f64>) -> SpeciesEngine<'static, f64, GridFn2D<f64>> {
    if lambda == 0.0 {
        Box::new(Identity(std::marker::PhantomData))
    } else {
        let axis = |a: Grid1D<f64>| DiffusionChernoff::new_const_a(lambda, lambda, a);
        Box::new(Strang2D::new(axis(g.x), axis(g.y)))
    }
}

/// Run a grid system with zero reaction (`fields` already `c_k·mode`).
fn grid_run<S: semiflow::NodalField<f64> + Clone>(
    cd: CrossDiffusion<f64>,
    engines: Vec<SpeciesEngine<'static, f64, S>>,
    fields: Vec<S>,
    (t, n): (f64, usize),
) -> Result<Species<S>, SemiflowError> {
    let k = fields.len();
    let kin = FnReaction::new(k, |_t: f64, _x: &[f64], _u: &[f64], du: &mut [f64]| {
        du.fill(0.0);
    });
    let rd = ReactionDiffusion::new(engines, Box::new(kin))?.with_cross_diffusion(cd)?;
    let src = Species::new(fields);
    let mut dst = src.clone();
    rd.evolve_into(0.0, t, n, &src, &mut dst, &mut ScratchPool::new())?;
    Ok(dst)
}

/// `max_i |u_i − amp·mode_i|` after `n` steps of `engine` from `mode`.
fn scalar_err<S: State<f64> + Clone>(
    engine: &dyn ChernoffFunction<f64, S = S>,
    u0: &S,
    (t, n): (f64, usize),
    err: impl Fn(&S) -> f64,
) -> f64 {
    let (mut u, mut v, mut scratch) = (u0.clone(), u0.clone(), ScratchPool::new());
    for _ in 0..n {
        engine
            .apply_into(t / n as f64, &u, &mut v, &mut scratch)
            .unwrap();
        std::mem::swap(&mut u, &mut v);
    }
    err(&u)
}

/// The modal bound `max_i Σ_k |P_ik|·|(P⁻¹c)_k|·err_k` of the cross-diffusion
/// error, `err_k` the scalar engine error of mode `k` on a unit-amplitude mode.
fn modal_bound(cd: &CrossDiffusion<f64>, c: &[f64], err_k: &[f64]) -> f64 {
    let k = c.len();
    let m = matvec(cd.inverse_modes(), c);
    let p = cd.modes();
    (0..k)
        .map(|i| {
            (0..k)
                .map(|l| p[i * k + l].abs() * m[l].abs() * err_k[l])
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

/// Absolute sup error of the species against `want_k·mode`.
fn abs_err(fields: &[&[f64]], want: &[f64], mode: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for (f, &w) in fields.iter().zip(want) {
        for (v, m) in f.iter().zip(mode) {
            worst = worst.max((v - w * m).abs());
        }
    }
    worst
}

const TPI: f64 = 2.0 * std::f64::consts::PI;

/// 1-D: (cross-diffusion error, modal bound) after `steps` steps to `T = 0.02`.
fn fourier_1d(d: &[f64], c: &[f64], steps: usize) -> (f64, f64) {
    let (t, g) = (0.02, periodic(64));
    let cd = CrossDiffusion::<f64>::new(d).unwrap();
    let s = GridFn1D::from_fn(g, |x| (TPI * x).sin());
    let err_k: Vec<f64> = cd
        .eigenvalues()
        .iter()
        .map(|&l| {
            let amp = (-TPI * TPI * l * t).exp();
            let e = grid_engine_1d(l, g);
            scalar_err(e.as_ref(), &s, (t, steps), |u| {
                abs_err(&[&u.values], &[amp], &s.values)
            })
        })
        .collect();
    let bound = modal_bound(&cd, c, &err_k);
    let engines = cd
        .eigenvalues()
        .iter()
        .map(|&l| grid_engine_1d(l, g))
        .collect();
    let fields = c
        .iter()
        .map(|&ck| GridFn1D::from_fn(g, |x| ck * (TPI * x).sin()))
        .collect();
    let u = grid_run(cd, engines, fields, (t, steps)).unwrap();
    let want = ring_exact(TPI * TPI, d, c, t, false);
    let vals: Vec<&[f64]> = u.fields.iter().map(|f| f.values.as_slice()).collect();
    (abs_err(&vals, &want, &s.values), bound)
}

/// 2-D: (cross-diffusion error, modal bound) after `steps` steps to `T = 0.02`.
fn fourier_2d(d: &[f64], c: &[f64], steps: usize) -> (f64, f64) {
    let (t, g) = (0.02, Grid2D::new(periodic(32), periodic(32)));
    let cd = CrossDiffusion::<f64>::new(d).unwrap();
    let s = GridFn2D::from_fn(g, |x, y| (TPI * x).sin() * (TPI * y).sin());
    let err_k: Vec<f64> = cd
        .eigenvalues()
        .iter()
        .map(|&l| {
            let amp = (-2.0 * TPI * TPI * l * t).exp();
            let e = grid_engine_2d(l, g);
            scalar_err(e.as_ref(), &s, (t, steps), |u| {
                abs_err(&[&u.values], &[amp], &s.values)
            })
        })
        .collect();
    let bound = modal_bound(&cd, c, &err_k);
    let engines = cd
        .eigenvalues()
        .iter()
        .map(|&l| grid_engine_2d(l, g))
        .collect();
    let fields = c
        .iter()
        .map(|&ck| GridFn2D {
            values: s.values.iter().map(|v| ck * v).collect(),
            grid: g,
        })
        .collect();
    let u = grid_run(cd, engines, fields, (t, steps)).unwrap();
    let want = ring_exact(2.0 * TPI * TPI, d, c, t, false);
    let vals: Vec<&[f64]> = u.fields.iter().map(|f| f.values.as_slice()).collect();
    (abs_err(&vals, &want, &s.values), bound)
}

/// `G_CROSSDIFF_FOURIER_GRID`: periodic `sin(2πx)·c` (1-D, 64 nodes) and
/// `sin(2πx)sin(2πy)·c` (2-D, 32²) with the grid engines (septic, Strang across
/// axes) against the closed form `e^{−Tk²D}c`: the error stays within the modal
/// bound `max_i Σ_k |P_ik||(P⁻¹c)_k|·err_k` built from each scalar mode engine's
/// own error `err_k` (plus `1e−14` rounding), the bound is informative (`≤ 1e−4`),
/// and the error converges with the engines' order 2 (slopes over 20, 40, 80 steps).
#[test]
fn g_crossdiff_fourier_grid() {
    for d in MATRICES {
        let k = (d.len() as f64).sqrt() as usize;
        let c: Vec<f64> = (0..k).map(|i| 1.0 - 0.7 * i as f64).collect();
        for (dim, run) in [
            (1, fourier_1d as fn(&[f64], &[f64], usize) -> (f64, f64)),
            (2, fourier_2d),
        ] {
            let res: Vec<(f64, f64)> = [20, 40, 80].iter().map(|&n| run(d, &c, n)).collect();
            let slopes = [(res[0].0 / res[1].0).log2(), (res[1].0 / res[2].0).log2()];
            eprintln!("fourier K={k} {dim}-D: (err, bound) {res:?} slopes {slopes:?}");
            for &(err, bound) in &res {
                assert!(
                    err <= bound + 1e-14 && bound <= 1e-4,
                    "D = {d:?} {dim}-D: {res:?}"
                );
            }
            for s in slopes {
                assert!(
                    (1.9..=2.1).contains(&s),
                    "D = {d:?} {dim}-D: slopes {slopes:?}"
                );
            }
        }
    }
}

/// `G_CROSSDIFF_REJECT`: complex, negative and defective `D` are rejected with
/// the reason; a non-zero Dirichlet value, species on different grids, and a
/// species-count mismatch are rejected at run time.
#[test]
fn g_crossdiff_reject() {
    let bad: [(&[f64], &str); 4] = [
        (&[1.0, -2.0, 2.0, 1.0], "complex"),
        (&[1.0, 2.0, 2.0, 1.0], "negative"),
        (&[1.0, 1.0, 0.0, 1.0], "defective"),
        (&[1.0, 1e-9, 0.0, 1.0 + 1e-12], "ill-conditioned"),
    ];
    for (d, why) in bad {
        match CrossDiffusion::<f64>::new(d) {
            Err(SemiflowError::DomainViolation { what, .. }) => {
                eprintln!("{why}: {what}");
            }
            other => panic!("{why}: {other:?}"),
        }
    }
    let d = [1.0, 0.5, 0.2, 0.6];
    let run = |gs: [Grid1D<f64>; 2]| {
        let cd = CrossDiffusion::<f64>::new(&d).unwrap();
        let engines = cd
            .eigenvalues()
            .iter()
            .map(|&l| grid_engine_1d(l, gs[0]))
            .collect();
        let fields = gs
            .iter()
            .map(|&g| GridFn1D::from_fn(g, |x| x * (1.0 - x)))
            .collect();
        grid_run(cd, engines, fields, (0.01, 2))
    };
    let base = Grid1D::new(0.0, 1.0, 33).unwrap();
    let dir = |v: f64| base.with_boundary(BoundaryPolicy::Dirichlet { value: v });
    assert!(run([base, base]).is_ok());
    assert!(run([dir(0.0), dir(0.0)]).is_ok());
    for gs in [[dir(1.0), dir(1.0)], [base, dir(0.0)], [base, periodic(33)]] {
        assert!(
            matches!(run(gs), Err(SemiflowError::DomainViolation { .. })),
            "{gs:?}"
        );
    }
    let cd3 = CrossDiffusion::<f64>::new(MATRICES[2]).unwrap();
    let engines = vec![grid_engine_1d(1.0, base), grid_engine_1d(1.0, base)];
    let kin = FnReaction::new(2, |_t: f64, _x: &[f64], _u: &[f64], du: &mut [f64]| {
        du.fill(0.0);
    });
    let rd = ReactionDiffusion::new(engines, Box::new(kin)).unwrap();
    assert!(rd.with_cross_diffusion(cd3).is_err());
}
