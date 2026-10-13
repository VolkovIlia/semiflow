//! `G_CROSSDIFF_REJECT` (`RELEASE_BLOCKING`, ADR-0213, math §70): invalid
//! cross-diffusion input is rejected with the REASON.
//!
//! * `CrossDiffusion::new`: complex, negative (also within the clamp tolerance),
//!   defective (`does not reproduce D`) and nearly defective (`cond₁(P) > 1e8`:
//!   `ill-conditioned`) matrices; `from_modes`: singular and ill-conditioned `P`.
//! * Run time (`ReactionDiffusion::with_cross_diffusion`): a non-zero Dirichlet
//!   value, and species that differ in exactly ONE of boundary policy, `xmin`,
//!   `xmax` or interpolant (1-D), in one axis (2-D), or live on different graphs;
//!   a species-count mismatch. Zero-Dirichlet and identical fields are accepted.

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use semiflow::{
    BoundaryPolicy, ChernoffFunction, CrossDiffusion, DiffusionChernoff, FnReaction, Graph,
    GraphKrylovChernoff, GraphSignal, Grid1D, Grid2D, GridFn1D, GridFn2D, Growth, InterpKind,
    KrylovPath, Laplacian, NodalField, ReactionDiffusion, ScratchPool, SemiflowError, Species,
    SpeciesEngine, State, Strang2D,
};

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

fn what<T: std::fmt::Debug>(r: Result<T, SemiflowError>) -> &'static str {
    match r {
        Err(SemiflowError::DomainViolation { what, .. }) => what,
        other => panic!("expected a DomainViolation, got {other:?}"),
    }
}

#[test]
fn g_crossdiff_reject_matrices() {
    let bad: [(&[f64], &str); 8] = [
        (&[1.0, -2.0, 2.0, 1.0], "complex eigenvalues"),
        (&[1.0, 2.0, 2.0, 1.0], "negative eigenvalue"),
        (&[1.0, 1.0, 1.0, 1.0 - 1e-8], "negative eigenvalue"),
        (&[1.0, -1e-8, 1e-8, 1.0], "complex eigenvalues"),
        (&[1.0, 1.0, 0.0, 1.0], "does not reproduce D"),
        (&[1.0, 1.0, 0.0, 1.0 + 1e-9], "ill-conditioned"),
        (
            &[1.0, 1.0, 0.0, 0.0, 1.0 + 1e-5, 1.0, 0.0, 0.0, 1.0 + 2e-5],
            "ill-conditioned",
        ),
        (&[1.0, 2.0, 3.0], "K×K"),
    ];
    for (d, why) in bad {
        let got = what(CrossDiffusion::<f64>::new(d));
        eprintln!("G_CROSSDIFF_REJECT {d:?}: {got}");
        assert!(got.contains(why), "{d:?}: expected {why:?}, got {got:?}");
    }
    let modes: [(&[f64], &str); 2] = [
        (&[1.0, 1.0, 1.0, 1.0 + 1e-10], "ill-conditioned"),
        (&[1.0, 1.0, 1.0, 1.0], "singular eigenvector matrix"),
    ];
    for (p, why) in modes {
        let got = what(CrossDiffusion::<f64>::from_modes(p, &[1.0, 2.0]));
        assert!(got.contains(why), "{p:?}: expected {why:?}, got {got:?}");
    }
}

const D: [f64; 4] = [1.0, 0.5, 0.2, 0.6];

/// Run a zero-reaction system of `fields` (engines from `engine(λ)`).
fn run<S: NodalField<f64> + Clone>(
    fields: Vec<S>,
    engine: impl Fn(f64) -> SpeciesEngine<'static, f64, S>,
) -> Result<(), SemiflowError> {
    let cd = CrossDiffusion::<f64>::new(&D)?;
    let engines = cd.eigenvalues().iter().map(|&l| engine(l)).collect();
    let kin = FnReaction::new(2, |_t: f64, _x: &[f64], _u: &[f64], du: &mut [f64]| {
        du.fill(0.0);
    });
    let rd = ReactionDiffusion::new(engines, Box::new(kin))?.with_cross_diffusion(cd)?;
    let src = Species::new(fields);
    let mut dst = src.clone();
    rd.evolve_into(0.0, 0.01, 2, &src, &mut dst, &mut ScratchPool::new())
}

fn engine_1d(g: Grid1D<f64>) -> impl Fn(f64) -> SpeciesEngine<'static, f64, GridFn1D<f64>> {
    move |l| -> SpeciesEngine<'static, f64, GridFn1D<f64>> {
        if l == 0.0 {
            Box::new(Identity(std::marker::PhantomData))
        } else {
            Box::new(DiffusionChernoff::new_const_a(l, l, g))
        }
    }
}

const MIX: &str = "cross-diffusion needs every species on the same nodes";

#[test]
fn g_crossdiff_reject_fields_1d() {
    let base = Grid1D::new(0.0, 1.0, 33).unwrap();
    let dir = |v: f64| base.with_boundary(BoundaryPolicy::Dirichlet { value: v });
    let fields = |gs: [Grid1D<f64>; 2]| {
        gs.iter()
            .map(|&g| GridFn1D::from_fn(g, |x| x * (1.0 - x)))
            .collect::<Vec<_>>()
    };
    let ok = |gs: [Grid1D<f64>; 2]| run(fields(gs), engine_1d(gs[0]));
    assert!(ok([base, base]).is_ok());
    assert!(ok([dir(0.0), dir(0.0)]).is_ok());
    for gs in [
        [dir(1.0), dir(1.0)],
        [base, dir(0.0)],
        [base, base.with_boundary(BoundaryPolicy::Neumann)],
        [base, Grid1D::new(0.0, 2.0, 33).unwrap()],
        [base, Grid1D::new(-1.0, 1.0, 33).unwrap()],
        [base, base.with_interp(InterpKind::CubicHermite)],
    ] {
        let got = what(ok(gs));
        assert!(got.contains(MIX), "{gs:?}: {got}");
    }
    let kin = FnReaction::new(2, |_t: f64, _x: &[f64], _u: &[f64], du: &mut [f64]| {
        du.fill(0.0);
    });
    let engines: Vec<SpeciesEngine<'static, f64, GridFn1D<f64>>> =
        vec![engine_1d(base)(1.0), engine_1d(base)(1.0)];
    let rd = ReactionDiffusion::new(engines, Box::new(kin)).unwrap();
    let three = CrossDiffusion::<f64>::new(&[0.8, 0.3, 0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.3]).unwrap();
    assert!(what(rd.with_cross_diffusion(three).map(|_| ())).contains("K×K"));
}

#[test]
fn g_crossdiff_reject_fields_2d_and_graph() {
    let base = Grid1D::new(0.0, 1.0, 17).unwrap();
    let (b2, other) = (
        Grid2D::new(base, base),
        Grid2D::new(base, Grid1D::new(0.0, 2.0, 17).unwrap()),
    );
    let engine_2d = move |l: f64| -> SpeciesEngine<'static, f64, GridFn2D<f64>> {
        let axis = |g: Grid1D<f64>| DiffusionChernoff::new_const_a(l, l, g);
        Box::new(Strang2D::new(axis(base), axis(base)))
    };
    let field = |g: Grid2D<f64>| GridFn2D::from_fn(g, |x, y| x * (1.0 - x) * y * (1.0 - y));
    assert!(run(vec![field(b2), field(b2)], engine_2d).is_ok());
    assert!(what(run(vec![field(b2), field(other)], engine_2d)).contains(MIX));
    // Graph signals mix only on one shared graph.
    let (g1, g2): (Arc<Graph<f64>>, Arc<Graph<f64>>) =
        (Arc::new(Graph::cycle(48)), Arc::new(Graph::cycle(48)));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g1));
    let engine_g = move |_l: f64| -> SpeciesEngine<'static, f64, GraphSignal<f64>> {
        Box::new(GraphKrylovChernoff::new(Arc::clone(&lap), KrylovPath::Chebyshev, 1e-12).unwrap())
    };
    let sig = |g: &Arc<Graph<f64>>| GraphSignal::from_fn(Arc::clone(g), |j| f64::from(j).sin());
    assert!(run(vec![sig(&g1), sig(&g1)], engine_g.clone()).is_ok());
    assert!(what(run(vec![sig(&g1), sig(&g2)], engine_g)).contains(MIX));
}
