//! Shared helpers of the stiff-kinetics gates (ADR-0212): an identity diffusion
//! engine (the gates isolate the reaction flow) and a 4-node evolution driver.
#![allow(dead_code)]

use semiflow::{
    ChernoffFunction, Grid1D, GridFn1D, Growth, Kinetics, ReactionDiffusion, ReactionSolver,
    RosenbrockOptions, ScratchPool, SemiflowError, Species, SpeciesEngine,
};

/// Nodes of every gate field (`Grid1D(0, 1, 4)`: `x = 0, 1/3, 2/3, 1`).
pub const NODES: usize = 4;

/// The identity "diffusion".
pub struct Identity;

impl ChernoffFunction<f64> for Identity {
    type S = GridFn1D<f64>;
    fn apply_into(
        &self,
        _tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        dst.values.copy_from_slice(&src.values);
        Ok(())
    }
    fn order(&self) -> u32 {
        u32::MAX
    }
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

/// Evolve per-species node values `u0[k]` (`NODES` each) over `[0, t]` in `n`
/// Strang steps; returns every species' node values.
pub fn evolve_nodes(
    kin: Box<dyn Kinetics<f64>>,
    solver: ReactionSolver,
    substeps: u32,
    u0: &[Vec<f64>],
    (t, n): (f64, usize),
) -> Result<Vec<Vec<f64>>, SemiflowError> {
    let grid = Grid1D::new(0.0, 1.0, NODES).expect("grid");
    let engines: Vec<SpeciesEngine<'static, f64, GridFn1D<f64>>> = u0
        .iter()
        .map(|_| Box::new(Identity) as SpeciesEngine<'static, f64, GridFn1D<f64>>)
        .collect();
    let rd = ReactionDiffusion::new(engines, kin)?
        .with_reaction_solver(solver)
        .with_reaction_substeps(substeps);
    let fields = u0
        .iter()
        .map(|v| GridFn1D::new(grid, v.clone()).expect("field"))
        .collect();
    let src = Species::new(fields);
    let mut dst = src.clone();
    rd.evolve_into(0.0, t, n, &src, &mut dst, &mut ScratchPool::new())?;
    Ok(dst.fields.into_iter().map(|f| f.values).collect())
}

/// [`evolve_nodes`] from a uniform field (one value per species); node 0 values.
pub fn evolve(
    kin: Box<dyn Kinetics<f64>>,
    solver: ReactionSolver,
    substeps: u32,
    u0: &[f64],
    job: (f64, usize),
) -> Result<Vec<f64>, SemiflowError> {
    let fields: Vec<Vec<f64>> = u0.iter().map(|&v| vec![v; NODES]).collect();
    let out = evolve_nodes(kin, solver, substeps, &fields, job)?;
    Ok(out.iter().map(|f| f[0]).collect())
}

/// `max |got − want| / max |want|`.
pub fn rel_sup(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |a, w| a.max(w.abs()));
    got.iter()
        .zip(want)
        .fold(0.0_f64, |a, (g, w)| a.max((g - w).abs()))
        / scale
}

/// The Rosenbrock solver with `rtol`, `atol`.
pub fn ros(rtol: f64, atol: f64) -> ReactionSolver {
    ReactionSolver::Rosenbrock(RosenbrockOptions::default().with_rtol(rtol).with_atol(atol))
}
