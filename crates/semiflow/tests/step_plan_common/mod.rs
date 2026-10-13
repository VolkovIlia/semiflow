//! Shared helpers of the step-plan gates (ADR-0215): the unplanned reference
//! wrapper `NoPlan`, the engines under test and a 1-D evolution driver.
#![allow(dead_code)]

use semiflow::{
    BoundaryPolicy, ChernoffFunction, ChernoffSemigroup, DiffusionChernoff, Evolver, Grid1D,
    GridFn1D, Growth, InterpKind, OobPolicy, ScratchPool, SemiflowError, ShiftChernoff1D,
};

/// Forwards `apply_into` only: the unplanned reference path.
#[derive(Clone)]
pub struct NoPlan<C>(pub C);

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

pub fn a_half(_: f64) -> f64 {
    0.5
}
pub fn zero(_: f64) -> f64 {
    0.0
}

/// The three storage variants of `DiffusionChernoff` on `grid`.
pub fn engines(grid: Grid1D<f64>) -> [(&'static str, DiffusionChernoff<f64>); 3] {
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

pub fn drift(x: f64) -> f64 {
    0.3 * x
}
pub fn sink(x: f64) -> f64 {
    -0.2 - 0.1 * x * x
}

/// `ShiftChernoff1D` with `a ≡ ½`, drift `0.3x`, reaction `−0.2 − 0.1x²`.
pub fn shift(grid: Grid1D<f64>) -> ShiftChernoff1D<f64> {
    ShiftChernoff1D::new(a_half, drift, sink, 0.3, grid)
}

pub const POLICIES: [BoundaryPolicy<f64>; 8] = [
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

pub fn interps() -> [InterpKind; 5] {
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
pub fn same(a: &Result<Vec<f64>, SemiflowError>, b: &Result<Vec<f64>, SemiflowError>) -> bool {
    match (a, b) {
        (Ok(x), Ok(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (Err(x), Err(y)) => core::mem::discriminant(x) == core::mem::discriminant(y),
        _ => false,
    }
}

pub fn bump(x: f64) -> f64 {
    (-2.0 * x * x).exp() + 0.1 * x
}

/// `Evolver` (3 steps, caller pool used twice) and `ChernoffSemigroup` (2 steps).
pub fn evolve_1d<C: ChernoffFunction<f64, S = GridFn1D<f64>>>(
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
