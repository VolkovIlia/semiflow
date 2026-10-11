//! `G_ND_SWEEP_BIT_EQUAL`, engine level (ADR-0214): `SmolyakGridND::apply_into`
//! and `AnisotropicShiftChernoffND::apply_into` with a constant dense `A`, constant
//! `b ≠ 0` and a variable `c` (separable path) — and with a variable `b` (per-node
//! path) — against the pre-ADR-0214 loop fed with the same Cholesky factors and
//! quadrature, bit for bit.

use alloc::{boxed::Box, vec::Vec};

use super::tests::{bits, field, grid, original, smolyak_quad, tensor_quad};
use super::ShiftStep;
use crate::{
    boundary::BoundaryPolicy,
    chernoff::ChernoffFunction,
    grid::InterpKind,
    grid_nd::{GridFnND, GridND},
    scratch::ScratchPool,
    shift_nd::{build_cholesky_cache, AnisotropicShiftChernoffND, SquareMatrix},
    smolyak::SmolyakGridND,
};

/// Constant SPD diffusion matrix with off-diagonal coupling.
fn a_const<const D: usize>(_x: &[f64; D], a: &mut SquareMatrix<f64, D>) {
    for i in 0..D {
        for j in 0..D {
            a.set(i, j, if i == j { 0.6 } else { 0.15 });
        }
    }
}

fn b_const<const D: usize>(_x: &[f64; D], b: &mut [f64; D]) {
    for (d, v) in b.iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let dd = d as f64;
        *v = 0.3 - 0.2 * dd;
    }
}

fn b_var<const D: usize>(x: &[f64; D], b: &mut [f64; D]) {
    b_const(x, b);
    b[0] += 0.1 * x[D - 1];
}

fn c_var<const D: usize>(x: &[f64; D]) -> f64 {
    0.2 * x[0] - 0.05
}

/// A drift closure as a plain function pointer.
type DriftFn<const D: usize> = fn(&[f64; D], &mut [f64; D]);
type DriftBox<const D: usize> = Box<dyn Fn(&[f64; D], &mut [f64; D]) + Send + Sync>;
type ReactionBox<const D: usize> = Box<dyn Fn(&[f64; D]) -> f64 + Send + Sync>;

/// The reference for an engine built from `(a_const, b, c_var)` on `g`.
fn reference<const D: usize>(
    g: &GridND<f64, D>,
    b: DriftFn<D>,
    quad: &[([f64; D], f64)],
    src: &GridFnND<f64, D>,
    tau: f64,
) -> Vec<u64> {
    let chol = build_cholesky_cache(&a_const::<D>, g).unwrap();
    let (bb, cc): (DriftBox<D>, ReactionBox<D>) = (Box::new(b), Box::new(c_var::<D>));
    let step = ShiftStep {
        tau,
        grid: g,
        b: &*bb,
        c: &*cc,
        chol: &chol,
        quad,
    };
    bits(&original(&step, src))
}

fn run<C: ChernoffFunction<f64, S = GridFnND<f64, D>>, const D: usize>(
    engine: &C,
    src: &GridFnND<f64, D>,
    tau: f64,
) -> Vec<u64> {
    let mut dst = src.clone();
    engine
        .apply_into(tau, src, &mut dst, &mut ScratchPool::new())
        .unwrap();
    bits(&dst.values)
}

/// Both engines on `g`, constant and variable `b`, against the verbatim loop.
fn engines_match<const D: usize>(g: &GridND<f64, D>, level: usize) {
    let tau = 0.12;
    let cases: [(DriftFn<D>, &str); 2] = [(b_const::<D>, "constant b"), (b_var::<D>, "variable b")];
    for (b, label) in cases {
        let src = field(g.clone());
        let smolyak =
            SmolyakGridND::with_level(a_const::<D>, b, c_var::<D>, g.clone(), level).unwrap();
        let want = reference(g, b, &smolyak_quad(level), &src, tau);
        assert_eq!(run(&smolyak, &src, tau), want, "Smolyak D={D}, {label}");
        let aniso =
            AnisotropicShiftChernoffND::new(a_const::<D>, b, c_var::<D>, g.clone()).unwrap();
        let want = reference(g, b, &tensor_quad(), &src, tau);
        assert_eq!(run(&aniso, &src, tau), want, "anisotropic D={D}, {label}");
    }
}

fn policy(d: usize) -> BoundaryPolicy<f64> {
    if d == 0 {
        BoundaryPolicy::LinearExtrapolate
    } else {
        BoundaryPolicy::Robin {
            alpha: 0.5,
            beta: 2.0,
        }
    }
}

#[test]
fn g_nd_sweep_engines_bit_equal() {
    engines_match(&grid::<f64, 2>([9, 7], InterpKind::CubicHermite, policy), 5);
    engines_match(&grid::<f64, 3>([6, 5, 5], InterpKind::Linear, policy), 5);
}
