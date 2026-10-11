//! `G_ND_SWEEP_BIT_EQUAL` (ADR-0214): the separable whole-array path (both
//! scratch modes), the per-node path and the pre-ADR-0214 per-node loop (copied
//! verbatim below) give the same bits. Coverage, exactly:
//! * D = 2 (9×7) and D = 3 (6×5×4): `CubicHermite` and `Linear` × all eight boundary
//!   policies (D = 3 rotated per axis) × τ ∈ {0.05, 0.6} × {Smolyak ℓ = 5 table,
//!   tensor Gauss–Hermite rule}, in f64; the D = 2 Smolyak matrix also in f32.
//! * D = 5 (4⁵), ℓ = 8 (341 nodes), τ = 0.2, three policy rotations: `Linear` on the
//!   full table, `CubicHermite` on its first 40 nodes.
//! * D = 6 (4⁶), ℓ = 9 (533 nodes), τ = 0.15: `Linear` on the first 160 nodes,
//!   `CubicHermite` on the first 8 and on every 41st node (13 nodes spanning every
//!   axis-0 run of the table) — the reference loop reads 4⁶ values per sample.
//!
//! Every case asserts non-vacuity `max|Sf − f| > 1e−3`. Engine-level wiring
//! (`apply_into` of both engines) is checked in `grid_nd_sweep_engine_tests.rs`.

use alloc::{boxed::Box, vec, vec::Vec};

use super::{per_node, same_bits, try_separable, ShiftStep, SWEEP_MAX_ELEMS};
use crate::{
    boundary::BoundaryPolicy,
    dual::Dual,
    float::{from_f64, SemiflowFloat},
    grid::{Grid1D, InterpKind},
    grid_nd::{GridFnND, GridND},
    scratch::ScratchPool,
    shift_nd::{flat_to_x, GaussHermiteTensor, SquareMatrix},
    smolyak::build_smolyak,
};

type Drift<const D: usize, F = f64> = Box<dyn Fn(&[F; D], &mut [F; D]) + Send + Sync>;
type Reaction<const D: usize, F = f64> = Box<dyn Fn(&[F; D]) -> F + Send + Sync>;

const POLICIES: [BoundaryPolicy<f64>; 8] = [
    BoundaryPolicy::Reflect,
    BoundaryPolicy::ZeroExtend,
    BoundaryPolicy::Periodic,
    BoundaryPolicy::LinearExtrapolate,
    BoundaryPolicy::Dirichlet { value: 0.7 },
    BoundaryPolicy::Neumann,
    BoundaryPolicy::Robin {
        alpha: 0.5,
        beta: 2.0,
    },
    BoundaryPolicy::OddReflect,
];

/// The engines' per-node loop before ADR-0214, verbatim (generic in `F`).
pub(super) fn original<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    src: &GridFnND<F, D>,
) -> Vec<F> {
    let tau = step.tau;
    let ns: [usize; D] = core::array::from_fn(|d| step.grid.axes[d].n);
    let total = src.values.len();
    let two_sqrt_tau = from_f64::<F>(2.0_f64) * tau.sqrt();
    let inv_pi_dhalf = crate::float::inv_pi_pow_half::<F>(D);
    let mut out = vec![F::zero(); total];
    for (flat, o) in out.iter_mut().enumerate() {
        let xk = flat_to_x::<F, D>(flat, &ns, step.grid);
        let mut b_val = [F::zero(); D];
        (step.b)(&xk, &mut b_val);
        let c_val = (step.c)(&xk);
        let exp_factor = (tau * c_val).libm_exp();
        let l_k = &step.chol[flat];
        let mut acc = F::zero();
        for &(eta_q, w_q) in step.quad {
            let mut l_eta = [F::zero(); D];
            l_k.mat_vec(&eta_q, &mut l_eta);
            let y_q: [F; D] =
                core::array::from_fn(|d| xk[d] + tau * b_val[d] + two_sqrt_tau * l_eta[d]);
            let f_y = src.sample(&y_q).unwrap_or(F::zero());
            acc += w_q * f_y;
        }
        *o = exp_factor * inv_pi_dhalf * acc;
    }
    out
}

/// A dense lower-triangular factor with distinct entries (not `I`: the offsets
/// of axis `d` then mix every `η_e`, `e ≤ d`).
fn factor<F: SemiflowFloat, const D: usize>() -> SquareMatrix<F, D> {
    let mut l = SquareMatrix::zero();
    for i in 0..D {
        for j in 0..=i {
            #[allow(clippy::cast_precision_loss)]
            let v = if i == j {
                1.0 + 0.05 * i as f64
            } else {
                0.21 + 0.07 * (i + j) as f64
            };
            l.set(i, j, from_f64(v));
        }
    }
    l
}

pub(super) fn smolyak_quad<F: SemiflowFloat, const D: usize>(level: usize) -> Vec<([F; D], F)> {
    build_smolyak::<D>(level)
        .unwrap()
        .into_iter()
        .map(|(node, w)| (core::array::from_fn(|d| from_f64(node[d])), from_f64(w)))
        .collect()
}

pub(super) fn tensor_quad<F: SemiflowFloat, const D: usize>() -> Vec<([F; D], F)> {
    let gh = GaussHermiteTensor::<F, D>::new();
    (0..gh.n_nodes())
        .map(|q| (gh.node(q), gh.weight(q)))
        .collect()
}

pub(super) fn grid<F: SemiflowFloat, const D: usize>(
    shape: [usize; D],
    interp: InterpKind,
    policy: impl Fn(usize) -> BoundaryPolicy<F>,
) -> GridND<F, D> {
    let axes = core::array::from_fn(|d| {
        Grid1D::new_generic(from_f64(-1.0), from_f64(1.3), shape[d])
            .unwrap()
            .with_boundary(policy(d))
    });
    GridND::new(axes).unwrap().with_interp(interp)
}

pub(super) fn field<F: SemiflowFloat, const D: usize>(g: GridND<F, D>) -> GridFnND<F, D> {
    #[allow(clippy::cast_precision_loss)]
    GridFnND::from_fn(g, |x| {
        let s: f64 = x
            .iter()
            .enumerate()
            .map(|(d, &v)| (1.3 * v.to_f64().unwrap_or(0.0) + d as f64).sin())
            .sum();
        from_f64(0.5 + s)
    })
}

pub(super) fn bits<F: SemiflowFloat>(v: &[F]) -> Vec<u64> {
    v.iter()
        .map(|x| x.to_f64().unwrap_or(f64::NAN).to_bits())
        .collect()
}

/// Policy `p` in `F` (the f64 table converted).
fn policy<F: SemiflowFloat>(p: usize) -> BoundaryPolicy<F> {
    match POLICIES[p % 8] {
        BoundaryPolicy::Dirichlet { value } => BoundaryPolicy::Dirichlet {
            value: from_f64(value),
        },
        BoundaryPolicy::Robin { alpha, beta } => BoundaryPolicy::Robin {
            alpha: from_f64(alpha),
            beta: from_f64(beta),
        },
        BoundaryPolicy::Reflect => BoundaryPolicy::Reflect,
        BoundaryPolicy::ZeroExtend => BoundaryPolicy::ZeroExtend,
        BoundaryPolicy::Periodic => BoundaryPolicy::Periodic,
        BoundaryPolicy::LinearExtrapolate => BoundaryPolicy::LinearExtrapolate,
        BoundaryPolicy::Neumann => BoundaryPolicy::Neumann,
        BoundaryPolicy::OddReflect => BoundaryPolicy::OddReflect,
    }
}

/// Every path agrees to the bit, and the step moves the field (`> 1e−3`).
fn check<F: SemiflowFloat, const D: usize>(
    src: &GridFnND<F, D>,
    quad: &[([F; D], F)],
    tau: f64,
    b0: [f64; D],
) {
    let b0: [F; D] = core::array::from_fn(|d| from_f64(b0[d]));
    let b: Drift<D, F> = Box::new(move |_x, out| *out = b0);
    let c: Reaction<D, F> = Box::new(|x| from_f64::<F>(0.3) * x[0] - from_f64(0.1));
    let chol = vec![factor::<F, D>(); src.values.len()];
    let step = ShiftStep {
        tau: from_f64(tau),
        grid: &src.grid,
        b: &*b,
        c: &*c,
        chol: &chol,
        quad,
    };
    let reference = original(&step, src);
    let want = bits(&reference);
    let mut dst = src.clone();
    per_node(&step, src, &mut dst);
    assert_eq!(bits(&dst.values), want, "per-node path, D={D}");
    let mut scratch = ScratchPool::new();
    for cap in [SWEEP_MAX_ELEMS, 3 * src.values.len()] {
        let mut dst = src.clone();
        assert!(
            try_separable(&step, src, &mut dst, &mut scratch, cap),
            "separable path refused, D={D}"
        );
        assert_eq!(bits(&dst.values), want, "separable path, D={D}, cap={cap}");
    }
    let moved = src
        .values
        .iter()
        .zip(&reference)
        .map(|(a, b)| (*a - *b).abs().to_f64().unwrap_or(0.0))
        .fold(0.0, f64::max);
    assert!(moved > 1e-3, "vacuous: D={D}, tau={tau}");
}

#[test]
fn g_nd_sweep_bit_equal_d2_d3_every_policy() {
    for interp in [InterpKind::CubicHermite, InterpKind::Linear] {
        for p in 0..8 {
            for tau in [0.05, 0.6] {
                let src2 = field::<f64, 2>(grid([9, 7], interp, |_| policy(p)));
                check(&src2, &smolyak_quad::<f64, 2>(5), tau, [0.4, -0.3]);
                check(&src2, &tensor_quad::<f64, 2>(), tau, [0.4, -0.3]);
                let src2f = field::<f32, 2>(grid([9, 7], interp, |_| policy(p)));
                check(&src2f, &smolyak_quad::<f32, 2>(5), tau, [0.4, -0.3]);
                let src3 = field::<f64, 3>(grid([6, 5, 4], interp, |d| policy(p + 3 * d)));
                check(&src3, &smolyak_quad::<f64, 3>(5), tau, [0.2, 0.0, -0.5]);
                check(&src3, &tensor_quad::<f64, 3>(), tau, [0.2, 0.0, -0.5]);
            }
        }
    }
}

#[test]
fn g_nd_sweep_bit_equal_d5_d6() {
    let q5 = smolyak_quad::<f64, 5>(8);
    for p in [3, 6, 7] {
        let src5 = field::<f64, 5>(grid([4; 5], InterpKind::Linear, |d| policy(p + d)));
        check(&src5, &q5, 0.2, [0.1, -0.2, 0.3, 0.0, 0.25]);
        let src5 = field::<f64, 5>(grid([4; 5], InterpKind::CubicHermite, |d| policy(p + d)));
        check(&src5, &q5[..40], 0.2, [0.1, -0.2, 0.3, 0.0, 0.25]);
    }
    let b6 = [0.1, 0.0, -0.1, 0.2, 0.0, 0.05];
    let full = smolyak_quad::<f64, 6>(9);
    let src6 = field::<f64, 6>(grid([4; 6], InterpKind::Linear, |d| policy(3 + d)));
    check(&src6, &full[..160], 0.15, b6);
    let src6 = field::<f64, 6>(grid([4; 6], InterpKind::CubicHermite, |d| policy(6 + d)));
    check(&src6, &full[..8], 0.15, b6);
    let strided: Vec<_> = full.iter().copied().step_by(41).collect();
    check(&src6, &strided, 0.15, b6);
}

/// NaN-carrying fields: non-NaN outputs keep exact bits, NaN outputs coincide as a
/// class (IEEE 754 / Rust leave the payload of NaN ⊕ NaN unspecified, ADR-0214).
#[test]
fn g_nd_sweep_nan_inputs_same_class() {
    let mut src = field::<f64, 2>(grid([9, 7], InterpKind::CubicHermite, |_| policy(3)));
    src.values[40] = f64::NAN;
    src.values[41] = f64::from_bits(0x7ff8_0000_0000_1234);
    src.values[12] = f64::INFINITY;
    src.values[13] = f64::NEG_INFINITY;
    let quad = smolyak_quad::<f64, 2>(5);
    let b: Drift<2> = Box::new(|_, out| *out = [0.4, -0.3]);
    let c: Reaction<2> = Box::new(|x| 0.3 * x[0] - 0.1);
    let chol = vec![factor::<f64, 2>(); src.values.len()];
    let step = ShiftStep {
        tau: 1e-4,
        grid: &src.grid,
        b: &*b,
        c: &*c,
        chol: &chol,
        quad: &quad,
    };
    let want = original(&step, &src);
    let mut got = src.clone();
    assert!(try_separable(
        &step,
        &src,
        &mut got,
        &mut ScratchPool::new(),
        SWEEP_MAX_ELEMS
    ));
    let mut nan = 0;
    for (k, (g, w)) in got.values.iter().zip(&want).enumerate() {
        if w.is_nan() || g.is_nan() {
            assert!(w.is_nan() && g.is_nan(), "node {k}: NaN class differs");
            nan += 1;
        } else {
            assert_eq!(g.to_bits(), w.to_bits(), "node {k}");
        }
    }
    assert!(
        nan > 0 && nan < want.len(),
        "the case must mix NaN and finite outputs ({nan})"
    );
}

/// Whether the separable path accepts drift `b`, factors `chol` and scratch `cap`.
fn accepts(
    src: &GridFnND<f64, 2>,
    b: &Drift<2>,
    chol: &[SquareMatrix<f64, 2>],
    cap: usize,
) -> bool {
    let quad = smolyak_quad::<f64, 2>(5);
    let c: Reaction<2> = Box::new(|_| 0.0);
    let step = ShiftStep {
        tau: 0.1,
        grid: &src.grid,
        b: &**b,
        c: &*c,
        chol,
        quad: &quad,
    };
    try_separable(&step, src, &mut src.clone(), &mut ScratchPool::new(), cap)
}

#[test]
fn g_nd_sweep_refuses_variable_coefficients() {
    let src = field(grid([6, 5], InterpKind::CubicHermite, |_| {
        BoundaryPolicy::Reflect
    }));
    let total = src.values.len();
    let constant: Drift<2> = Box::new(|_, out| *out = [0.1, 0.0]);
    let chol = vec![factor::<f64, 2>(); total];
    assert!(accepts(&src, &constant, &chol, SWEEP_MAX_ELEMS));
    // Variable drift.
    let variable: Drift<2> = Box::new(|x, out| *out = [0.1 * x[0], 0.0]);
    assert!(!accepts(&src, &variable, &chol, SWEEP_MAX_ELEMS));
    // `-0.0` vs `+0.0` drift is not "the same bits" either.
    let signed_zero: Drift<2> =
        Box::new(|x, out| *out = [if x[0] > 0.0 { 0.0 } else { -0.0 }, 0.0]);
    assert!(!accepts(&src, &signed_zero, &chol, SWEEP_MAX_ELEMS));
    // Variable factor: one node differs in the last bit.
    let mut bumped = chol.clone();
    let v = bumped[7].get(1, 0);
    bumped[7].set(1, 0, f64::from_bits(v.to_bits() + 1));
    assert!(!accepts(&src, &constant, &bumped, SWEEP_MAX_ELEMS));
    // Scratch cap: refused below 3·N, accepted at exactly 3·N.
    assert!(!accepts(&src, &constant, &chol, 3 * total - 1));
    assert!(accepts(&src, &constant, &chol, 3 * total));
}

#[test]
fn same_bits_only_for_primitive_floats() {
    assert!(same_bits(&1.5_f64, &1.5_f64) && same_bits(&1.5_f32, &1.5_f32));
    assert!(!same_bits(&0.0_f64, &-0.0_f64));
    let d = Dual::<f64>::new(1.0, 0.0);
    assert!(
        !same_bits(&d, &d),
        "Dual has no bitwise test: path disabled"
    );
    assert!(<Dual<f64> as SemiflowFloat>::libm_exp(d).value > 0.0);
}
