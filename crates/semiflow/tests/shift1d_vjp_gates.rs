//! `G_SHIFT1D_*` — coefficient-field gradients for `ShiftChernoff1D` (ADR-0197).
//!
//! Three gates, in increasing cost and decreasing sharpness:
//!
//! * `G_SHIFT1D_WEIGHTS_ORACLE` — the measured interpolation weight rows
//!   reproduce the sampler exactly. This is what de-risks the whole feature: it
//!   verifies the compact-support assumption exhaustively across boundary
//!   policies and interpolation kinds, at machine precision, for ~30 lines.
//! * `G_SHIFT1D_TRANSPOSE_ID` — `⟨Sᵀλ, u⟩ == ⟨λ, S u⟩` at machine precision.
//!   Catches a sign or fold error in the scatter that a gradient check would
//!   only see as a small bias.
//! * `G_SHIFT1D_COEFF_FD` — the gradient against central finite differences,
//!   per field.

// Test-local index arithmetic and short math names (`a`/`b`/`c` are the
// coefficient fields the module under test names them).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use semiflow::{
    shift1d_vjp::{
        shift1d_coeff_gradient, shift1d_forward, Shift1DProblem, ShiftCoeffField, ShiftCoeffs,
    },
    BoundaryPolicy, Grid1D, InterpKind,
};

const XMIN: f64 = -2.0;
const XMAX: f64 = 2.0;

fn policies() -> [BoundaryPolicy<f64>; 4] {
    [
        BoundaryPolicy::Reflect,
        BoundaryPolicy::ZeroExtend,
        BoundaryPolicy::Periodic,
        BoundaryPolicy::LinearExtrapolate,
    ]
}

fn smooth(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((n - 1) as f64);
            (seed * x).sin() + 0.5 * (0.7 * x).cos() + 1.0
        })
        .collect()
}

fn coeffs(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..n)
        .map(|i| 0.25 + 0.15 * ((i as f64) * 0.31).sin().abs())
        .collect();
    let b: Vec<f64> = (0..n).map(|i| 0.10 * ((i as f64) * 0.17).sin()).collect();
    let c: Vec<f64> = (0..n).map(|i| -0.05 * ((i as f64) * 0.23).cos()).collect();
    (a, b, c)
}

/// `G_SHIFT1D_WEIGHTS_ORACLE` — the measured rows reproduce the sampler.
///
/// The gradient path never hand-transposes an interpolation stencil; it measures
/// the weight row by probing the sampler with unit basis vectors. The assumption
/// that carries risk is *compact support* — that the probe set covers every node
/// the sampler can touch, including boundary folds. This gate falsifies that
/// directly: if any node were missed, `Σ_j w_j·u_j` would differ from
/// `sample(u, y)`.
#[test]
fn g_shift1d_weights_oracle() {
    for kind in [
        InterpKind::CubicHermite,
        InterpKind::SepticHermite,
        InterpKind::OctonicHermite,
    ] {
        for policy in policies() {
            for n in [16_usize, 33] {
                let grid = Grid1D::new(XMIN, XMAX, n)
                    .unwrap()
                    .with_boundary(policy)
                    .with_interp(kind);
                let u = smooth(n, 1.3);
                let dx = grid.dx();
                // Sweep well outside the domain in both directions so the
                // boundary folds are exercised, not just the interior.
                for k in -20..=(20 + 20 * (n as i32)) {
                    let y = XMIN - 5.0 * dx + (k as f64) * dx * 0.37;
                    let direct = grid.interp(&u, y).unwrap();
                    let via = semiflow::shift1d_vjp::weight_row_dot(&grid, y, &u).unwrap();
                    assert!(
                        (direct - via).abs() <= 1e-12 * direct.abs().max(1.0),
                        "kind={kind:?} policy={policy:?} n={n} y={y:.4}: \
                         sampler={direct:.17e} weights={via:.17e}"
                    );
                }
            }
        }
    }
}

/// `G_SHIFT1D_TRANSPOSE_ID` — `⟨Sᵀλ, u⟩ == ⟨λ, Su⟩`.
///
/// Non-vacuity: `S` here has variable `a`, `b`, `c`, so it is genuinely
/// non-self-adjoint — the test separately asserts `Sᵀλ ≠ Sλ`.
#[test]
fn g_shift1d_transpose_id() {
    for policy in policies() {
        let n = 48;
        let grid = Grid1D::new(XMIN, XMAX, n).unwrap().with_boundary(policy);
        let (a, b, c) = coeffs(n);
        let co = ShiftCoeffs {
            a: &a,
            b: &b,
            c: &c,
        };
        let u = smooth(n, 1.1);
        let lam = smooth(n, 2.3);
        let tau = 0.01;

        let su = semiflow::shift1d_vjp::forward_once(&grid, &co, tau, &u).unwrap();
        let stl = semiflow::shift1d_vjp::adjoint_once(&grid, &co, tau, &lam).unwrap();
        let lhs: f64 = stl.iter().zip(u.iter()).map(|(p, q)| p * q).sum();
        let rhs: f64 = lam.iter().zip(su.iter()).map(|(p, q)| p * q).sum();
        assert!(
            (lhs - rhs).abs() <= 1e-11 * lhs.abs().max(1.0),
            "policy={policy:?}: <S^T l, u>={lhs:.17e} != <l, S u>={rhs:.17e}"
        );

        let sl = semiflow::shift1d_vjp::forward_once(&grid, &co, tau, &lam).unwrap();
        assert!(
            stl.iter().zip(sl.iter()).any(|(p, q)| (p - q).abs() > 1e-9),
            "policy={policy:?}: S is self-adjoint on this datum — the gate is vacuous"
        );
    }
}

/// The three coefficient fields, borrowed.
#[derive(Clone, Copy)]
struct Fields<'a> {
    a: &'a [f64],
    b: &'a [f64],
    c: &'a [f64],
}

/// `J` as a function of the three coefficient fields.
type LossFn<'a> = dyn Fn(&[f64], &[f64], &[f64]) -> f64 + 'a;

/// Largest absolute component-wise difference between two gradients.
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Central-difference gradient of `loss` w.r.t. every node of one field.
///
/// `eps = 1e-6` sits between the `O(eps²)` truncation term and the
/// `O(machine-eps · |J| / eps)` roundoff term for a loss of order 1.
fn fd_gradient(field: ShiftCoeffField, fields: Fields<'_>, loss: &LossFn<'_>) -> Vec<f64> {
    let (a, b, c) = (fields.a, fields.b, fields.c);
    let eps = 1e-6;
    let mut out = vec![0.0_f64; a.len()];
    for (i, slot) in out.iter_mut().enumerate() {
        let (mut ap, mut bp, mut cp) = (a.to_vec(), b.to_vec(), c.to_vec());
        let (mut am, mut bm, mut cm) = (a.to_vec(), b.to_vec(), c.to_vec());
        match field {
            ShiftCoeffField::A => (ap[i], am[i]) = (ap[i] + eps, am[i] - eps),
            ShiftCoeffField::B => (bp[i], bm[i]) = (bp[i] + eps, bm[i] - eps),
            ShiftCoeffField::C => (cp[i], cm[i]) = (cp[i] + eps, cm[i] - eps),
        }
        *slot = (loss(&ap, &bp, &cp) - loss(&am, &bm, &cm)) / (2.0 * eps);
    }
    out
}

/// `G_SHIFT1D_COEFF_FD` — gradient vs central finite differences, per field.
///
/// `J(u_n) = ½‖u_n‖²`, so the cotangent is `∂J/∂u_n = u_n` — supplied by the
/// caller, as `edge_weight_grad` requires.
#[test]
#[ignore = "G_SHIFT1D_COEFF_FD — 3 fields x n parameters x 2 solves; Pattern B slow gate"]
fn g_shift1d_coeff_fd() {
    let n = 24;
    let grid = Grid1D::new(XMIN, XMAX, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect);
    let (a, b, c) = coeffs(n);
    let u0 = smooth(n, 1.7);
    let (tau, n_steps) = (0.004, 6);

    let loss = |a: &[f64], b: &[f64], c: &[f64]| -> f64 {
        let p = Shift1DProblem {
            grid: &grid,
            coeffs: ShiftCoeffs { a, b, c },
            tau,
            n_steps,
        };
        let un = shift1d_forward(&p, &u0).unwrap();
        0.5 * un.iter().map(|v| v * v).sum::<f64>()
    };

    let fields = Fields {
        a: &a,
        b: &b,
        c: &c,
    };
    for field in [ShiftCoeffField::A, ShiftCoeffField::B, ShiftCoeffField::C] {
        let p = Shift1DProblem {
            grid: &grid,
            coeffs: ShiftCoeffs {
                a: &a,
                b: &b,
                c: &c,
            },
            tau,
            n_steps,
        };
        let un = shift1d_forward(&p, &u0).unwrap();
        let mut grad = vec![0.0_f64; n];
        shift1d_coeff_gradient(&p, field, &u0, &un, &mut grad).unwrap();

        // Vector comparison, not per-component relative error: components of a
        // gradient can legitimately be ~0, where a per-component ratio is
        // dominated by the FD floor rather than by any error in the adjoint.
        let fd_grad = fd_gradient(field, fields, &loss);
        let scale = fd_grad.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            scale > 1e-6,
            "field={field:?}: FD gradient is identically zero"
        );
        let worst = max_abs_diff(&grad, &fd_grad) / scale;
        assert!(
            worst <= 1e-6,
            "field={field:?}: max|adjoint - fd| / max|fd| = {worst:.3e}\n adjoint={grad:?}\n      fd={fd_grad:?}"
        );
    }
}

/// `∂/∂a` is refused where it is undefined.
#[test]
fn shift1d_vjp_rejects_degenerate_a() {
    let n = 16;
    let grid = Grid1D::new(XMIN, XMAX, n).unwrap();
    let (mut a, b, c) = coeffs(n);
    a[4] = 0.0;
    let p = Shift1DProblem {
        grid: &grid,
        coeffs: ShiftCoeffs {
            a: &a,
            b: &b,
            c: &c,
        },
        tau: 0.01,
        n_steps: 2,
    };
    let u0 = smooth(n, 1.0);
    let mut grad = vec![0.0_f64; n];
    assert!(
        shift1d_coeff_gradient(&p, ShiftCoeffField::A, &u0, &u0, &mut grad).is_err(),
        "d/da must be refused at a_i = 0 (the sqrt(tau/a) chain factor diverges)"
    );
    // b and c carry no such factor and stay available.
    assert!(shift1d_coeff_gradient(&p, ShiftCoeffField::C, &u0, &u0, &mut grad).is_ok());
}
