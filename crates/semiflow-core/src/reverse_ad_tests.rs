// sqrt(n_steps).ceil().max(1.0) as usize: result is always >= 1.0 so non-negative cast is safe.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use super::*;
use crate::{DiffusionChernoff, Dual, Grid1D, GridFn1D, InterpKind};

// Small grid — fast, deterministic.
const N_GRID: usize = 24;
const X_MIN: f64 = -4.0;
const X_MAX: f64 = 4.0;
const THETA: f64 = 0.4;
const TAU: f64 = 0.05;
const N_STEPS: usize = 8;
// Loose tolerance for unit tests — tight gate lives in g_reverse_ad.rs.
const FD_TOL: f64 = 1e-5;

fn a_const_dual(x: Dual<f64>) -> Dual<f64> {
    let _ = x;
    Dual::variable(THETA)
}
fn zero_dual(x: Dual<f64>) -> Dual<f64> {
    let _ = x;
    Dual::constant(0.0)
}

fn make_f64_kernel() -> (DiffusionChernoff<f64>, Grid1D<f64>) {
    let grid = Grid1D::new(X_MIN, X_MAX, N_GRID)
        .expect("grid ok")
        .with_interp(InterpKind::CubicHermite);
    let k = DiffusionChernoff::with_closure(|_| THETA, |_| 0.0_f64, |_| 0.0_f64, THETA, grid);
    (k, grid)
}

fn make_dual_kernel() -> DiffusionChernoff<Dual<f64>> {
    let grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("grid ok")
            .with_interp(InterpKind::CubicHermite);
    DiffusionChernoff::<Dual<f64>>::new(
        a_const_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA,
        grid,
    )
}

/// Central-FD helper: evaluate J(θ) = ‖(`F_θ(τ))ⁿ` u₀‖².
fn eval_loss_theta(theta: f64, grid: Grid1D<f64>, u0: &GridFn1D<f64>) -> f64 {
    let k = DiffusionChernoff::with_closure(move |_| theta, |_| 0.0_f64, |_| 0.0_f64, theta, grid);
    let mut u = u0.clone();
    for _ in 0..N_STEPS {
        u = k.apply_f(TAU, &u).expect("step");
    }
    u.values.iter().map(|&v| v * v).sum()
}

/// Unit test: reverse grad matches central-FD within loose tolerance.
#[test]
fn test_value_and_grad_k1_matches_fd() {
    let (kernel, grid) = make_f64_kernel();
    let dual_kernel = make_dual_kernel();
    let sched = CheckpointSchedule::sqrt_n(N_STEPS);
    let rc = ReverseChernoff::new(kernel, dual_kernel, sched);

    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);

    let (_, grad_rev) = rc
        .value_and_grad_k1(TAU, N_STEPS, &u0, &target)
        .expect("reverse AD");

    let h = 1e-4_f64;
    let lp = eval_loss_theta(THETA + h, grid, &u0);
    let lm = eval_loss_theta(THETA - h, grid, &u0);
    let grad_fd = (lp - lm) / (2.0 * h);

    let rel = (grad_rev - grad_fd).abs() / (grad_fd.abs() + 1e-30);
    assert!(
        rel < FD_TOL,
        "reverse {grad_rev:.6e} vs FD {grad_fd:.6e}, rel={rel:.3e} > {FD_TOL:.0e}"
    );
}

/// Unit test: `TransposeApply` on constant-a kernel is self-adjoint (F^⊤ = F).
/// Round-trip F then F^⊤ gives ‖F^⊤ F u − u‖ = O(τ²) for small τ.
#[test]
// u_fwd/u_bwd are standard forward/backward names for adjoint tests.
#[allow(clippy::similar_names)]
fn test_transpose_self_adjoint_const_a() {
    let (kernel, grid) = make_f64_kernel();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let u_fwd = kernel.apply_f(TAU, &u0).expect("fwd");
    let u_bwd = kernel.apply_transpose_step(TAU, &u_fwd).expect("bwd");
    let dx = grid.dx();
    let residual: f64 = u_bwd
        .values
        .iter()
        .zip(u0.values.iter())
        .map(|(a, b)| (a - b) * (a - b) * dx)
        .sum::<f64>()
        .sqrt();
    // Expect O(τ²) = O(0.0025) residual — well under 0.1.
    assert!(
        residual < 0.1,
        "round-trip residual {residual:.3e} too large (τ={TAU})"
    );
}

/// Unit test: recomputed segment states match full-forward states bit-exactly.
#[test]
fn test_checkpoint_reconstruction_bit_exact() {
    let (kernel, grid) = make_f64_kernel();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let sched = CheckpointSchedule::sqrt_n(N_STEPS);

    // Reference states from independent forward sweep.
    let mut ref_states: Vec<GridFn1D<f64>> = Vec::with_capacity(N_STEPS + 1);
    let mut u = u0.clone();
    ref_states.push(u.clone());
    for _ in 0..N_STEPS {
        u = kernel.apply_f(TAU, &u).expect("ref fwd");
        ref_states.push(u.clone());
    }

    let fwd = |t: f64, u: &GridFn1D<f64>| kernel.apply_f(t, u);
    let (_, checkpoints) =
        forward_with_checkpoints(&fwd, TAU, &u0, N_STEPS, &sched).expect("ck fwd");

    for k in 1..=N_STEPS {
        let ck_base = ((k - 1) / sched.stride) * sched.stride;
        let ck_idx = ck_base / sched.stride;
        let seg =
            recompute_segment(&fwd, TAU, &checkpoints[ck_idx], ck_base, k - 1).expect("recompute");
        let recomp = seg.last().expect("non-empty");
        let reference = &ref_states[k - 1];
        for (i, (a, b)) in recomp
            .values
            .iter()
            .zip(reference.values.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "checkpoint recompute NOT bit-exact at k={k}, node={i}"
            );
        }
    }
}

/// Unit test: `CheckpointSchedule::sqrt_n` gives correct stride and count.
#[test]
fn test_checkpoint_schedule() {
    let s64 = CheckpointSchedule::sqrt_n(64);
    assert_eq!(s64.stride, 8); // ceil(sqrt(64)) = 8
    assert_eq!(s64.checkpoint_count(), 8); // (64-1)/8 + 1 = 8

    let s256 = CheckpointSchedule::sqrt_n(256);
    assert_eq!(s256.stride, 16); // ceil(sqrt(256)) = 16

    let s1 = CheckpointSchedule::sqrt_n(1);
    assert_eq!(s1.stride, 1);
    assert_eq!(s1.checkpoint_count(), 1);
}
