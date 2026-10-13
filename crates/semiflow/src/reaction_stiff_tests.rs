//! Unit tests of the ROS2 reaction flow (ADR-0212). The gates live in
//! `tests/g_stiff_kinetics.rs`.

use alloc::{vec, vec::Vec};

use super::*;
use crate::reaction::{Brusselator, FnReaction, GrayScott, Nagumo};

fn flow(
    kin: &dyn Kinetics<f64>,
    opts: &RosenbrockOptions,
    h: f64,
    u: &mut [f64],
) -> Result<usize, SemiflowError> {
    let n = u.len() / kin.species();
    let mut scratch = ScratchPool::new();
    let mut buf = RosBuffers::new(kin.species(), n, &mut scratch);
    let r = ros2_flow(kin, opts, 0.0, h, n, &[], u, &mut buf);
    buf.release(&mut scratch);
    r
}

#[test]
fn options_are_validated() {
    let ok = RosenbrockOptions::default();
    assert!(ok.validate().is_ok());
    for bad in [
        ok.with_rtol(0.0),
        ok.with_rtol(1.0),
        ok.with_rtol(f64::NAN),
        ok.with_atol(-1.0),
        ok.with_atol(f64::INFINITY),
        ok.with_max_steps(0),
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
}

#[test]
fn zero_step_is_the_identity_and_logistic_decay_is_accurate() {
    let kin = Nagumo { a: 0.25 };
    let mut u = vec![0.1, 0.6, 0.9];
    let before = u.clone();
    assert_eq!(
        flow(&kin, &RosenbrockOptions::default(), 0.0, &mut u).unwrap(),
        0
    );
    assert_eq!(u, before);
    // u′ = −u (FnReaction: forward-difference Jacobian) to t = 2.
    let decay = FnReaction::new(1, |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        du[0] = -u[0];
    });
    let mut v = vec![1.0, 2.0];
    let opts = RosenbrockOptions::default()
        .with_rtol(1e-8)
        .with_atol(1e-14);
    flow(&decay, &opts, 2.0, &mut v).unwrap();
    for (got, want) in v.iter().zip([(-2.0_f64).exp(), 2.0 * (-2.0_f64).exp()]) {
        assert!((got - want).abs() <= 1e-7 * want, "{got} vs {want}");
    }
}

#[test]
fn step_cap_is_reported() {
    let stiff = FnReaction::new(1, |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        du[0] = -1e3 * (u[0] - 1.0) * (u[0] + 2.0);
    });
    let mut u: Vec<f64> = vec![0.5; 3];
    let opts = RosenbrockOptions::default()
        .with_rtol(1e-10)
        .with_max_steps(3);
    let r = flow(&stiff, &opts, 10.0, &mut u);
    assert!(
        matches!(r, Err(SemiflowError::ConvergenceFailed { max_iter: 3, .. })),
        "{r:?}"
    );
}

/// `u′ = j·u` with the analytic Jacobian `j`.
struct Linear1(f64);

impl Kinetics<f64> for Linear1 {
    fn species(&self) -> usize {
        1
    }
    fn eval_batch(
        &self,
        _t: f64,
        n: usize,
        _x: &[f64],
        u: &[f64],
        du: &mut [f64],
    ) -> Result<(), SemiflowError> {
        for i in 0..n {
            du[i] = self.0 * u[i];
        }
        Ok(())
    }
    fn jacobian_batch(
        &self,
        _t: f64,
        _n: usize,
        _x: &[f64],
        _u: &[f64],
        jac: &mut [f64],
    ) -> Option<Result<(), SemiflowError>> {
        jac.fill(self.0);
        Some(Ok(()))
    }
}

/// §69.2: a singular `W` or a non-finite stage is a rejection (×0.2), not an error.
#[test]
fn singular_w_and_overflowing_stage_are_rejected_not_errors() {
    let opts = RosenbrockOptions::default().with_atol(1e-12);
    // The first try `hs = h` gives `W = 1 − γhj = 0` exactly.
    let h = 1.0;
    let j = 1.0 / (gamma::<f64>() * h);
    assert_eq!(
        (gamma::<f64>() * h * j).to_bits(),
        1.0_f64.to_bits(),
        "W not singular"
    );
    let mut u = vec![1.0];
    let steps = flow(&Linear1(j), &opts, h, &mut u).unwrap();
    let want = libm::exp(j * h);
    assert!(
        steps > 1 && (u[0] - want).abs() <= 1e-5 * want,
        "{steps} {u:?}"
    );
    // u′ = eᵘ from 0 over h < 1 (the blow-up time): the first stage overflows.
    let blow = FnReaction::new(1, |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        du[0] = libm::exp(u[0]);
    });
    let h = 1.0 / gamma::<f64>() - 4e-4;
    let mut v = vec![0.0];
    let steps = flow(&blow, &opts, h, &mut v).unwrap();
    let want = -libm::log(1.0 - h);
    assert!(
        steps > 1 && (v[0] - want).abs() <= 1e-5 * want,
        "{steps} {v:?}"
    );
}

/// A non-finite `f(t, u)` at an accepted state, or a step that stalls on the
/// half-step clock, fails at once with `DomainViolation` (no retry to the cap).
#[test]
fn non_finite_state_and_stalled_steps_fail_fast() {
    let root = FnReaction::new(1, |_t: f64, _x: &[f64], u: &[f64], du: &mut [f64]| {
        du[0] = -libm::sqrt(u[0]);
    });
    let never = RosenbrockOptions::default().with_max_steps(usize::MAX);
    let mut u = vec![-1e-12, 0.5];
    let r = flow(&root, &never, 1.0, &mut u);
    assert!(
        matches!(r, Err(SemiflowError::DomainViolation { .. })),
        "{r:?}"
    );
    // u = (1 − t/2)² reaches 0 at t = 2; beyond it f is NaN.
    for atol in [1e-9, 0.0] {
        let mut v = vec![1.0];
        let r = flow(&root, &never.with_atol(atol), 3.0, &mut v);
        assert!(
            matches!(r, Err(SemiflowError::DomainViolation { .. })),
            "{atol}: {r:?}"
        );
    }
}

/// `atol = 0` with a species that is exactly 0 and stays 0: zero error, accepted.
#[test]
fn zero_atol_with_a_vanishing_species_is_accepted() {
    let kin = GrayScott {
        feed: 0.04,
        kill: 0.06,
    };
    let run = |atol: f64| {
        let mut u = vec![1.0, 0.5, 1.0, 0.0, 0.25, 0.0];
        flow(
            &kin,
            &RosenbrockOptions::default().with_atol(atol),
            1.0,
            &mut u,
        )
        .unwrap();
        u
    };
    let (zero, tiny) = (run(0.0), run(1e-30));
    assert_eq!(zero, tiny);
    assert!(zero[3] == 0.0 && zero[5] == 0.0, "{zero:?}");
}

/// The forward-difference Jacobian (§69.3 layout, per node) against Brusselator's
/// analytic one at 64 random, non-uniform node states.
#[test]
fn forward_difference_jacobian_matches_the_analytic_one() {
    struct NoJac(Brusselator);
    impl Kinetics<f64> for NoJac {
        fn species(&self) -> usize {
            2
        }
        fn eval_batch(
            &self,
            t: f64,
            n: usize,
            x: &[f64],
            u: &[f64],
            du: &mut [f64],
        ) -> Result<(), SemiflowError> {
            self.0.eval_batch(t, n, x, u, du)
        }
    }
    let (bru, n) = (Brusselator { a: 1.0, b: 3.0 }, 64);
    #[allow(clippy::cast_precision_loss)] // e < 128
    let u: Vec<f64> = (0..2 * n)
        .map(|e| 0.1 + 3.0 * libm::sin(1.7 * e as f64).abs())
        .collect();
    let mut scratch = ScratchPool::new();
    let mut b = RosBuffers::new(2, n, &mut scratch);
    bru.eval_batch(0.0, n, &[], &u, &mut b.f0).unwrap();
    jacobian(&NoJac(bru), 0.0, n, &[], &u, &mut b).unwrap();
    let mut an = vec![0.0; 4 * n];
    bru.jacobian_batch(0.0, n, &[], &u, &mut an)
        .unwrap()
        .unwrap();
    for (e, (&fd, &a)) in b.jac.iter().zip(&an).enumerate() {
        assert!(
            (fd - a).abs() <= 1e-6 * a.abs().max(1.0),
            "entry {e}: {fd} vs {a}"
        );
    }
    b.release(&mut scratch);
}
