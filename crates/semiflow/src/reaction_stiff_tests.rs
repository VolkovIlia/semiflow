//! Unit tests of the ROS2 reaction flow (ADR-0212). The gates live in
//! `tests/g_stiff_kinetics.rs`.

use alloc::{vec, vec::Vec};

use super::*;
use crate::reaction::{FnReaction, Nagumo};

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
