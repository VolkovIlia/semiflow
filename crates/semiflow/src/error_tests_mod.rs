// Unit tests for `error.rs`.
//
// Covers Display formatting and Clone/Debug for every variant.
// Each test asserts on actual message text so a regression in the formatter fails the test.

use super::*;

// ── Display coverage ──────────────────────────────────────────────────────────

#[test]
fn domain_violation_display() {
    let e = SemiflowError::DomainViolation {
        what: "test invariant",
        value: 3.14,
    };
    let s = e.to_string();
    assert!(s.contains("domain violation"), "got: {s}");
    assert!(s.contains("test invariant"), "got: {s}");
    assert!(s.contains("3.14"), "got: {s}");
}

#[test]
fn grid_underresolved_display() {
    let e = SemiflowError::GridUnderresolved {
        suggested_n: 64,
        shift_dx_ratio: 2.5,
    };
    let s = e.to_string();
    assert!(s.contains("under-resolved"), "got: {s}");
    assert!(s.contains("64"), "got: {s}");
    assert!(s.contains("2.500"), "got: {s}");
}

#[test]
fn convergence_failed_display() {
    let e = SemiflowError::ConvergenceFailed {
        last_residual: 1e-3,
        max_iter: 100,
    };
    let s = e.to_string();
    assert!(s.contains("convergence failed"), "got: {s}");
    assert!(s.contains("100"), "got: {s}");
}

#[test]
fn unsupported_display() {
    let e = SemiflowError::Unsupported {
        feature: "my-feature",
    };
    let s = e.to_string();
    assert!(s.contains("my-feature"), "got: {s}");
}

#[test]
fn cfl_violated_display() {
    let e = SemiflowError::CflViolated {
        tau: 0.1,
        dx_squared: 0.01,
        a_norm_bound: 1.5,
    };
    let s = e.to_string();
    assert!(s.contains("CFL violated"), "got: {s}");
    assert!(s.contains("1.500e0"), "got: {s}");
}

#[test]
fn adaptive_step_rejected_display() {
    let e = SemiflowError::AdaptiveStepRejected {
        last_tau: 1e-5,
        last_err: 2e-3,
        steps_attempted: 50,
    };
    let s = e.to_string();
    assert!(s.contains("adaptive PI"), "got: {s}");
    assert!(s.contains("50"), "got: {s}");
}

#[test]
fn out_of_magnus_radius_display() {
    let e = SemiflowError::OutOfMagnusRadius {
        tau: 0.5,
        rho_estimate: 4.0,
    };
    let s = e.to_string();
    assert!(s.contains("Magnus"), "got: {s}");
    // product = 2.0 >= pi/2
    assert!(s.contains("2.000"), "got: {s}");
}

#[test]
fn unsupported_operation_display() {
    let e = SemiflowError::UnsupportedOperation {
        what: "no transpose",
    };
    let s = e.to_string();
    assert!(s.contains("no transpose"), "got: {s}");
}

#[test]
fn varcoef_out_of_class_display() {
    let e = SemiflowError::VarCoefOutOfClass {
        detail: "a_axis non-positive",
    };
    let s = e.to_string();
    assert!(s.contains("a_axis non-positive"), "got: {s}");
}

// ── Clone + Debug ─────────────────────────────────────────────────────────────

#[test]
fn clone_and_debug_round_trip() {
    let e = SemiflowError::DomainViolation {
        what: "clone_test",
        value: 0.0,
    };
    let cloned = e.clone();
    let debug = alloc::format!("{:?}", cloned);
    assert!(debug.contains("DomainViolation"), "got: {debug}");
}
