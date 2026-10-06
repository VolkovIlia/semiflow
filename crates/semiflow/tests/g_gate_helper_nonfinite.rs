//! Regression: ADR-0202 gate helpers must reject non-finite data instead of
//! letting `f64::max` swallow NaN (an all-NaN result would read as "exact").
#![allow(dead_code, clippy::float_cmp)]

mod phi_dense;
mod spdr_common;

#[test]
#[should_panic(expected = "non-finite")]
fn phi_sup_diff_rejects_nan() {
    let _ = phi_dense::sup_diff(&[f64::NAN], &[1.0]);
}

#[test]
#[should_panic(expected = "non-finite")]
fn phi_sup_diff_rejects_inf() {
    let _ = phi_dense::sup_diff(&[1.0], &[f64::INFINITY]);
}

#[test]
#[should_panic(expected = "non-finite")]
fn phi_sup_rejects_nan() {
    let _ = phi_dense::sup(&[f64::NAN]);
}

#[test]
#[should_panic(expected = "non-finite")]
fn spdr_sup_rejects_nan() {
    let _ = spdr_common::sup(&[f64::NAN]);
}

#[test]
#[should_panic(expected = "non-finite")]
fn spdr_rel_sup_err_rejects_nan() {
    let _ = spdr_common::rel_sup_err(&[f64::NAN], &[1.0]);
}

#[test]
fn finite_inputs_unchanged() {
    assert_eq!(phi_dense::sup_diff(&[1.0, 2.0], &[1.5, 2.0]), 0.5);
    assert_eq!(phi_dense::sup(&[-3.0, 2.0]), 3.0);
    assert_eq!(spdr_common::sup(&[-3.0, 2.0]), 3.0);
    assert_eq!(spdr_common::rel_sup_err(&[1.0], &[2.0]), 0.5);
}
