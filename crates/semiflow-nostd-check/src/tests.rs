//! Host run of every scenario (`cargo test -p semiflow-nostd-check --release`).

use std::println;

use super::*;

fn run(name: &str) {
    let scenario = scenarios()
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no scenario named {name}"));
    match (scenario.run)() {
        Ok(o) => println!("PASS {name} err={:e} tol={:e}", o.err, o.tol),
        Err(e) => panic!("FAIL {name} {e}"),
    }
}

#[test]
fn heat_shift1d() {
    run("heat_shift1d");
}

#[test]
fn heat_diffusion() {
    run("heat_diffusion");
}

#[test]
fn strang_heat_decay() {
    run("strang_heat_decay");
}

#[test]
fn truncated_exp_heat() {
    run("truncated_exp_heat");
}

#[test]
fn chebyshev_bc_reflect() {
    run("chebyshev_bc_reflect");
}

#[test]
fn chebyshev_bc_periodic() {
    run("chebyshev_bc_periodic");
}

#[test]
fn schrodinger_unitarity() {
    run("schrodinger_unitarity");
}

#[test]
fn expmv_div_form() {
    run("expmv_div_form");
}

#[test]
fn graph_krylov_chebyshev() {
    run("graph_krylov_chebyshev");
}

#[test]
fn symop_implicit_euler_pcg() {
    run("symop_implicit_euler_pcg");
}

#[test]
fn subordinated_gamma() {
    run("subordinated_gamma");
}

#[test]
fn gridless_moments() {
    run("gridless_moments");
}

#[test]
fn smolyak_heat_d2() {
    run("smolyak_heat_d2");
}

#[test]
fn reverse_ad_sqrt_n() {
    run("reverse_ad_sqrt_n");
}

#[test]
fn run_all_passes_and_names_are_unique() {
    let mut seen = std::vec::Vec::new();
    let summary = run_all(|s, r| {
        assert!(!seen.contains(&s.name), "duplicate scenario {}", s.name);
        seen.push(s.name);
        assert!(r.is_ok(), "{} failed: {}", s.name, r.as_ref().unwrap_err());
    });
    assert_eq!(summary.passed, scenarios().len());
    assert!(summary.all_passed());
}

#[test]
fn check_rejects_nan_and_excess() {
    assert!(check(f64::NAN, 1.0).is_err());
    assert!(check(2.0, 1.0).is_err());
    assert!(check(1.0, 1.0).is_ok());
    assert!(sup_diff(&[0.0, f64::NAN], &[0.0, 0.0]).is_nan());
    assert!(sup_diff(&[0.0], &[0.0, 0.0]).is_nan());
}
