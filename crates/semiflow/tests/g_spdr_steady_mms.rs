//! `G_SPDR_STEADY_MMS` (`RELEASE_BLOCKING`, ADR-0202 + Amendment 1, math §62.3).
//!
//! `u = 2 + cos(πx)`, `k = 1 + x²`, `c = 1 + x` on `[0,1]` (Neumann); FV system
//! `(A + diag(c⊙m)) u = m⊙s`, `m = [½,1,…,1,½]`, solved by `SpdResolvent(λ = 0)`.
//! Oracle: `scripts/verify_spdr_mms.py`. It derives, without any discrete solve, the
//! leading global error `u_h − u = h²w`, where `w` solves the continuous error
//! equation driven by the face-flux truncation and the half-cell boundary defect
//! `k w'(1) = −π²/2`. That gives the a-priori constant `C* = max|w| = 3.506917`.
//! Its independent Thomas solve gives the errors pinned below (2 %).
//!
//! Gate (Amendment 1, Gate-Change-Approved-By: ai-solutions-architect): slope band,
//! `|err(n)·(n−1)²/C* − 1| ≤ 2 %` for every n, and `err(513) ≤ 1.365e-5 = 1.02·C*·h²`.
//! The original `err(513) ≤ 1e-5` was set before implementation from the interior
//! truncation alone (`C = 1.565`). It left out the O(h) half-cell defect at x = 1,
//! where `k'(1) ≠ 0`.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::similar_names
)]

use std::f64::consts::PI;

use semiflow::{
    assemble_conservative_csr_1d, boundary::BoundaryPolicy, grid::Grid1D, scratch::ScratchPool,
    SpdSolver,
};

mod spdr_common;
use spdr_common::half_cell_mass;

/// A-priori leading error constant `C* = max|w|` from `scripts/verify_spdr_mms.py` (Part 2).
const C_STAR: f64 = 3.506_917;

/// `err(513)` bound: `1.02·C*/512²`. The 2 % covers the O(h²)-relative remainder,
/// which is measured at 9e-7 at n = 513.
const ERR_513_MAX: f64 = 1.365e-5;

/// `(n, err)` pinned from `scripts/verify_spdr_mms.py` (Part 3).
const ORACLE: [(usize, f64); 4] = [
    (65, 8.562_246e-4),
    (129, 2.140_480e-4),
    (257, 5.351_148e-5),
    (513, 1.337_784e-5),
];

fn source(x: f64) -> f64 {
    2.0 * PI * x * (PI * x).sin()
        + (x + 1.0) * ((PI * x).cos() + 2.0)
        + PI * PI * (x * x + 1.0) * (PI * x).cos()
}

fn max_err(n: usize) -> f64 {
    let grid = Grid1D::new(0.0, 1.0, n).unwrap();
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let k: Vec<f64> = xs.iter().map(|x| 1.0 + x * x).collect();
    let a = assemble_conservative_csr_1d(grid, &k, None, BoundaryPolicy::Neumann).unwrap();
    let m = half_cell_mass(n);
    let cm: Vec<f64> = xs.iter().zip(&m).map(|(x, mi)| (1.0 + x) * mi).collect();
    let op = a.with_diagonal(&cm).unwrap();
    let rhs: Vec<f64> = xs.iter().zip(&m).map(|(&x, mi)| mi * source(x)).collect();
    let r = op.resolvent(0.0, None, SpdSolver::Auto, 1e-12).unwrap();
    let mut u = vec![0.0; n];
    r.solve_into(&rhs, &mut u, &mut ScratchPool::new()).unwrap();
    assert!(
        u.iter().all(|v| v.is_finite()),
        "max_err: non-finite solution (f64::max would swallow NaN)"
    );
    xs.iter()
        .zip(&u)
        .map(|(&x, ui)| (ui - (2.0 + (PI * x).cos())).abs())
        .fold(0.0, f64::max)
}

fn ols_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let lx: Vec<f64> = ns.iter().map(|&n| (1.0 / (n - 1) as f64).ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
    let (mx, my) = (lx.iter().sum::<f64>() / 4.0, ly.iter().sum::<f64>() / 4.0);
    let num: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
    num / lx.iter().map(|a| (a - mx).powi(2)).sum::<f64>()
}

#[test]
fn g_spdr_steady_mms() {
    let ns: Vec<usize> = ORACLE.iter().map(|o| o.0).collect();
    let errs: Vec<f64> = ns.iter().map(|&n| max_err(n)).collect();
    for ((n, want), got) in ORACLE.iter().zip(&errs) {
        eprintln!(
            "n={n}: err={got:.6e} (oracle {want:.6e}) C={:.4}",
            got * ((n - 1) as f64).powi(2)
        );
        assert!(
            (got - want).abs() <= 0.02 * want,
            "n={n}: {got:.6e} vs oracle {want:.6e}"
        );
        let c_n = got * ((n - 1) as f64).powi(2);
        assert!(
            (c_n / C_STAR - 1.0).abs() <= 0.02,
            "n={n}: error constant {c_n:.4} vs a-priori C* = {C_STAR}"
        );
    }
    assert!(
        errs[3] <= ERR_513_MAX,
        "err(513) = {:.4e} > {ERR_513_MAX:e}",
        errs[3]
    );
    let slope = ols_slope(&ns, &errs);
    eprintln!("G_SPDR_STEADY_MMS: OLS slope = {slope:.4}");
    // dx = 1/(n-1): slope of log err vs log dx is +2, i.e. -2 vs log n (band [-2.2,-1.8]).
    assert!(
        (1.8..=2.2).contains(&slope),
        "slope {slope:.4} outside the order-2 band"
    );
}
