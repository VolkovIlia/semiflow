//! `G_SPDR_LINEAR_COST` (ADVISORY, ADR-0202, math §62.2.a, `slow-tests`).
//!
//! Median of 7 `solve_into` wall times on the tridiagonal path (factor built outside
//! the timer) at `n = 2^16` and `n = 2^20`; expected ratio 16, gate `<= 32`.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::similar_names
)]
#![cfg(feature = "slow-tests")]

use std::time::Instant;

use semiflow::{
    assemble_conservative_csr_1d, boundary::BoundaryPolicy, grid::Grid1D, scratch::ScratchPool,
    ResolventMethod, SpdSolver,
};

fn median_solve_secs(n: usize) -> f64 {
    let grid = Grid1D::new(0.0, 1.0, n).unwrap();
    let k = vec![1.0; n];
    let a = assemble_conservative_csr_1d(grid, &k, None, BoundaryPolicy::Neumann).unwrap();
    let op = a.with_diagonal(&vec![1.0; n]).unwrap();
    let r = op.resolvent(0.0, None, SpdSolver::Auto, 1e-12).unwrap();
    assert_eq!(r.method(), ResolventMethod::Tridiagonal);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];
    let mut sc = ScratchPool::new();
    let mut t: Vec<f64> = (0..7)
        .map(|_| {
            let t0 = Instant::now();
            r.solve_into(&b, &mut x, &mut sc).unwrap();
            t0.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[3]
}

#[test]
fn g_spdr_linear_cost() {
    let (t16, t20) = (median_solve_secs(1 << 16), median_solve_secs(1 << 20));
    let ratio = t20 / t16;
    eprintln!("G_SPDR_LINEAR_COST: t(2^16)={t16:.3e}s t(2^20)={t20:.3e}s ratio={ratio:.2}");
    assert!(ratio <= 32.0, "ratio {ratio:.2} > 32 (advisory)");
}
