//! `G_SPDR_PHI1_LIMIT` (`RELEASE_BLOCKING`, ADR-0202, math §62.1.c).
//!
//! `SpdResolvent(λ=0).solve(s)` vs `τ·φ₁(τG)·M⁻¹s` at `τ = 40/λ_min` (polynomial path
//! via the Wave-1 `CsrGenerator`). `A + diag(c)` with uniform `c` and uniform mass
//! `m`: the constant mode is the lowest eigenvector of the Neumann carrier, so
//! `λ_min(M⁻¹(A + cI)) = c/m` exactly.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::similar_names
)]

use semiflow::{generator_action::CsrGenerator, phi_action, scratch::ScratchPool, SpdSolver};

mod spdr_common;
use spdr_common::{lcg, op_1d, rel_sup_err};

const N: usize = 12;

#[test]
fn g_spdr_phi1_limit() {
    let (c, m) = (2.0_f64, 1.25_f64);
    let op = op_1d(N).with_diagonal(&vec![c; N]).unwrap();
    let mass = vec![m; N];
    let s = lcg(11, N);
    let r = op
        .resolvent(0.0, Some(&mass), SpdSolver::Auto, 1e-12)
        .unwrap();
    let mut x = vec![0.0; N];
    r.solve_into(&s, &mut x, &mut ScratchPool::new()).unwrap();
    // τ·φ₁(τG)·M⁻¹ s,  G = −M⁻¹ A.
    let tau = 40.0 / (c / m);
    let gen = CsrGenerator::from_symmetric(&op, Some(&mass)).unwrap();
    let rhs: Vec<f64> = s.iter().map(|v| v / m).collect();
    let mut y = vec![0.0; N];
    phi_action(&gen, 1, tau, &rhs, &mut y, &mut ScratchPool::new()).unwrap();
    let got: Vec<f64> = y.iter().map(|v| tau * v).collect();
    let err = rel_sup_err(&got, &x);
    eprintln!("G_SPDR_PHI1_LIMIT: tau={tau:.3} rel sup diff = {err:.3e}");
    assert!(err <= 1e-10, "{err:.3e} > 1e-10");
}
