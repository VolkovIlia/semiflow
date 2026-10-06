//! `G_SPDR_IMPLICIT_CROSS` (`RELEASE_BLOCKING`, ADR-0202, math §62.1.a).
//!
//! `(I + ΔtA)⁻¹ v = λ (λI + A)⁻¹ v`, `λ = 1/Δt`: the new resolvent (tridiagonal and a
//! forced Pcg{Jacobi}) vs the ADR-0190 `KrylovPath::ImplicitEuler{n_steps: 1}`.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::similar_names
)]

use semiflow::{graph_expmv_krylov, scratch::ScratchPool, KrylovPath, Precond, SpdSolver};

mod spdr_common;
use spdr_common::{lcg, op_1d, rel_sup_err};

const N: usize = 12;

#[test]
fn g_spdr_implicit_cross() {
    let op = op_1d(N);
    let v = lcg(3, N);
    for dt in [1e-3_f64, 1.0, 1e3] {
        let mut want = vec![0.0; N];
        let path = KrylovPath::ImplicitEuler {
            n_steps: 1,
            // Default cap (16 at n = 12) is too small for S = I + 1e3·A (κ ≈ 1e6): the
            // ADR-0190 path itself returns ConvergenceFailed there, so the cap is raised.
            cg_max_iter: Some(20 * N),
        };
        graph_expmv_krylov(&op, dt, &v, &mut want, path, 1e-13, &mut ScratchPool::new())
            .expect("implicit euler");
        let lam = 1.0 / dt;
        for solver in [
            SpdSolver::Auto,
            SpdSolver::Pcg {
                precond: Precond::Jacobi,
                max_iter: None,
            },
        ] {
            let r = op.resolvent(lam, None, solver, 1e-13).unwrap();
            let mut x = vec![0.0; N];
            r.solve_into(&v, &mut x, &mut ScratchPool::new()).unwrap();
            let got: Vec<f64> = x.iter().map(|xi| lam * xi).collect();
            let err = rel_sup_err(&got, &want);
            eprintln!("dt={dt:e} {:?}: rel diff {err:.3e}", r.method());
            assert!(err <= 1e-10, "dt={dt}: {err:.3e} > 1e-10");
        }
    }
}
