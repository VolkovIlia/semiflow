//! `G_ETD_ADJOINT_FD` (`RELEASE_BLOCKING`): [`NonlinearityDiff`] correctness for
//! [`AllenCahn`] against finite-difference and adjoint identity.
//!
//! Two checks:
//!  1. `jvp` relative error vs central FD ≤ 1e-6.
//!  2. Adjoint identity `⟨vjp(u,w), du⟩ = ⟨w, jvp(u,du)⟩` relative error ≤ 1e-6.

use semiflow::{
    nonlinearity::{Nonlinearity, NonlinearityDiff},
    AllenCahn,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Central-difference approximation of `J_N(u) · du`.
fn fd_jvp(nl: AllenCahn<f64>, u: &[f64], du: &[f64], eps: f64) -> Vec<f64> {
    let n = u.len();
    let u_p: Vec<f64> = u.iter().zip(du).map(|(ui, dui)| ui + eps * dui).collect();
    let u_m: Vec<f64> = u.iter().zip(du).map(|(ui, dui)| ui - eps * dui).collect();
    let mut n_p = vec![0.0; n];
    let mut n_m = vec![0.0; n];
    nl.eval(&u_p, &mut n_p).unwrap();
    nl.eval(&u_m, &mut n_m).unwrap();
    n_p.iter().zip(&n_m).map(|(a, b)| (a - b) / (2.0 * eps)).collect()
}

/// Supremum of `|a[i] − b[i]| / (|b[i]| + ε_floor)`.
fn sup_rel_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / (y.abs() + 1e-15))
        .fold(0.0_f64, f64::max)
}

/// Euclidean inner product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow gate: G_ETD_ADJOINT_FD"]
#[allow(clippy::cast_precision_loss)]
fn g_etd_adjoint_fd() {
    let n = 16_usize;
    let u:  Vec<f64> = (0..n).map(|i| 0.4 * (i as f64 / n as f64)).collect();
    let du: Vec<f64> = (0..n).map(|i| 0.1 * ((i as f64) * 0.7).cos()).collect();
    let w:  Vec<f64> = (0..n).map(|i| 0.2 * ((i as f64) * 1.3).sin()).collect();

    let nl = AllenCahn::<f64>::new();

    // --- JVP vs FD ---
    let mut jvp_out = vec![0.0_f64; n];
    nl.jvp(&u, &du, &mut jvp_out).unwrap();
    let jvp_fd = fd_jvp(nl, &u, &du, 1e-7);
    let rel_err_jvp = sup_rel_err(&jvp_out, &jvp_fd);

    // --- adjoint identity ---
    let mut vjp_out = vec![0.0_f64; n];
    nl.vjp(&u, &w, &mut vjp_out).unwrap();
    let lhs = dot(&vjp_out, &du);
    let rhs = dot(&w, &jvp_out);
    let rel_err_adj = (lhs - rhs).abs() / (rhs.abs() + 1e-15);

    assert!(
        rel_err_jvp <= 1e-6,
        "G_ETD_ADJOINT_FD: JVP rel-err {rel_err_jvp:.2e} > 1e-6",
    );
    assert!(
        rel_err_adj <= 1e-6,
        "G_ETD_ADJOINT_FD: adjoint rel-err {rel_err_adj:.2e} > 1e-6",
    );
}
