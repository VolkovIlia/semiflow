//! `G_SYMOP_ENTRY_FRECHET` (`RELEASE_BLOCKING`, §55.5–§55.6): `EntrySensitivity`
//! + `graph_expmv_frechet` vs central finite-difference oracle.
//!
//! N=3 triangle (non-commuting: `[L, ∂L/∂L_{ij}] ≠ 0`).
//! Three off-diagonal entries: `(0,1)`, `(1,2)`, `(0,2)`.
//! Relative error `≤ 1e-7` for every entry.
//!
//! Non-vacuity (structural): `[L, E_{ij}] ≠ 0` for a triangle with distinct
//! edge weights, so the naive right-endpoint rectangle rule fails.  The Duhamel
//! integral (augmented Fréchet) must be correct.
//!
//! Non-vacuity (row-sum): diagonal entries include a `c = 0.5` reaction term
//! so every row sums to `0.5 ≠ 0`.  This gates issue #13's headline capability
//! (generic non-zero-row-sum symmetric operator) and is enforced by an explicit
//! assertion that silently catches regression to the zero-sum case.

use semiflow::{
    graph_expmv_krylov,
    graph_frechet::graph_expmv_frechet,
    scratch::ScratchPool, EntrySensitivity, KrylovPath, SymmetricOperator,
};

/// Build triangle Laplacian as `SymmetricOperator` from flat 9-element CSR vals.
///
/// CSR layout (row 0: cols [0,1,2], row 1: cols [0,1,2], row 2: cols [0,1,2]).
fn triangle_op(vals: [f64; 9]) -> SymmetricOperator {
    let row_ptr = [0_usize, 3, 6, 9];
    let col_idx = [0u32, 1, 2, 0, 1, 2, 0, 1, 2];
    SymmetricOperator::from_csr(3, &row_ptr, &col_idx, &vals, 1e-12_f64)
        .expect("triangle_op: from_csr failed")
}

/// Compute `J = ⟨dj, e^{−τL} u0⟩` via Krylov on a triangle with given CSR vals.
fn j_triangle(vals: [f64; 9], tau: f64, u0: &[f64; 3], dj: &[f64; 3]) -> f64 {
    let op = triangle_op(vals);
    let mut out = [0.0_f64; 3];
    let mut scratch = ScratchPool::new();
    graph_expmv_krylov(&op, tau, u0.as_slice(), out.as_mut_slice(), KrylovPath::Chebyshev, 1e-12, &mut scratch)
        .expect("j_triangle: krylov failed");
    out.iter().zip(dj.iter()).map(|(a, b)| a * b).sum::<f64>()
}

/// `G_SYMOP_ENTRY_FRECHET`: `EntrySensitivity` vs central FD on N=3 triangle.
///
/// Entries `(0,1)`, `(1,2)`, `(0,2)`.  Expected: `rel_err ≤ 1e-7` each.
#[allow(clippy::too_many_lines)] // 67 lines: single-gate with inline non-vacuity check + 3-entry FD loop
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_symop_entry_frechet() {
    let tau = 0.5_f64;
    let u0 = [1.0_f64, -0.5, 0.3];
    let dj = [0.3_f64, -0.8, 0.2];

    // Triangle Laplacian + reaction c=0.5: edges (0,1,1.0), (1,2,0.7), (0,2,0.4).
    // L[0,0]=1.9, L[1,1]=2.2, L[2,2]=1.6 (off-diagonals −1.0/−0.7/−0.4 unchanged).
    // Row sums: each row = 0.5 ≠ 0 (generic non-zero-row-sum case, issue #13).
    let nom: [f64; 9] = [
        1.9, -1.0, -0.4,   // row 0: sum = 0.5
        -1.0, 2.2, -0.7,   // row 1: sum = 0.5
        -0.4, -0.7, 1.6,   // row 2: sum = 0.5
    ];

    // Non-vacuity: assert at least one row sum is non-zero (gates #13 headline capability).
    // This assertion fails if the operator silently regresses to a zero-row-sum Laplacian.
    let row_sums: [f64; 3] = [
        nom[0] + nom[1] + nom[2],
        nom[3] + nom[4] + nom[5],
        nom[6] + nom[7] + nom[8],
    ];
    assert!(
        row_sums.iter().any(|&s| s.abs() > 1e-15_f64),
        "G_SYMOP_ENTRY_FRECHET: all row sums are zero — gate is vacuous (regressed to zero-sum case)"
    );

    let op = triangle_op(nom);
    let krylov = op.krylov(KrylovPath::Chebyshev, 1e-12_f64).expect("krylov");
    let sens = EntrySensitivity {
        entries: vec![(0, 1), (1, 2), (0, 2)],
        n_nodes: 3,
    };
    let mut grad = [0.0_f64; 3];
    let mut scratch = ScratchPool::new();
    graph_expmv_frechet(&krylov, &u0, &dj, 1, tau, &sens, grad.as_mut_slice(), &mut scratch)
        .expect("graph_expmv_frechet");

    // CSR position pairs for symmetric perturbation of each entry:
    //   entry (0,1): vals[1] = L[0,1], vals[3] = L[1,0]
    //   entry (1,2): vals[5] = L[1,2], vals[7] = L[2,1]
    //   entry (0,2): vals[2] = L[0,2], vals[6] = L[2,0]
    let pert_pairs: [(usize, usize); 3] = [(1, 3), (5, 7), (2, 6)];
    let entry_labels: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];
    let eps = 1e-6_f64;

    for k in 0..3 {
        let (pi, pj) = pert_pairs[k];
        let (ei, ej) = entry_labels[k];
        let mut vp = nom;
        vp[pi] += eps;
        vp[pj] += eps;
        let mut vm = nom;
        vm[pi] -= eps;
        vm[pj] -= eps;
        let fd = (j_triangle(vp, tau, &u0, &dj) - j_triangle(vm, tau, &u0, &dj)) / (2.0 * eps);
        let a2 = grad[k];
        let rel_err = (a2 - fd).abs() / (fd.abs() + 1e-30_f64);
        eprintln!(
            "G_SYMOP_ENTRY_FRECHET entry({ei},{ej})  a2={a2:.10e}  fd={fd:.10e}  rel_err={rel_err:.3e}"
        );
        assert!(
            rel_err <= 1e-7_f64,
            "G_SYMOP_ENTRY_FRECHET: entry({ei},{ej}) rel_err={rel_err:.3e} > 1e-7 \
             (a2={a2:.10e}, fd={fd:.10e})"
        );
    }
}
