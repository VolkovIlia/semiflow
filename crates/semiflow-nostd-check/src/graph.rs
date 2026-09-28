//! Graph / sparse-operator exponentials against the dense Padé reference.

use alloc::{sync::Arc, vec::Vec};

use semiflow::{
    dense_csr_expmv_ref, dense_graph_expmv_ref, graph_expmv_krylov, ChernoffFunction, Graph,
    GraphKrylovChernoff, GraphSignal, KrylovPath, Laplacian, ScratchPool, SymmetricOperator,
};

use crate::{check, m, sup_diff, ScenarioResult};

/// `GraphKrylovChernoff` (Chebyshev path) vs `dense_graph_expmv_ref`.
///
/// `tests/graph_expmv_dense.rs` (`G_GRAPH_EXPMV_DENSE`): `Graph::path(10)`,
/// combinatorial Laplacian, `τ = 1`, solver tolerance `1e-12`, signal
/// `exp(-(i−5)²/2)`, gate `sup_error ≤ 1e-10`.
pub(crate) fn krylov_chebyshev() -> ScenarioResult {
    let g = Arc::new(Graph::<f64>::path(10));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let krylov = GraphKrylovChernoff::new(Arc::clone(&lap), KrylovPath::Chebyshev, 1e-12)?;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = f64::from(i) - 5.0;
        m::exp(-0.5 * x * x)
    });
    let mut dst = GraphSignal::zeros(Arc::clone(&g));
    krylov.apply_into(1.0, &src, &mut dst, &mut ScratchPool::new())?;
    let mut dense = alloc::vec![0.0; 10];
    dense_graph_expmv_ref(&lap, 1.0, src.values(), &mut dense)?;
    check(sup_diff(dst.values(), &dense), 1e-10)
}

/// CSR of the N = 10 Robin 1-D Laplacian of `tests/g_symop_implicit_dense.rs`:
/// off-diagonals `−1`, diagonal `2.5` (ends `2.0`).
fn robin_n10_csr() -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let mut row_ptr = alloc::vec![0_usize];
    let (mut cols, mut vals) = (Vec::new(), Vec::new());
    for i in 0_u32..10 {
        if i > 0 {
            cols.push(i - 1);
            vals.push(-1.0);
        }
        cols.push(i);
        vals.push(if i == 0 || i == 9 { 2.0 } else { 2.5 });
        if i < 9 {
            cols.push(i + 1);
            vals.push(-1.0);
        }
        row_ptr.push(cols.len());
    }
    (row_ptr, cols, vals)
}

/// Implicit-Euler `exp(-τA)v` (PCG solves, `symmetric_operator` + `pcg`) vs
/// the dense Padé reference.
///
/// `tests/g_symop_implicit_dense.rs` (`G_SYMOP_IMPLICIT_DENSE`, gate a):
/// `τ = 0.01`, `n_steps = 200`, `v_i = sin((i+1)π/11)`, `sup_error ≤ 1e-6`.
pub(crate) fn implicit_euler_pcg() -> ScenarioResult {
    let (row_ptr, cols, vals) = robin_n10_csr();
    let op = SymmetricOperator::from_csr(10, &row_ptr, &cols, &vals, 1e-10)?;
    let v: Vec<f64> = (1_u32..=10)
        .map(|i| m::sin(f64::from(i) * core::f64::consts::PI / 11.0))
        .collect();
    let mut exact = alloc::vec![0.0; 10];
    dense_csr_expmv_ref(&op, 0.01, &v, &mut exact)?;
    let path = KrylovPath::ImplicitEuler {
        n_steps: 200,
        cg_max_iter: None,
    };
    let mut approx = alloc::vec![0.0; 10];
    graph_expmv_krylov(
        &op,
        0.01,
        &v,
        &mut approx,
        path,
        1e-12,
        &mut ScratchPool::new(),
    )?;
    check(sup_diff(&approx, &exact), 1e-6)
}
