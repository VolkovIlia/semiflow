//! Graph / sparse-operator exponentials against the dense Padé reference.

use alloc::{sync::Arc, vec::Vec};

use semiflow::{
    dense_csr_expmv_ref, dense_graph_expmv_ref, graph_expmv_frechet, graph_expmv_frechet_plan,
    graph_expmv_krylov, ChernoffFunction, EdgeWeightSensitivity, Graph, GraphKrylovChernoff,
    GraphSignal, KrylovPath, Laplacian, ScratchPool, SymmetricOperator,
};

use crate::{check, digest_of, m, sup_diff, Failure, ScenarioResult};

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
    check(
        sup_diff(dst.values(), &dense),
        1e-10,
        digest_of(dst.values()),
    )
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
    check(sup_diff(&approx, &exact), 1e-6, digest_of(&approx))
}

/// Number of nodes of the stiff path of [`frechet_large_t`].
const FRECHET_N: usize = 10;

/// Alternating conductances `{1e3, 1, 1e3, ...}` of the `n - 1` path edges.
fn stiff_path_weights() -> Vec<f64> {
    (0..FRECHET_N - 1)
        .map(|k| if k % 2 == 0 { 1.0e3 } else { 1.0 })
        .collect()
}

/// Path graph on `FRECHET_N` nodes with edge `k` = `(k, k + 1)` of weight `weights[k]`.
fn stiff_path(weights: &[f64]) -> Result<Graph<f64>, semiflow::SemiflowError> {
    let edges = (0_u32..).zip(weights).map(|(k, &w)| (k, k + 1, w));
    Graph::<f64>::from_edges(FRECHET_N, edges)
}

/// `J(w) = <dj, e^{-t L(w)} u0>` with the dense Padé reference.
fn frechet_j(
    weights: &[f64],
    t: f64,
    u0: &[f64],
    dj: &[f64],
) -> Result<f64, semiflow::SemiflowError> {
    let g = stiff_path(weights)?;
    let lap = Laplacian::assemble_combinatorial(&g);
    let mut out = alloc::vec![0.0; FRECHET_N];
    dense_graph_expmv_ref(&lap, t, u0, &mut out)?;
    Ok(out.iter().zip(dj).map(|(a, b)| a * b).sum())
}

/// Fréchet gradient at `ρ̄t ≈ 2e3` (math §63, ADR-0203) vs a central difference.
///
/// `G_FRECHET_LARGE_T_NOSTD_DIGEST`: path(10), alternating weights `{1e3, 1}`,
/// `t = 1`, Chebyshev `tol = 1e-12`, `EdgeWeightSensitivity` on all 9 edges, one
/// channel. The graded mesh has `K ≥ 10` panels per half and no decay skip (the
/// zero mode of the combinatorial Laplacian keeps the far vector alive). The
/// directional derivative along `δw = (1, 2, …, 9)` is compared with
/// `[J(w + εδw) − J(w − εδw)] / 2ε`, `ε = 1e-5` (`dense_graph_expmv_ref`);
/// gate `|error| ≤ 1e-6`. The digest covers the 9 gradient entries, so the mesh
/// (no transcendental calls) and the sweep must be bit-identical in every build.
pub(crate) fn frechet_large_t() -> ScenarioResult {
    let t = 1.0;
    let weights = stiff_path_weights();
    let g = stiff_path(&weights)?;
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let gk = GraphKrylovChernoff::new(Arc::clone(&lap), KrylovPath::Chebyshev, 1e-12)?;
    let u0: Vec<f64> = (0_u32..10)
        .map(|i| m::sin(0.7 * f64::from(i) + 0.3))
        .collect();
    let dj: Vec<f64> = (0_u32..10).map(|i| m::cos(0.4 * f64::from(i))).collect();
    let sens = EdgeWeightSensitivity {
        params: (0..FRECHET_N - 1).map(|k| (k, k + 1)).collect(),
        n_nodes: FRECHET_N,
    };
    // The point of the scenario: a deep graded mesh (K >= 10), not one panel.
    let plan = graph_expmv_frechet_plan(gk.lambda_max_bound(), t, gk.tol(), &gk.path());
    if plan.panels_per_half < 11 {
        return Err(Failure::Invariant("mesh has fewer than 11 panels per half"));
    }
    let mut grad = alloc::vec![0.0; FRECHET_N - 1];
    graph_expmv_frechet(
        &gk,
        &u0,
        &dj,
        1,
        t,
        &sens,
        &mut grad,
        &mut ScratchPool::new(),
    )?;
    let dir: Vec<f64> = (1_u32..10).map(f64::from).collect();
    let shifted = |sign: f64| -> Vec<f64> {
        weights
            .iter()
            .zip(&dir)
            .map(|(w, d)| w + sign * 1e-5 * d)
            .collect()
    };
    let fd =
        (frechet_j(&shifted(1.0), t, &u0, &dj)? - frechet_j(&shifted(-1.0), t, &u0, &dj)?) / 2e-5;
    let analytic: f64 = grad.iter().zip(&dir).map(|(g, d)| g * d).sum();
    check(m::abs(analytic - fd), 1e-6, digest_of(&grad))
}
