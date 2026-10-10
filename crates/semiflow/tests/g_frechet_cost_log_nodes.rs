//! `G_FRECHET_COST_LOG_NODES` (ADVISORY, ADR-0203, §63.6): the Duhamel node count
//! grows like `log(ρ̄t)` and the all-edges gradient costs the same actions as a
//! single edge.
//!
//! Parts (i) (node counts) and (ii) (`spmv_upper ≤ B(ρ̄t)` of §63.6.c, Amendment 4,
//! never looser than the Amendment-1 bound) are exact plan arithmetic and are
//! asserted; part (iii) (wall-time ratio) is
//! printed with `eprintln!` because CI hosts are noisy.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)] // test arithmetic on small indices/counts (< 2^32), never on user data

use std::{collections::HashSet, sync::Arc, time::Instant};

use semiflow::{
    graph_expmv_frechet, graph_expmv_frechet_plan, graph_expmv_matvec_count, EdgeWeightSensitivity,
    Graph, GraphKrylovChernoff, KrylovPath, Laplacian, ScratchPool,
};

/// `K = ⌈log_{1.5}(ρ̄t/4)⌉₊`, `n_nodes = 16(K + 1)` (§63.6).
fn nodes_formula(rho_t: f64) -> u32 {
    let k = (rho_t / 4.0).ln().div_euclid((1.5_f64).ln());
    let mut k = k.max(0.0) as u32;
    while 4.0 * 1.5_f64.powi(k as i32) < rho_t {
        k += 1;
    }
    16 * (k + 1)
}

fn part_i_node_counts() {
    let path = KrylovPath::Chebyshev;
    let expected = [(1.0, 16), (10.0, 64), (1e2, 144), (1e4, 336), (1e6, 512)];
    for (rho_t, nodes) in expected {
        let plan = graph_expmv_frechet_plan(rho_t, 1.0_f64, 1e-12, &path);
        assert_eq!(plan.n_nodes, nodes, "plan at rho_bar t = {rho_t}");
        assert_eq!(
            nodes_formula(rho_t),
            nodes,
            "closed form at rho_bar t = {rho_t}"
        );
    }
    let hi = graph_expmv_frechet_plan(1e6, 1.0_f64, 1e-12, &path).n_nodes;
    let lo = graph_expmv_frechet_plan(1e2, 1.0_f64, 1e-12, &path).n_nodes;
    assert!(hi <= 4 * lo, "n_nodes(1e6)/n_nodes(1e2) = {hi}/{lo} > 4");
}

/// `B(ρ̄t) = √(6L·N·ρ̄t) + N·2L/3` of §63.6.c (Amendment 4), `N = 2 + 32(K+1)`
/// calls, `L = ln(8/tol)`: each call costs `m(z) ≤ 2L/3 + √(2Lz)` (Bennett tail of
/// the Skellam law, gate `G_CHEB_SQRT_COST`) and `Σ√zᵢ ≤ √(N·Σzᵢ)`, `Σzᵢ ≤ 3ρ̄t`.
fn spmv_bound(rho_t: f64, panels: u32, tol: f64) -> f64 {
    let ln8 = (8.0 / tol).ln();
    let calls = 2.0 + 32.0 * f64::from(panels);
    (6.0 * ln8 * calls * rho_t).sqrt() + calls * 2.0 * ln8 / 3.0
}

/// Amendment-1 bound with `Z_SAFE = 200`, `m_Z = 101` (the substep kernel).
fn legacy_spmv_bound(rho_t: f64, panels: u32) -> f64 {
    101.0 * (3.0 * rho_t / 200.0 + 2.0 + 32.0 * f64::from(panels))
}

fn part_ii_spmv_bound() {
    let path = KrylovPath::Chebyshev;
    let tol = 1e-12;
    for rho_t in [1e3, 1e4, 1e5, 1e6] {
        let plan = graph_expmv_frechet_plan(rho_t, 1.0_f64, tol, &path);
        let (s, m) = graph_expmv_matvec_count(rho_t, 1.0_f64, tol, &path);
        let c_t = (u64::from(s) * u64::from(m)) as f64;
        let bound = spmv_bound(rho_t, plan.panels_per_half, tol);
        let legacy = legacy_spmv_bound(rho_t, plan.panels_per_half);
        let upper = plan.spmv_upper as f64;
        eprintln!(
            "ADVISORY (ii) rho_t={rho_t:e} spmv_upper={upper:e} B={bound:e} \
             spmv_upper/C(t)={:.2} B/C(t)={:.2} B/B_legacy={:.3}",
            upper / c_t,
            bound / c_t,
            bound / legacy
        );
        assert!(
            upper <= bound,
            "spmv_upper {upper:e} > B {bound:e} at rho_t = {rho_t:e}"
        );
        assert!(
            bound <= legacy,
            "re-derived B {bound:e} looser than Amendment-1 B {legacy:e}"
        );
    }
}

fn random_graph(n: usize) -> (Arc<Graph<f64>>, Vec<(usize, usize)>) {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB) >> 11
    };
    let mut seen = HashSet::new();
    let mut edges = Vec::new();
    for i in 0..n {
        let mut targets = vec![(i + 1) % n];
        targets.extend((0..2).map(|_| next() as usize % n));
        for j in targets {
            let (a, b) = (i.min(j), i.max(j));
            if a != b && seen.insert((a, b)) {
                edges.push((a as u32, b as u32, 0.5 + (next() % 1000) as f64 / 1000.0));
            }
        }
    }
    let pairs = edges
        .iter()
        .map(|&(a, b, _)| (a as usize, b as usize))
        .collect();
    (
        Arc::new(Graph::<f64>::from_edges(n, edges).expect("graph")),
        pairs,
    )
}

fn wall(gk: &GraphKrylovChernoff<f64>, params: Vec<(usize, usize)>, t: f64, n: usize) -> f64 {
    let sens = EdgeWeightSensitivity { params, n_nodes: n };
    let u0: Vec<f64> = (0..n).map(|i| (0.01 * i as f64).sin()).collect();
    let dj: Vec<f64> = (0..n).map(|i| (0.013 * i as f64).cos()).collect();
    let mut grad = vec![0.0; sens.params.len()];
    let mut scratch = ScratchPool::new();
    let start = Instant::now();
    graph_expmv_frechet(gk, &u0, &dj, 1, t, &sens, &mut grad, &mut scratch).expect("frechet");
    start.elapsed().as_secs_f64()
}

fn part_iii_wall_time() {
    let n = 2000;
    let (g, all) = random_graph(n);
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let gk = GraphKrylovChernoff::new(lap, KrylovPath::Chebyshev, 1e-12).expect("gk");
    let t = 1e3 / gk.lambda_max_bound();
    let (t_one, t_all) = (wall(&gk, vec![all[0]], t, n), wall(&gk, all.clone(), t, n));
    let ratio = t_all / t_one;
    eprintln!("ADVISORY (iii) |E|={} one-edge={t_one:.3}s all-edges={t_all:.3}s ratio={ratio:.2} (limit 2)", all.len());
}

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_frechet_cost_log_nodes() {
    part_i_node_counts();
    part_ii_spmv_bound();
    part_iii_wall_time();
}
