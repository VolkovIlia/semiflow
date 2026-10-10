//! `G_FRECHET_LARGE_T_ORACLE` (`RELEASE_BLOCKING`, ADR-0203, §63.7, Amendment 1):
//! the Fréchet gradient of `graph_expmv_frechet` against a Daleckii-Krein eigen
//! oracle at `λ_max t ∈ {1, 10, 1e2, 1e4, 1e6}`.
//!
//! Bound per parameter: `|g_k − g_k^ref| ≤ τ_k = (ε_Q + 2n²u)·G_k + η·N_k`
//! with `N_chain` and the rounding weight `W` (Amendment 4) from
//! `graph_expmv_frechet_plan` (never measured).
//!
//! Non-vacuity is PER CARRIER and per point, never pooled (Amendment 1):
//! - F3-edge (clustered network, `W` solved per point): at least 2 of the 3
//!   bridges have `|g_ref| ≥ 1e3·τ_k`;
//! - F2-entry: at least one `k` with `|g_ref| ≥ 1e3·τ_k`;
//! - for each carrier, at `λ_max t ≥ 10` the pre-ADR-0203 one-panel GL8 rule, run
//!   with EXACT eigen propagators, violates the propagator-free `τ_k^Q` for some `k`.
//!
//! F1-edge and F2-edge are accuracy-only: on a connected graph with generic
//! signals every edge gradient decays like `e^{−λ₂t}`, so their vacuity at large
//! `λ_max t` is expected; it is reported (`informative=`), not asserted.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)] // test arithmetic on small indices/counts (< 2^32), never on user data

mod frechet_oracle_common;

use std::{ops::Range, sync::Arc, time::Instant};

use frechet_oracle_common::{
    csr_of, f3_matrix, f3_point, f3_signals, jacobi_eigh, legacy_gl8, max_row_nnz, nan_max, norm2,
    oracle, random_graph, stiff_contrast, test_vectors, violates, BoundInputs, Dense, Eig, Stencil,
    F3_BRIDGES, F3_N,
};
use semiflow::{
    graph_expmv_frechet, graph_expmv_frechet_plan, EdgeWeightSensitivity, EntrySensitivity,
    FrechetPlan, Graph, GraphKrylovChernoff, KrylovPath, Laplacian, ScratchPool, SymmetricOperator,
};

const N: usize = 12;
const TOL: f64 = 1e-14;
const GRID: [f64; 5] = [1.0, 10.0, 1e2, 1e4, 1e6];
/// `W` of fixture F2, chosen so `λ_max/λ_min ∈ [5e5, 2e6]` (asserted below).
const FAST: f64 = 3e3;

/// What a case must demonstrate beyond accuracy (Amendment 1).
struct Carrier {
    /// Parameters that must be informative (`|g_ref| ≥ 1e3·τ_k`).
    focus: Range<usize>,
    /// Minimum number of informative parameters inside `focus`.
    min_informative: usize,
}

struct Case {
    name: &'static str,
    gk: GraphKrylovChernoff<f64>,
    eig: Eig,
    row_nnz: usize,
    stencils: Vec<Stencil>,
    entry_params: bool,
    d: Vec<f64>,
    v: Vec<f64>,
    /// `None`: accuracy-only case.
    carrier: Option<Carrier>,
}

fn grad_of(case: &Case, t: f64) -> Vec<f64> {
    let pairs: Vec<(usize, usize)> = case.stencils.iter().map(Stencil::pair).collect();
    let mut grad = vec![0.0; pairs.len()];
    let mut scratch = ScratchPool::new();
    let (u0, dj) = (&case.v, &case.d);
    let run = if case.entry_params {
        let sens = EntrySensitivity {
            entries: pairs,
            n_nodes: N,
        };
        graph_expmv_frechet(&case.gk, u0, dj, 1, t, &sens, &mut grad, &mut scratch)
    } else {
        let sens = EdgeWeightSensitivity {
            params: pairs,
            n_nodes: N,
        };
        graph_expmv_frechet(&case.gk, u0, dj, 1, t, &sens, &mut grad, &mut scratch)
    };
    run.expect("graph_expmv_frechet");
    grad
}

/// Outcome of one `(case, λ_max t)` point.
struct Point {
    fails: Vec<String>,
}

/// Running sums of one point (the table row).
#[derive(Default)]
struct Row {
    informative: usize,
    focus_informative: usize,
    legacy_violations: usize,
    rel_vs_g: f64,
    legacy_rel: f64,
    err_vs_tau: f64,
    legacy_vs_tau: f64,
}

/// §63.7.a inputs of a case at time `t`, with the plan they come from.
fn bound_for(case: &Case, t: f64) -> (BoundInputs, FrechetPlan) {
    let rho = case.gk.lambda_max_bound();
    let plan = graph_expmv_frechet_plan(rho, t, case.gk.tol(), &case.gk.path());
    let bound = BoundInputs {
        n: N,
        row_nnz: case.row_nnz,
        tol: case.gk.tol(),
        n_chain: plan.n_chain,
        chain_weight: plan.chain_weight,
        rho_t: rho * t,
        t,
    };
    (bound, plan)
}

/// Compare one point's gradient with the oracle and the legacy replica.
///
/// All maxima propagate NaN and every comparison is written `!(x <= y)`, so a
/// NaN anywhere is a violation, never a silent pass.
fn score(
    case: &Case,
    (bound, lt): (&BoundInputs, f64),
    grad: &[f64],
    refs: &[(f64, f64)],
    legacy: &[f64],
) -> (Row, Vec<String>) {
    let (nd, nv) = (norm2(&case.d), norm2(&case.v));
    let (mut row, mut fails) = (Row::default(), Vec::new());
    for (k, st) in case.stencils.iter().enumerate() {
        let (g_ref, big_g) = refs[k];
        let tau = bound.tau(big_g, st.norm2(), nd, nv);
        let err = (grad[k] - g_ref).abs();
        if violates(err, tau) {
            let name = case.name;
            fails.push(format!(
                "{name} lt={lt:e} k={k} err={err:e} > tau={tau:e} (G={big_g:e})"
            ));
        }
        row.err_vs_tau = nan_max(row.err_vs_tau, err / tau);
        let legacy_err = (legacy[k] - g_ref).abs();
        let tau_q = bound.tau_quad(big_g, st.norm2(), nd, nv);
        row.legacy_violations += usize::from(violates(legacy_err, tau_q));
        row.legacy_vs_tau = nan_max(row.legacy_vs_tau, legacy_err / tau);
        if g_ref.abs() >= 1e3 * tau {
            row.informative += 1;
            row.focus_informative +=
                usize::from(case.carrier.as_ref().is_some_and(|c| c.focus.contains(&k)));
            row.rel_vs_g = nan_max(row.rel_vs_g, err / big_g);
            row.legacy_rel = nan_max(row.legacy_rel, legacy_err / big_g);
        }
    }
    (row, fails)
}

/// Per-carrier non-vacuity (a) and (b) of Amendment 1.
fn carrier_checks(case: &Case, lt: f64, row: &Row) -> Vec<String> {
    let Some(carrier) = &case.carrier else {
        return Vec::new();
    };
    let mut fails = Vec::new();
    if row.focus_informative < carrier.min_informative {
        fails.push(format!(
            "{} lt={lt:e}: {} informative parameters in {:?}, need {} (vacuous)",
            case.name, row.focus_informative, carrier.focus, carrier.min_informative
        ));
    }
    if lt >= 10.0 && row.legacy_violations == 0 {
        fails.push(format!(
            "{} lt={lt:e}: legacy rule not rejected (gate vacuous)",
            case.name
        ));
    }
    fails
}

/// One `(case, λ_max t)` point at time `t`: checks `τ_k`, prints a table row.
///
/// The table's relative errors are `max_k |g_k − g_k^ref| / G_k` over the
/// informative parameters only (`|g_ref| ≥ 1e3·τ_k`); a decayed gradient has
/// no meaningful relative error. `legacy/tau` is the legacy rule's error over
/// the FULL `τ_k` (informational).
fn check_point(case: &Case, t: f64, lt: f64) -> Point {
    let refs = oracle(&case.eig, &case.d, &case.v, t, &case.stencils);
    let (bound, plan) = bound_for(case, t);
    let start = Instant::now();
    let grad = grad_of(case, t);
    let wall = start.elapsed();
    let legacy = legacy_gl8(&case.eig, &case.d, &case.v, t, &case.stencils);
    let (row, mut fails) = score(case, (&bound, lt), &grad, &refs, &legacy);
    fails.extend(carrier_checks(case, lt, &row));
    println!(
        "TABLE {:<9} lt={lt:>6.0e} rel_vs_G={:9.2e} legacy_exactprop_rel={:9.2e} \
         max_err/tau={:9.2e} legacy_err/tau={:9.2e} nodes={:4} wall_ms={:9.3} \
         informative={} focus_informative={} legacy_violations={}",
        case.name,
        row.rel_vs_g,
        row.legacy_rel,
        row.err_vs_tau,
        row.legacy_vs_tau,
        plan.n_nodes,
        wall.as_secs_f64() * 1e3,
        row.informative,
        row.focus_informative,
        row.legacy_violations,
    );
    Point { fails }
}

/// Build the CSR-Laplacian solver of a weighted edge list through `Graph`.
fn graph_krylov(edges: &[(usize, usize, f64)]) -> (GraphKrylovChernoff<f64>, usize) {
    let triples = edges.iter().map(|&(i, j, w)| (i as u32, j as u32, w));
    let g = Arc::new(Graph::<f64>::from_edges(N, triples).expect("graph"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let row_nnz = max_row_nnz(lap.row_ptr());
    (
        GraphKrylovChernoff::new(lap, KrylovPath::Chebyshev, TOL).expect("krylov"),
        row_nnz,
    )
}

fn f1_case() -> Case {
    let (mat, edges) = random_graph(N, 1);
    let (gk, row_nnz) = graph_krylov(&edges);
    let (v, d) = test_vectors(N, 11);
    Case {
        name: "F1-edge",
        gk,
        eig: jacobi_eigh(&mat),
        row_nnz,
        stencils: edges.iter().map(|&(i, j, _)| Stencil::Edge(i, j)).collect(),
        entry_params: false,
        d,
        v,
        carrier: None,
    }
}

/// F3-edge at stiffness `w` (Amendment 1): the edge carrier.
fn f3_case(w: f64) -> Case {
    let (mat, edges) = f3_matrix(w);
    let (gk, row_nnz) = graph_krylov(&edges);
    let (v, d) = f3_signals();
    assert_eq!(v.len(), F3_N);
    Case {
        name: "F3-edge",
        gk,
        eig: jacobi_eigh(&mat),
        row_nnz,
        stencils: edges.iter().map(|&(i, j, _)| Stencil::Edge(i, j)).collect(),
        entry_params: false,
        d,
        v,
        carrier: Some(Carrier {
            focus: F3_BRIDGES,
            min_informative: 2,
        }),
    }
}

fn operator_of(mat: &Dense) -> (GraphKrylovChernoff<f64>, usize) {
    let (row_ptr, cols, vals) = csr_of(mat);
    let op = SymmetricOperator::from_csr(N, &row_ptr, &cols, &vals, 1e-10).expect("operator");
    (
        op.krylov(KrylovPath::Chebyshev, TOL).expect("krylov"),
        max_row_nnz(&row_ptr),
    )
}

fn f2_cases() -> (Case, Case) {
    let (mat, edges) = stiff_contrast(N, FAST, 0);
    let eig = jacobi_eigh(&mat);
    let ratio = eig.lam[N - 1] / eig.lam[0];
    assert!(
        (5e5..=2e6).contains(&ratio),
        "F2 stiffness ratio {ratio:e} outside [5e5, 2e6]"
    );
    let (v, d) = test_vectors(N, 22);
    let mut entries: Vec<Stencil> = Vec::new();
    for i in 0..N {
        for j in i + 1..N {
            entries.push(Stencil::Entry(i, j));
        }
    }
    entries.extend([Stencil::Entry(0, 0), Stencil::Entry(N / 2, N / 2)]);
    let n_entries = entries.len();
    let mk = |name, stencils, entry_params, carrier| {
        let (gk, row_nnz) = operator_of(&mat);
        Case {
            name,
            gk,
            eig: jacobi_eigh(&mat),
            row_nnz,
            stencils,
            entry_params,
            d: d.clone(),
            v: v.clone(),
            carrier,
        }
    };
    let edge_st = edges.iter().map(|&(i, j)| Stencil::Edge(i, j)).collect();
    let entry_carrier = Carrier {
        focus: 0..n_entries,
        min_informative: 1,
    };
    (
        mk("F2-edge", edge_st, false, None),
        mk("F2-entry", entries, true, Some(entry_carrier)),
    )
}

/// `λ_max t = lt` time of a static case.
fn time_of(case: &Case, lt: f64) -> f64 {
    lt / case.eig.lam.last().expect("eigenvalues")
}

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_frechet_large_t_oracle() {
    let (f2_edge, f2_entry) = f2_cases();
    let statics = [f1_case(), f2_edge, f2_entry];
    let mut fails = Vec::new();
    for lt in GRID {
        for case in &statics {
            fails.extend(check_point(case, time_of(case, lt), lt).fails);
        }
        let (w, t) = f3_point(lt);
        println!("F3 point lt={lt:e}: W={w:.4e} t={t:.4e}");
        fails.extend(check_point(&f3_case(w), t, lt).fails);
    }
    assert!(
        fails.is_empty(),
        "{} violations:\n{}",
        fails.len(),
        fails.join("\n")
    );
}

/// A NaN anywhere in the comparison is a violation, never a silent pass.
#[test]
fn nan_is_a_violation() {
    assert!(violates(f64::NAN, 1.0));
    assert!(violates(1.0, f64::NAN));
    assert!(violates(f64::INFINITY, f64::INFINITY * 0.0));
    assert!(!violates(0.5, 1.0));
    assert!(nan_max(1.0, f64::NAN).is_nan());
    assert!(nan_max(f64::NAN, 1.0).is_nan());
    // End-to-end: a NaN gradient entry injected into `score` fails the gate.
    let case = f3_case(1.0);
    let t = time_of(&case, 1.0);
    let refs = oracle(&case.eig, &case.d, &case.v, t, &case.stencils);
    let (bound, _) = bound_for(&case, t);
    let mut grad = grad_of(&case, t);
    let legacy = legacy_gl8(&case.eig, &case.d, &case.v, t, &case.stencils);
    let (_, clean) = score(&case, (&bound, 1.0), &grad, &refs, &legacy);
    assert_eq!(clean, Vec::<String>::new(), "clean gradient must pass");
    grad[3] = f64::NAN;
    let (row, fails) = score(&case, (&bound, 1.0), &grad, &refs, &legacy);
    assert_eq!(fails.len(), 1, "NaN gradient entry must be reported");
    assert!(
        row.err_vs_tau.is_nan(),
        "NaN must propagate into the row maximum"
    );
}
