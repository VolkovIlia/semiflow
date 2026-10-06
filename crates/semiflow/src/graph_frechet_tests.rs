//! In-crate gates of `graph_frechet.rs` (ADR-0203 Amendment 2, math §63.7.b).
//!
//! - `g_frechet_sweep_exact_prop` (`G_FRECHET_SWEEP_EXACT_PROP`, `RELEASE_BLOCKING`):
//!   the library's OWN sweep, driven by the exact-eigenvalue propagator
//!   `P_E(τ)x = Ṽ(e^{−τλ̃} ⊙ Ṽᵀx)` built from the oracle's Jacobi decomposition,
//!   against `τ_k^E`. Oracle and propagator describe the same matrix, so the
//!   eigensolver backward error cancels and no `ρ̄t` floor is needed.
//! - `production_bit_identical_after_refactor`: the public `graph_expmv_frechet`
//!   still produces the bits recorded before the sweep was made generic.
//!
//! The helper module is shared with the integration gates under `tests/`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)] // test arithmetic on small indices/counts (< 2^32), never on user data

#[path = "../tests/frechet_oracle_common/mod.rs"]
mod common;

use std::{cell::Cell, eprintln, ops::Range, string::String, sync::Arc, vec::Vec};

use common::{
    f3_matrix, f3_point, f3_signals, jacobi_eigh, legacy_gl8, nan_max, norm2, oracle, propagate,
    random_graph, stiff_contrast, test_vectors, violates, Dense, Eig, Stencil, EPS_Q, F3_BRIDGES,
    U,
};

use crate::{
    error::SemiflowError,
    graph_expmv_frechet, graph_expmv_frechet_plan,
    graph_frechet::{frechet_sweep, FrechetPropagator, SweepParams},
    graph_sensitivity::GeneratorSensitivity,
    EdgeWeightSensitivity, EntrySensitivity, Graph, GraphKrylovChernoff, KrylovPath, Laplacian,
    ScratchPool,
};

#[path = "graph_frechet_skip_tests.rs"]
mod skip;

const N: usize = 12;
const GRID: [f64; 5] = [1.0, 10.0, 1e2, 1e4, 1e6];

/// `P_E(τ)x = Ṽ(e^{−τλ̃} ⊙ Ṽᵀx)` from the oracle's own decomposition; counts calls.
struct ExactProp<'a> {
    eig: &'a Eig,
    calls: Cell<usize>,
}

impl FrechetPropagator<f64> for ExactProp<'_> {
    fn propagate(
        &self,
        tau: f64,
        src: &[f64],
        dst: &mut [f64],
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        self.calls.set(self.calls.get() + 1);
        dst.copy_from_slice(&propagate(self.eig, tau, src));
        Ok(())
    }
}

struct Fixture {
    name: &'static str,
    eig: Eig,
    /// Gershgorin bound of the dense matrix (what production caches).
    rho: f64,
    stencils: Vec<Stencil>,
    entry_params: bool,
    d: Vec<f64>,
    v: Vec<f64>,
    /// `(parameters that must be informative, minimum count)`; `None`: accuracy only.
    carrier: Option<(Range<usize>, usize)>,
}

fn gershgorin(mat: &Dense) -> f64 {
    mat.iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn edge_stencils(pairs: impl IntoIterator<Item = (usize, usize)>) -> Vec<Stencil> {
    pairs
        .into_iter()
        .map(|(i, j)| Stencil::Edge(i, j))
        .collect()
}

fn f1_fixture(lt: f64) -> (Fixture, f64) {
    let (mat, edges) = random_graph(N, 1);
    let (v, d) = test_vectors(N, 11);
    let eig = jacobi_eigh(&mat);
    let t = lt / eig.lam[N - 1];
    let fx = Fixture {
        name: "F1-edge",
        rho: gershgorin(&mat),
        stencils: edge_stencils(edges.iter().map(|&(i, j, _)| (i, j))),
        entry_params: false,
        carrier: None,
        eig,
        d,
        v,
    };
    (fx, t)
}

/// F2 edge (accuracy only) and entry (carrier) fixtures.
fn f2_fixtures(lt: f64) -> [(Fixture, f64); 2] {
    let (mat, edges) = stiff_contrast(N, 3e3, 0);
    let (v, d) = test_vectors(N, 22);
    let t = lt / jacobi_eigh(&mat).lam[N - 1];
    let mut entries: Vec<Stencil> = Vec::new();
    for i in 0..N {
        for j in i + 1..N {
            entries.push(Stencil::Entry(i, j));
        }
    }
    entries.extend([Stencil::Entry(0, 0), Stencil::Entry(N / 2, N / 2)]);
    let n_entries = entries.len();
    let mk = |name, stencils, entry_params, carrier| Fixture {
        name,
        eig: jacobi_eigh(&mat),
        rho: gershgorin(&mat),
        stencils,
        entry_params,
        d: d.clone(),
        v: v.clone(),
        carrier,
    };
    let edge_st = edge_stencils(edges.iter().copied());
    [
        (mk("F2-edge", edge_st, false, None), t),
        (mk("F2-entry", entries, true, Some((0..n_entries, 1))), t),
    ]
}

/// F3 (Amendment 1): `W` and `t` solved per point.
fn f3_fixture(lt: f64) -> (Fixture, f64) {
    let (w, t) = f3_point(lt);
    let (mat, edges) = f3_matrix(w);
    let (v, d) = f3_signals();
    let fx = Fixture {
        name: "F3-edge",
        eig: jacobi_eigh(&mat),
        rho: gershgorin(&mat),
        stencils: edge_stencils(edges.iter().map(|&(i, j, _)| (i, j))),
        entry_params: false,
        d,
        v,
        carrier: Some((F3_BRIDGES, 2)),
    };
    (fx, t)
}

/// Fixtures of one `λ_max t = lt` point, each with its time `t`.
fn fixtures_at(lt: f64) -> Vec<(Fixture, f64)> {
    let [f2_edge, f2_entry] = f2_fixtures(lt);
    std::vec![f1_fixture(lt), f2_edge, f2_entry, f3_fixture(lt)]
}

/// The library sweep driven by `P_E` with skip predicate `skip`; returns gradient and call count.
fn sweep_with<S: Fn(f64, f64) -> bool>(fx: &Fixture, t: f64, skip: &S) -> (Vec<f64>, usize) {
    let pairs: Vec<(usize, usize)> = fx.stencils.iter().map(Stencil::pair).collect();
    let mut grad = std::vec![0.0; pairs.len()];
    let par = SweepParams {
        n: N,
        n_cols: 1,
        t,
        rho: fx.rho,
    };
    let mut scratch = ScratchPool::new();
    let prop = ExactProp {
        eig: &fx.eig,
        calls: Cell::new(0),
    };
    let cols = (fx.v.as_slice(), fx.d.as_slice());
    let run = if fx.entry_params {
        let sens = EntrySensitivity {
            entries: pairs,
            n_nodes: N,
        };
        frechet_sweep(&prop, &sens, skip, par, cols, &mut grad, &mut scratch)
    } else {
        let sens = EdgeWeightSensitivity {
            params: pairs,
            n_nodes: N,
        };
        frechet_sweep(&prop, &sens, skip, par, cols, &mut grad, &mut scratch)
    };
    run.expect("frechet_sweep");
    (grad, prop.calls.get())
}

/// Part A: the decay skip is OFF (the predicate never fires).
fn sweep_skip_off(fx: &Fixture, t: f64) -> Vec<f64> {
    sweep_with(fx, t, &|_, _| false).0
}

/// `τ_k^E` of §63.7.b; `ε_skip·N_k` is added for the skip-on bound of §63.7.c (`0` = skip off).
fn tau_e(big_g: f64, n_k: f64, panels: u32, n_nodes: u32, eps_skip: f64) -> f64 {
    let nn = N as f64;
    let kappa = 2.0 * nn.powf(1.5) + nn + 2.0;
    let l_e = 1.0 + 8.0 * f64::from(panels);
    (EPS_Q + 2.0 * nn * nn * U) * big_g
        + (2.0 * l_e * kappa + f64::from(n_nodes) + 4.0) * U * n_k
        + eps_skip * n_k
}

/// Outcome of one `(fixture, λ_max t)` part-A run.
#[derive(Default)]
struct Run {
    err_vs_tau: f64,
    legacy_vs_tau: f64,
    informative_focus: usize,
    legacy_violations: usize,
    fails: Vec<String>,
}

fn run_point(fx: &Fixture, t: f64, lt: f64) -> Run {
    let refs = oracle(&fx.eig, &fx.d, &fx.v, t, &fx.stencils);
    let grad = sweep_skip_off(fx, t);
    let legacy = legacy_gl8(&fx.eig, &fx.d, &fx.v, t, &fx.stencils);
    let plan = graph_expmv_frechet_plan(fx.rho, t, 1e-14, &KrylovPath::Chebyshev);
    let (nd, nv) = (norm2(&fx.d), norm2(&fx.v));
    let mut run = Run::default();
    for (k, st) in fx.stencils.iter().enumerate() {
        let (g_ref, big_g) = refs[k];
        let n_k = t * st.norm2() * nd * nv;
        let tau = tau_e(big_g, n_k, plan.panels_per_half, plan.n_nodes, 0.0);
        let err = (grad[k] - g_ref).abs();
        if violates(err, tau) {
            run.fails.push(format!(
                "{} lt={lt:e} k={k} err={err:e} > tau_E={tau:e}",
                fx.name
            ));
        }
        run.err_vs_tau = nan_max(run.err_vs_tau, err / tau);
        let legacy_err = (legacy[k] - g_ref).abs();
        run.legacy_vs_tau = nan_max(run.legacy_vs_tau, legacy_err / tau);
        run.legacy_violations += usize::from(violates(legacy_err, tau));
        let in_focus = fx.carrier.as_ref().is_some_and(|(r, _)| r.contains(&k));
        run.informative_focus += usize::from(in_focus && g_ref.abs() >= 1e3 * tau);
    }
    run
}

/// Part A (§63.7.b, skip off): F1/F2/F3 against `τ_k^E`, per-carrier non-vacuity.
fn part_a() -> Vec<String> {
    let mut fails: Vec<String> = Vec::new();
    for lt in GRID {
        for (fx, t) in fixtures_at(lt) {
            let run = run_point(&fx, t, lt);
            eprintln!(
                "SWEEP_E {:<9} lt={lt:>6.0e} max_err/tau_E={:9.2e} legacy_err/tau_E={:9.2e} \
                 informative_focus={} legacy_violations={}",
                fx.name,
                run.err_vs_tau,
                run.legacy_vs_tau,
                run.informative_focus,
                run.legacy_violations
            );
            fails.extend(run.fails);
            let Some((focus, min_informative)) = &fx.carrier else {
                continue;
            };
            if run.informative_focus < *min_informative {
                fails.push(format!(
                    "{} lt={lt:e}: {} informative in {focus:?}, need {min_informative}",
                    fx.name, run.informative_focus
                ));
            }
            if lt >= 10.0 && run.legacy_violations == 0 {
                fails.push(format!(
                    "{} lt={lt:e}: legacy rule does not violate tau_E (vacuous)",
                    fx.name
                ));
            }
        }
    }
    fails
}

/// `G_FRECHET_SWEEP_EXACT_PROP` (§63.7.b part A, §63.7.c part B).
#[test]
fn g_frechet_sweep_exact_prop() {
    let mut fails = part_a();
    fails.extend(skip::part_b());
    assert!(
        fails.is_empty(),
        "{} violations:\n{}",
        fails.len(),
        fails.join("\n")
    );
}

/// NaN is a violation of the sweep gate comparison and survives the maxima.
#[test]
fn sweep_gate_rejects_nan() {
    let tau = tau_e(1.0, 1.0, 32, 512, 0.0);
    assert!(violates(f64::NAN, tau));
    assert!(violates(1.0, f64::NAN));
    assert!(nan_max(0.5, f64::NAN).is_nan());
    let (fx, t) = fixtures_at(10.0).pop().expect("F3 fixture");
    let fails = run_point(&fx, t, 10.0).fails;
    assert_eq!(
        fails,
        Vec::<String>::new(),
        "F3 at lt=10 must pass the sweep gate"
    );
}

// ---------------------------------------------------------------------------
// Production path is bit-identical after the generic refactor
// ---------------------------------------------------------------------------

/// FNV-1a 64 over the bits of `xs`.
fn fnv(xs: &[f64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Gradient bits of the public API on F3 at `W = 3.6013`, `t = 6.8096`
/// (`λ_max t ≈ 1e2`, 160 nodes), Edge then Entry sensitivities, Chebyshev `1e-12`.
fn production_hash() -> u64 {
    let edges = f3_matrix(3.6013).1;
    let triples = edges.iter().map(|&(i, j, w)| (i as u32, j as u32, w));
    let g = Arc::new(Graph::<f64>::from_edges(N, triples).expect("graph"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let gk = GraphKrylovChernoff::new(lap, KrylovPath::Chebyshev, 1e-12).expect("gk");
    let u0: Vec<f64> = (0..12).map(|i| (0.7 * f64::from(i) + 0.3).sin()).collect();
    let dj: Vec<f64> = (0..12).map(|i| (0.4 * f64::from(i)).cos()).collect();
    let mut all = Vec::new();
    let edge = EdgeWeightSensitivity {
        params: edges.iter().map(|&(i, j, _)| (i, j)).collect(),
        n_nodes: N,
    };
    all.extend(public_grad(&gk, &u0, &dj, &edge));
    let entry = EntrySensitivity {
        entries: std::vec![(0, 1), (2, 3), (5, 5), (0, 11)],
        n_nodes: N,
    };
    all.extend(public_grad(&gk, &u0, &dj, &entry));
    fnv(&all)
}

fn public_grad<P: GeneratorSensitivity<f64>>(
    gk: &GraphKrylovChernoff<f64>,
    u0: &[f64],
    dj: &[f64],
    sens: &P,
) -> Vec<f64> {
    let mut grad = std::vec![0.0; sens.n_params()];
    graph_expmv_frechet(
        gk,
        u0,
        dj,
        1,
        6.8096,
        sens,
        &mut grad,
        &mut ScratchPool::new(),
    )
    .expect("graph_expmv_frechet");
    grad
}

/// Hash recorded from the pre-refactor implementation (same fixture, same flags).
const PRE_REFACTOR_HASH: u64 = 0x6d87_193c_3a06_6bb0;

#[test]
fn production_bit_identical_after_refactor() {
    assert_eq!(
        production_hash(),
        PRE_REFACTOR_HASH,
        "graph_expmv_frechet results changed bitwise after the sweep refactor"
    );
}
