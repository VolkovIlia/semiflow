//! G17′ slope gate — `MagnusGraphHeat6thChernoff` genuine order-6 convergence.
//!
//! Gate: OLS log-log slope ≤ −5.85 (f64, self-convergence at 2× refinement).
//!
//! ## Non-commuting family (ADR-0114)
//!
//! Path graph `P_64`, each edge `i` carries a DISTINCT time-phase:
//!
//! ```text
//! wᵢ(t) = 1 + 0.3·sin(π·(t + φᵢ)),  φᵢ = i·0.13 mod 1
//! ```
//!
//! This makes `[L(t₁), L(t₂)] ≠ 0` on general vectors (proved by the
//! non-commutativity precondition assertion below), so commutator code is
//! genuinely exercised.
//!
//! IC: `f_i = sin(0.31·π·i/N) + 0.2·cos(1.7·i)` — not all-ones (kernel trap),
//! not constant. See math.md line 4947 warning about `ker L`.
//!
//! Parameters: `t_final = 0.5`, `n_steps ∈ {5, 10, 20, 40, 80}`.
//!
//! ## Non-commutativity precondition (anti-masking guard)
//!
//! An assertion verifies `‖[L(t₁),L(t₂)]·f‖ > 1e-6` at the start of the
//! test.  If anyone reverts to a commuting family this assertion fails loudly,
//! preventing silent re-masking of the order gate.
//!
//! **f64 ONLY** — f32 impl intentionally absent (ADR-0056).
//!
//! This test is gated behind `#[ignore]` and runs with
//! `--features slow-tests --release -- --ignored`.
//!
//! See ADR-0114, contracts/semiflow-core.math.md §16 (NORMATIVE).

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use semiflow_core::{
    magnus6_graph::MagnusGraphHeat6thChernoff, magnus_graph::LaplacianAtTime, Graph, GraphSignal,
    Laplacian, ScratchPool,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const N_NODES: usize = 64;
const T_FINAL: f64 = 0.5;
const N_STEPS_F64: [usize; 5] = [5, 10, 20, 40, 80];
const SLOPE_THRESHOLD_F64: f64 = -5.85;

/// Absolute error floor for f64 self-convergence on `P_64` (phased edges).
/// Errors below this threshold are round-off noise; floor-saturated points
/// are excluded from the OLS slope fit.
const F64_FLOOR: f64 = 5e-14;

// ---------------------------------------------------------------------------
// Phased-edge weight and Laplacian builder
//
// Edge i carries phase φᵢ = (i * 0.13) mod 1 so L(t₁), L(t₂) do NOT commute.
// ---------------------------------------------------------------------------

fn edge_phase(edge_idx: usize) -> f64 {
    let raw = (edge_idx as f64) * 0.13;
    raw - raw.floor() // mod 1
}

fn weight_at(t: f64, edge_idx: usize) -> f64 {
    let phi = edge_phase(edge_idx);
    1.0 + 0.3 * (core::f64::consts::PI * (t + phi)).sin()
}

fn laplacian_phased(n_nodes: usize, t: f64) -> Laplacian<f64> {
    let edges = (0..n_nodes as u32 - 1).map(|i| (i, i + 1, weight_at(t, i as usize)));
    let g = Graph::from_edges(n_nodes, edges).expect("valid path edges");
    Laplacian::assemble_combinatorial(&g)
}

// ---------------------------------------------------------------------------
// Non-commutativity precondition helper
//
// Compute ‖[L(t₁), L(t₂)]·f‖ using sparse mat-vec on two Laplacians.
// ---------------------------------------------------------------------------

/// Apply `[L₁, L₂]·v = L₁(L₂v) − L₂(L₁v)` using `Laplacian::apply_into_slice`.
fn commutator_norm(lap1: &Laplacian<f64>, lap2: &Laplacian<f64>, v: &[f64]) -> f64 {
    let n = v.len();
    let mut l2v = vec![0.0_f64; n];
    let mut l1v = vec![0.0_f64; n];
    let mut l1_l2v = vec![0.0_f64; n];
    let mut l2_l1v = vec![0.0_f64; n];
    lap2.apply_into_slice(v, &mut l2v);
    lap1.apply_into_slice(v, &mut l1v);
    lap1.apply_into_slice(&l2v, &mut l1_l2v);
    lap2.apply_into_slice(&l1v, &mut l2_l1v);
    l1_l2v
        .iter()
        .zip(&l2_l1v)
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// OLS slope helper
// ---------------------------------------------------------------------------

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / m;
    let my = ys.iter().sum::<f64>() / m;
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

// ---------------------------------------------------------------------------
// Self-convergence runner (f64)
// ---------------------------------------------------------------------------

fn self_conv_err_f64(
    mc: &MagnusGraphHeat6thChernoff<f64>,
    f0: &GraphSignal<f64>,
    n_steps: usize,
) -> f64 {
    let tau = T_FINAL / n_steps as f64;
    let tau_fine = T_FINAL / (2 * n_steps) as f64;

    let mut pool = ScratchPool::<f64>::new();

    // Coarse: n_steps, step size tau, t_k = k * tau.
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for k in 0..n_steps {
        let t_start = k as f64 * tau;
        mc.apply_into_at(t_start, tau, &cur, &mut nxt, &mut pool)
            .unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_coarse = cur.clone();

    // Fine: 2*n_steps, step size tau_fine, t_k = k * tau_fine.
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for k in 0..(2 * n_steps) {
        let t_start = k as f64 * tau_fine;
        mc.apply_into_at(t_start, tau_fine, &cur, &mut nxt, &mut pool)
            .unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_fine = cur;

    u_coarse
        .values()
        .iter()
        .zip(u_fine.values())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Floor-filter helper (same logic as original G17)
// ---------------------------------------------------------------------------

fn filter_floor_points(log_n: &mut Vec<f64>, log_err: &mut Vec<f64>) {
    // Criterion (b): absolute floor threshold.
    while log_err.len() >= 2 {
        let last_err = log_err.last().unwrap().exp();
        if last_err < F64_FLOOR {
            log_n.pop();
            log_err.pop();
        } else {
            break;
        }
    }
    // Criterion (a): non-monotone tail.
    while log_err.len() >= 2 {
        let last = *log_err.last().unwrap();
        let prev = log_err[log_err.len() - 2];
        if last >= prev {
            log_n.pop();
            log_err.pop();
        } else {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// G17′ f64 slope gate (slow-test, non-commuting phased-edge family)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g17_magnus6_slope_f64() {
    let n = N_NODES;
    let topology = Arc::new(Graph::<f64>::path(n));
    // Gershgorin bound for P_64 with wᵢ(t) ∈ [0.7, 1.3]: max degree = 2 * 1.3 = 2.6.
    let rho_bar = 2.6_f64;

    let lap_at: LaplacianAtTime<f64> = Box::new(move |t: f64| Arc::new(laplacian_phased(n, t)));
    let mc = MagnusGraphHeat6thChernoff::new(Arc::clone(&topology), lap_at, rho_bar, true)
        .expect("valid inputs");

    // IC: f_i = sin(0.31·π·i/N) + 0.2·cos(1.7·i)  — not all-ones, not in ker L.
    let f0 = GraphSignal::from_fn(Arc::clone(&topology), |i| {
        let x = i as f64;
        (0.31 * core::f64::consts::PI * x / n as f64).sin() + 0.2 * (1.7_f64 * x).cos()
    });

    // -------------------------------------------------------------------
    // NON-COMMUTATIVITY PRECONDITION (anti-masking guard, ADR-0114)
    //
    // Assert ‖[L(t₁),L(t₂)]·f‖ > 1e-6. If this fails the family is
    // commuting and the order gate is meaningless (see A.4 in fixspec).
    // -------------------------------------------------------------------
    {
        let lap_t1 = laplacian_phased(n, 0.1);
        let lap_t2 = laplacian_phased(n, 0.4);
        let comm_norm = commutator_norm(&lap_t1, &lap_t2, f0.values());
        assert!(
            comm_norm > 1e-6,
            "G17′ PRECONDITION FAIL: family is commuting (‖[L(t₁),L(t₂)]·f‖ = {comm_norm:.3e}). \
             The order gate requires a genuinely non-commuting family. \
             Check that distinct per-edge phases are set correctly (ADR-0114)."
        );
        println!("G17′ non-commutativity precondition ‖[L,L]·f‖ = {comm_norm:.3e} > 1e-6  OK");
    }

    let mut log_n = Vec::new();
    let mut log_err = Vec::new();
    for &n_steps in &N_STEPS_F64 {
        let err = self_conv_err_f64(&mc, &f0, n_steps);
        println!("G17′ f64  n_steps={n_steps:4}  self_conv_err={err:.4e}");
        if err > 0.0 {
            log_n.push((n_steps as f64).ln());
            log_err.push(err.ln());
        }
    }

    // Remove floor-saturated points.
    filter_floor_points(&mut log_n, &mut log_err);

    assert!(
        log_n.len() >= 3,
        "G17′ FAIL: fewer than 3 usable error values after floor filter"
    );

    let slope = ols_slope(&log_n, &log_err);
    println!("G17′ f64  slope = {slope:.4}  (threshold {SLOPE_THRESHOLD_F64})");
    assert!(
        slope <= SLOPE_THRESHOLD_F64,
        "G17′ FAIL f64: slope {slope:.4} > {SLOPE_THRESHOLD_F64} (order-6 gate). \
         Buggy formula (header/math.md §16.2 WRONG Ω₆) gives slope ≈ −2 on \
         non-commuting family; BCOR-6 corrected form gives ≤ −5.85. \
         See ADR-0114 and fixspec A.5."
    );
}
