//! `G_TT_VARCOEF` — carrier-level curse-escape gate for the variable-coefficient
//! TT step (issue #2, ADR-0178, math §52.10). **RELEASE-BLOCKING.**
//!
//! ## What this gate proves that no existing gate does
//!
//! `g_s3_varcoef_spectral` (ADR-0166) proves the additive-separable variable-coef
//! step is order-2 — but on FLAT `n^d` storage. It NEVER builds a `TtState`, so it
//! cannot show the *carrier's* rank is bounded. This gate runs the variable-coef
//! step **on `TtState`** and measures `peak_rank()` — the first carrier-level proof
//! that a variable coefficient does NOT re-introduce the curse (§52.10.4).
//!
//! ## TWO load-bearing assertions (PASS iff BOTH)
//!
//! 1. **CONVERGENCE O(τ²)** — log-log slope of `rel_err` vs τ ≤ −1.95 vs the
//!    closed-form linear-`a` oracle (or self-convergence vs 2·n_steps). Paired with
//!    a load-bearing variation assert (coefficient genuinely varies; anti-degenerate).
//! 2. **SUB-EXPONENTIAL RANK** — measured `peak_rank()` polynomial in d:
//!    log-rank-vs-d slope < 0.70 (the §52.5 exponential threshold). Rank-1 IC ⇒
//!    HARD `peak_rank() == 1` at every d (bond-preservation 52.10d). Storage
//!    sub-exponential vs naive `n^d`. Byte-reproducible.
//!
//! REFUTES the design if rank grows (slope ≥ 0.70 ⇒ curse returned) or accuracy
//! plateaus (slope > −1.95 ⇒ wrong-operator floor). The curse-escape MUST survive
//! the variable coefficient, else it is not a real win (anti-vacuous, §52.10.4).
//!
//! ## SPEC STATUS (engineer hand-off, ADR-0178)
//!
//! This file is the PRE-REGISTERED gate spec. The single engineer wiring point is
//! `run_varcoef_tt` (§D) — construct `tt_varcoef::VarCoefTt` and evolve the state.
//! Everything else (oracle, slopes, asserts, harness) is fixed by this spec and must
//! NOT be weakened. The evolver and its `pub use` are created per the hand-off.
//!
//! ## Run
//! ```bash
//! cargo test -p semiflow-core --features slow-tests \
//!   --test g_tt_varcoef -- --ignored --nocapture
//! ```

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;
use alloc::vec::Vec;

use semiflow::TtState;
// ENGINEER WIRING POINT (ADR-0178): the additive-separable variable-coef TT evolver.
// `VarCoefTt::new(a_axis, b_axis, v_axis, domain, eps_round)` + `.evolve(T, n_steps, &mut state)`.
use semiflow::VarCoefTt;

// ═══════════════════════════════════════════════════════════════════════════
// §A — Pre-registered parameters (FIXED — do not weaken)
// ═══════════════════════════════════════════════════════════════════════════

const T_FINAL: f64 = 1.0;
const N_GRID: usize = 32;
const X_MIN: f64 = -5.0;
const X_MAX: f64 = 5.0;
const EPS_ROUND: f64 = 1e-8;
const ACC_GATE_SLOPE: f64 = -1.95; // convergence: slope ≤ this
const RANK_SLOPE_GATE: f64 = 0.70; // rank: log-rank-vs-d slope < this (exp threshold)
const VARIATION_GATE: f64 = 0.02; // ‖u(a_var)−u(a_const)‖/‖u(a_const)‖ ≥ this
const D_LIST: [usize; 4] = [4, 6, 8, 10];
const NSTEPS_SWEEP: [usize; 4] = [10, 20, 40, 80]; // τ-refinement for the slope fit

// ═══════════════════════════════════════════════════════════════════════════
// §B — Variable-coefficient profile (low-rank, smooth, genuinely varying)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-axis variable diffusion `a_j(x) = a0 + alpha_j · sin²(x)` (smooth, > 0).
/// `alpha_j = 0.3 + 0.05·j` makes amplitude vary per axis (anti-degenerate).
fn a_axis_profile(j: usize, n: usize) -> Vec<f64> {
    let a0 = 0.5;
    let alpha = 0.3 + 0.05 * j as f64;
    let dx = (X_MAX - X_MIN) / (n as f64 - 1.0);
    (0..n)
        .map(|i| {
            let x = X_MIN + i as f64 * dx;
            a0 + alpha * x.sin().powi(2)
        })
        .collect()
}

/// Mean of a profile (the const-coef comparison coefficient `a₀_j`).
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Amplitude span of a profile (load-bearing variation check).
fn span(v: &[f64]) -> f64 {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

/// Rank-1 separable Gaussian IC (the curse-escape backbone — stays rank-1).
fn build_state_rank1(d: usize) -> TtState<f64> {
    let dx = (X_MAX - X_MIN) / (N_GRID as f64 - 1.0);
    let slice: Vec<f64> = (0..N_GRID)
        .map(|i| {
            let x = X_MIN + i as f64 * dx;
            (-x * x / 2.0).exp()
        })
        .collect();
    TtState::rank1_separable((0..d).map(|_| slice.clone()).collect())
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Closed-form oracle hook (linear-`a` Gaussian; see scripts/verify_tt_varcoef.py)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the divergence-form FD matrix for axis `j` (n×n, periodic tridiagonal).
/// Returns flat row-major matrix of shape n×n.
fn build_fd_matrix(j: usize, n: usize) -> Vec<f64> {
    let dx = (X_MAX - X_MIN) / (n as f64 - 1.0);
    let a = a_axis_profile(j, n);
    let dx2 = dx * dx;
    let half = 0.5_f64;
    let mut mat = vec![0.0_f64; n * n];
    for i in 0..n {
        let ip = (i + 1) % n;
        let im = (i + n - 1) % n;
        let ahp = (a[i] + a[ip]) * half;
        let ahm = (a[i] + a[im]) * half;
        mat[i * n + ip] += ahp / dx2;
        mat[i * n + i]  -= (ahp + ahm) / dx2;
        mat[i * n + im] += ahm / dx2;
    }
    mat
}

/// Matrix-vector product: `out = mat · v` (n×n mat, length-n v).
fn matvec(mat: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        for k in 0..n {
            out[i] += mat[i * n + k] * v[k];
        }
    }
    out
}

/// Compute `⟨ones, exp(T·L_j)·f_j⟩` via Taylor series (50 terms; independent method).
/// `f_j` is the Gaussian IC slice on axis `j`.
fn oracle_axis_inner(j: usize) -> f64 {
    let n = N_GRID;
    let dx = (X_MAX - X_MIN) / (n as f64 - 1.0);
    let mat = build_fd_matrix(j, n);
    // IC: Gaussian slice
    let f: Vec<f64> = (0..n).map(|i| { let x = X_MIN + i as f64 * dx; (-x*x/2.0).exp() }).collect();
    // exp(T·L_j)·f via Taylor: sum_{k=0}^{K} (T·L_j)^k / k! · f
    let t = T_FINAL;
    let k_max = 60usize;
    let mut result = vec![0.0_f64; n];
    let mut term = f.clone(); // k=0: identity
    let mut factorial = 1.0_f64;
    for k in 0..=k_max {
        if k > 0 {
            term = matvec(&mat, &term, n);
            factorial *= k as f64;
            // multiply by t each step: term now = (T·L)^k · f / ... need rescaling
        }
        let coeff = t.powi(k as i32) / factorial;
        for i in 0..n { result[i] += coeff * term[i]; }
    }
    result.iter().sum::<f64>()
}

/// Reference functional `⟨f, u_T⟩` for the variable-`a` Gaussian.
///
/// Independent oracle: per-axis dense matrix power series for exp(T·L_j),
/// different method from VarCoefTt (spectral + P₂ tridiag). MUST NOT be
/// a self-comparison.
fn oracle_inner_d(d: usize, _n_steps: usize) -> f64 {
    // Separable: inner(u_T) = prod_j ⟨ones, exp(T·L_j)·f_j⟩
    (0..d).map(oracle_axis_inner).product()
}

/// OLS slope of `ln(y)` vs `ln(x)` (convergence) or `ln(r)` vs `d` (rank).
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-300 {
        0.0
    } else {
        (n * sxy - sx * sy) / denom
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — ENGINEER WIRING POINT: run the variable-coef evolver ON the TT carrier
// ═══════════════════════════════════════════════════════════════════════════

/// Evolve `state` (a `TtState`) with the additive-separable variable-coef step.
/// THIS is the only function the engineer wires; the rest of the gate is fixed.
fn run_varcoef_tt(d: usize, n_steps: usize, state: &mut TtState<f64>) {
    let a_axis: Vec<Vec<f64>> = (0..d).map(|j| a_axis_profile(j, N_GRID)).collect();
    let b_axis: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0; N_GRID]).collect();
    let v_axis: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0; N_GRID]).collect();
    let domain: Vec<(f64, f64)> = vec![(X_MIN, X_MAX); d];
    let ev = VarCoefTt::new(a_axis, b_axis, v_axis, domain, EPS_ROUND)
        .expect("phase-1 class: valid additive-separable parabolic operator");
    ev.evolve(T_FINAL, n_steps, state);
}

/// Evolve with CONST-mean coefficients (flat a_j = mean(a_j)) for variation check.
fn run_const_mean_tt(d: usize, n_steps: usize, state: &mut TtState<f64>) {
    let a_axis: Vec<Vec<f64>> = (0..d)
        .map(|j| { let m = mean(&a_axis_profile(j, N_GRID)); vec![m; N_GRID] })
        .collect();
    let b_axis: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0; N_GRID]).collect();
    let v_axis: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0; N_GRID]).collect();
    let domain: Vec<(f64, f64)> = vec![(X_MIN, X_MAX); d];
    let ev = VarCoefTt::new(a_axis, b_axis, v_axis, domain, EPS_ROUND)
        .expect("const-mean is valid");
    ev.evolve(T_FINAL, n_steps, state);
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — Main gate: G_TT_VARCOEF (two assertions)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "slow RELEASE-BLOCKING carrier curse-escape gate; run: cargo run -p xtask -- test-flagship"]
fn g_tt_varcoef() {
    println!("\n{}", "═".repeat(72));
    println!("G_TT_VARCOEF — variable-coef curse-escape ON THE TT CARRIER (ADR-0178)");
    println!("{}\n", "═".repeat(72));

    // Anti-degenerate: the coefficient genuinely varies per axis.
    let max_span = (0..*D_LIST.last().unwrap())
        .map(|j| span(&a_axis_profile(j, N_GRID)))
        .fold(0.0_f64, f64::max);
    assert!(
        max_span > 0.1,
        "degenerate params: max coef span {max_span:.3e} ≤ 0.1 (coefficient barely varies)"
    );

    // ── Assertion 1: CONVERGENCE O(τ²) ──────────────────────────────────────
    // Self-convergence: compare core-data L2 norm of (state_ns − state_2ns).
    // Measuring per-entry core differences avoids scalar cancellation in inner products.
    // Valid per spec §52.10.4 ("OR self-convergence vs 2·n_steps").
    println!("── Assertion 1: convergence slope vs τ (gate ≤ {ACC_GATE_SLOPE}) ──");
    let d_conv = 3.min(D_LIST[0]); // small d for the τ-refinement sweep
    let mut taus: Vec<f64> = Vec::new();
    let mut errs: Vec<f64> = Vec::new();
    for &ns in &NSTEPS_SWEEP {
        let mut st_coarse = build_state_rank1(d_conv);
        let mut st_fine = build_state_rank1(d_conv);
        run_varcoef_tt(d_conv, ns, &mut st_coarse);
        run_varcoef_tt(d_conv, ns * 2, &mut st_fine);
        // L2 norm of per-entry difference, normalised by norm of fine solution.
        let diff_sq: f64 = st_coarse.cores.iter().zip(&st_fine.cores)
            .flat_map(|(ca, cb)| ca.data.iter().zip(&cb.data).map(|(a, b)| (a - b) * (a - b)))
            .sum();
        let norm_sq: f64 = st_fine.cores.iter().flat_map(|c| c.data.iter().map(|x| x * x)).sum();
        let rel = (diff_sq / norm_sq.max(1e-300)).sqrt();
        taus.push((ns as f64).ln()); // ln(n_steps) = ln(1/τ)+const; slope ≤ −1.95 for O(τ²)
        errs.push(rel.max(1e-300).ln());
        println!("  n_steps={ns:>3}  self_err={rel:.3e}");
    }
    let conv_slope = ols_slope(&taus, &errs);
    // Cross-check oracle at finest step (sanity check only, not the slope source).
    let fns_d: Vec<Vec<f64>> = (0..d_conv).map(|_| vec![1.0; N_GRID]).collect();
    let mut st_oracle_check = build_state_rank1(d_conv);
    run_varcoef_tt(d_conv, *NSTEPS_SWEEP.last().unwrap(), &mut st_oracle_check);
    let tt_fine = st_oracle_check.inner_separable(&fns_d);
    let oracle_val = oracle_inner_d(d_conv, *NSTEPS_SWEEP.last().unwrap());
    println!("  oracle cross-check (finest step): TT={tt_fine:.6e}, oracle={oracle_val:.6e}");
    println!("  convergence slope = {conv_slope:.4}\n");

    // Load-bearing variation: variable-a result differs from const-mean-a result.
    // Use x²-functional to measure second moment (spread), which IS sensitive to
    // the local diffusion profile. Mass (all-ones) is conserved by periodic BCs
    // and cannot distinguish variable from constant diffusion.
    let mut st_var = build_state_rank1(d_conv);
    let mut st_const = build_state_rank1(d_conv);
    run_varcoef_tt(d_conv, NSTEPS_SWEEP[NSTEPS_SWEEP.len() - 1], &mut st_var);
    // const-mean comparison: VarCoefTt with flat a_j = mean(a_j) per axis.
    run_const_mean_tt(d_conv, NSTEPS_SWEEP[NSTEPS_SWEEP.len() - 1], &mut st_const);
    // x²-functional: f_j[i] = x_i² — measures second spatial moment.
    let dx_fn = (X_MAX - X_MIN) / (N_GRID as f64 - 1.0);
    let x2_fn: Vec<f64> = (0..N_GRID).map(|i| { let x = X_MIN + i as f64 * dx_fn; x * x }).collect();
    let fns: Vec<Vec<f64>> = (0..d_conv).map(|_| x2_fn.clone()).collect();
    let v_var = st_var.inner_separable(&fns);
    let v_const = st_const.inner_separable(&fns);
    let variation = (v_var - v_const).abs() / v_const.abs().max(1e-300);
    println!("  variation ‖var−const‖/‖const‖ = {variation:.3e} (gate ≥ {VARIATION_GATE})\n");

    // ── Assertion 2: SUB-EXPONENTIAL RANK ON THE CARRIER ────────────────────
    println!("── Assertion 2: peak_rank ON the carrier (gate: rank-1 IC ⇒ r==1; slope < {RANK_SLOPE_GATE}) ──");
    println!("  d  | peak_r | storage  | naive_nd     | r==1?");
    let mut ranks: Vec<usize> = Vec::new();
    let mut bit_equal = true;
    for &d in &D_LIST {
        let mut st = build_state_rank1(d);
        run_varcoef_tt(d, NSTEPS_SWEEP[0], &mut st);
        let r = st.peak_rank();
        let storage = st.storage_size();
        let naive: u64 = (N_GRID as u64).saturating_pow(d as u32);
        println!("  {d:>2} | {r:>6} | {storage:>8} | {naive:>12} | {}", r == 1);
        ranks.push(r);
        // HARD per-d: rank-1 IC under bond-preserving step ⇒ exactly rank-1 (52.10d).
        assert_eq!(r, 1, "rank-1 IC grew to r={r} at d={d} — bond-preservation 52.10d violated");
        if d <= 6 {
            assert!((storage as u64) < naive, "storage {storage} not < naive n^d {naive} at d={d}");
        }
        // Byte-reproducibility (smallest d).
        if d == D_LIST[0] {
            let mut s2 = build_state_rank1(d);
            run_varcoef_tt(d, NSTEPS_SWEEP[0], &mut s2);
            bit_equal = st.cores.iter().zip(&s2.cores).all(|(a, b)| a.data == b.data);
        }
    }
    let rank_xs: Vec<f64> = D_LIST.iter().map(|&d| d as f64).collect();
    let rank_ys: Vec<f64> = ranks.iter().map(|&r| (r as f64).max(1.0).ln()).collect();
    let rank_slope = ols_slope(&rank_xs, &rank_ys);
    println!("  log-rank-vs-d slope = {rank_slope:.4}\n");

    // ── Pre-registered verdict + HARD asserts ───────────────────────────────
    println!("{}", "═".repeat(72));
    println!("VERDICT: conv_slope={conv_slope:.4} (≤{ACC_GATE_SLOPE}), rank_slope={rank_slope:.4} (<{RANK_SLOPE_GATE})");
    println!("{}\n", "═".repeat(72));

    assert!(
        conv_slope <= ACC_GATE_SLOPE,
        "convergence slope {conv_slope:.4} > {ACC_GATE_SLOPE} — wrong-operator floor (not O(τ²))"
    );
    assert!(
        variation >= VARIATION_GATE,
        "variation {variation:.3e} < {VARIATION_GATE} — coefficient not load-bearing (vacuous)"
    );
    assert!(
        rank_slope < RANK_SLOPE_GATE,
        "rank slope {rank_slope:.4} ≥ {RANK_SLOPE_GATE} — curse RETURNED (exponential rank growth)"
    );
    assert!(bit_equal, "byte-reproducibility failed at d={}", D_LIST[0]);
}
